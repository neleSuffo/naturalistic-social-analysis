import librosa
import os
import datetime
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GRU, Dense, Dropout, BatchNormalization, Reshape, Permute, multiply, Lambda, Activation, Flatten, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, f1_score # Added for detailed test evaluation
from collections import Counter # For calculating class weights

# --- Custom Keras Metrics for Macro-Average F1, Precision, Recall ---

class MacroPrecision(tf.keras.metrics.Metric):
    def __init__(self, name='macro_precision', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        # Initialize internal state to indicate if variables are built
        self._built = False 

    # The build method will be called by Keras.
    def build(self, input_shape):
        # Prevent rebuilding if already built
        if self._built: 
            return
        
        num_classes = input_shape[-1]
        if num_classes is None:
            # If num_classes is None here (e.g., from model.summary()), we can't build yet.
            # We'll build lazily in update_state when concrete shapes are available.
            return

        self.num_classes = num_classes
        # Use trainable=False for metric variables as they are not model parameters
        self.per_class_tp = self.add_weight(name='per_class_tp', shape=(self.num_classes,), initializer='zeros', trainable=False)
        self.per_class_fp = self.add_weight(name='per_class_fp', shape=(self.num_classes,), initializer='zeros', trainable=False)
        self._built = True # Mark as built

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Lazily build variables if not already built
        if not self._built:
            # y_true.shape will have concrete dimensions during update_state call
            self.build(y_true.shape)
            # If build still couldn't determine num_classes (e.g., if y_true.shape[-1] is None),
            # this would still be problematic. But for sequence data, y_true.shape[-1] should be concrete.
            if not self._built: # Check again if build was successful
                raise RuntimeError("MacroPrecision metric could not determine num_classes during build.")

        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)

        self.per_class_tp.assign_add(tp)
        self.per_class_fp.assign_add(fp)

    def result(self):
        # Return a default value if not built yet (e.g., when model.summary() is printed)
        if not self._built:
            return tf.constant(0.0, dtype=tf.float32) 
            
        precision_per_class = tf.where(
            tf.math.equal(self.per_class_tp + self.per_class_fp, 0),
            0.0,
            self.per_class_tp / (self.per_class_tp + self.per_class_fp)
        )
        return tf.reduce_mean(precision_per_class)

    def reset_state(self):
        # Reset only if variables have been built
        if self._built:
            self.per_class_tp.assign(tf.zeros(self.num_classes))
            self.per_class_fp.assign(tf.zeros(self.num_classes))
        # If not built, there's no state to reset, so do nothing.


class MacroRecall(tf.keras.metrics.Metric):
    def __init__(self, name='macro_recall', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self._built = False

    def build(self, input_shape):
        if self._built:
            return
        
        num_classes = input_shape[-1]
        if num_classes is None:
            return

        self.num_classes = num_classes
        self.per_class_tp = self.add_weight(name='per_class_tp', shape=(self.num_classes,), initializer='zeros', trainable=False)
        self.per_class_fn = self.add_weight(name='per_class_fn', shape=(self.num_classes,), initializer='zeros', trainable=False)
        self._built = True
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self._built:
            self.build(y_true.shape)
            if not self._built:
                raise RuntimeError("MacroRecall metric could not determine num_classes during build.")

        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

        self.per_class_tp.assign_add(tp)
        self.per_class_fn.assign_add(fn)

    def result(self):
        if not self._built:
            return tf.constant(0.0, dtype=tf.float32)

        recall_per_class = tf.where(
            tf.math.equal(self.per_class_tp + self.per_class_fn, 0),
            0.0,
            self.per_class_tp / (self.per_class_tp + self.per_class_fn)
        )
        return tf.reduce_mean(recall_per_class)

    def reset_state(self):
        if self._built:
            self.per_class_tp.assign(tf.zeros(self.num_classes))
            self.per_class_fn.assign(tf.zeros(self.num_classes))


class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='macro_f1', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self._built = False

    def build(self, input_shape):
        if self._built:
            return
            
        num_classes = input_shape[-1]
        if num_classes is None:
            return

        self.num_classes = num_classes
        self.per_class_tp = self.add_weight(name='per_class_tp', shape=(self.num_classes,), initializer='zeros', trainable=False)
        self.per_class_fp = self.add_weight(name='per_class_fp', shape=(self.num_classes,), initializer='zeros', trainable=False)
        self.per_class_fn = self.add_weight(name='per_class_fn', shape=(self.num_classes,), initializer='zeros', trainable=False)
        self._built = True
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self._built:
            self.build(y_true.shape)
            if not self._built:
                raise RuntimeError("MacroF1Score metric could not determine num_classes during build.")

        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
        self.per_class_tp.assign_add(tp)
        self.per_class_fp.assign_add(fp)
        self.per_class_fn.assign_add(fn)

    def result(self):
        if not self._built:
            return tf.constant(0.0, dtype=tf.float32)

        precision_per_class = tf.where(
            tf.math.equal(self.per_class_tp + self.per_class_fp, 0),
            0.0,
            self.per_class_tp / (self.per_class_tp + self.per_class_fp)
        )
        recall_per_class = tf.where(
            tf.math.equal(self.per_class_tp + self.per_class_fn, 0),
            0.0,
            self.per_class_tp / (self.per_class_tp + self.per_class_fn)
        )
        f1_per_class = tf.where(
            tf.math.equal(precision_per_class + recall_per_class, 0),
            0.0,
            2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
        )
        return tf.reduce_mean(f1_per_class)

    def reset_state(self):
        if self._built:
            self.per_class_tp.assign(tf.zeros(self.num_classes))
            self.per_class_fp.assign(tf.zeros(self.num_classes))
            self.per_class_fn.assign(tf.zeros(self.num_classes))
        
# --- Custom Focal Loss ---
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha # Alpha for the positive class (1-alpha for negative)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip predictions to prevent NaN's in log
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate p_t (probability of the true class)
        # If y_true is 1, p_t = y_pred
        # If y_true is 0, p_t = 1 - y_pred
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))

        # Calculate alpha_t (weighting factor for imbalance)
        # If y_true is 1, alpha_t = alpha
        # If y_true is 0, alpha_t = 1 - alpha
        alpha_t = (y_true * self.alpha) + ((1 - y_true) * (1 - self.alpha))

        # Calculate the focusing term (1 - p_t)^gamma
        focal_weight = tf.pow(1. - p_t, self.gamma)

        # Calculate the base cross-entropy term
        bce = -tf.math.log(p_t)

        # Combine all terms
        loss = alpha_t * focal_weight * bce
        
        # Average loss over all samples and classes
        return tf.reduce_mean(loss)

    def get_config(self):
        # This method is crucial for saving and loading the model with custom objects
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config
    
# --- 1. Feature Extraction (Mel-spectrograms from fixed-duration windows) ---
def extract_mel_spectrogram_fixed_window(audio_path, start_time, duration, sr=16000, n_mels=128, hop_length=512, fixed_time_steps=None):
    """
    Extracts a Mel-spectrogram from a fixed-duration segment of an audio file.
    Pads with zeros if the segment is shorter than the requested duration.
    
    Args:
        audio_path (str): Path to the audio file.
        start_time (float): Start time of the segment in seconds.
        duration (float): Duration of the segment in seconds.
        sr (int): Sampling rate.
        n_mels (int): Number of Mel bands.
        hop_length (int): Hop length for Mel spectrogram calculation.
        fixed_time_steps (int, optional): The exact number of time frames the spectrogram should have.
                                          If None, it's calculated from duration.
    Returns:
        np.ndarray: Mel-spectrogram (n_mels, fixed_time_steps) or a zero array on error.
    """
    if fixed_time_steps is None:
        fixed_time_steps = int(np.ceil(duration * sr / hop_length))

    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr, offset=start_time, duration=duration)
        
        if sr_loaded != sr:
            y = librosa.resample(y, orig_sr=sr_loaded, target_sr=sr)

        expected_samples = int(duration * sr)
        if len(y) < expected_samples:
            y = np.pad(y, (0, expected_samples - len(y)), 'constant')
        elif len(y) > expected_samples:
            y = y[:expected_samples]

        if len(y) == 0:
            print(f"Warning: Empty audio segment from {audio_path} [{start_time}s, duration {duration}s]. Returning zeros.")
            return np.zeros((n_mels, fixed_time_steps), dtype=np.float32)

        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        if mel_spectrogram_db.shape[1] < fixed_time_steps:
            pad_width = fixed_time_steps - mel_spectrogram_db.shape[1]
            mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, pad_width)), 'constant')
        elif mel_spectrogram_db.shape[1] > fixed_time_steps:
            mel_spectrogram_db = mel_spectrogram_db[:, :fixed_time_steps]

        return mel_spectrogram_db
    except Exception as e:
        print(f"Error processing segment from {audio_path} [{start_time}s, duration {duration}s]: {e}. Returning zeros.")
        return np.zeros((n_mels, fixed_time_steps), dtype=np.float32)

# --- 2. RTTM Parsing and Multi-Label Data Preparation (Slightly adjusted) ---
def parse_rttm_for_multi_label(rttm_path, audio_files_dir, valid_rttm_labels, window_duration, window_step, sr, n_mels, hop_length):
    """
    Parses a single RTTM file and generates fixed-duration windows with multi-hot labels.
    Only includes labels specified in valid_rttm_labels.
    
    Args:
        rttm_path (str): Path to the RTTM file.
        audio_files_dir (str): Directory where all audio files are located.
        valid_rttm_labels (list): List of RTTM speaker_id strings to include (e.g., ['OHS', 'CDS', 'KCHI']).
        window_duration (float): The fixed duration of each analysis window in seconds.
        window_step (float): The step size between consecutive windows in seconds.
        sr (int): Sampling rate (for librosa.get_duration).
        n_mels (int): Number of Mel bands (for fixed_time_steps calculation).
        hop_length (int): Hop length (for fixed_time_steps calculation).
                                                 
    Returns:
        tuple: (list of window data, list of unique class labels found).
    """
    try:
        rttm_df = pd.read_csv(rttm_path, sep=' ', header=None, 
                              names=['type', 'file_id', 'channel', 'start', 'duration', 
                                     'NA1', 'NA2', 'speaker_id', 'NA3', 'NA4'])
    except Exception as e:
        print(f"Error reading RTTM file {rttm_path}: {e}")
        return [], []


    all_window_data = []
    all_unique_labels = set()

    unique_file_ids = rttm_df['file_id'].unique()

    print(f"Processing {len(unique_file_ids)} audio files from RTTM: {os.path.basename(rttm_path)}...")
    for file_id in tqdm(unique_file_ids):
        audio_path = os.path.join(audio_files_dir, f"{file_id}.wav") # Adjust extension if needed

        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for RTTM entry: {audio_path}. Skipping.")
            continue

        file_segments = rttm_df[rttm_df['file_id'] == file_id].copy()
        file_segments['end'] = file_segments['start'] + file_segments['duration']

        try:
            audio_duration = librosa.get_duration(path=audio_path)
        except Exception as e:
            print(f"Warning: Could not get duration for {audio_path}: {e}. Skipping file.")
            continue
            
        max_time_rttm = file_segments['end'].max() if not file_segments.empty else 0
        analysis_end_time = min(audio_duration, max_time_rttm)

        current_time = 0.0
        while current_time < analysis_end_time:
            window_start = current_time
            window_end = min(current_time + window_duration, audio_duration, analysis_end_time) 

            active_speaker_ids = set()
            for _, row in file_segments.iterrows():
                segment_start = row['start']
                segment_end = row['end']
                
                if max(window_start, segment_start) < min(window_end, segment_end):
                    active_speaker_ids.add(row['speaker_id'])

            # Filter active speaker_ids to only include valid RTTM labels
            active_labels = {sid for sid in active_speaker_ids if sid in valid_rttm_labels}
            all_unique_labels.update(active_labels) # Update overall unique labels seen

            # Only add segments that have at least one valid mapped label AND actual duration
            if active_labels and (window_end - window_start) > 0:
                all_window_data.append({
                    'audio_path': audio_path,
                    'start': window_start,
                    'duration': window_duration, 
                    'labels': sorted(list(active_labels)) 
                })

            current_time += window_step
            
    return all_window_data, sorted(list(all_unique_labels))


# --- 3. Deep Learning Model Architecture (CNN-GRU with Attention) ---
def build_model_multi_label(n_mels, fixed_time_steps, num_classes):
    # **** CRITICAL CHANGE HERE ****
    # Define the Input layer with the channel dimension (1) directly
    input_mel = Input(shape=(n_mels, fixed_time_steps, 1), name='mel_spectrogram_input') 
    x = input_mel

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    reduced_n_mels = n_mels // 4
    reduced_time_steps = fixed_time_steps // 4
    channels_after_cnn = 64

    x = Permute((2, 1, 3))(x) 
    x = Reshape((reduced_time_steps, reduced_n_mels * channels_after_cnn))(x)
    
    x = GRU(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = GRU(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    attention = Dense(1, activation='tanh')(x)
    attention = Flatten()(attention) 
    attention = Activation('softmax')(attention) 
    attention = RepeatVector(128)(attention) # Corrected: Uses fixed GRU units as repetition factor
    attention = Permute((2, 1))(attention) 

    sent_representation = multiply([x, attention])
    sent_representation = Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(sent_representation) 

    x = Dense(128, activation='relu')(sent_representation)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_mel, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=FocalLoss(gamma=2.0, alpha=0.25),
                  metrics=[
                    'accuracy',
                    'precision',
                    'recall',
                  ])
    return model

# --- Data Generator for fixed-duration windows ---
class AudioSegmentDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, segments_data, mlb, n_mels, hop_length, sr, window_duration, fixed_time_steps, batch_size=32, shuffle=True):
        self.segments_data = segments_data
        self.mlb = mlb 
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sr = sr
        self.window_duration = window_duration 
        self.fixed_time_steps = fixed_time_steps 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Calculate number of batches, ensuring it's an integer
        if len(self.segments_data) == 0:
            return 0
        return int(np.floor(len(self.segments_data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_segments = [self.segments_data[k] for k in indexes]

        X_batch = []
        y_batch = []

        for segment in batch_segments:
            mel = extract_mel_spectrogram_fixed_window(
                segment['audio_path'], segment['start'], segment['duration'], 
                sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length, fixed_time_steps=self.fixed_time_steps
            )
            # Ensure each individual mel is 2D (n_mels, fixed_time_steps)
            if mel.ndim != 2:
                # If it's something unexpected, try to reshape or handle it.
                # For example, if it's (n_mels, fixed_time_steps, 1) already, squeeze it.
                if mel.ndim == 3 and mel.shape[-1] == 1:
                    mel = mel.squeeze(axis=-1)
                else:
                    print(f"Warning: Individual mel spectrogram has unexpected dimensions {mel.shape}. Expected 2D. Skipping this sample.")
                    continue # Skip this problematic sample
            X_batch.append(mel)
            
            multi_hot_labels = self.mlb.transform([segment['labels']])[0]
            y_batch.append(multi_hot_labels)
                
        if not X_batch:
            return np.array([]), np.array([])
            
        X_batch_np = np.array(X_batch) # This will create (batch_size, n_mels, fixed_time_steps)

        # Now explicitly expand for the channel dimension
        # This assumes X_batch_np is (batch_size, n_mels, fixed_time_steps) after np.array(X_batch)
        X_batch_final = np.expand_dims(X_batch_np, -1) 

        # --- DEBUG PRINT ---
        # print(f"DEBUG: Final batch shape from generator: {X_batch_final.shape}") # Commenting out to reduce spam
        # --- END DEBUG PRINT ---

        return X_batch_final, np.array(y_batch)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.segments_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    AUDIO_FILES_DIR = '/home/nele_pauline_suffo/ProcessedData/childlens_audio'
    
    # Define paths to your pre-split RTTM files
    TRAIN_RTTM_FILE = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/train_small.rttm'
    VAL_RTTM_FILE = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/dev_small.rttm'
    TEST_RTTM_FILE = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/test_small.rttm'
    
    SR = 16000
    N_MELS = 128
    HOP_LENGTH = 512
    WINDOW_DURATION = 0.5 # seconds (e.g., 500ms analysis window)
    WINDOW_STEP = 0.25    # seconds (e.g., 250ms hop between windows for more data)

    # --- Define the specific RTTM labels you want to include in classification ---
    # Any speaker_id in the RTTM not in this list will be ignored.
    VALID_RTTM_LABELS = ['OHS', 'CDS', 'KCHI'] 

    # Initialize class_weights_for_keras here to ensure it's always defined
    class_weights_for_keras = {} 

    # --- 1. Parse RTTM files for each split ---
    print("--- Processing Training Data ---")
    train_segments, train_unique_labels = parse_rttm_for_multi_label(
        TRAIN_RTTM_FILE, AUDIO_FILES_DIR, VALID_RTTM_LABELS,
        WINDOW_DURATION, WINDOW_STEP, SR, N_MELS, HOP_LENGTH 
    )
    if not train_segments:
        print("No training segments found. Exiting.")
        exit()

    print("\n--- Processing Validation Data ---")
    val_segments, val_unique_labels = parse_rttm_for_multi_label(
        VAL_RTTM_FILE, AUDIO_FILES_DIR, VALID_RTTM_LABELS,
        WINDOW_DURATION, WINDOW_STEP, SR, N_MELS, HOP_LENGTH
    )
    if not val_segments:
        print("No validation segments found. Exiting.")
        exit()

    print("\n--- Processing Test Data ---")
    test_segments, test_unique_labels = [], []
    if os.path.exists(TEST_RTTM_FILE):
        test_segments, test_unique_labels = parse_rttm_for_multi_label(
            TEST_RTTM_FILE, AUDIO_FILES_DIR, VALID_RTTM_LABELS,
            WINDOW_DURATION, WINDOW_STEP, SR, N_MELS, HOP_LENGTH
        )
        if not test_segments:
            print("No test segments found from the provided file.")
    else:
        print(f"Test RTTM file not found at {TEST_RTTM_FILE}. Skipping test data processing.")


    # --- 2. Initialize MultiLabelBinarizer based on ALL labels ---
    all_possible_target_labels_seen = sorted(list(set(train_unique_labels + val_unique_labels + test_unique_labels)))
    mlb = MultiLabelBinarizer(classes=all_possible_target_labels_seen)
    mlb.fit([[]]) # Fit with an empty list to initialize classes correctly
    num_classes = len(mlb.classes_)
    print(f"\nDetected {num_classes} unique target classes across all splits: {mlb.classes_}")

    # Exit if no valid classes are found, as model building would fail
    if num_classes == 0:
        print("Error: No valid target classes detected after RTTM parsing with specified VALID_RTTM_LABELS. Cannot proceed.")
        exit()

    # --- 3. Determine Fixed Time Steps for Model Input ---
    FIXED_TIME_STEPS = int(np.ceil(WINDOW_DURATION * SR / HOP_LENGTH))
    print(f"Fixed Time Steps for Mel-spectrogram input (for {WINDOW_DURATION}s windows): {FIXED_TIME_STEPS}")
    # Update this print statement to reflect the new input shape expectation
    print(f"Model input shape: ({N_MELS}, {FIXED_TIME_STEPS}, 1)") # Expected by model

    # --- 4. Build Multi-Label Model ---
    model = build_model_multi_label(n_mels=N_MELS, fixed_time_steps=FIXED_TIME_STEPS, num_classes=num_classes)
    model.summary()

    # --- 5. Calculate class_weights for Focal Loss (if desired, optional but recommended) ---
    total_training_windows = len(train_segments)
    
    if total_training_windows == 0:
        print("Warning: No training segments available. Class weights will be set to 1.0 for all classes.")
        for i in range(num_classes):
            class_weights_for_keras[i] = 1.0
    else:
        # Count occurrences of each valid label in the training set
        class_counts_for_weights = {label: 0 for label in mlb.classes_} # Initialize counts for all known classes
        for seg in train_segments:
            for label in seg['labels']:
                if label in class_counts_for_weights: # Ensure only counts for relevant classes
                    class_counts_for_weights[label] += 1
        
        # Calculate inverse frequency weights
        for i, class_name in enumerate(mlb.classes_):
            count = class_counts_for_weights.get(class_name, 0) # Get count, default to 0 if class not in training data
            if count > 0:
                class_weights_for_keras[i] = total_training_windows / (num_classes * count)
            else:
                # If a class is never present in training data, assign a default weight (e.g., 1.0)
                # This ensures it's not "inf" and doesn't crash, but it won't be actively trained on.
                class_weights_for_keras[i] = 1.0 
        
        # Normalize weights so their average is 1.0 (optional, but can help with stability)
        if len(class_weights_for_keras) > 0: # Ensure there are weights to average
            avg_weight = sum(class_weights_for_keras.values()) / len(class_weights_for_keras)
            class_weights_for_keras = {k: v / avg_weight for k, v in class_weights_for_keras.items()}
        else:
            print("Warning: No class weights to normalize. This should ideally not happen if num_classes > 0.")
            # If this warning shows up, it means num_classes was > 0 but class_weights_for_keras ended up empty.
            # This implies an issue in how mlb.classes_ or class_counts_for_weights were populated.

    print("\nCalculated Keras Class Weights (for model.fit):", class_weights_for_keras)


    # --- 6. Create Data Generators ---
    print(f"\nTotal training segments: {len(train_segments)}")
    print(f"Total validation segments: {len(val_segments)}")
    if test_segments:
        print(f"Total test segments: {len(test_segments)}")


    train_generator = AudioSegmentDataGenerator(
        train_segments, mlb, N_MELS, HOP_LENGTH, SR, WINDOW_DURATION, FIXED_TIME_STEPS, batch_size=32, shuffle=True
    )
    val_generator = AudioSegmentDataGenerator(
        val_segments, mlb, N_MELS, HOP_LENGTH, SR, WINDOW_DURATION, FIXED_TIME_STEPS, batch_size=32, shuffle=False
    )
    
    test_generator = None
    if test_segments:
        test_generator = AudioSegmentDataGenerator(
            test_segments, mlb, N_MELS, HOP_LENGTH, SR, WINDOW_DURATION, FIXED_TIME_STEPS, batch_size=32, shuffle=False
        )

    # Callbacks for training
    # It's better to monitor val_macro_f1 as it's more relevant for multi-label classification
    early_stopping = EarlyStopping(monitor='val_macro_f1', patience=15, mode='max', restore_best_weights=True) 
    reduce_lr = ReduceLROnPlateau(monitor='val_macro_f1', factor=0.2, patience=7, min_lr=0.00001, mode='max') 

    print("\nTraining the model...")
    # Check if generators have any batches before fitting
    if len(train_generator) == 0:
        print("Error: Training generator has no batches. Training cannot proceed.")
        exit()
    if len(val_generator) == 0:
        print("Error: Validation generator has no batches. Training cannot proceed.")
        exit()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=100, 
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights_for_keras
    )

    # Evaluate the model on the validation set (final validation metrics)
    print("\nEvaluating the model on validation data (final validation metrics)...")
    val_results = model.evaluate(val_generator)
    val_metrics_dict = dict(zip(model.metrics_names, val_results))
    for name, value in val_metrics_dict.items():
        print(f"Validation {name}: {value:.4f}")

    # --- Evaluate on the separate test set and output per-class metrics ---
    if test_generator and len(test_generator) > 0:
        print("\n--- Detailed Evaluation on TEST data ---")
        
        # 1. Get predictions from the model
        test_predictions = model.predict(test_generator)
        
        # 2. Get true labels from the test generator
        test_true_labels = []
        for i in tqdm(range(len(test_generator)), desc="Collecting true test labels"):
            _, labels = test_generator[i]
            test_true_labels.extend(labels)
        test_true_labels = np.array(test_true_labels)

        # Ensure that test_predictions and test_true_labels have matching numbers of samples
        if test_predictions.shape[0] != test_true_labels.shape[0]:
            print(f"Warning: Mismatch between number of predictions ({test_predictions.shape[0]}) and true labels ({test_true_labels.shape[0]}) for test set. This might indicate an issue with generator or prediction pipeline.")
            # Adjust test_true_labels to match predictions length if necessary (e.g., if some samples were skipped)
            min_samples = min(test_predictions.shape[0], test_true_labels.shape[0])
            test_predictions = test_predictions[:min_samples]
            test_true_labels = test_true_labels[:min_samples]

        # 3. Binarize predictions using a threshold (e.5)
        prediction_threshold = 0.5 
        test_pred_binary = (test_predictions > prediction_threshold).astype(int) 

        # 4. Calculate per-class Precision, Recall, and F1-score
        # Ensure that test_true_labels has actual positive instances for calculation.
        # This can sometimes be an issue with very small test sets or rare classes.
        if test_true_labels.sum() == 0:
            print("Warning: No positive instances in test_true_labels. Skipping detailed metrics calculation.")
        else:
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                test_true_labels, test_pred_binary, average=None, zero_division=0 
            )
            
            # Also calculate macro averages using sklearn for consistency with per-class output
            macro_precision = precision_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
            macro_recall = recall_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
            macro_f1 = f1_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)

            # And overall (micro) accuracy as a reference
            overall_accuracy = accuracy_score(test_true_labels, test_pred_binary)

            print(f"\nTest Overall Accuracy (threshold {prediction_threshold}): {overall_accuracy:.4f}")
            print(f"Test Macro Precision (sklearn): {macro_precision:.4f}")
            print(f"Test Macro Recall (sklearn): {macro_recall:.4f}")
            print(f"Test Macro F1-score (sklearn): {macro_f1:.4f}")
            print("\n--- Per-Class Metrics on Test Set (threshold 0.5) ---")
            print("{:<15} {:<10} {:<10} {:<10}".format("Class", "Precision", "Recall", "F1-Score"))
            print("-" * 55)
            for i, class_name in enumerate(mlb.classes_):
                print(f"{class_name:<15} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} {f1_per_class[i]:<10.4f}")
            print("-" * 55)
    else:
        print("Skipping detailed test evaluation as test_generator is not available or empty.")

    # You can save the model
    # To ensure it saves with custom objects, pass custom_objects dict
    model.save('/home/nele_pauline_suffo/projects/leuphana-IPE/src/models/audio_classification/multi_label_speech_type_classifier.h5',
               custom_objects={'FocalLoss': FocalLoss})