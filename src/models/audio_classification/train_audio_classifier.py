import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
import pandas as pd
import librosa
import os
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score

# --- Custom Focal Loss ---
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip predictions to prevent NaN's in log
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate p_t (probability of the true class)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))

        # Calculate alpha_t (weighting factor for imbalance)
        alpha_t = (y_true * self.alpha) + ((1 - y_true) * (1 - self.alpha))

        # Calculate the focusing term (1 - p_t)^gamma
        focal_weight = tf.pow(1. - p_t, self.gamma)

        # Calculate the base cross-entropy term
        bce = -tf.math.log(p_t)

        # Combine all terms
        loss = alpha_t * focal_weight * bce
        
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config

# --- Custom Macro F1 Metric ---
class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, threshold=0.5, name='macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        
        # Create precision and recall metrics for each class
        self.precisions = []
        self.recalls = []
        for i in range(num_classes):
            self.precisions.append(Precision(thresholds=threshold, name=f'precision_{i}'))
            self.recalls.append(Recall(thresholds=threshold, name=f'recall_{i}'))

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update each class-specific precision and recall
        for i in range(self.num_classes):
            y_true_class = y_true[:, i]
            y_pred_class = y_pred[:, i]
            self.precisions[i].update_state(y_true_class, y_pred_class, sample_weight)
            self.recalls[i].update_state(y_true_class, y_pred_class, sample_weight)

    def result(self):
        # Calculate F1 for each class and return macro average
        f1_scores = []
        for i in range(self.num_classes):
            p = self.precisions[i].result()
            r = self.recalls[i].result()
            f1 = 2 * (p * r) / (p + r + tf.keras.backend.epsilon())
            f1_scores.append(f1)
        
        return tf.reduce_mean(f1_scores)

    def reset_state(self):
        for i in range(self.num_classes):
            self.precisions[i].reset_state()
            self.recalls[i].reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'threshold': self.threshold
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        num_classes = config.pop('num_classes')
        threshold = config.pop('threshold', 0.5)
        return cls(num_classes=num_classes, threshold=threshold, **config)
    
# --- Feature Extraction ---
def extract_mel_spectrogram_fixed_window(audio_path, start_time, duration, sr=16000, n_mels=128, hop_length=512, fixed_time_steps=None):
    """
    Extracts a Mel-spectrogram from a fixed-duration segment of an audio file.
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

# --- RTTM Parsing ---
def parse_rttm_for_multi_label(rttm_path, audio_files_dir, valid_rttm_labels, window_duration, window_step, sr, n_mels, hop_length):
    """
    Parses a single RTTM file and generates fixed-duration windows with multi-hot labels.
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
        audio_path = os.path.join(audio_files_dir, f"{file_id}.wav")

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
            all_unique_labels.update(active_labels)

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

# --- Deep Learning Model Architecture ---
def build_model_multi_label(n_mels, fixed_time_steps, num_classes):
    input_mel = Input(shape=(n_mels, fixed_time_steps, 1), name='mel_spectrogram_input') 
    x = input_mel

    # CNN layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Calculate dimensions after CNN
    reduced_n_mels = n_mels // 4
    reduced_time_steps = fixed_time_steps // 4
    channels_after_cnn = 64

    # Reshape for RNN
    x = Permute((2, 1, 3))(x) 
    x = Reshape((reduced_time_steps, reduced_n_mels * channels_after_cnn))(x)
    
    # RNN layers
    x = GRU(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = GRU(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    # Attention mechanism
    attention = Dense(1, activation='tanh')(x)
    attention = Flatten()(attention) 
    attention = Activation('softmax')(attention) 
    attention = RepeatVector(128)(attention)
    attention = Permute((2, 1))(attention) 

    sent_representation = multiply([x, attention])
    sent_representation = Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(sent_representation) 

    # Final dense layers
    x = Dense(128, activation='relu')(sent_representation)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_mel, outputs=output)
    
    # Create custom metrics
    macro_f1 = MacroF1Score(num_classes=num_classes, name='macro_f1')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            macro_f1
        ]
    )
    return model

# --- Data Generator ---
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
                sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length, 
                fixed_time_steps=self.fixed_time_steps
            )
            
            # Ensure each mel is 2D
            if mel.ndim != 2:
                if mel.ndim == 3 and mel.shape[-1] == 1:
                    mel = mel.squeeze(axis=-1)
                else:
                    print(f"Warning: Unexpected mel dimensions {mel.shape}. Skipping sample.")
                    continue
                    
            X_batch.append(mel)
            multi_hot_labels = self.mlb.transform([segment['labels']])[0]
            y_batch.append(multi_hot_labels)
                
        if not X_batch:
            return np.array([]), np.array([])
            
        X_batch_np = np.array(X_batch)
        X_batch_final = np.expand_dims(X_batch_np, -1)

        return X_batch_final, np.array(y_batch)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.segments_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

# --- Main Execution ---
if __name__ == "__main__":
    # Configuration
    AUDIO_FILES_DIR = '/home/nele_pauline_suffo/ProcessedData/childlens_audio'
    TRAIN_RTTM_FILE = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/train_small.rttm'
    VAL_RTTM_FILE = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/dev_small.rttm'
    TEST_RTTM_FILE = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/test_small.rttm'
    
    SR = 16000
    N_MELS = 128
    HOP_LENGTH = 512
    WINDOW_DURATION = 0.5
    WINDOW_STEP = 0.25
    VALID_RTTM_LABELS = ['OHS', 'CDS', 'KCHI']

    class_weights_for_keras = {}

    # Parse RTTM files
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

    # Initialize MultiLabelBinarizer
    all_possible_target_labels_seen = sorted(list(set(train_unique_labels + val_unique_labels + test_unique_labels)))
    mlb = MultiLabelBinarizer(classes=all_possible_target_labels_seen)
    mlb.fit([[]])
    num_classes = len(mlb.classes_)
    print(f"\nDetected {num_classes} unique target classes: {mlb.classes_}")

    if num_classes == 0:
        print("Error: No valid target classes detected. Cannot proceed.")
        exit()

    # Calculate fixed time steps
    FIXED_TIME_STEPS = int(np.ceil(WINDOW_DURATION * SR / HOP_LENGTH))
    print(f"Fixed Time Steps: {FIXED_TIME_STEPS}")
    print(f"Model input shape: ({N_MELS}, {FIXED_TIME_STEPS}, 1)")

    # Build model
    model = build_model_multi_label(n_mels=N_MELS, fixed_time_steps=FIXED_TIME_STEPS, num_classes=num_classes)
    model.summary()

    # Calculate class weights
    total_training_windows = len(train_segments)
    
    if total_training_windows == 0:
        print("Warning: No training segments available.")
        for i in range(num_classes):
            class_weights_for_keras[i] = 1.0
    else:
        class_counts_for_weights = {label: 0 for label in mlb.classes_}
        for seg in train_segments:
            for label in seg['labels']:
                if label in class_counts_for_weights:
                    class_counts_for_weights[label] += 1
        
        for i, class_name in enumerate(mlb.classes_):
            count = class_counts_for_weights.get(class_name, 0)
            if count > 0:
                class_weights_for_keras[i] = total_training_windows / (num_classes * count)
            else:
                class_weights_for_keras[i] = 1.0
        
        # Normalize weights
        if len(class_weights_for_keras) > 0:
            avg_weight = sum(class_weights_for_keras.values()) / len(class_weights_for_keras)
            class_weights_for_keras = {k: v / avg_weight for k, v in class_weights_for_keras.items()}

    print(f"\nClass distribution in training data:")
    for i, class_name in enumerate(mlb.classes_):
        count = sum(1 for seg in train_segments if class_name in seg['labels'])
        percentage = (count / len(train_segments)) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    print(f"\nCalculated Class Weights: {class_weights_for_keras}")

    # Create data generators
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_segments)} segments")
    print(f"  Validation: {len(val_segments)} segments")
    if test_segments:
        print(f"  Test: {len(test_segments)} segments")

    train_generator = AudioSegmentDataGenerator(
        train_segments, mlb, N_MELS, HOP_LENGTH, SR, WINDOW_DURATION, FIXED_TIME_STEPS, 
        batch_size=32, shuffle=True
    )
    val_generator = AudioSegmentDataGenerator(
        val_segments, mlb, N_MELS, HOP_LENGTH, SR, WINDOW_DURATION, FIXED_TIME_STEPS, 
        batch_size=32, shuffle=False
    )
    
    test_generator = None
    if test_segments:
        test_generator = AudioSegmentDataGenerator(
            test_segments, mlb, N_MELS, HOP_LENGTH, SR, WINDOW_DURATION, FIXED_TIME_STEPS, 
            batch_size=32, shuffle=False
        )

    # Training callbacks
    early_stopping = EarlyStopping(monitor='val_macro_f1', patience=15, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_macro_f1', factor=0.2, patience=7, min_lr=0.00001, mode='max')

    print("\nStarting training...")
    if len(train_generator) == 0 or len(val_generator) == 0:
        print("Error: No batches available for training/validation.")
        exit()

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=100,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights_for_keras,
        verbose=1
    )

    # Final validation evaluation
    print("\n" + "="*60)
    print("FINAL VALIDATION RESULTS")
    print("="*60)
    val_results = model.evaluate(val_generator, verbose=0)
    val_metrics_dict = dict(zip(model.metrics_names, val_results))
    for name, value in val_metrics_dict.items():
        print(f"Validation {name}: {value:.4f}")

    # Detailed test evaluation
    if test_generator and len(test_generator) > 0:
        print("\n" + "="*60)
        print("DETAILED TEST SET EVALUATION")
        print("="*60)
        
        # Get predictions and true labels
        test_predictions = model.predict(test_generator, verbose=1)
        
        test_true_labels = []
        for i in tqdm(range(len(test_generator)), desc="Collecting true labels"):
            _, labels = test_generator[i]
            test_true_labels.extend(labels)
        test_true_labels = np.array(test_true_labels)

        # Handle size mismatch
        if test_predictions.shape[0] != test_true_labels.shape[0]:
            min_samples = min(test_predictions.shape[0], test_true_labels.shape[0])
            test_predictions = test_predictions[:min_samples]
            test_true_labels = test_true_labels[:min_samples]
            print(f"Adjusted to {min_samples} samples due to size mismatch.")

        # Apply threshold
        prediction_threshold = 0.5
        test_pred_binary = (test_predictions > prediction_threshold).astype(int)

        # Calculate metrics
        if test_true_labels.sum() > 0:
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                test_true_labels, test_pred_binary, average=None, zero_division=0
            )
            
            # Macro averages
            macro_precision = precision_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
            macro_recall = recall_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
            macro_f1 = f1_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
            
            # Subset accuracy (exact match ratio for multi-label)
            subset_accuracy = accuracy_score(test_true_labels, test_pred_binary)

            print(f"\nTest Metrics (threshold {prediction_threshold}):")
            print(f"  Subset Accuracy (exact match): {subset_accuracy:.4f}")
            print(f"  Macro Precision: {macro_precision:.4f}")
            print(f"  Macro Recall: {macro_recall:.4f}")
            print(f"  Macro F1-score: {macro_f1:.4f}")
            
            print(f"\nPer-Class Results:")
            print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-" * 65)
            for i, class_name in enumerate(mlb.classes_):
                print(f"{class_name:<15} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
                      f"{f1_per_class[i]:<10.4f} {support_per_class[i]:<10}")
            print("-" * 65)
            print(f"{'MACRO AVG':<15} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f}")
            
        else:
            print("Warning: No positive instances in test set for metrics calculation.")

    # Save the model
    model_save_path = '/home/nele_pauline_suffo/projects/leuphana-IPE/src/models/audio_classification/multi_label_speech_type_classifier.h5'
    model.save(model_save_path, custom_objects={'FocalLoss': FocalLoss, 'MacroF1Score': MacroF1Score})
    print(f"\nModel saved to: {model_save_path}")
    
    print("\nTraining completed successfully!")