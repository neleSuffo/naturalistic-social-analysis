import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import json
import csv
import time
import shutil
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tqdm import tqdm
import argparse

# Environment setup
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Custom Focal Loss ---
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=3.0, alpha=0.5, name='improved_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = tf.pow(1. - p_t, self.gamma)
        loss = alpha_t * focal_weight * ce
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config

# --- Custom Macro F1 Metric ---
class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, threshold=0.5, name='macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.precisions = [Precision(thresholds=threshold, name=f'precision_{i}') for i in range(num_classes)]
        self.recalls = [Recall(thresholds=threshold, name=f'recall_{i}') for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_classes):
            y_true_class = y_true[..., i]
            y_pred_class = y_pred[..., i]
            self.precisions[i].update_state(y_true_class, y_pred_class, sample_weight)
            self.recalls[i].update_state(y_true_class, y_pred_class, sample_weight)

    def result(self):
        f1_scores = []
        for i in range(self.num_classes):
            p = self.precisions[i].result()
            r = self.recalls[i].result()
            f1 = 2 * (p * r) / (p + r + tf.keras.backend.epsilon())
            f1_scores.append(f1)
        return tf.reduce_mean(f1_scores) if f1_scores else tf.constant(0.0, dtype=tf.float32)

    def reset_state(self):
        for i in range(self.num_classes):
            self.precisions[i].reset_state()
            self.recalls[i].reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes, 'threshold': self.threshold})
        return config

    @classmethod
    def from_config(cls, config):
        num_classes = config.pop('num_classes')
        threshold = config.pop('threshold', 0.5)
        return cls(num_classes=num_classes, threshold=threshold, **config)

# --- Threshold Optimizer ---
class ThresholdOptimizer(tf.keras.callbacks.Callback):
    def __init__(self, validation_generator, mlb_classes):
        super().__init__()
        self.validation_generator = validation_generator
        self.mlb_classes = mlb_classes
        self.best_thresholds = [0.5] * len(mlb_classes)
        self.best_f1 = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            self.optimize_thresholds()

    def optimize_thresholds(self):
        predictions = self.model.predict(self.validation_generator, verbose=0)
        true_labels = []
        for i in range(len(self.validation_generator)):
            _, labels = self.validation_generator[i]
            true_labels.extend(labels)
        true_labels = np.array(true_labels)
        
        best_thresholds = []
        for class_idx in range(len(self.mlb_classes)):
            best_threshold = 0.5
            best_class_f1 = 0.0
            for threshold in np.arange(0.1, 0.9, 0.01):  # Finer granularity
                pred_binary = (predictions[:, class_idx] > threshold).astype(int)
                f1 = f1_score(true_labels[:, class_idx], pred_binary, zero_division=0)
                if f1 > best_class_f1:
                    best_class_f1 = f1
                    best_threshold = threshold
            best_thresholds.append(best_threshold)
        
        self.best_thresholds = best_thresholds
        print(f"Optimized thresholds: {dict(zip(self.mlb_classes, best_thresholds))}")
        # Save thresholds
        with open(os.path.join(self.model.log_dir, 'thresholds.json'), 'w') as f:
            json.dump(dict(zip(self.mlb_classes, best_thresholds)), f)

# --- Feature Extraction ---
def extract_enhanced_features(audio_path, start_time, duration, sr=16000, n_mels=256, hop_length=512, fixed_time_steps=None):
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
            return np.zeros((n_mels + 13, fixed_time_steps), dtype=np.float32)  # +13 for MFCC
        
        y = y / (np.max(np.abs(y)) + 1e-6)
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])
        
        # Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=2048, fmin=20, fmax=10000
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=80)
        mel_spectrogram_db = 2 * (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min() + 1e-6) - 1
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc = 2 * (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-6) - 1
        
        # Concatenate features
        combined = np.concatenate([mel_spectrogram_db, mfcc], axis=0)
        
        if combined.shape[1] < fixed_time_steps:
            pad_width = fixed_time_steps - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_width)), 'constant', constant_values=-1)
        elif combined.shape[1] > fixed_time_steps:
            combined = combined[:, :fixed_time_steps]
        
        return combined
    
    except Exception as e:
        print(f"Error processing segment from {audio_path} [{start_time}s, duration {duration}s]: {e}")
        return np.zeros((n_mels + 13, fixed_time_steps), dtype=np.float32)

# --- RTTM Parsing ---
def parse_rttm_for_multi_label_and_save(rttm_path, audio_files_dir, valid_rttm_labels, window_duration, window_step, sr, n_mels, hop_length, output_segments_path):
    try:
        rttm_df = pd.read_csv(rttm_path, sep=' ', header=None, 
                              names=['type', 'file_id', 'channel', 'start', 'duration', 'NA1', 'NA2', 'speaker_id', 'NA3', 'NA4'])
    except Exception as e:
        print(f"Error reading RTTM file {rttm_path}: {e}")
        return [], []
    
    all_unique_labels = set()
    unique_file_ids = rttm_df['file_id'].unique()
    print(f"Processing {len(unique_file_ids)} audio files from RTTM: {os.path.basename(rttm_path)}...")
    
    segment_counter = 0
    with open(output_segments_path, 'w', newline='') as f_out:
        for file_id in tqdm(unique_file_ids):
            audio_path = os.path.join(audio_files_dir, f"{file_id}.wav")
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}. Skipping.")
                continue
            
            file_segments = rttm_df[rttm_df['file_id'] == file_id].copy()
            file_segments['end'] = file_segments['start'] + file_segments['duration']
            
            try:
                audio_duration = librosa.get_duration(path=audio_path)
            except Exception as e:
                print(f"Warning: Could not get duration for {audio_path}: {e}. Skipping.")
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
                
                active_labels = {sid for sid in active_speaker_ids if sid in valid_rttm_labels}
                all_unique_labels.update(active_labels)
                
                if active_labels and (window_end - window_start) > 0:
                    segment_data = {
                        'audio_path': audio_path,
                        'start': window_start,
                        'duration': window_duration,
                        'labels': sorted(list(active_labels))
                    }
                    f_out.write(json.dumps(segment_data) + '\n')
                    segment_counter += 1
                
                current_time += window_step
                
    return segment_counter, sorted(list(all_unique_labels))

# --- Deep Learning Model Architecture ---
def build_model_multi_label(n_mels, fixed_time_steps, num_classes):
    input_mel = Input(shape=(n_mels + 13, fixed_time_steps, 1), name='mel_spectrogram_input')
    x = input_mel

    def conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
        shortcut = x
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        if shortcut.shape[-1] != filters or strides != (1, 1):
            shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    x = conv_block(x, 32)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = conv_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = conv_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    x = conv_block(x, 256)  # Added deeper block
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    reduced_n_mels = (n_mels + 13) // 16
    reduced_time_steps = fixed_time_steps // 16
    channels_after_cnn = 256

    x = Permute((2, 1, 3))(x)
    x = Reshape((reduced_time_steps, reduced_n_mels * channels_after_cnn))(x)
    
    x = Bidirectional(GRU(256, return_sequences=True, dropout=0.3))(x)  # Increased units
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.3))(x)

    def multi_head_attention(x, num_heads=4):
        head_size = x.shape[-1] // num_heads
        heads = []
        for i in range(num_heads):
            attention = Dense(1, activation='tanh')(x)
            attention = Flatten()(attention)
            attention = Activation('softmax')(attention)
            attention = RepeatVector(x.shape[-1])(attention)
            attention = Permute((2, 1))(attention)
            head = multiply([x, attention])
            head = Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(head)
            heads.append(head)
        return Concatenate()(heads) if len(heads) > 1 else heads[0]

    x = multi_head_attention(x)

    dense_input = x
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    if dense_input.shape[-1] == 256:
        x = Add()([x, dense_input])
    
    class_outputs = []
    for i in range(num_classes):
        class_branch = Dense(128, activation='relu', name=f'class_{i}_dense')(x)
        class_branch = Dropout(0.3)(class_branch)
        class_output = Dense(1, activation='sigmoid', name=f'class_{i}_output')(class_branch)
        class_outputs.append(class_output)
    
    output = Concatenate(name='combined_output')(class_outputs)

    model = Model(inputs=input_mel, outputs=output)
    
    macro_f1 = MacroF1Score(num_classes=num_classes, name='macro_f1')
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), macro_f1]
    )
    model.log_dir = None  # For ThresholdOptimizer
    return model

# --- Class Weights ---
def calculate_balanced_class_weights(train_segments, mlb):
    class_counts = {label: 0 for label in mlb.classes_}
    for seg in train_segments:
        for label in seg['labels']:
            if label in class_counts:
                class_counts[label] += 1
    
    class_weights_for_keras = {}
    total_samples = len(train_segments)
    for i, class_name in enumerate(mlb.classes_):
        count = class_counts.get(class_name, 1)
        weight = (total_samples / (len(mlb.classes_) * count))  # Balanced weighting
        class_weights_for_keras[i] = weight
    
    return class_weights_for_keras

# --- Data Generator ---
class EnhancedAudioSegmentDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, segments_file_path, mlb, n_mels, hop_length, sr, window_duration, fixed_time_steps, 
                 batch_size=32, shuffle=True, augment=False):
        self.segments_file_path = segments_file_path
        self.mlb = mlb
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sr = sr
        self.window_duration = window_duration
        self.fixed_time_steps = fixed_time_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.segments_data = self._load_segments_metadata()
        self.on_epoch_end()

    def __len__(self):
        return (len(self.segments_data) + self.batch_size - 1) // self.batch_size

    def _load_segments_metadata(self):
        segments = []
        with open(self.segments_file_path, 'r') as f:
            for line in f:
                segments.append(json.loads(line.strip()))
        return segments

    def augment_spectrogram(self, mel_spec):
        if not self.augment:
            return mel_spec
        augmented = mel_spec.copy()
        if np.random.random() < 0.5:
            time_mask_width = np.random.randint(1, min(20, mel_spec.shape[1] // 4))
            time_mask_start = np.random.randint(0, mel_spec.shape[1] - time_mask_width)
            augmented[:, time_mask_start:time_mask_start + time_mask_width] = -1
        if np.random.random() < 0.5:
            freq_mask_width = np.random.randint(1, min(15, mel_spec.shape[0] // 4))
            freq_mask_start = np.random.randint(0, mel_spec.shape[0] - freq_mask_width)
            augmented[freq_mask_start:freq_mask_start + freq_mask_width, :] = -1
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.05, mel_spec.shape)
            augmented = np.clip(augmented + noise, -1, 1)
        # Pitch shift (new augmentation)
        if np.random.random() < 0.3:
            n_steps = np.random.uniform(-2, 2)
            augmented = librosa.effects.pitch_shift(augmented, sr=self.sr, n_steps=n_steps)
        return augmented

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_segments = [self.segments_data[k] for k in indexes]
        X_batch = []
        y_batch = []
        for segment in batch_segments:
            mel = extract_enhanced_features(
                segment['audio_path'], segment['start'], segment['duration'],
                sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length,
                fixed_time_steps=self.fixed_time_steps
            )
            mel = self.augment_spectrogram(mel)
            if mel.ndim != 2:
                if mel.ndim == 3 and mel.shape[-1] == 1:
                    mel = mel.squeeze(axis=-1)
                else:
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

# --- Cosine Annealing LR ---
def cosine_annealing_lr(epoch):
    max_lr = 0.001
    min_lr = 0.00001
    epochs_per_cycle = 20
    cos_inner = (np.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
    lr = min_lr + (max_lr - min_lr) * (1 + np.cos(cos_inner)) / 2
    return lr

# --- Training Callbacks ---
def train_model(run_dir, val_generator, mlb_classes):
    callbacks = [
        EarlyStopping(monitor='val_macro_f1', patience=30, mode='max', restore_best_weights=True, verbose=1),
        LearningRateScheduler(cosine_annealing_lr, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(run_dir, 'best_model.h5'),
            monitor='val_macro_f1',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ThresholdOptimizer(val_generator, mlb_classes),
        TrainingLogger(RUN_DIR, mlb_classes)
    ]
    return callbacks

# --- Custom History and Plotting Callback ---
class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, mlb_classes):
        super().__init__()
        self.log_dir = log_dir
        self.csv_file_path = os.path.join(log_dir, 'results.csv')
        self.start_time = 0
        self.epoch_times = []
        self.mlb_classes = mlb_classes
        self.history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'macro_f1': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_macro_f1': [],
            'lr': []
        }
        
        self.csv_headers = [
            'epoch', 'time_sec', 
            'train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_macro_f1',
            'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_macro_f1',
            'learning_rate'
        ]
        
        # Initialize CSV file with headers
        with open(self.csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"Training started. Logs will be saved to: {self.log_dir}")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        self.epoch_times.append(elapsed_time)

        # Get learning rate from optimizer
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

        # Store metrics in history
        self.history['loss'].append(logs.get('loss'))
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['precision'].append(logs.get('precision'))
        self.history['recall'].append(logs.get('recall'))
        self.history['macro_f1'].append(logs.get('macro_f1'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))
        self.history['val_precision'].append(logs.get('val_precision'))
        self.history['val_recall'].append(logs.get('val_recall'))
        self.history['val_macro_f1'].append(logs.get('val_macro_f1'))
        self.history['lr'].append(current_lr)

        # Write to CSV
        row = [
            epoch + 1, # epoch starts from 0 in callback, but 1 in CSV
            elapsed_time,
            logs.get('loss', 0.0), logs.get('accuracy', 0.0), logs.get('precision', 0.0), logs.get('recall', 0.0), logs.get('macro_f1', 0.0),
            logs.get('val_loss', 0.0), logs.get('val_accuracy', 0.0), logs.get('val_precision', 0.0), logs.get('val_recall', 0.0), logs.get('val_macro_f1', 0.0),
            current_lr
        ]
        with open(self.csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        print(f"Epoch {epoch+1} completed. Elapsed time: {elapsed_time:.2f}s")

    def on_train_end(self, logs=None):
        print("Training finished. Generating plots...")
        self.plot_metrics()

    def plot_metrics(self):
        epochs = range(1, len(self.history['loss']) + 1)
        
        # Plot Loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.history['loss'], label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'loss_plot.png'))
        plt.close()

        # Plot Macro F1
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.history['macro_f1'], label='Training Macro F1')
        plt.plot(epochs, self.history['val_macro_f1'], label='Validation Macro F1')
        plt.title('Training and Validation Macro F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Macro F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'macro_f1_plot.png'))
        plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    # Configuration
    AUDIO_FILES_DIR = '/home/nele_pauline_suffo/ProcessedData/childlens_audio'
    TRAIN_RTTM_FILE = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/train_small.rttm'
    VAL_RTTM_FILE = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/dev.rttm'
    TEST_RTTM_FILE = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/test.rttm'
    
    SR = 16000
    N_MELS = 256
    HOP_LENGTH = 512
    WINDOW_DURATION = 3.0  # Reduced to minimize label overlap
    WINDOW_STEP = 1.0
    VALID_RTTM_LABELS = ['OHS', 'CDS', 'KCHI']
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_DIR = os.path.join('/home/nele_pauline_suffo/outputs/audio_classification/runs', timestamp)
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"Training run output directory: {RUN_DIR}")
    
    window_duration_str = str(WINDOW_DURATION).replace('.', 'p')
    window_step_str = str(WINDOW_STEP).replace('.', 'p')
    train_segments_file = os.path.join(os.path.dirname(RUN_DIR), f'train_segments_w{window_duration_str}_s{window_step_str}.jsonl')
    val_segments_file = os.path.join(os.path.dirname(RUN_DIR), f'val_segments_w{window_duration_str}_s{window_step_str}.jsonl')
    test_segments_file = os.path.join(os.path.dirname(RUN_DIR), f'test_segments_w{window_duration_str}_s{window_step_str}.jsonl')
    
    current_script_path = os.path.abspath(__file__)
    shutil.copy(current_script_path, os.path.join(RUN_DIR, 'train_audio_classifier_improved.py'))
    
    if (os.path.exists(train_segments_file) and 
        os.path.exists(val_segments_file) and 
        os.path.exists(test_segments_file)):
        print("Existing segment files found. Skipping RTTM extraction...")
        with open(train_segments_file, 'r') as f:
            num_train_segments = sum(1 for line in f)
        with open(val_segments_file, 'r') as f:
            num_val_segments = sum(1 for line in f)
        with open(test_segments_file, 'r') as f:
            num_test_segments = sum(1 for line in f)
        train_unique_labels = set()
        val_unique_labels = set()
        test_unique_labels = set()
        with open(train_segments_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                train_unique_labels.update(data['labels'])
        with open(val_segments_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                val_unique_labels.update(data['labels'])
        with open(test_segments_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                test_unique_labels.update(data['labels'])
        train_unique_labels = sorted(list(train_unique_labels))
        val_unique_labels = sorted(list(val_unique_labels))
        test_unique_labels = sorted(list(test_unique_labels))
        print(f"Loaded existing segment files:")
        print(f"  Training: {num_train_segments} segments")
        print(f"  Validation: {num_val_segments} segments")
        print(f"  Test: {num_test_segments} segments")
    else:
        print("Segment files not found. Extracting from RTTM files...")
        print("--- Processing Training Data ---")
        num_train_segments, train_unique_labels = parse_rttm_for_multi_label_and_save(
            TRAIN_RTTM_FILE, AUDIO_FILES_DIR, VALID_RTTM_LABELS,
            WINDOW_DURATION, WINDOW_STEP, SR, N_MELS, HOP_LENGTH, train_segments_file
        )
        print("\n--- Processing Validation Data ---")
        num_val_segments, val_unique_labels = parse_rttm_for_multi_label_and_save(
            VAL_RTTM_FILE, AUDIO_FILES_DIR, VALID_RTTM_LABELS,
            WINDOW_DURATION, WINDOW_STEP, SR, N_MELS, HOP_LENGTH, val_segments_file
        )
        print("\n--- Processing Test Data ---")
        num_test_segments, test_unique_labels = parse_rttm_for_multi_label_and_save(
            TEST_RTTM_FILE, AUDIO_FILES_DIR, VALID_RTTM_LABELS,
            WINDOW_DURATION, WINDOW_STEP, SR, N_MELS, HOP_LENGTH, test_segments_file
        )
    
    if num_train_segments == 0 or num_val_segments == 0 or num_test_segments == 0:
        print("Error: Missing required segment data. Exiting.")
        exit()
        
    all_possible_target_labels_seen = sorted(list(set(train_unique_labels + val_unique_labels + test_unique_labels)))
    mlb = MultiLabelBinarizer(classes=all_possible_target_labels_seen)
    mlb.fit([[]])
    num_classes = len(mlb.classes_)
    print(f"\nDetected {num_classes} unique target classes: {mlb.classes_}")
    
    if num_classes == 0:
        print("Error: No valid target classes detected. Cannot proceed.")
        exit()
    
    FIXED_TIME_STEPS = int(np.ceil(WINDOW_DURATION * SR / HOP_LENGTH))
    print(f"Fixed Time Steps: {FIXED_TIME_STEPS}")
    print(f"Model input shape: ({N_MELS + 13}, {FIXED_TIME_STEPS}, 1)")
    
    model = build_model_multi_label(n_mels=N_MELS, fixed_time_steps=FIXED_TIME_STEPS, num_classes=num_classes)
    model.log_dir = RUN_DIR
    model.summary()
    
    segments = {"train": [], "val": [], "test": []}
    segment_files = {"train": train_segments_file, "val": val_segments_file, "test": test_segments_file}
    for split, file_path in segment_files.items():
        with open(file_path, 'r') as f:
            for line in f:
                segments[split].append(json.loads(line.strip()))
    
    train_segments = segments["train"]
    val_segments = segments["val"]
    test_segments = segments["test"]
    
    class_weights_for_keras = calculate_balanced_class_weights(train_segments, mlb)
    print(f"\nClass distribution in training data:")
    for i, class_name in enumerate(mlb.classes_):
        count = sum(1 for seg in train_segments if class_name in seg['labels'])
        percentage = (count / len(train_segments)) * 100 if len(train_segments) > 0 else 0
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    print(f"\nCalculated Class Weights: {class_weights_for_keras}")
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_segments)} segments")
    print(f"  Validation: {len(val_segments)} segments")
    if test_segments:
        print(f"  Test: {len(test_segments)} segments")
    
    train_generator = EnhancedAudioSegmentDataGenerator(
        train_segments_file, mlb, N_MELS, HOP_LENGTH, SR, WINDOW_DURATION, FIXED_TIME_STEPS,
        batch_size=32, shuffle=True, augment=True
    )
    val_generator = EnhancedAudioSegmentDataGenerator(
        val_segments_file, mlb, N_MELS, HOP_LENGTH, SR, WINDOW_DURATION, FIXED_TIME_STEPS,
        batch_size=32, shuffle=False, augment=False
    )
    test_generator = None
    if test_segments:
        test_generator = EnhancedAudioSegmentDataGenerator(
            test_segments_file, mlb, N_MELS, HOP_LENGTH, SR, WINDOW_DURATION, FIXED_TIME_STEPS,
            batch_size=32, shuffle=False, augment=False
        )
        print(f"Test generator created successfully.")
    
    callbacks = train_model(RUN_DIR, val_generator, mlb.classes_)
    
    print("\nStarting training...")
    if len(train_generator) == 0 or len(val_generator) == 0:
        print("Error: No batches available for training/validation. Check your RTTM files and data paths.")
        exit()
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=100,  # Increased epochs
        callbacks=callbacks,
        class_weight=class_weights_for_keras,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("FINAL VALIDATION RESULTS (from best restored weights)")
    print("="*60)
    val_results = model.evaluate(val_generator, verbose=0)
    val_metrics_dict = dict(zip(model.metrics_names, val_results))
    for name, value in val_metrics_dict.items():
        print(f"Validation {name}: {value:.4f}")
    
    if test_generator and len(test_generator) > 0:
        print("\n" + "="*60)
        print("DETAILED TEST SET EVALUATION")
        print("="*60)
        test_predictions = model.predict(test_generator, verbose=1)
        test_true_labels = []
        for i in tqdm(range(len(test_generator)), desc="Collecting true labels"):
            _, labels = test_generator[i]
            test_true_labels.extend(labels)
        test_true_labels = np.array(test_true_labels)
        
        if test_predictions.shape[0] != test_true_labels.shape[0]:
            print(f"Warning: Test predictions ({test_predictions.shape[0]}) and true labels ({test_true_labels.shape[0]}) mismatch.")
            min_samples = min(test_predictions.shape[0], test_true_labels.shape[0])
            test_predictions = test_predictions[:min_samples]
            test_true_labels = test_true_labels[:min_samples]
            print(f"Adjusted to {min_samples} samples for evaluation.")
        
        # Load optimized thresholds
        try:
            with open(os.path.join(RUN_DIR, 'thresholds.json'), 'r') as f:
                thresholds_dict = json.load(f)
            thresholds = [thresholds_dict[class_name] for class_name in mlb.classes_]
        except:
            thresholds = [0.5] * num_classes
        
        test_pred_binary = np.array([(test_predictions[:, i] > thresholds[i]).astype(int) for i in range(num_classes)]).T
        
        if test_true_labels.sum() > 0:
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                test_true_labels, test_pred_binary, average=None, zero_division=0
            )
            macro_precision = precision_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
            macro_recall = recall_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
            macro_f1 = f1_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
            subset_accuracy = accuracy_score(test_true_labels, test_pred_binary)
            
            print(f"\nTest Metrics (thresholds {thresholds}):")
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
    
    print("\nTraining completed successfully! Check the run directory for logs and plots.")