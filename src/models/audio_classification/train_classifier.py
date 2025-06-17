import torch
import torch.nn as nn
import torchaudio
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Load BERT for semantic features
bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")

class CNNBranch(nn.Module):
    def __init__(self):
        super(CNNBranch, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        x = self.conv_layers(x)  # [B, 256, H, W]
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B, 256]
        x = self.fc(x)  # [B, 256]
        return x

class FeatureBranch(nn.Module):
    def __init__(self, input_dim=800):  # 32 prosodic + 768 semantic
        super(FeatureBranch, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.fc_layers(x)  # [B, 256]

class TransformerBranch(nn.Module):
    def __init__(self, input_dim=512, num_layers=4, num_heads=8, ff_dim=1024):
        super(TransformerBranch, self).__init__()
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 512)

    def forward(self, x):
        x = x.unsqueeze(1) + self.pos_encoder  # [B, 1, 512]
        x = self.transformer(x)  # [B, 1, 512]
        x = x.mean(dim=1)  # [B, 512]
        x = self.fc(x)  # [B, 512]
        return x

class HybridCNNTransformer(nn.Module):
    def __init__(self):
        super(HybridCNNTransformer, self).__init__()
        self.cnn_branch = CNNBranch()
        self.feature_branch = FeatureBranch()
        self.transformer_branch = TransformerBranch()
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3)  # 3 classes: CS, CDS, OHS
        )

    def forward(self, spectrogram, features):
        cnn_out = self.cnn_branch(spectrogram)  # [B, 256]
        feature_out = self.feature_branch(features)  # [B, 256]
        combined = torch.cat([cnn_out, feature_out], dim=-1)  # [B, 512]
        transformer_out = self.transformer_branch(combined)  # [B, 512]
        logits = self.classifier(transformer_out)  # [B, 3]
        return logits

# Data preprocessing
def preprocess_segment(audio, sr=16000, duration=5.0):
    # Resample and pad/truncate to fixed length
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
    target_samples = int(duration * 16000)
    if audio.shape[1] < target_samples:
        audio = torch.nn.functional.pad(audio, (0, target_samples - audio.shape[1]))
    else:
        audio = audio[:, :target_samples]
    
    # Log-mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=80, win_length=400, hop_length=160
    )
    spectrogram = mel_transform(audio)  # [1, 80, T]
    spectrogram = torch.log(spectrogram + 1e-9).unsqueeze(0)  # [1, 1, 80, T]

    # Prosodic features (from your script)
    prosodic_features = extract_prosodic_features(audio.numpy()[0], 16000)

    # Semantic features (transcription + BERT)
    text = transcribe_audio_segment(audio.numpy()[0], 16000)
    semantic_features = extract_semantic_features(text, use_bert=True)

    features = np.concatenate([prosodic_features, semantic_features])
    return spectrogram, torch.tensor(features, dtype=torch.float32)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.25, 0.25, 0.5], gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Training loop (simplified)
model = HybridCNNTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = FocalLoss()

for epoch in range(50):
    model.train()
    for spectrogram, features, labels in train_loader:
        spectrogram, features, labels = spectrogram.to(device), features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(spectrogram, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()