#!/usr/bin/env python3
"""
SMART GPU Training for FMA - Focus on Generalization, Not Overfitting
"""

import modal
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tqdm
from sklearn.feature_selection import SelectKBest, f_classif

# Modal config
GPU_CONFIG = "A100-80GB"
image = modal.Image.debian_slim().pip_install([
    "numpy", "torch", "torchvision", "tqdm", "scikit-learn", "pandas", "onnx"
])

# Dataset with Stronger Augmentation
class FMADataset(Dataset):
    def __init__(self, features_path="/fma_vol/optimized_features.npz", augment=True, k_best=60):
        data = np.load(features_path)
        self.X = data['features'].astype(np.float32)
        self.y = data['genres'].astype(np.int64)
        self.augment = augment
        
        # FEATURE SELECTION - Add this block
        if k_best and self.X.shape[1] > k_best:
            print(f"Selecting top {k_best} features from {self.X.shape[1]}...")
            selector = SelectKBest(f_classif, k=k_best)
            self.X = selector.fit_transform(self.X, self.y)
            print(f"Reduced to {self.X.shape[1]} features")
        
        # Then normalize as before
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
        print(f"Loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Class distribution: {np.bincount(self.y)}")


    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        # For temporal model: create sequence of 10 feature vectors
        # Use overlapping windows from the dataset
        seq_length = 10
        if idx + seq_length > len(self.X):
            idx = len(self.X) - seq_length  # Adjust if near end
        
        sequence = self.X[idx:idx+seq_length]  # [10, 60]
        sequence = sequence.reshape(-1)  # Flatten to [600]
        
        # Use the label from the middle of the sequence for stability
        label_idx = idx + seq_length // 2
        label = self.y[label_idx]
        
        return torch.from_numpy(sequence).float(), torch.tensor(label)

class TemporalGenreClassifier(nn.Module):
    def __init__(self, input_len=60, num_classes=16, temporal_size=10):  # Note: 60 features now!
        super().__init__()
        self.temporal_size = temporal_size
        self.input_len = input_len  # Store this
        
        # CNN for local patterns within each time step
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 128, 3, padding=1), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(), nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # LSTM for temporal context across windows
        self.lstm = nn.LSTM(256, 128, 2, batch_first=True, dropout=0.3, bidirectional=True)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.float()  # ← ADD THIS LINE - convert to float32
        
        batch_size = x.shape[0]
        
        # Reshape to: [batch, temporal_size, features]
        # Total features = temporal_size * input_len (10 * 60 = 600)
        x = x.view(batch_size, self.temporal_size, self.input_len)  # [batch, 10, 60]
        x = x.unsqueeze(2)  # [batch, 10, 1, 60]
        x = x.view(batch_size * self.temporal_size, 1, self.input_len)  # [batch*10, 1, 60]
        
        # Extract features for each time step
        x = self.feature_extractor(x)  # [batch*10, 256, 1]
        x = x.squeeze(-1)  # [batch*10, 256]
        x = x.view(batch_size, self.temporal_size, -1)  # [batch, 10, 256]
        
        # Temporal modeling
        lstm_out, (hn, cn) = self.lstm(x)
        h_last = torch.cat((hn[-2], hn[-1]), dim=1)  # [batch, 256]
        
        return self.classifier(h_last)
    
# SMART Training with Overfitting Detection
def train_model(epochs=200, batch_size=256, lr=1e-4):  # Smaller batch, lower LR
    # After creating dataset, use the actual feature count
    full_dataset = FMADataset(augment=False, k_best=60)  # Try 60, 70, or None for all
    
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(full_dataset.y), y=full_dataset.y)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights: {class_weights}")
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    train_dataset, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    train_dataset.dataset.augment = True  # Augment only training
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalGenreClassifier(input_len=full_dataset.X.shape[1]).to(device)  # ONLY ONCE
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # STRONGER Regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)  # 10x more weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # SMART Tracking
    best_val_acc = 0
    best_gap = float('inf')  # Track overfitting gap
    best_epoch = 0
    patience = 50
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_correct = train_total = 0
        
        for X, y in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_correct += (outputs.argmax(1) == y).sum().item()
            train_total += y.size(0)
        
        train_acc = train_correct / train_total
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Calculate overfitting gap
        overfitting_gap = train_acc - val_acc
        
        # SMART Model Selection
        improved = False
        if val_acc > best_val_acc:
            improved = True
        elif val_acc >= best_val_acc - 0.01 and overfitting_gap < best_gap:
            improved = True  # Similar accuracy but better generalization
        
        if improved:
            best_val_acc = val_acc
            best_gap = overfitting_gap
            best_epoch = epoch
            torch.save(model.state_dict(), "/fma_vol/best_model.pth")
            print(f"🎯 NEW BEST! Val: {val_acc:.4f}, Gap: {overfitting_gap:.4f}")
        
        print(f"Epoch {epoch+1}: Train: {train_acc:.4f}, Val: {val_acc:.4f}, "
              f"Gap: {overfitting_gap:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Early stopping if severe overfitting
        if overfitting_gap > 0.15:  # 15% gap is too much
            print(f"🛑 Stopping - severe overfitting (gap: {overfitting_gap:.4f})")
            break
        
        # Patience based on no improvement
        if epoch - best_epoch > patience:
            print(f"⏹️ Early stopping - no improvement for {patience} epochs")
            break

    # Load best model
    model.load_state_dict(torch.load("/fma_vol/best_model.pth"))
    final_val_acc = evaluate(model, val_loader, criterion, device)[1]
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Epoch: {best_epoch + 1}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    
    # Export
    dummy_input = torch.randn(1, 1, full_dataset.X.shape[1], device=device)
    torch.onnx.export(model, dummy_input, "/fma_vol/genre_model.onnx", 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    
    print("✅ Training complete! Model exported.")

def evaluate(model, loader, criterion, device):
    model.eval()
    loss = correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss += criterion(outputs, y).item() * X.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss/total, correct/total

# Modal app
app = modal.App("fma-trainer-smart")

@app.function(image=image, gpu=GPU_CONFIG, timeout=3600*12,
              volumes={"/fma_vol": modal.Volume.from_name("fma_features_volume")})
def main():
    train_model(epochs=200, batch_size=256, lr=1e-4)

if __name__ == "__main__":
    main()