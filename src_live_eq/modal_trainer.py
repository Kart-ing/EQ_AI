#!/usr/bin/env python3
"""
Modal GPU Training for EQ Neural Network
Using precomputed features (.npy) stored in a Modal Volume
"""

import modal

# -----------------------------
# 1️⃣ Create Modal App & Image
# -----------------------------
app = modal.App("eq-training-npy")

image = modal.Image.debian_slim().pip_install([
    "torch>=2.0.0",
    "numpy",
    "tqdm"
])

# GPU configuration
GPU_CONFIG = "A100-40GB"

# -----------------------------
# 2️⃣ Volume for precomputed features
# -----------------------------
# Make sure you create this volume first via CLI:
#   modal volume create eq_features_volume
features_volume = modal.Volume.from_name("eq_features_volume")

# Paths inside the container
FEATURES_PATH = "/mnt/features/features/features.npy"
TARGETS_PATH  = "/mnt/features/features/targets.npy"


# -----------------------------
# 3️⃣ Training Function
# -----------------------------
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    memory=32*1024,
    volumes={"/mnt/features": features_volume},
)

def train_eq_model_npy(
    features_path: str = FEATURES_PATH,
    targets_path: str = TARGETS_PATH,
    batch_size: int = 256,
    epochs: int = 100,
    learning_rate: float = 0.0005
):
    """Train EQ model on Modal A100 using precomputed features"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # -----------------------------
    # Dataset
    # -----------------------------
    class EQNpyDataset(Dataset):
        def __init__(self, features_file, targets_file):
            self.X = np.load(features_file)
            self.y = np.load(targets_file)

            self.X = (self.X - self.X.mean(axis=0)) / (self.X.std(axis=0) + 1e-8)
            self.y = self.y / 1000.0

            print(f"Loaded {len(self.X)} samples")

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return (
                torch.FloatTensor(self.X[idx]),
                torch.FloatTensor(self.y[idx])
            )

    # -----------------------------
    # Model
    # -----------------------------
    # -----------------------------
# Model
# -----------------------------
    class EQNeuralNetwork(nn.Module):
        def __init__(self, input_size=12):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 512),   # Wider first layer
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),              # Reduced dropout

                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.15),

                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.1),

                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.1),

                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.05),

                nn.Linear(32, 16),
                nn.ReLU(),
                nn.BatchNorm1d(16),
                nn.Dropout(0.05),

                nn.Linear(16, 8),
                nn.ReLU(),

                nn.Linear(8, 15)              # output
            )

        def forward(self, features):
            return self.network(features)



    # -----------------------------
    # Load dataset
    # -----------------------------
    dataset = EQNpyDataset(features_path, targets_path)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # -----------------------------
    # Initialize model, optimizer, scheduler, loss
    # -----------------------------
    model = EQNeuralNetwork(input_size=12).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    loss_fn = nn.L1Loss()

    # 🔹 Scheduler: Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,       # reduce LR by half
        patience=5
    )

    best_val_loss = float('inf')

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for features, targets in train_pbar:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(features)
                loss = loss_fn(outputs, targets)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Update scheduler
        scheduler.step(avg_val_loss)


        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), '/tmp/best_eq_model.pth')
            print(f"New best model saved! Val loss: {best_val_loss:.4f}")

    # Return model
    with open('/tmp/best_eq_model.pth', 'rb') as f:
        model_data = f.read()

    return {
        'model_data': model_data,
        'final_train_loss': avg_train_loss,
        'best_val_loss': best_val_loss,
        'epochs_completed': epochs,
        'total_params': total_params
    }

# -----------------------------
# 4️⃣ Local entrypoint
# -----------------------------
@app.local_entrypoint()
def main():
    result = train_eq_model_npy.remote()

    if result:
        print(f"Training finished. Best val loss: {result['best_val_loss']:.4f}")
        with open('fma_eq_model_npy.pth', 'wb') as f:
            f.write(result['model_data'])
        print("Model saved as 'fma_eq_model_npy.pth'")