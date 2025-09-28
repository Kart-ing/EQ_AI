#!/usr/bin/env python3
"""
Convert EQ Neural Network PyTorch model to ONNX format
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os

# -----------------------------
# Model Architecture (must match training)
# -----------------------------
class EQNeuralNetwork(nn.Module):
    def __init__(self, input_size=12):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

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
            nn.Linear(8, 15)
        )

    def forward(self, features):
        return self.network(features)

# -----------------------------
# Conversion Function
# -----------------------------
def convert_pth_to_onnx(
    pth_path: str,
    onnx_path: str,
    input_size: int = 12,
    batch_size: int = 1,  # Dynamic batch size support
    opset_version: int = 13
):
    """
    Convert PyTorch .pth model to ONNX format
    
    Args:
        pth_path: Path to the saved PyTorch model
        onnx_path: Output path for ONNX model
        input_size: Size of input features (default: 12)
        batch_size: Batch size for export (use -1 for dynamic)
        opset_version: ONNX opset version
    """
    
    # Check if input file exists
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model file not found: {pth_path}")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model architecture
    model = EQNeuralNetwork(input_size=input_size).to(device)
    
    # Load trained weights
    try:
        # For models saved with model.state_dict()
        model.load_state_dict(torch.load(pth_path, map_location=device))
    except:
        # For models saved with entire model
        model = torch.load(pth_path, map_location=device)
    
    model.eval()
    print(f"Model loaded successfully from {pth_path}")
    
    # Create dummy input for tracing
    if batch_size == -1:
        # Dynamic batch size
        dummy_input = torch.randn(1, input_size, device=device)
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        # Fixed batch size
        dummy_input = torch.randn(batch_size, input_size, device=device)
        dynamic_axes = None
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Export to ONNX
    torch.onnx.export(
        model,                    # Model to export
        dummy_input,              # Example input
        onnx_path,                # Output file
        export_params=True,       # Store trained parameters
        opset_version=opset_version,  # ONNX opset version
        do_constant_folding=True, # Optimize constants
        input_names=['input'],    # Input name
        output_names=['output'],  # Output name
        dynamic_axes=dynamic_axes, # Dynamic axes (if batch_size=-1)
        verbose=True              # Print conversion details
    )
    
    print(f"✓ ONNX model successfully exported to: {onnx_path}")
    
    # Verify the ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # Print model info
        print(f"✓ ONNX model input: {onnx_model.graph.input[0]}")
        print(f"✓ ONNX model output: {onnx_model.graph.output[0]}")
        
    except ImportError:
        print("⚠ ONNX not installed, skipping model verification")
    except Exception as e:
        print(f"⚠ ONNX model verification failed: {e}")

# -----------------------------
# Additional Utility: Test Inference
# -----------------------------
def test_model_inference(pth_path: str, input_size: int = 12):
    """Test the model with sample input to verify it works"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = EQNeuralNetwork(input_size=input_size).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    
    # Create test input
    test_input = torch.randn(1, input_size, device=device)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Sample output values: {output[0][:5]}")  # First 5 values
    
    return output

# -----------------------------
# Command Line Interface
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--pth_path', type=str, required=True,
                       help='Path to input .pth model file')
    parser.add_argument('--onnx_path', type=str, default='eq_model.onnx',
                       help='Path for output ONNX model (default: eq_model.onnx)')
    parser.add_argument('--input_size', type=int, default=12,
                       help='Input feature size (default: 12)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for export (-1 for dynamic, default: 1)')
    parser.add_argument('--opset', type=int, default=13,
                       help='ONNX opset version (default: 13)')
    parser.add_argument('--test', action='store_true',
                       help='Test model inference before conversion')
    
    args = parser.parse_args()
    
    # Test model if requested
    if args.test:
        print("Testing model inference...")
        test_model_inference(args.pth_path, args.input_size)
        print()
    
    # Convert to ONNX
    convert_pth_to_onnx(
        pth_path=args.pth_path,
        onnx_path=args.onnx_path,
        input_size=args.input_size,
        batch_size=args.batch_size,
        opset_version=args.opset
    )

if __name__ == "__main__":
    main()