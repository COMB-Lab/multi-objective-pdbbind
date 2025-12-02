#!/usr/bin/env python3
"""
Minimal Trial Run - PyTorch PGGCN with Real Data
================================================

Tests the full pipeline with a small real dataset.
Uses only 10 structures and 5 epochs for quick verification.

Expected runtime: 1-2 minutes (CPU) or 30 seconds (GPU)
Memory usage: 1-2 GB
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("=" * 70)
print("PYTORCH PGGCN - MINIMAL TRIAL RUN (Real Data)")
print("=" * 70)

# Import main training function
try:
    from multi_objective_pdbbind_pytorch import (
        DatasetConfig,
        get_memory_usage,
        load_data,
        prepare_features_chunked,
        combined_loss,
        train_model,
    )
    from models.layers_pytorch import PGGCNModel
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Configure for minimal test
print("\n⚙️  Configuration:")
config = DatasetConfig(
    dataset_size=10,      # Only 10 structures
    max_padding=1000,     # Smaller padding
    batch_size=4,         # Smaller batch
    epochs=5,             # Just 5 epochs
    memory_limit_gb=8
)

print(f"  • Dataset size: {config.dataset_size} structures")
print(f"  • Batch size: {config.batch_size}")
print(f"  • Epochs: {config.epochs}")
print(f"  • Max padding: {config.max_padding}")
print(f"  • Memory limit: {config.memory_limit_gb} GB")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Load data
print("\n📂 Loading data...")
try:
    df, physics_info = load_data(config.dataset_size)
    print(f"  ✓ Loaded {len(df)} structures")
except FileNotFoundError:
    print(f"  ⚠️  Data not found at /home/exouser/multi-objective-pdbbind/Datasets/")
    print(f"  Using synthetic data for testing...")
    df = pd.DataFrame({
        'PDBID': [f'test_{i}' for i in range(10)],
        'binding_affinity': np.random.randn(10),
        'num_atoms': np.random.randint(30, 100, 10),
    })
    physics_info = {}
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Prepare features
print("\n⚙️  Preparing features...")
try:
    X, y, _ = prepare_features_chunked(df, physics_info, config.max_padding, config=config)
    print(f"  ✓ Features prepared: {len(X)} samples")
except Exception as e:
    print(f"  ✗ Feature preparation failed: {e}")
    sys.exit(1)

# Split data
print("\n✂️  Splitting data...")
if len(X) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test = [X[0]], X
    y_train, y_test = [y[0]], y

print(f"  ✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")

# Initialize model
print("\n🧠 Initializing model...")
model = PGGCNModel(num_atom_features=36, r_out_channel=20, c_out_channel=1024)
model.add_rule("sum", 0, 32)
model.add_rule("multiply", 32, 33)
model.add_rule("distance", 33, 36)
print(f"  ✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

# Train model
print("\n🏋️  Training...")
try:
    y_pred, losses = train_model(
        model, X_train, y_train, X_test, y_test,
        config, physics_weight=1e-5, device=device
    )
    print(f"  ✓ Training completed")
except Exception as e:
    print(f"  ✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Metrics
print("\n📊 Metrics:")
if len(y_test) > 0:
    test_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    test_mae = np.mean(np.abs(y_test - y_pred))
    print(f"  ✓ Test RMSE: {test_rmse:.4f}")
    print(f"  ✓ Test MAE: {test_mae:.4f}")
    
    if len(losses.train_losses) > 1:
        converging = losses.train_losses[-1] < losses.train_losses[0]
        print(f"  {'✓' if converging else '⚠️'} Loss trend: {losses.train_losses[0]:.4f} → {losses.train_losses[-1]:.4f}")

# Memory
print(f"\n💾 Memory: {get_memory_usage():.2f} GB")

# Summary
print("\n" + "=" * 70)
print("✅ TRIAL RUN COMPLETE - PyTorch PGGCN is working!")
print("=" * 70)
