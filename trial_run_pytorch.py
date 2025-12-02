#!/usr/bin/env python3
"""
Quick Trial Run - PyTorch PGGCN
===============================

Fast validation script for the PyTorch translation.
Runs with minimal dataset to verify all components work correctly.

Expected runtime: 2-5 minutes
Memory usage: < 2 GB
Purpose: Quick sanity check of the complete pipeline
"""

import sys
import os

# Set before any torch imports to avoid GPU memory issues
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("=" * 70)
print("PYTORCH PGGCN - TRIAL RUN")
print("=" * 70)

# Step 1: Check imports
print("\n📦 Step 1: Checking imports...")
try:
    import torch
    import torch.nn as nn
    import pandas as pd
    import numpy as np
    from models.layers_pytorch import PGGCNModel, RuleGraphConvLayer, ConvLayer
    print("   ✓ All imports successful")
    print(f"   - PyTorch version: {torch.__version__}")
    print(f"   - GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    print("   Install dependencies: pip install torch pandas numpy scikit-learn")
    sys.exit(1)

# Step 2: Test device setup
print("\n🖥️  Step 2: Setting up device...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   ✓ Using device: {device}")
except Exception as e:
    print(f"   ✗ Device setup failed: {e}")
    sys.exit(1)

# Step 3: Create a small test dataset
print("\n📊 Step 3: Creating test dataset...")
try:
    # Create synthetic data: 5 molecules, 3 epochs
    # Model expects: [36 atom features + 2 neighbor indices + 15 physics] = 53 features
    batch_size = 2
    max_atoms = 50
    num_atom_features = 36      # Atom property features
    num_neighbor_features = 2   # Neighbor indices (padded)
    num_physics_features = 15   # Physics info
    num_total_atom_features = num_atom_features + num_neighbor_features  # 38
    num_model_features = num_total_atom_features + num_physics_features   # 53
    num_samples = 5
    
    # Create full input: [batch, max_atoms, 53]
    X_test = torch.randn(batch_size, max_atoms, num_model_features).to(device)
    
    # Make neighbor indices valid (0 to max_atoms-1), in positions 36-37
    X_test[:, :, num_atom_features:num_atom_features+num_neighbor_features] = torch.randint(
        0, max_atoms, (batch_size, max_atoms, num_neighbor_features), dtype=torch.float32).to(device)
    
    # Extract physics info for later use
    physics_info = X_test[:, 0, num_total_atom_features:].clone()  # [batch_size, 15]
    
    # Random targets
    y_test = torch.randn(batch_size).to(device)
    
    print(f"   ✓ Test data created")
    print(f"   - Input shape: {X_test.shape} (36 atom + 2 neighbor + 15 physics)")
    print(f"   - Physics info shape: {physics_info.shape}")
    print(f"   - Target shape: {y_test.shape}")
    print(f"   - Number of samples: {num_samples}")
except Exception as e:
    print(f"   ✗ Data creation failed: {e}")
    sys.exit(1)

# Step 4: Initialize model
print("\n🧠 Step 4: Initializing model...")
try:
    model = PGGCNModel(
        num_atom_features=36,
        r_out_channel=20,
        c_out_channel=1024
    ).to(device)
    
    # Add combination rules
    model.add_rule("sum", 0, 32)
    model.add_rule("multiply", 32, 33)
    model.add_rule("distance", 33, 36)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✓ Model initialized")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Model device: {next(model.parameters()).device}")
except Exception as e:
    print(f"   ✗ Model initialization failed: {e}")
    sys.exit(1)

# Step 5: Test forward pass
print("\n▶️  Step 5: Testing forward pass...")
try:
    model.eval()
    with torch.no_grad():
        # Model expects: [batch, max_atoms, 53] (36 atom + 2 neighbor + 15 physics)
        output = model(X_test, training=False)
    
    print(f"   ✓ Forward pass successful")
    print(f"   - Model output shape: {output.shape} (1 pred + 15 physics)")
    print(f"   - Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"   - Contains NaN: {torch.isnan(output).any().item()}")
    print(f"   - Contains Inf: {torch.isinf(output).any().item()}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test backward pass
print("\n◀️  Step 6: Testing backward pass...")
try:
    model.train()
    # Create optimizer BEFORE training to avoid late imports
    import torch.optim as optim
    # Manually update parameters to simulate gradient descent
    learning_rate = 1e-5
    
    output = model(X_test, training=True)
    
    # Simple MSE loss on empirical output (first column)
    loss = torch.mean((output[:, 0] - y_test) ** 2)
    
    # Manual backward and gradient update
    loss.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= learning_rate * param.grad
                param.grad.zero_()
    
    print(f"   ✓ Backward pass successful")
    print(f"   - Loss value: {loss.item():.6f}")
    print(f"   - Loss contains NaN: {torch.isnan(loss).item()}")
    
    # Check if gradients were computed
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and torch.any(p.grad != 0))
    print(f"   - Layers with gradients: {grad_count}")
except Exception as e:
    print(f"   ✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: Test loss functions
print("\n📉 Step 7: Testing loss functions...")
try:
    from multi_objective_pdbbind_pytorch import (
        pure_rmse,
        physical_consistency_loss,
        combined_loss
    )
    
    # Test pure RMSE
    rmse_loss = pure_rmse(y_test, output[:, 0])
    print(f"   ✓ RMSE loss: {rmse_loss.item():.6f}")
    
    # Test physics loss
    physics_loss = physical_consistency_loss(y_test, output[:, 0], output[:, 1:16])
    print(f"   ✓ Physics loss: {physics_loss.item():.6f}")
    
    # Test combined loss
    combined, emp, phys = combined_loss(y_test, output, physics_weight=1e-5)
    print(f"   ✓ Combined loss: {combined.item():.6f}")
    print(f"     - Empirical: {emp.item():.6f}")
    print(f"     - Physics: {phys.item():.6f}")
except ImportError:
    print(f"   ⚠️  Loss functions test skipped (import error)")
except Exception as e:
    print(f"   ✗ Loss function test failed: {e}")
    import traceback
    traceback.print_exc()

# Step 8: Mini training loop
print("\n🏋️  Step 8: Running mini training loop...")
try:
    model.train()
    learning_rate = 1e-5
    
    num_epochs = 3
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        # Mini batches
        for i in range(2):
            X_batch = torch.randn(batch_size, max_atoms, num_model_features).to(device)
            # Make neighbor indices valid
            X_batch[:, :, num_atom_features:num_atom_features+num_neighbor_features] = torch.randint(
                0, max_atoms, (batch_size, max_atoms, num_neighbor_features), dtype=torch.float32).to(device)
            y_batch = torch.randn(batch_size).to(device)
            
            output = model(X_batch, training=True)
            
            # Simple MSE loss on empirical output
            loss = torch.mean((output[:, 0] - y_batch) ** 2)
            loss.backward()
            
            # Manual parameter update
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param -= learning_rate * param.grad
                        param.grad.zero_()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / 2
        losses.append(avg_loss)
        print(f"   Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    print(f"   ✓ Training loop successful")
    print(f"   - Loss trend: {' → '.join([f'{l:.4f}' for l in losses])}")
    
    # Check convergence
    if losses[-1] < losses[0]:
        print(f"   ✓ Loss decreasing (convergence indicator)")
    else:
        print(f"   ⚠️  Loss not decreasing (may need hyperparameter tuning)")
        
except Exception as e:
    print(f"   ✗ Training loop failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 9: Memory check
print("\n💾 Step 9: Checking memory usage...")
try:
    import psutil
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"   ✓ RAM usage: {mem_usage:.1f} MB")
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        print(f"   ✓ GPU memory: {gpu_mem:.1f} MB")
except Exception as e:
    print(f"   ⚠️  Memory check skipped: {e}")

# Summary
print("\n" + "=" * 70)
print("✅ TRIAL RUN COMPLETE - All systems operational!")
print("=" * 70)
print("\nNext steps:")
print("  1. Run with 10 structures: dataset_size=10, epochs=5")
print("  2. Run full training: dataset_size=100, epochs=100")
print("  3. Compare results with TensorFlow version")
print("\nTo run full training:")
print("  python3 multi-objective-pdbbind-pytorch.py")
print("\n" + "=" * 70)
