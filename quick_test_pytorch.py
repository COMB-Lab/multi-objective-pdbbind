#!/usr/bin/env python3
"""
Quick test of PyTorch PGGCN training with real PDBbind data.
Tests with 10 structures and 5 epochs to verify everything works.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle

from multi_objective_pdbbind_pytorch import (
    load_data, prepare_features_chunked, DatasetConfig, 
    combined_loss, LossTracker, get_memory_usage
)
from models.layers_pytorch import PGGCNModel


def main():
    print('🚀 Quick PyTorch PGGCN Training Test')
    print('=' * 70)
    print('Configuration: 10 structures, 5 epochs, single physics weight')
    print('=' * 70 + '\n')
    
    # Configuration
    config = DatasetConfig(
        dataset_size=10,
        batch_size=2,
        epochs=5,
        learning_rate=1e-5,
        physics_weights=[1e-5]  # Single weight for quick test
    )
    
    print(config)
    
    # Load data
    print('Loading data...')
    df, physics_info = load_data(dataset_size=10)
    print(f'  ✓ Loaded {len(df)} complexes\n')
    
    # Prepare features
    print('Preparing features...')
    X, y, physics_all = prepare_features_chunked(
        df, physics_info, max_padding=3000, config=config
    )
    print(f'  ✓ Prepared {len(X)} samples\n')
    
    # Train-test split
    X_train, X_test, y_train, y_test, physics_train, physics_test = train_test_split(
        X, y, physics_all, test_size=0.2, random_state=42
    )
    
    print(f'Data Split:')
    print(f'  - Train: {len(X_train)} samples')
    print(f'  - Test:  {len(X_test)} samples\n')
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    model = PGGCNModel(num_atom_features=36, r_out_channel=20, c_out_channel=1024)
    model.add_rule('sum', 0, 32)
    model.add_rule('multiply', 32, 33)
    model.add_rule('distance', 33, 36)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}\n')
    
    # Training loop
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    tracker = LossTracker()
    
    print('Starting training...')
    print('-' * 70)
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for i in range(0, len(X_train), config.batch_size):
            batch_end = min(i + config.batch_size, len(X_train))
            batch_X = X_train[i:batch_end]
            batch_y = torch.tensor(y_train[i:batch_end], dtype=torch.float32).to(device)
            
            # Adaptive padding
            max_atoms = max(x.shape[0] for x in batch_X)
            padded_X = []
            for x in batch_X:
                if x.shape[0] < max_atoms:
                    x = torch.nn.functional.pad(x, (0, 0, 0, max_atoms - x.shape[0]))
                padded_X.append(x)
            batch_X = torch.stack(padded_X).to(device)
            
            optimizer.zero_grad()
            model_output = model(batch_X, training=True)
            total_loss, emp_loss, phys_loss = combined_loss(batch_y, model_output, physics_weight=1e-5)
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        tracker.update(avg_loss)
        print(f'Epoch {epoch+1}/{config.epochs}: Loss = {avg_loss:.4f}')
    
    print('-' * 70)
    
    # Evaluation
    print('\n✓ Training completed successfully!')
    print(f'  Final loss: {avg_loss:.4f}')
    print(f'  Memory usage: {get_memory_usage():.2f} GB\n')
    
    # Test inference
    print('Running inference on test set...')
    model.eval()
    with torch.no_grad():
        predictions = []
        for i in range(0, len(X_test), config.batch_size):
            batch_end = min(i + config.batch_size, len(X_test))
            batch_X = X_test[i:batch_end]
            
            max_atoms = max(x.shape[0] for x in batch_X)
            padded_X = []
            for x in batch_X:
                if x.shape[0] < max_atoms:
                    x = torch.nn.functional.pad(x, (0, 0, 0, max_atoms - x.shape[0]))
                padded_X.append(x)
            batch_X = torch.stack(padded_X).to(device)
            
            output = model(batch_X, training=False)
            predictions.extend(output[:, 0].cpu().numpy())
    
    predictions = np.array(predictions)
    test_rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    test_mae = np.mean(np.abs(y_test - predictions))
    
    print(f'  ✓ Test RMSE: {test_rmse:.4f}')
    print(f'  ✓ Test MAE:  {test_mae:.4f}\n')
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"PGGCN_quick_test_pytorch_{timestamp}.pkl"
    
    results = {
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'losses': tracker.train_losses,
        'y_test': y_test,
        'y_pred': predictions,
        'config': config.__dict__,
        'timestamp': timestamp,
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f'💾 Results saved to {results_file}')
    print('\n✅ Quick test completed successfully!')


if __name__ == '__main__':
    main()
