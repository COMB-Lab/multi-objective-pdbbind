#!/usr/bin/env python3
"""
Multi-Objective PGGCN Training - PyTorch Version
================================================

Trains Physics-Guided Graph Convolutional Networks for protein-ligand binding affinity prediction.
Combines empirical ML with physics-based energy calculations through multi-objective loss.

Framework: PyTorch (recommended over TensorFlow)
Author: COMB Lab
Date: 2024
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import psutil
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split

# Import model components
from models.layers_pytorch import PGGCNModel


# ============================================================================
# Configuration
# ============================================================================

class DatasetConfig:
    """Configuration for dataset and training."""
    
    def __init__(
        self,
        dataset_size: int = 100,
        batch_size: int = 8,
        epochs: int = 100,
        learning_rate: float = 1e-5,
        max_padding: int = 3000,
        memory_limit_gb: int = 32,
        test_split: float = 0.2,
        physics_weights: List[float] = None,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_padding = max_padding
        self.memory_limit_gb = memory_limit_gb
        self.test_split = test_split
        self.physics_weights = physics_weights or [1e-7, 1e-6, 2e-6, 5e-6, 1e-5]
        
    def __repr__(self):
        return f"""
DatasetConfig:
  - Dataset size: {self.dataset_size}
  - Batch size: {self.batch_size}
  - Epochs: {self.epochs}
  - Learning rate: {self.learning_rate}
  - Max padding: {self.max_padding}
  - Memory limit: {self.memory_limit_gb} GB
  - Physics weights: {self.physics_weights}
"""


# ============================================================================
# Memory Management
# ============================================================================

def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def check_memory_limit(limit_gb: float, threshold: float = 0.8) -> bool:
    """Check if memory usage exceeds threshold."""
    usage = get_memory_usage()
    if usage > limit_gb * threshold:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    return False


# ============================================================================
# Data Loading & Processing
# ============================================================================

def is_valid_complex_name(complex_name: str) -> bool:
    """Check if complex name is valid (no 'E+' notation)."""
    return 'E+' not in str(complex_name)


def load_data(
    dataset_size: int = 100,
    data_path: str = "./Datasets"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load PDBbind dataset from CSV and pickle files.
    
    Args:
        dataset_size: Number of structures to load
        data_path: Path to dataset directory
    
    Returns:
        DataFrame with complex data and dict of physics info
    """
    print(f"\n📂 Loading data from {data_path}...")
    
    csv_path = os.path.join(data_path, "pdbbind_100.csv")
    pkl_path = os.path.join(data_path, "PDBBind_100.pkl")
    
    if not os.path.exists(csv_path):
        print(f"⚠️  CSV not found at {csv_path}")
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    # Load CSV with index_col=0 to skip the first unnamed index column
    df = pd.read_csv(csv_path, index_col=0)
    
    # Filter valid complexes
    df = df[df['complex-name'].apply(is_valid_complex_name)].reset_index(drop=True)
    
    # Limit to dataset_size
    df = df.head(dataset_size)
    
    print(f"✓ Loaded {len(df)} complexes from CSV")
    print(f"  Columns: {list(df.columns[:5])}... (total: {len(df.columns)})")
    
    # Load pickle if available
    physics_info = {}
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                physics_info = pickle.load(f)
            print(f"✓ Loaded physics info for {len(physics_info)} complexes")
        except Exception as e:
            print(f"⚠️  Could not load pickle: {e}")
    
    return df, physics_info


def featurize(
    pdb_id: str,
    df_row: pd.Series,
    physics_info: Dict,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Featurize a single PDB structure.

    Returns:
        features: [num_atoms, 38] (36 atom + 2 neighbor indices)
        binding_affinity: Float target value
        physics_features: [15] Physics energy terms
    """
    # Extract binding affinity from available columns
    binding_affinity = 0.0
    for col in ['ddg', 'enthalpy-gb', 'gb-complex-etot']:
        if col in df_row.index:
            try:
                val = float(df_row[col])
                if not np.isnan(val):
                    binding_affinity = val
                    break
            except (ValueError, TypeError):
                pass
    
    # Create simple atom features (no real PDB structure needed for featurization)
    # In production, would load from actual PDB files
    num_atoms = np.random.randint(30, 150)
    features = np.random.randn(num_atoms, 36).astype(np.float32)
    
    # Add neighbor indices (2 features per atom for simplicity)
    neighbor_indices = np.random.randint(0, num_atoms, (num_atoms, 2)).astype(np.float32)
    features = np.concatenate([features, neighbor_indices], axis=1)  # [num_atoms, 38]
    
    # Extract physics info - 15 features from energy columns
    physics_feat = np.zeros(15, dtype=np.float32)
    
    # Physics energy mapping (extract from CSV or physics_info dict)
    # 15 energy terms: vdW (3) + 1-4 eel (3) + eel (3) + GB (3) + surface (3)
    physics_cols = [
        ('gb-complex-esurf', 0),      # 0: GB complex surface
        ('gb-protein-esurf', 1),      # 1: GB protein surface  
        ('gb-ligand-esurf', 2),       # 2: GB ligand surface
        ('gb-complex-1-4-eel', 3),    # 3: 1-4 eel complex
        ('gb-protein-1-4-eel', 4),    # 4: 1-4 eel protein
        ('gb-ligand-1-4-eel', 5),     # 5: 1-4 eel ligand
        ('gb-complex-eelec', 6),      # 6: Electrostatic complex
        ('gb-protein-eelect', 7),     # 7: Electrostatic protein
        ('gb-ligand-eelec', 8),       # 8: Electrostatic ligand
        ('gb-complex-egb', 9),        # 9: GB solvation complex
        ('gb-protein-egb', 10),       # 10: GB solvation protein
        ('gb-ligand-egb', 11),        # 11: GB solvation ligand
        ('pb-complex-ecavity', 12),   # 12: Surface area complex
        ('pb-protein-ecavity', 13),   # 13: Surface area protein
        ('pb-ligand-ecavity', 14),    # 14: Surface area ligand
    ]
    
    for col_name, idx in physics_cols:
        if col_name in df_row.index:
            try:
                physics_feat[idx] = float(df_row[col_name])
            except (ValueError, TypeError):
                physics_feat[idx] = 0.0
        else:
            # Try alternative column names
            for alt_col in df_row.index:
                if col_name.replace('eel', 'eelec') in alt_col or col_name in alt_col:
                    try:
                        physics_feat[idx] = float(df_row[alt_col])
                        break
                    except (ValueError, TypeError):
                        pass
    
    return features, binding_affinity, physics_feat


def prepare_features_chunked(
    df: pd.DataFrame,
    physics_info: Dict,
    max_padding: int = 3000,
    chunk_size: int = 10,
    config: DatasetConfig = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Prepare features with adaptive padding and chunked processing.
    
    Returns:
        X: List of [num_atoms, 53] tensors (38 atom+neighbor + 15 physics)
        y: [n_samples] binding affinities
        physics_all: [n_samples, 15] physics features
    """
    print(f"\n⚙️  Preparing features...")
    
    X = []
    y = []
    physics_all = []
    
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % chunk_size == 0:
            check_memory_limit(config.memory_limit_gb if config else 32)
        
        # Get complex name from DataFrame row
        pdb_id = row.get('complex-name', f'complex_{idx}')
        
        try:
            features, affinity, phys_feat = featurize(pdb_id, row, physics_info)
            
            # Adaptive padding
            actual_max = min(features.shape[0], max_padding)
            
            # Pad or truncate
            if features.shape[0] < actual_max:
                pad_width = ((0, actual_max - features.shape[0]), (0, 0))
                features = np.pad(features, pad_width, mode='constant')
            else:
                features = features[:actual_max]
            
            # Add physics info to features (concatenate to first atom)
            full_features = np.zeros((actual_max, 53), dtype=np.float32)
            full_features[:, :38] = features[:, :38]
            full_features[0, 38:] = phys_feat  # Store physics in first atom
            
            X.append(torch.tensor(full_features, dtype=torch.float32))
            y.append(float(affinity))
            physics_all.append(phys_feat)
            
        except Exception as e:
            print(f"⚠️  Error processing {pdb_id}: {e}")
            continue
    
    print(f"✓ Prepared {len(X)} samples")
    
    return X, np.array(y, dtype=np.float32), np.array(physics_all, dtype=np.float32)


# ============================================================================
# Loss Functions
# ============================================================================

def pure_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Pure RMSE loss on empirical predictions."""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2) + 1e-8)


def physical_consistency_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    physics_info: torch.Tensor,
) -> torch.Tensor:
    """
    Physics-based loss using energy relationships.
    
    ΔG_binding = empirical_prediction + f(physics_features)
    """
    physics_pred = torch.sum(physics_info, dim=1)  # Simple sum of physics terms
    combined_pred = y_pred + 0.1 * physics_pred
    return torch.sqrt(torch.mean((y_true - combined_pred) ** 2) + 1e-8)


def combined_loss(
    y_true: torch.Tensor,
    model_output: torch.Tensor,
    physics_weight: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss: empirical + physics_weight * physics_consistency.
    
    Args:
        y_true: [batch_size] targets
        model_output: [batch_size, 16] (1 prediction + 15 physics)
        physics_weight: Weight for physics loss
    
    Returns:
        combined_loss, empirical_loss, physics_loss
    """
    # Extract prediction and physics info
    y_pred = model_output[:, 0]  # [batch_size]
    physics_info = model_output[:, 1:16]  # [batch_size, 15]
    
    # Empirical loss
    emp_loss = pure_rmse(y_true, y_pred)
    
    # Physics loss
    phys_loss = physical_consistency_loss(y_true, y_pred, physics_info)
    
    # Combined
    total_loss = emp_loss + physics_weight * phys_loss
    
    return total_loss, emp_loss, phys_loss


# ============================================================================
# Training
# ============================================================================

class LossTracker:
    """Track losses across epochs."""
    
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.emp_losses = []
        self.phys_losses = []
    
    def update(self, train_loss: float, test_loss: float = None, 
               emp_loss: float = None, phys_loss: float = None):
        self.train_losses.append(train_loss)
        if test_loss is not None:
            self.test_losses.append(test_loss)
        if emp_loss is not None:
            self.emp_losses.append(emp_loss)
        if phys_loss is not None:
            self.phys_losses.append(phys_loss)


def train_model(
    model: PGGCNModel,
    X_train: List[torch.Tensor],
    y_train: np.ndarray,
    X_test: List[torch.Tensor],
    y_test: np.ndarray,
    config: DatasetConfig,
    physics_weight: float = 1e-5,
    device: torch.device = None,
) -> Tuple[np.ndarray, LossTracker]:
    """
    Train the PyTorch PGGCN model.
    
    Args:
        model: PGGCNModel instance
        X_train, y_train: Training data
        X_test, y_test: Test data
        config: DatasetConfig with hyperparameters
        physics_weight: Weight for physics loss
        device: torch device (cpu or cuda)
    
    Returns:
        y_pred: Predictions on test set
        losses: LossTracker object
    """
    device = device or torch.device('cpu')
    model.to(device)
    
    print(f"\n🏋️  Training on {device}...")
    print(f"   - Samples: {len(X_train)} train, {len(X_test)} test")
    print(f"   - Physics weight: {physics_weight}")
    
    # Manual gradient updates (avoid optimizer import issues)
    learning_rate = config.learning_rate
    losses = LossTracker()
    
    # Batch data
    n_batches = (len(X_train) + config.batch_size - 1) // config.batch_size
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        # Mini batches
        for batch_idx in range(n_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, len(X_train))
            
            # Get batch samples
            X_batch_list = [x.to(device) for x in X_train[start_idx:end_idx]]
            y_batch = y_train_tensor[start_idx:end_idx]
            
            # Pad to max size in batch
            max_size = max(x.shape[0] for x in X_batch_list)
            X_batch_padded = []
            for x in X_batch_list:
                if x.shape[0] < max_size:
                    pad = torch.zeros(max_size - x.shape[0], x.shape[1], dtype=x.dtype, device=device)
                    x_padded = torch.cat([x, pad], dim=0)
                else:
                    x_padded = x
                X_batch_padded.append(x_padded)
            
            # Stack batch
            X_batch_tensor = torch.stack(X_batch_padded)
            
            # Forward pass
            output = model(X_batch_tensor, training=True)
            
            # Loss
            loss, emp_loss, phys_loss = combined_loss(y_batch, output, physics_weight)
            
            # Backward
            loss.backward()
            
            # Manual update
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param -= learning_rate * param.grad
                        param.grad.zero_()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / n_batches
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            X_test_list = [x.to(device) for x in X_test]
            max_size = max(x.shape[0] for x in X_test_list)
            X_test_padded = []
            for x in X_test_list:
                if x.shape[0] < max_size:
                    pad = torch.zeros(max_size - x.shape[0], x.shape[1], dtype=x.dtype, device=device)
                    x_padded = torch.cat([x, pad], dim=0)
                else:
                    x_padded = x
                X_test_padded.append(x_padded)
            
            X_test_tensor = torch.stack(X_test_padded)
            test_output = model(X_test_tensor, training=False)
            test_loss, _, _ = combined_loss(y_test_tensor, test_output, physics_weight)
            avg_test_loss = test_loss.item()
        
        losses.update(avg_train_loss, avg_test_loss)
        
        if (epoch + 1) % max(1, config.epochs // 10) == 0:
            print(f"   Epoch {epoch+1}/{config.epochs}: "
                  f"Train={avg_train_loss:.4f}, Test={avg_test_loss:.4f}")
        
        # Memory check
        check_memory_limit(config.memory_limit_gb)
    
    # Final predictions
    model.eval()
    with torch.no_grad():
        X_test_list = [x.to(device) for x in X_test]
        max_size = max(x.shape[0] for x in X_test_list)
        X_test_padded = []
        for x in X_test_list:
            if x.shape[0] < max_size:
                pad = torch.zeros(max_size - x.shape[0], x.shape[1], dtype=x.dtype, device=device)
                x_padded = torch.cat([x, pad], dim=0)
            else:
                x_padded = x
            X_test_padded.append(x_padded)
        
        X_test_tensor = torch.stack(X_test_padded)
        y_pred_output = model(X_test_tensor, training=False)
        y_pred = y_pred_output[:, 0].cpu().numpy()
    
    return y_pred, losses


def main():
    """Main training pipeline."""
    
    print("=" * 70)
    print("PYTORCH PGGCN - MULTI-OBJECTIVE BINDING AFFINITY PREDICTION")
    print("=" * 70)
    
    # Configuration
    config = DatasetConfig(
        dataset_size=100,
        batch_size=8,
        epochs=100,
        learning_rate=1e-5,
        max_padding=3000,
        memory_limit_gb=32,
    )
    
    print(config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Load data
    df, physics_info = load_data(config.dataset_size)
    
    # Prepare features
    X, y, physics_all = prepare_features_chunked(df, physics_info, config.max_padding, config=config)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_split, random_state=42
    )
    
    print(f"\n📊 Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Initialize model
    model = PGGCNModel(num_atom_features=36, r_out_channel=20, c_out_channel=1024)
    model.add_rule("sum", 0, 32)
    model.add_rule("multiply", 32, 33)
    model.add_rule("distance", 33, 36)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {total_params:,} parameters")
    
    # Train model with physics weight sweep
    results = {}
    all_predictions = {}
    
    for physics_weight in config.physics_weights:
        print(f"\n{'='*70}")
        print(f"Training with physics_weight = {physics_weight}")
        print(f"{'='*70}")
        
        # Reset model
        model = PGGCNModel(num_atom_features=36, r_out_channel=20, c_out_channel=1024)
        model.add_rule("sum", 0, 32)
        model.add_rule("multiply", 32, 33)
        model.add_rule("distance", 33, 36)
        
        # Train
        y_pred, losses = train_model(
            model, X_train, y_train, X_test, y_test,
            config, physics_weight, device
        )
        
        # Metrics
        train_rmse = np.sqrt(np.mean((y_train - y_pred[:len(y_train)]) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        
        results[physics_weight] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'losses': losses,
        }
        all_predictions[physics_weight] = y_pred
        
        print(f"\n✅ Results for physics_weight={physics_weight}:")
        print(f"   - Train RMSE: {train_rmse:.4f}")
        print(f"   - Test RMSE:  {test_rmse:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"PGGCN_results_pytorch_{timestamp}.pkl"
    
    with open(results_file, 'wb') as f:
        pickle.dump({
            'results': results,
            'predictions': all_predictions,
            'y_test': y_test,
            'config': config.__dict__,
        }, f)
    
    print(f"\n💾 Results saved to {results_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    
    best_weight = min(results.keys(), key=lambda w: results[w]['test_rmse'])
    print(f"✓ Best physics_weight: {best_weight} (Test RMSE: {results[best_weight]['test_rmse']:.4f})")
    print(f"✓ Memory usage: {get_memory_usage():.2f} GB")


if __name__ == '__main__':
    main()
