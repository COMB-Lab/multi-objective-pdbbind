#!/usr/bin/env python3
"""
PyTorch implementation of PGGCN training for PDBbind dataset.
Predicts protein-ligand binding free energies using physics-guided neural networks.

Ported from: multi-objective-pdbbind.py (TensorFlow version)
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import copy
import gc
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from rdkit import Chem
from deepchem.feat.graph_features import atom_features as get_atom_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import importlib

# Import PyTorch layers
from models.layers_pytorch import RuleGraphConvLayer, ConvLayer, PGGCNModel, PCGradOptimizer

# ============================================================================
# CONFIGURATION AND MEMORY MANAGEMENT
# ============================================================================

class DatasetConfig:
    """Configuration class for dataset and memory management."""
    def __init__(self, dataset_size=100, max_padding=3000, batch_size=8, 
                 epochs=100, memory_limit_gb=16, preserve_full_structures=True):
        self.dataset_size = dataset_size
        self.max_padding = max_padding
        self.batch_size = batch_size
        self.epochs = epochs
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.preserve_full_structures = preserve_full_structures


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)


def check_memory_limit(config):
    """Check if memory usage is approaching the limit."""
    current_memory = get_memory_usage()
    if current_memory > config.memory_limit_gb * 0.8:
        print(f"Warning: Memory usage ({current_memory:.2f} GB) approaching limit ({config.memory_limit_gb} GB)")
        gc.collect()
        return True
    return False


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(config):
    """Load and preprocess the PDBbind dataset with configurable size."""
    print(f"Loading data for {config.dataset_size} structures...")
    print(f"Memory limit: {config.memory_limit_gb} GB")
    
    df = pd.read_csv('/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind_100.csv')
    PDBs = pickle.load(open('/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_100.pkl', 'rb'))
    
    print(f"Original dataset size: {len(df)}")
    
    # Remove entries with NaN values in ddg column
    initial_size = len(df)
    df = df.dropna(subset=['ddg'])
    print(f"Removed {initial_size - len(df)} entries with NaN in ddg column")
    
    # Remove problematic complex names containing 'E+' (scientific notation)
    def is_valid_complex_name(name):
        if pd.isna(name):
            return False
        name_str = str(name)
        if 'E+' in name_str:
            return False
        return True
    
    valid_mask = df['complex-name'].apply(is_valid_complex_name)
    invalid_count = (~valid_mask).sum()
    df = df[valid_mask]
    print(f"Removed {invalid_count} entries with 'E+' in complex names")
    
    print(f"Cleaned dataset size: {len(df)}")
    
    # Get the specified number of structures
    selected_rows = df.head(config.dataset_size)
    print(f"Selected {len(selected_rows)} structures:")
    print(selected_rows[['complex-name', 'ddg']].tail(10))
    if len(selected_rows) > 10:
        print(f"... and {len(selected_rows) - 10} more structures")
    
    df = selected_rows

    # Filter PDBs to match available keys
    pdb_keys = set(PDBs.keys())
    df_filtered = df[df['complex-name'].isin(pdb_keys)]
    print(f"Filtered dataframe length: {len(df_filtered)}")
    
    # Select relevant columns
    physics_columns = ['pb-protein-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals', 
                      'gb-protein-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
                      'gb-protein-eelect', 'gb-ligand-eelec', 'gb-complex-eelec', 
                      'gb-protein-egb', 'gb-ligand-egb', 'gb-complex-egb', 
                      'gb-protein-esurf', 'gb-ligand-esurf', 'gb-complex-esurf']
    
    required_columns = ['complex-name'] + physics_columns + ['ddg']
    df_final = df_filtered[required_columns]
    
    keys_of_interest = df_final['complex-name'].tolist()
    filtered_PDBs = {k: PDBs[k] for k in keys_of_interest if k in PDBs}
    
    # Final validation
    final_keys = set(df_final['complex-name'].tolist())
    pdb_keys = set(filtered_PDBs.keys())
    common_keys = final_keys.intersection(pdb_keys)
    missing_pdb = final_keys - pdb_keys
    missing_csv = pdb_keys - final_keys
    
    if missing_pdb:
        print(f"Warning: {len(missing_pdb)} complexes in CSV but missing PDB structures")
    if missing_csv:
        print(f"Warning: {len(missing_csv)} PDB structures without corresponding CSV data")
    
    df_final = df_final[df_final['complex-name'].isin(common_keys)]
    filtered_PDBs = {k: v for k, v in filtered_PDBs.items() if k in common_keys}
    
    print(f"Final validated dataset: {len(df_final)} structures with both CSV and PDB data")
    print(f"Filtered PDBs count: {len(filtered_PDBs)}")
    print(f"Current memory usage: {get_memory_usage():.2f} GB")
    
    print(f"\nData cleaning summary:")
    print(f"  Original entries: {initial_size}")
    print(f"  After all cleaning: {len(df_final)}")
    print(f"  Data retention rate: {len(df_final)/initial_size*100:.1f}%")
    
    return df_final, filtered_PDBs


def extract_physics_info(df_final, filtered_PDBs):
    """Extract physics information for each PDB structure."""
    info = []
    for pdb in list(filtered_PDBs.keys()):
        physics_data = df_final[df_final['complex-name'] == pdb][['pb-protein-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals', 
                                                                  'gb-protein-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
                                                                  'gb-protein-eelect', 'gb-ligand-eelec', 'gb-complex-eelec', 
                                                                  'gb-protein-egb', 'gb-ligand-egb', 'gb-complex-egb', 
                                                                  'gb-protein-esurf', 'gb-ligand-esurf', 'gb-complex-esurf']].to_numpy()[0]
        info.append(physics_data)
    return info


def featurize(molecule, info):
    """
    Featurize a molecule with atom features and physics information.
    
    Args:
        molecule: RDKit molecule object
        info: Physics information array (15 values)
        
    Returns:
        numpy array of shape [num_atoms, 53] with atom features + physics info
    """
    atom_features = []
    for atom in molecule.GetAtoms():
        new_feature = get_atom_features(atom).tolist()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
        new_feature += [position.x, position.y, position.z]
        
        # Add neighbor information (up to 2 neighbors)
        neighbors = list(atom.GetNeighbors())
        for neighbor in neighbors[:2]:
            new_feature.append(float(neighbor.GetIdx()))
        # Pad with -1 if fewer than 2 neighbors
        for _ in range(2 - len(neighbors)):
            new_feature.append(-1.0)
        
        atom_features.append(np.concatenate([new_feature, info], axis=0))
    
    return np.array(atom_features, dtype=np.float32)


def prepare_features_chunked(df_final, filtered_PDBs, info, config, chunk_size=10):
    """Prepare feature matrices X and target values y with chunked processing."""
    print(f"Preparing features for {len(filtered_PDBs)} structures...")
    print(f"Processing in chunks of {chunk_size} for memory efficiency")
    
    X = []
    y = []
    atom_counts = []
    
    pdb_list = list(filtered_PDBs.keys())
    
    # Process in chunks
    for chunk_start in range(0, len(pdb_list), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(pdb_list))
        chunk_pdbs = pdb_list[chunk_start:chunk_end]
        
        print(f"Processing chunk {chunk_start//chunk_size + 1}/{math.ceil(len(pdb_list)/chunk_size)}: "
              f"structures {chunk_start+1}-{chunk_end}")
        
        for i, pdb in enumerate(chunk_pdbs):
            global_i = chunk_start + i
            features = featurize(filtered_PDBs[pdb], info[global_i])
            X.append(features)
            y.append(df_final[df_final['complex-name'] == pdb]['ddg'].to_numpy()[0])
            atom_counts.append(features.shape[0])
            
            if global_i % 5 == 0:
                print(f"  Processed {global_i+1}/{len(pdb_list)}: {pdb} ({features.shape[0]} atoms)")
        
        if check_memory_limit(config):
            print("Memory usage high, forcing garbage collection...")
            gc.collect()
    
    # Calculate statistics
    max_atoms = max(atom_counts)
    avg_atoms = np.mean(atom_counts)
    max_features = X[0].shape[1] if X else 0
    
    print(f"Dataset statistics:")
    print(f"  Max atoms: {max_atoms}")
    print(f"  Average atoms: {avg_atoms:.1f}")
    print(f"  Max features: {max_features}")
    print(f"  Total structures: {len(X)}")
    print(f"Current memory usage: {get_memory_usage():.2f} GB")
    
    return X, y, max_atoms


def pad_sequences_adaptive(X, config, max_atoms_actual):
    """Adaptive padding based on actual data and memory constraints."""
    estimated_memory_gb = (max_atoms_actual * 53 * 4 * len(X) * config.batch_size) / (1024**3)
    
    if estimated_memory_gb < config.memory_limit_gb * 0.7:
        max_length = max_atoms_actual
        print(f"Using actual max atoms: {max_length} (estimated memory: {estimated_memory_gb:.1f} GB)")
    else:
        max_length = min(max_atoms_actual, config.max_padding)
        print(f"Memory constraint active: using {max_length} atoms (actual max: {max_atoms_actual})")
        print(f"WARNING: {max_atoms_actual - max_length} atoms will be truncated from largest structures!")
    
    current_memory = get_memory_usage()
    if current_memory > config.memory_limit_gb * 0.6:
        max_length = min(max_length, config.max_padding // 2)
        print(f"CRITICAL: Memory usage high, reducing to {max_length} atoms")
    
    print(f"Final padding decision: {max_length} atoms")
    
    for i in range(len(X)):
        original_atoms = X[i].shape[0]
        if X[i].shape[0] < max_length:
            padding_size = max_length - X[i].shape[0]
            padding = np.zeros([padding_size, X[i].shape[1]], dtype=np.float32)
            X[i] = np.concatenate([X[i], padding], axis=0).astype(np.float32)
        elif X[i].shape[0] > max_length:
            X[i] = X[i][:max_length].astype(np.float32)
            print(f"  Truncated structure {i} from {X[i].shape[0]} to {max_length} atoms")
    
    return np.array(X, dtype=np.float32)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def pure_rmse(y_true, y_pred):
    """Pure root mean squared error."""
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def physical_consistency_loss(y_true, y_pred, physics_info):
    """
    Physics-based consistency loss function.
    Calculates difference between predicted binding affinity and physics-based calculation.
    """
    y_true = y_true.view(-1, 1)
    y_pred = y_pred.view(-1, 1)
    
    # Extract energy components from physics_info
    device = physics_info.device
    
    # Indices for host, guest, complex contributions
    host_indices = torch.tensor([0, 3, 6, 9, 12], device=device)
    guest_indices = torch.tensor([1, 4, 7, 10, 13], device=device)
    complex_indices = torch.tensor([2, 5, 8, 11, 14], device=device)
    
    # Calculate ΔG based on physics
    dG_physics = (torch.sum(torch.index_select(physics_info, 1, complex_indices), dim=1, keepdim=True) - 
                  (torch.sum(torch.index_select(physics_info, 1, host_indices), dim=1, keepdim=True) +
                   torch.sum(torch.index_select(physics_info, 1, guest_indices), dim=1, keepdim=True)))
    
    phy_loss = torch.sqrt(torch.mean((y_pred - dG_physics) ** 2))
    return phy_loss


def combined_loss(y_true, y_pred, physics_weight=0.0003):
    """
    Combined loss function with empirical and physics components.
    
    Args:
        y_true: True binding affinities [batch_size]
        y_pred: Model output [batch_size, 16] with [prediction, physics_info]
        physics_weight: Weight for physics loss component
    
    Returns:
        Combined loss value
    """
    prediction = y_pred[:, 0]
    physics_info = y_pred[:, 1:16]
    
    # Calculate individual loss components
    empirical_loss = pure_rmse(y_true, prediction)
    physics_loss = physical_consistency_loss(y_true, prediction, physics_info)
    
    # Combine losses with weights
    total_loss = empirical_loss + (physics_weight * physics_loss)
    
    return total_loss, empirical_loss, physics_loss


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class LossTracker:
    """Track loss and learning rate history during training."""
    
    def __init__(self):
        self.total_losses = []
        self.empirical_losses = []
        self.physics_losses = []
        self.learning_rates = []
    
    def record(self, total_loss, empirical_loss, physics_loss, lr):
        """Record loss values for an epoch."""
        self.total_losses.append(total_loss)
        self.empirical_losses.append(empirical_loss)
        self.physics_losses.append(physics_loss)
        self.learning_rates.append(lr)


def plot_training_results(loss_tracker, config):
    """Plot training loss over epochs with configuration info."""
    if not loss_tracker.total_losses:
        print("No training loss data to plot.")
        return
    
    try:
        plt.figure(figsize=(15, 10))
        
        epoch_length = range(1, len(loss_tracker.total_losses) + 1)
        
        # Total loss
        plt.subplot(2, 3, 1)
        plt.plot(epoch_length, loss_tracker.total_losses, 'b-', label='Total Loss', linewidth=2)
        plt.title(f'Total Loss Over Epochs\n({config.dataset_size} structures)', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        if loss_tracker.learning_rates:
            plt.subplot(2, 3, 2)
            plt.plot(epoch_length, loss_tracker.learning_rates, 'r-', label='Learning Rate', linewidth=2)
            plt.title('Learning Rate Over Epochs', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Loss trend (last 10 epochs)
        if len(loss_tracker.total_losses) > 10:
            plt.subplot(2, 3, 3)
            recent_losses = loss_tracker.total_losses[-10:]
            recent_epochs = list(range(len(loss_tracker.total_losses)-9, len(loss_tracker.total_losses)+1))
            plt.plot(recent_epochs, recent_losses, 'g-', label='Recent Loss', linewidth=2, marker='o')
            plt.title('Loss Trend (Last 10 Epochs)', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Configuration info
        plt.subplot(2, 3, 4)
        plt.text(0.1, 0.9, f"Dataset Size: {config.dataset_size}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.8, f"Batch Size: {config.batch_size}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.7, f"Max Padding: {config.max_padding}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.6, f"Epochs: {config.epochs}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.5, f"Memory Limit: {config.memory_limit_gb} GB", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.1, 0.4, f"Final Loss: {loss_tracker.total_losses[-1]:.6f}", transform=plt.gca().transAxes, fontsize=12)
        plt.title('Training Configuration', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        filename = f'training_results_{config.dataset_size}structures_pytorch.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
        plt.close()
        
        print(f"\nTraining Summary ({config.dataset_size} structures):")
        print(f"  Initial Loss: {loss_tracker.total_losses[0]:.6f}")
        print(f"  Final Loss: {loss_tracker.total_losses[-1]:.6f}")
        print(f"  Loss Reduction: {((loss_tracker.total_losses[0] - loss_tracker.total_losses[-1]) / loss_tracker.total_losses[0] * 100):.2f}%")
        print(f"  Total Epochs: {len(loss_tracker.total_losses)}")
    
    except Exception as e:
        print(f"Error creating plot: {e}")


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function with PyTorch."""
    print("=" * 70)
    print("Starting PGGCN PyTorch Training Script")
    print("=" * 70)
    
    # Configuration
    config = DatasetConfig(
        dataset_size=100,
        max_padding=3000,
        batch_size=8,
        epochs=100,
        memory_limit_gb=32
    )
    
    print(f"Configuration:")
    print(f"  Dataset size: {config.dataset_size} structures")
    print(f"  Max padding: {config.max_padding} atoms")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Memory limit: {config.memory_limit_gb} GB")
    print(f"  Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Load and preprocess data
    df_final, filtered_PDBs = load_data(config)
    info = extract_physics_info(df_final, filtered_PDBs)
    
    # Prepare features
    X, y, max_atoms_actual = prepare_features_chunked(df_final, filtered_PDBs, info, config)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    print(f"Data split: {len(X_train)} training, {len(X_test)} testing")
    
    # Physics hyperparameter sweep
    physics_hyperparam = [1e-5, 1e-6, 1e-7, 2e-6, 5e-6]
    print(f"Physics hyperparameter sweep: {physics_hyperparam}")
    
    all_results = []
    start = time.time()
    
    sweep_logfile = "physics_weight_sweep_summary_pytorch.log"
    with open(sweep_logfile, "w") as sweep_log:
        sweep_log.write("# physics_weight, final_train_loss, test_loss, mean_abs_diff, epochs_trained\n")
        
        for physics_weight in physics_hyperparam:
            print(f"\n{'='*50}")
            print(f"Training with physics_weight: {physics_weight}")
            print(f"{'='*50}")
            
            # Initialize model
            model = PGGCNModel(num_atom_features=36, r_out_channel=20, c_out_channel=1024)
            model.add_rule("sum", 0, 32)
            model.add_rule("multiply", 32, 33)
            model.add_rule("distance", 33, 36)
            model = model.to(device)
            
            # Optimizer and scheduler
            optimizer = Adam(model.parameters(), lr=1e-5)
            scheduler = ExponentialLR(optimizer, gamma=0.85)
            
            # Prepare training data
            X_train_padded = pad_sequences_adaptive(copy.deepcopy(X_train), config, max_atoms_actual)
            X_train_tensor = torch.from_numpy(X_train_padded).to(device)
            y_train_tensor = torch.from_numpy(np.array(y_train, dtype=np.float32)).to(device)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            
            loss_tracker = LossTracker()
            patience = 10
            best_loss = float('inf')
            patience_counter = 0
            
            print(f"Starting training...")
            
            for epoch in range(config.epochs):
                model.train()
                epoch_loss = 0
                epoch_empirical_loss = 0
                epoch_physics_loss = 0
                
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    y_pred = model(X_batch, training=True)
                    
                    # Compute loss
                    total_loss, empirical_loss, physics_loss = combined_loss(
                        y_batch, y_pred, physics_weight
                    )
                    
                    # Backward pass
                    total_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    epoch_empirical_loss += empirical_loss.item()
                    epoch_physics_loss += physics_loss.item()
                
                # Average loss over batches
                epoch_loss /= len(train_loader)
                epoch_empirical_loss /= len(train_loader)
                epoch_physics_loss /= len(train_loader)
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                
                loss_tracker.record(epoch_loss, epoch_empirical_loss, epoch_physics_loss, current_lr)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{config.epochs}: "
                          f"Loss={epoch_loss:.6f}, Empirical={epoch_empirical_loss:.6f}, "
                          f"Physics={epoch_physics_loss:.6f}")
                
                # Early stopping
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Make predictions on test set
            model.eval()
            with torch.no_grad():
                X_test_padded = pad_sequences_adaptive(copy.deepcopy(X_test), config, max_atoms_actual)
                X_test_tensor = torch.from_numpy(X_test_padded).to(device)
                y_test_tensor = torch.from_numpy(np.array(y_test, dtype=np.float32)).to(device)
                
                y_pred_test_full = model(X_test_tensor, training=False)
                y_pred_test = y_pred_test_full[:, 0].cpu().numpy()
                
                # Calculate test loss
                test_loss_val, _, _ = combined_loss(y_test_tensor, y_pred_test_full, physics_weight)
                test_loss_value = test_loss_val.item()
            
            # Calculate metrics
            y_difference = np.mean(np.abs(np.abs(np.array(y_test)) - np.abs(y_pred_test)))
            final_train_loss = loss_tracker.total_losses[-1]
            
            print(f"\nResults:")
            print(f"  Mean absolute difference: {y_difference:.6f}")
            print(f"  Test loss: {test_loss_value:.6f}")
            print(f"  Final training loss: {final_train_loss:.6f}")
            
            sweep_log.write(f"{physics_weight}, {final_train_loss:.6f}, {test_loss_value:.6f}, {y_difference:.6f}, {len(loss_tracker.total_losses)}\n")
            sweep_log.flush()
            
            # Store results
            results = {
                'physics_weight': physics_weight,
                'dataset_size': config.dataset_size,
                'final_train_loss': final_train_loss,
                'test_loss': test_loss_value,
                'mean_abs_diff': y_difference,
                'epochs_trained': len(loss_tracker.total_losses),
                'y_true_test': np.array(y_test),
                'y_pred_test': y_pred_test,
                'training_history': {
                    'total_losses': loss_tracker.total_losses,
                    'learning_rates': loss_tracker.learning_rates
                },
                'config': {
                    'dataset_size': config.dataset_size,
                    'max_padding': config.max_padding,
                    'batch_size': config.batch_size,
                    'epochs': config.epochs,
                    'memory_limit_gb': config.memory_limit_gb
                }
            }
            all_results.append(results)
            
            # Plot results
            plot_training_results(loss_tracker, config)
            
            # Cleanup
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
            gc.collect()
    
    end = time.time()
    runtime_minutes = (end - start) / 60
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Dataset size: {config.dataset_size} structures")
    print(f"Total runtime: {runtime_minutes:.2f} minutes")
    print(f"Average time per structure: {runtime_minutes/config.dataset_size:.2f} minutes")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pickle_filename = f'PGGCN_results_pytorch_{config.dataset_size}structures_{timestamp}.pkl'
    
    results_data = {
        'experiment_info': {
            'timestamp': timestamp,
            'dataset_size': config.dataset_size,
            'total_runtime_minutes': runtime_minutes,
            'framework': 'PyTorch',
            'device': str(device),
            'config': {
                'dataset_size': config.dataset_size,
                'max_padding': config.max_padding,
                'batch_size': config.batch_size,
                'epochs': config.epochs,
                'memory_limit_gb': config.memory_limit_gb
            }
        },
        'all_results': all_results,
        'summary_metrics': {
            'best_physics_weight': min(all_results, key=lambda x: x['test_loss'])['physics_weight'],
            'best_test_loss': min(all_results, key=lambda x: x['test_loss'])['test_loss'],
            'best_train_loss': min(all_results, key=lambda x: x['final_train_loss'])['final_train_loss'],
            'best_mae': min(all_results, key=lambda x: x['mean_abs_diff'])['mean_abs_diff']
        }
    }
    
    try:
        with open(pickle_filename, 'wb') as f:
            pickle.dump(results_data, f)
        print(f"\n{'='*70}")
        print(f"RESULTS SAVED TO PICKLE FILE")
        print(f"{'='*70}")
        print(f"Filename: {pickle_filename}")
        print(f"Contents saved:")
        print(f"  - All training results and metrics")
        print(f"  - True vs predicted values for test set")
        print(f"  - Training history (loss curves, learning rates)")
        print(f"  - Configuration parameters")
        print(f"  - Framework information (PyTorch)")
    except Exception as e:
        print(f"Error saving pickle file: {e}")
    
    print(f"\nFinal Results Summary:")
    for result in all_results:
        print(f"  Physics weight {result['physics_weight']}: "
              f"Train loss {result['final_train_loss']:.6f}, "
              f"Test loss {result['test_loss']:.6f}, "
              f"MAE {result['mean_abs_diff']:.6f}")


if __name__ == "__main__":
    main()
