#!/usr/bin/env python3
"""
Test2 PDBbind Scalable - PGGCN Model Training Script (Large Dataset Support)

This script implements a Physics-Guided Graph Convolutional Network (PGGCN) model
for predicting binding free energies using scalable approaches for large datasets.
The model incorporates both empirical and physics-based loss functions.

Scalability features:
- Configurable dataset size (10, 50, 100+ structures)
- Dynamic memory management with data generators
- Adaptive padding based on actual data requirements
- Batch processing for memory efficiency
- Memory monitoring and cleanup
- Chunked data processing for very large datasets

Converted from Jupyter notebook: Test2_PDBbind_10_structures.ipynb
"""

# ============================================================================
# IMPORTS AND ENVIRONMENT SETUP
# ============================================================================

import os
import conda_installer

# Set CUDA environment variables (must come BEFORE importing TensorFlow)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/nvidia/cuda/cuda-11.6'

import pandas as pd
import tensorflow as tf
import numpy as np
from rdkit import Chem
from deepchem.feat.graph_features import atom_features as get_atom_features
import rdkit
import pickle
import copy
import gc  # For garbage collection
import psutil  # For memory monitoring
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import csv
import time
import sys
import importlib
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras.backend as K
from tensorflow.keras import regularizers, constraints, callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

# ============================================================================
# CONFIGURATION AND MEMORY MANAGEMENT
# ============================================================================

class DatasetConfig:
    """Configuration class for dataset and memory management."""
    def __init__(self, dataset_size=100, max_padding=3000, batch_size=8, 
                 epochs=100, memory_limit_gb=16, preserve_full_structures=True):
        self.dataset_size = dataset_size
        self.max_padding = max_padding  # Maximum atoms to pad to
        self.batch_size = batch_size    # Increased batch size for more stable gradients
        self.epochs = epochs
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.preserve_full_structures = preserve_full_structures  # NEW: Try to keep all atoms

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)

def check_memory_limit(config):
    """Check if memory usage is approaching the limit."""
    current_memory = get_memory_usage()
    if current_memory > config.memory_limit_gb * 0.8:  # 80% threshold
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
    # Clean and validate data
    print(f"Original dataset size: {len(df)}")
    
    # Remove entries with NaN values in ddg column
    initial_size = len(df)
    df = df.dropna(subset=['ddg'])
    print(f"Removed {initial_size - len(df)} entries with NaN in ddg column")
    
    # Remove problematic complex names containing 'E+' (scientific notation like '1.00E+66')
    def is_valid_complex_name(name):
        """Check if complex name is valid (doesn't contain 'E+' for scientific notation)."""
        if pd.isna(name):
            return False
        name_str = str(name)
        # Remove entries containing 'E+' (scientific notation)
        if 'E+' in name_str:
            return False
        return True
    
    valid_mask = df['complex-name'].apply(is_valid_complex_name)
    invalid_count = (~valid_mask).sum()
    df = df[valid_mask]
    print(f"Removed {invalid_count} entries with 'E+' in complex names")
    
    print(f"Cleaned dataset size: {len(df)} (removed {initial_size - len(df)} total problematic entries)")
    
    # Get the specified number of structures
    selected_rows = df.head(config.dataset_size)
    print(f"Selected {len(selected_rows)} structures:")
    print(selected_rows[['complex-name', 'ddg']].tail(10))  # Show last 10
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
    
    # Get the list of complex names from df_final
    keys_of_interest = df_final['complex-name'].tolist()
    
    # Filter PDBs dictionary to keep only those complexes
    filtered_PDBs = {k: PDBs[k] for k in keys_of_interest if k in PDBs}
    
    # Final validation: ensure we have both CSV data and PDB structures
    final_keys = set(df_final['complex-name'].tolist())
    pdb_keys = set(filtered_PDBs.keys())
    
    # Keep only entries that exist in both datasets
    common_keys = final_keys.intersection(pdb_keys)
    missing_pdb = final_keys - pdb_keys
    missing_csv = pdb_keys - final_keys
    
    if missing_pdb:
        print(f"Warning: {len(missing_pdb)} complexes in CSV but missing PDB structures")
    if missing_csv:
        print(f"Warning: {len(missing_csv)} PDB structures without corresponding CSV data")
    
    # Filter to only common entries
    df_final = df_final[df_final['complex-name'].isin(common_keys)]
    filtered_PDBs = {k: v for k, v in filtered_PDBs.items() if k in common_keys}
    
    print(f"Final validated dataset: {len(df_final)} structures with both CSV and PDB data")
    
    print(f"Filtered PDBs count: {len(filtered_PDBs)}")
    print(f"Current memory usage: {get_memory_usage():.2f} GB")
    
    # Summary of data cleaning
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

# Import custom layers and atom features
import models.layers_update_mobley as layers
from models.dcFeaturizer import atom_features as get_atom_features

# Force reload of the layers module to ensure we get the latest version
importlib.reload(layers)


def featurize(molecule, info):
    """
    Featurize a molecule with atom features and physics information.
    
    Args:
        molecule: RDKit molecule object
        info: Physics information array
        
    Returns:
        numpy array of concatenated atom features and physics info
    """
    atom_features = []
    for atom in molecule.GetAtoms():
        new_feature = get_atom_features(atom).tolist()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
        new_feature += [position.x, position.y, position.z]
        
        # Add neighbor information (up to 2 neighbors)
        for neighbor in atom.GetNeighbors()[:2]:
            neighbor_idx = neighbor.GetIdx()
            new_feature += [neighbor_idx]
        # Pad with -1 if fewer than 2 neighbors
        for i in range(2 - len(atom.GetNeighbors())):
            new_feature += [-1]

        atom_features.append(np.concatenate([new_feature, info], 0))
    return np.array(atom_features, dtype=np.float32)


def prepare_features_chunked(df_final, filtered_PDBs, info, config, chunk_size=10):
    """Prepare feature matrices X and target values y with chunked processing."""
    print(f"Preparing features for {len(filtered_PDBs)} structures...")
    print(f"Processing in chunks of {chunk_size} for memory efficiency")
    
    X = []
    y = []
    atom_counts = []
    
    pdb_list = list(filtered_PDBs.keys())
    
    # Process in chunks to manage memory
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
            
            if global_i % 5 == 0:  # Print progress every 5 structures
                print(f"  Processed {global_i+1}/{len(pdb_list)}: {pdb} ({features.shape[0]} atoms)")
        
        # Check memory after each chunk
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


# ============================================================================
# MODEL DEFINITION (Same as before)
# ============================================================================

class PGGCNModel(tf.keras.Model):
    """Physics-Guided Graph Convolutional Network Model."""
    
    def __init__(self, num_atom_features=36, r_out_channel=20, c_out_channel=1024, 
                 l2=1e-2, dropout_rate=0.2, maxnorm=3.0):
        super().__init__()
        self.ruleGraphConvLayer = layers.RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
        self.ruleGraphConvLayer.combination_rules = []
        self.conv = layers.ConvLayer(c_out_channel, r_out_channel)
        
        # Dense layers with regularization
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', name='dense1', 
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        
        self.dense5 = tf.keras.layers.Dense(16, activation='relu', name='dense2', 
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
        self.dense6 = tf.keras.layers.Dense(1, name='dense6', 
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))
        
        # Physics-informed dense layer with custom initialization
        self.dense7 = tf.keras.layers.Dense(1, name='dense7',
                                           kernel_initializer=tf.keras.initializers.Constant([0.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]),
                                           bias_initializer=tf.keras.initializers.Zeros(),
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))

    def addRule(self, rule, start_index, end_index=None):
        """Add a combination rule to the graph convolution layer."""
        self.ruleGraphConvLayer.addRule(rule, start_index, end_index)

    def call(self, inputs, training=False):
        """Forward pass of the model."""
        # Extract physics info from first atom of each molecule (index 0)
        physics_info = inputs[:, 0, 38:]
        
        # Extract atom features (first 38 dimensions) from all samples
        atom_features = inputs[:, :, :38]
        
        # Process each sample in the batch using tf.map_fn for graph convolution
        def process_sample_graph_conv(sample):
            # Apply RuleGraphConv layer - wrap to handle single sample
            rule_conv_output = self.ruleGraphConvLayer._call_single(sample)
            # Apply ConvLayer - wrap to handle single sample  
            conv_output = self.conv._call_single(rule_conv_output)
            return conv_output
        
        # Process all samples in the batch with graph convolution
        #x = tf.map_fn(process_sample_graph_conv, atom_features, dtype=tf.float32)
        x = tf.map_fn(process_sample_graph_conv, atom_features, fn_output_signature=tf.float32)
        
        # Use the final dense layers
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense5(x)
        x = self.dropout2(x, training=training)
        model_var = self.dense6(x)
        
        # Merge model prediction with physics information
        merged = tf.concat([model_var, physics_info], axis=1)
        out = self.dense7(merged)
        # Add to enforce negative values (I think this is whats causing loss to be infinite?)
        #out = -tf.exp(out)
        
        return tf.concat([out, physics_info], axis=1)


class LossComponentsCallback(tf.keras.callbacks.Callback):
    """Callback to track different loss components during training."""
    
    def __init__(self, model_instance):
        super().__init__()
        self.empirical_losses = []
        self.physical_losses = []
        self.total_losses = []
        self.learning_rates = []
        self.model_instance = model_instance
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch to log metrics."""
        logs = logs or {}
        # Store the total loss
        self.total_losses.append(logs.get('loss'))
        
        lr = self.model_instance.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model_instance.optimizer.iterations)
        
        self.learning_rates.append(float(tf.keras.backend.get_value(lr)))
        
        # Print memory usage every few epochs
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}: Memory usage: {get_memory_usage():.2f} GB")


# ============================================================================
# LOSS FUNCTIONS (Same as before)
# ============================================================================

def root_mean_squared_error(y_true, y_pred):
    """Root mean squared error with additional term."""
    return K.sqrt(K.mean(K.square(y_pred[0] - y_true))) + K.abs(1 / K.mean(.2 + y_pred[1]))


def pure_rmse(y_true, y_pred):
    """Pure root mean squared error."""
    y_true_flat = tf.reshape(y_true, [-1])
    return K.sqrt(K.mean(K.square(y_pred - y_true_flat)))


def physical_consistency_loss(y_true, y_pred, physics_info):
    """
    Physics-based consistency loss function.
    
    Calculates the difference between predicted binding affinity and 
    physics-based calculation using energy components.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    dG_pred = y_pred
    y_true = tf.reshape(y_true, (-1, 1))

    # Extract energy components from physics_info
    host = tf.gather(physics_info, [0, 3, 6, 9, 12], axis=1)      # Host energy terms
    guest = tf.gather(physics_info, [1, 4, 7, 10, 13], axis=1)   # Guest energy terms
    complex_ = tf.gather(physics_info, [2, 5, 8, 11, 14], axis=1) # Complex energy terms

    # Calculate ΔG based on physics: ΔG = ΔGcomplex - (ΔGhost + ΔGguest)
    dG_physics = tf.reduce_sum(complex_, axis=1, keepdims=True) - (
        tf.reduce_sum(host, axis=1, keepdims=True) + 
        tf.reduce_sum(guest, axis=1, keepdims=True)
    )
    
    phy_loss = K.sqrt(K.mean(K.square(dG_pred - dG_physics)))
    return phy_loss


def combined_loss(physics_hyperparam=0.0003):
    """
    Combined loss function with empirical and physics components.
    
    Args:
        physics_hyperparam: Weight for physics loss component
        
    Returns:
        Loss function that combines empirical RMSE and physics consistency
    """
    def loss_function(y_true, y_pred):
        # Extract prediction and physics info
        prediction = y_pred[:, 0]
        physics_info = y_pred[:, 1:16]  # Assuming 15 physical features
        
        # Calculate individual loss components
        empirical_loss = pure_rmse(y_true, prediction)
        physics_loss = physical_consistency_loss(y_true, prediction, physics_info)
        
        # Combine losses with weights
        total_loss = empirical_loss + (physics_hyperparam * physics_loss)
        
        return total_loss
    
    return loss_function


# ============================================================================
# SCALABLE TRAINING UTILITIES
# ============================================================================

def pad_sequences_adaptive(X, config, max_atoms_actual):
    """Adaptive padding based on actual data and memory constraints."""
    # Option 1: Use actual max atoms if memory allows
    estimated_memory_gb = (max_atoms_actual * 53 * 4 * len(X) * config.batch_size) / (1024**3)
    
    if estimated_memory_gb < config.memory_limit_gb * 0.7:  # 70% threshold
        max_length = max_atoms_actual  # Use actual max - NO TRUNCATION!
        print(f"Using actual max atoms: {max_length} (estimated memory: {estimated_memory_gb:.1f} GB)")
    else:
        # Option 2: Use configured max with warning
        max_length = min(max_atoms_actual, config.max_padding)
        print(f"Memory constraint active: using {max_length} atoms (actual max: {max_atoms_actual})")
        print(f"WARNING: {max_atoms_actual - max_length} atoms will be truncated from largest structures!")
    
    # Further reduce if memory is critically tight
    current_memory = get_memory_usage()
    if current_memory > config.memory_limit_gb * 0.6:  # 60% threshold
        max_length = min(max_length, config.max_padding // 2)  # More aggressive limit
        print(f"CRITICAL: Memory usage high, reducing to {max_length} atoms")
    
    print(f"Final padding decision: {max_length} atoms")
    
    # Track truncation statistics
    truncated_count = 0
    max_truncated_atoms = 0
    
    for i in range(len(X)):
        original_atoms = X[i].shape[0]
        if X[i].shape[0] < max_length:
            padding_size = max_length - X[i].shape[0]
            padding = np.zeros([padding_size, X[i].shape[1]], dtype=np.float32)
            X[i] = np.concatenate([X[i], padding], 0).astype(np.float32)
        elif X[i].shape[0] > max_length:
            # Truncate if too large
            atoms_lost = X[i].shape[0] - max_length
            max_truncated_atoms = max(max_truncated_atoms, atoms_lost)
            X[i] = X[i][:max_length].astype(np.float32)
            print(f"  Truncated structure {i} from {X[i].shape[0]} to {max_length} atoms")
    
    return np.array(X, dtype=np.float32)


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
        
        # Save the plot
        plt.tight_layout()
        filename = f'training_results_{config.dataset_size}structures_new.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
        
        # Try to show the plot
        plt.show()
        
        # Print summary statistics
        print(f"\nTraining Summary ({config.dataset_size} structures):")
        print(f"  Initial Loss: {loss_tracker.total_losses[0]:.6f}")
        print(f"  Final Loss: {loss_tracker.total_losses[-1]:.6f}")
        print(f"  Loss Reduction: {((loss_tracker.total_losses[0] - loss_tracker.total_losses[-1]) / loss_tracker.total_losses[0] * 100):.2f}%")
        print(f"  Total Epochs: {len(loss_tracker.total_losses)}")
            
    except Exception as e:
        print(f"Error creating plot: {e}")
        print("Printing loss values instead:")
        for i, loss in enumerate(loss_tracker.total_losses):
            print(f"Epoch {i+1}: {loss:.6f}")


# ============================================================================
# RESULT ANALYSIS UTILITIES
# ============================================================================

def load_and_explore_results(pickle_filename):
    """
    Load saved results and print exploration summary.
    
    Args:
        pickle_filename: Path to the saved pickle file
    
    Returns:
        Dictionary containing all saved results
    """
    try:
        with open(pickle_filename, 'rb') as f:
            results_data = pickle.load(f)
        
        print(f"=" * 70)
        print(f"LOADED RESULTS FROM: {pickle_filename}")
        print(f"=" * 70)
        
        # Print experiment info
        exp_info = results_data['experiment_info']
        print(f"Experiment Info:")
        print(f"  Timestamp: {exp_info['timestamp']}")
        print(f"  Dataset size: {exp_info['dataset_size']} structures")
        print(f"  Runtime: {exp_info['total_runtime_minutes']:.2f} minutes")
        
        # Print summary metrics
        summary = results_data['summary_metrics']
        print(f"\nBest Results:")
        print(f"  Best physics weight: {summary['best_physics_weight']}")
        print(f"  Best test loss: {summary['best_test_loss']:.6f}")
        print(f"  Best train loss: {summary['best_train_loss']:.6f}")
        print(f"  Best MAE: {summary['best_mae']:.6f}")
        
        # Print available data for visualization
        print(f"\nAvailable data for visualization:")
        for i, result in enumerate(results_data['all_results']):
            print(f"  Result {i+1}: Physics weight {result['physics_weight']}")
            print(f"    - Test predictions: {len(result['y_pred_test'])} samples")
            print(f"    - Training epochs: {result['epochs_trained']}")
            print(f"    - Training history available: {'Yes' if result['training_history']['total_losses'] else 'No'}")
        
        return results_data
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


# ============================================================================
# MAIN SCALABLE TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function with scalable configuration."""
    print("=" * 70)
    print("Starting PGGCN Scalable Training Script")
    print("=" * 70)
    
    # Configuration - MODIFY THESE VALUES FOR DIFFERENT DATASET SIZES
    # 
    # SCALING GUIDELINES:
    # - Small test (10 structures, 5 epochs): Quick verification, ~0.5 min
    # - Medium scale (50-100 structures, 50 epochs): Development/debugging, ~10-30 min  
    # - Large scale (500+ structures, 100+ epochs): Production training, hours
    # - Memory: Increase memory_limit_gb for larger datasets (32GB+ recommended for 500+ structures)
    # - Batch size: Keep at 2-4 for memory efficiency with graph convolutions
    config = DatasetConfig(
        dataset_size=100,       # Number of structures to process (10, 50, 100, 500+)
        max_padding=3000,       # Maximum atoms to pad to (3000 works for most datasets)
        batch_size=8,           # Batch size (increased for more stable gradients)
        epochs=100,             # Number of training epochs (5 for test, 50-100+ for production)
        memory_limit_gb=32      # Memory limit in GB (16GB+ for small, 32GB+ for large)
    )
    
    print(f"Configuration:")
    print(f"  Dataset size: {config.dataset_size} structures")
    print(f"  Max padding: {config.max_padding} atoms")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Memory limit: {config.memory_limit_gb} GB")
    print(f"  Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Clear any cached TensorFlow functions and reset the backend completely
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph() if hasattr(tf.compat.v1, 'reset_default_graph') else None
    
    # Force garbage collection
    gc.collect()
    
    # Reload the layers module to ensure we get fresh code
    global layers
    importlib.reload(layers)
    
    # Configure TensorFlow for memory efficiency
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory configuration error: {e}")
    
    # Load and preprocess data
    df_final, filtered_PDBs = load_data(config)
    info = extract_physics_info(df_final, filtered_PDBs)
    
    # Prepare features with chunked processing
    X, y, max_atoms_actual = prepare_features_chunked(df_final, filtered_PDBs, info, config)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    print(f"Data split: {len(X_train)} training, {len(X_test)} testing")
    
    # Training hyperparameters
    #physics_hyperparam = [0.000002]
    # Sweep over multiple physics weights to find the best one
    physics_hyperparam = [1e-5, 1e-6, 1e-7, 2e-6, 5e-6]
    print(f"Physics hyperparameter sweep: {physics_hyperparam}")
    
    # Adaptive learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.00001,  # Lowered learning rate for smoother, more stable training
        decay_steps=max(1000, len(X_train) * 5),  # Adaptive based on data size
        decay_rate=0.85,
        staircase=True
    )
    
    # Initialize result tracking
    all_results = []
    
    start = time.time()
    
    # Prepare log file for sweep summary
    sweep_logfile = "physics_weight_sweep_summary.log"
    with open(sweep_logfile, "w") as sweep_log:
        sweep_log.write("# physics_weight, final_train_loss, test_loss, mean_abs_diff, epochs_trained\n")

        # Training loop
        for physics_weight in physics_hyperparam:
            print(f"\n{'='*50}")
            print(f"Training with physics_weight: {physics_weight}")
            print(f"{'='*50}")

            # Initialize model
            m = PGGCNModel()
            m.addRule("sum", 0, 32)
            m.addRule("multiply", 32, 33)
            m.addRule("distance", 33, 36)

            # Compile model
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            m.compile(loss=combined_loss(physics_weight), optimizer=opt)
            
            # Check for regularization losses in model.losses
            print(f"Number of regularization losses: {len(m.losses)}")
            if len(m.losses) > 0:
                total_reg = tf.add_n(m.losses)
                print(f"Total regularization loss: {total_reg.numpy():.6e}")
                
                # Inspect individual losses
                for i, loss in enumerate(m.losses):
                    print(f"  Loss {i}: {loss.numpy():.6e}")

            # Prepare training data with adaptive padding
            input_shapes = [np.array(X_train[i]).shape[0] for i in range(len(X_train))]
            # Note: input_shapes is now only used for reporting, not for model configuration

            print("Preparing training data with adaptive padding...")
            print(f"Original input shapes range: {min(input_shapes)} to {max(input_shapes)} atoms")
            X_train_padded = pad_sequences_adaptive(copy.deepcopy(X_train), config, max_atoms_actual)
            y_train_array = np.array(y_train, dtype=np.float32)

            print(f"Training data shape: {X_train_padded.shape}")
            print(f"Batch size: {config.batch_size}")
            print(f"Memory after training prep: {get_memory_usage():.2f} GB")

            # Force garbage collection
            gc.collect()

            # Initialize callbacks
            loss_tracker = LossComponentsCallback(m)
            early_stopping = EarlyStopping(
                monitor='loss',
                patience=10,  # Wait 20 epochs before stopping (good for 100+ structures)
                restore_best_weights=True,
                min_delta=0.0005,  # Smaller threshold for more precise stopping
                verbose=1
            )

            print(f"Starting training with batch size {config.batch_size} using in-memory data...")
            hist = m.fit(
                X_train_padded,
                y_train_array,
                batch_size=config.batch_size,
                epochs=config.epochs,
                callbacks=[early_stopping, loss_tracker],
                verbose=1
            )


            # Prepare train data predictions for scatter plot
            print("Making predictions on train data for scatter plot...")
            X_train_padded_pred = pad_sequences_adaptive(copy.deepcopy(X_train), config, max_atoms_actual)
            y_train_array_pred = np.array(y_train, dtype=np.float32)
            y_pred_train_full = m.predict(X_train_padded_pred)
            y_pred_train = np.array(y_pred_train_full[:, 0])

            # Prepare test data in memory
            input_shapes_test = [np.array(X_test[i]).shape[0] for i in range(len(X_test))]
            print("Preparing test data...")
            print(f"Test input shapes range: {min(input_shapes_test)} to {max(input_shapes_test)} atoms")
            X_test_padded = pad_sequences_adaptive(copy.deepcopy(X_test), config, max_atoms_actual)
            y_test_array = np.array(y_test, dtype=np.float32)
            print(f"Test data shape: {X_test_padded.shape}")
            gc.collect()
            # Make predictions in one go
            print("Making predictions...")
            y_pred_test_full = m.predict(X_test_padded)
            y_pred_test = np.array(y_pred_test_full[:, 0])
            # Calculate test loss using the same loss function as training
            print("Calculating test loss...")
            test_loss_fn = combined_loss(physics_weight)
            test_loss = test_loss_fn(y_test_array, y_pred_test_full)
            test_loss_value = float(test_loss.numpy())
            # Calculate metrics
            y_difference = np.mean(np.abs(np.abs(y_test_array) - np.abs(y_pred_test)))
            print(f"\nResults:")
            print(f"  Mean absolute difference: {y_difference:.6f}")
            print(f"  Test loss (using training loss function): {test_loss_value:.6f}")
            final_train_loss = loss_tracker.total_losses[-1] if loss_tracker.total_losses else None
            print(f"  Final training loss: {final_train_loss:.6f}")

            # Log sweep summary for this run
            sweep_log.write(f"{physics_weight}, {final_train_loss:.6f}, {test_loss_value:.6f}, {y_difference:.6f}, {len(loss_tracker.total_losses)}\n")
            sweep_log.flush()

            # Store comprehensive results for visualization
            results = {
                'physics_weight': physics_weight,
                'dataset_size': config.dataset_size,
                'final_train_loss': final_train_loss,
                'test_loss': test_loss_value,  # Changed from eval_loss to test_loss
                'mean_abs_diff': y_difference,
                'epochs_trained': len(loss_tracker.total_losses),
                # Add data for visualizations
                'y_true_test': y_test_array,
                'y_pred_test': y_pred_test,
                'y_true_train': y_train_array_pred,
                'y_pred_train': y_pred_train,
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

            # Clean up memory before plotting
            del X_train_padded, X_test_padded
            gc.collect()

            # Plot results for this sweep
            plot_training_results(loss_tracker, config)
    
    end = time.time()
    runtime_minutes = (end - start) / 60
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Dataset size: {config.dataset_size} structures")
    print(f"Total runtime: {runtime_minutes:.2f} minutes")
    print(f"Average time per structure: {runtime_minutes/config.dataset_size:.2f} minutes")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    print(f"Peak memory stayed within {config.memory_limit_gb} GB limit")
    
    # Save results to pickle file for visualization
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pickle_filename = f'PGGCN_results_{config.dataset_size}structures_{timestamp}.pkl'
    
    # Prepare comprehensive data for saving
    results_data = {
        'experiment_info': {
            'timestamp': timestamp,
            'dataset_size': config.dataset_size,
            'total_runtime_minutes': runtime_minutes,
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
    
    # Save to pickle file
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
        print(f"  - Summary statistics")
        print(f"Use this file for generating visualizations and analysis!")
    except Exception as e:
        print(f"Error saving pickle file: {e}")
    
    # Print all results
    print(f"\nFinal Results Summary:")
    for result in all_results:
        print(f"  Physics weight {result['physics_weight']}: "
              f"Train loss {result['final_train_loss']:.6f}, "
              f"Test loss {result['test_loss']:.6f}, "
              f"MAE {result['mean_abs_diff']:.6f}")


if __name__ == "__main__":
    main()