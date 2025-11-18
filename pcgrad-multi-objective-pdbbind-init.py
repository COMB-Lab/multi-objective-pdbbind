#!/usr/bin/env python3
"""
PCGRad PDBbind Scalable - PGGCN Model Training Script (Large Dataset Support)

This script implements a Physics-Guided Graph Convolutional Network (PGGCN) model
for predicting binding free energies using scalable approaches for large datasets.
The model incorporates both empirical and physics-based loss functions.

"""

# ============================================================================
# IMPORTS AND ENVIRONMENT SETUP
# ============================================================================

import os
import sys

# Set CUDA environment variables BEFORE importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth

import pandas as pd
import tensorflow as tf
import numpy as np
from rdkit import Chem
from deepchem.feat.graph_features import atom_features as get_atom_features
import rdkit
import pickle
import copy
import gc
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import time
import importlib
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras.backend as K
from tensorflow.keras import regularizers, constraints, callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

# Configure TensorFlow for GPU
# print("Configuring TensorFlow for GPU usage...")
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print(f"Found {len(gpus)} GPU(s), memory growth enabled")
#         mixed_precision.set_global_policy('mixed_float16')
#         tf.config.optimizer.set_jit(True)  # Enable XLA
#     except RuntimeError as e:
#         print(f"GPU configuration error: {e}")
# else:
#     print("No GPUs found, using CPU")

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
    print(f"Selected {len(selected_rows)} structures")
    df = selected_rows

    # Filter PDBs to match available keys
    pdb_keys = set(PDBs.keys())
    df_filtered = df[df['complex-name'].isin(pdb_keys)]
    
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
    
    df_final = df_final[df_final['complex-name'].isin(common_keys)]
    filtered_PDBs = {k: v for k, v in filtered_PDBs.items() if k in common_keys}
    
    print(f"Final validated dataset: {len(df_final)} structures")
    print(f"Current memory usage: {get_memory_usage():.2f} GB")
    
    return df_final, filtered_PDBs


def extract_physics_info(df_final, filtered_PDBs):
    """Extract physics information for each PDB structure."""
    info = []
    for pdb in list(filtered_PDBs.keys()):
        physics_data = df_final[df_final['complex-name'] == pdb][
            ['pb-protein-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals', 
             'gb-protein-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
             'gb-protein-eelect', 'gb-ligand-eelec', 'gb-complex-eelec', 
             'gb-protein-egb', 'gb-ligand-egb', 'gb-complex-egb', 
             'gb-protein-esurf', 'gb-ligand-esurf', 'gb-complex-esurf']
        ].to_numpy()[0]
        info.append(physics_data)
    return info

# Import custom layers
import models.layers_update_mobley as layers
from models.dcFeaturizer import atom_features as get_atom_features

importlib.reload(layers)


def featurize(molecule, info):
    """Featurize a molecule with atom features and physics information."""
    atom_features = []
    for atom in molecule.GetAtoms():
        new_feature = get_atom_features(atom).tolist()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
        new_feature += [position.x, position.y, position.z]
        
        # Add neighbor information
        for neighbor in atom.GetNeighbors()[:2]:
            neighbor_idx = neighbor.GetIdx()
            new_feature += [neighbor_idx]
        for i in range(2 - len(atom.GetNeighbors())):
            new_feature += [-1]

        atom_features.append(np.concatenate([new_feature, info], 0))
    return np.array(atom_features, dtype=np.float32)


def prepare_features_chunked(df_final, filtered_PDBs, info, config, chunk_size=10):
    """Prepare feature matrices X and target values y with chunked processing."""
    print(f"Preparing features for {len(filtered_PDBs)} structures...")
    
    X = []
    y = []
    atom_counts = []
    
    pdb_list = list(filtered_PDBs.keys())
    
    for chunk_start in range(0, len(pdb_list), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(pdb_list))
        chunk_pdbs = pdb_list[chunk_start:chunk_end]
        
        for i, pdb in enumerate(chunk_pdbs):
            global_i = chunk_start + i
            features = featurize(filtered_PDBs[pdb], info[global_i])
            X.append(features)
            y.append(df_final[df_final['complex-name'] == pdb]['ddg'].to_numpy()[0])
            atom_counts.append(features.shape[0])
        
        if check_memory_limit(config):
            gc.collect()
    
    max_atoms = max(atom_counts)
    avg_atoms = np.mean(atom_counts)
    
    print(f"Dataset statistics:")
    print(f"  Max atoms: {max_atoms}")
    print(f"  Average atoms: {avg_atoms:.1f}")
    print(f"  Total structures: {len(X)}")
    
    return X, y, max_atoms


# ===========================================================================
# PCGrad Class 
# ===========================================================================
class PCGrad(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, name="PCGrad", **kwargs):
        kwargs.pop('learning_rate', None)
        super().__init__(name=name, **kwargs)
        self._optimizer = optimizer
    
    @property
    def learning_rate(self):
        return self._optimizer.learning_rate

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        return self._optimizer.apply_gradients(grads_and_vars, name, **kwargs)
    
    def build(self, var_list):
        """Build the optimizer"""
        super().build(var_list)
        if hasattr(self._optimizer, 'build'):
            self._optimizer.build(var_list)

    def get_config(self):
        config = super().get_config()
        config.update({"optimizer": tf.keras.optimizers.serialize(self._optimizer)})
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer_config = config.pop("optimizer")
        optimizer = tf.keras.optimizers.deserialize(optimizer_config, custom_objects=custom_objects)
        return cls(optimizer, **config)

    def compute_gradients(self, losses, tape, var_list, weights=None):
        """Compute PCGrad projected gradients from a list of task losses."""
        assert isinstance(losses, list), "loss must be a list of task losses"
        
        grads_task = []
        for loss in losses:
            grads = tape.gradient(loss, var_list)
            grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, var_list)]
            grads_task.append(grads)

        def flatten(grads):
            return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

        flat_grads_task = [flatten(g) for g in grads_task]
        flat_grads_task = tf.stack(flat_grads_task)
        flat_grads_task = tf.random.shuffle(flat_grads_task)

        def project(g, others):
            for o in others:
                dot = tf.reduce_sum(g * o)
                if dot < 0:
                    g -= dot / (tf.reduce_sum(o * o) + 1e-12) * o
            return g

        projected = []
        for i in range(len(flat_grads_task)):
            others = tf.concat([flat_grads_task[:i], flat_grads_task[i+1:]], axis=0)
            projected.append(project(flat_grads_task[i], others))
        projected = tf.stack(projected)

        if weights is not None:
            weighted_projected = [w * p for w, p in zip(weights, projected)]
            mean_grad = tf.reduce_sum(tf.stack(weighted_projected), axis=0)
        else:
            mean_grad = tf.reduce_mean(projected, axis=0)

        reshaped_grads = []
        idx = 0
        for v in var_list:
            shape = tf.shape(v)
            size = tf.reduce_prod(shape)
            reshaped_grads.append(tf.reshape(mean_grad[idx:idx + size], shape))
            idx += size

        reshaped_grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in reshaped_grads]
        
        return list(zip(reshaped_grads, var_list))


#====================================================
# Model definition - FIXED INDENTATION
#===================================================

class PGGCN_Hybrid(tf.keras.Model):
    def __init__(self, num_atom_features=36, r_out_channel=20, c_out_channel=1024, 
                 l2=1e-4, dropout_rate=0.2, maxnorm=3.0):
        super().__init__()
        self.ruleGraphConvLayer = layers.RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
        self.ruleGraphConvLayer.combination_rules = []
        self.conv = layers.ConvLayer(c_out_channel, r_out_channel)
        
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
        
        self.dense7 = tf.keras.layers.Dense(1, name='dense7',
                                           kernel_initializer=tf.keras.initializers.Constant([.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]),
                                           bias_initializer=tf.keras.initializers.Zeros(),
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))
        
        self.i_s = None

    def addRule(self, rule, start_index, end_index=None):
        self.ruleGraphConvLayer.addRule(rule, start_index, end_index)
    
    def set_input_shapes(self, i_s):
        self.i_s = i_s

    def call(self, inputs):
        """        
        Args:
            inputs: Tensor of shape [batch_size, max_atoms, num_features (53)]
            training: Boolean for training mode
        
        Returns:
            Tensor of shape [batch_size, 16] (prediction + physics info)
        """        
        # Extract physics info from first atom of each molecule
        physics_info = inputs[:, 0, 38:]  # Shape: [batch_size, 15]
        
        # Process each sample separately through graph conv layers
        x_a = []
        
        for i in range(len(self.i_s)):
            x_a.append(inputs[i][:self.i_s[i], :38])
        
        x = self.ruleGraphConvLayer(x_a)
        final_weights = mn.ruleGraphConvLayer.w_s #capture after training

        # Apply dense layers
        x = self.conv
        x = self.desnse1(x)
        x = self.dense5(x)
        model_var = self.dense6(x)
        merged = tf.concat([model_var, physics_info], axis=1)
        out = self.dense7(merged)
       
        # Add to enforce negative values (this is causing problems, loss turns infinite)
        #out = -tf.exp(out)
        return out


empirical_loss_value = tf.Variable(0.0, trainable=False, dtype=tf.float32)
physics_loss_value = tf.Variable(0.0, trainable=False, dtype=tf.float32)


class LossComponentsCallback_Hybrid(tf.keras.callbacks.Callback):
    def __init__(self, model_instance):
        super().__init__()
        self.empirical_losses = []
        self.physical_losses = []
        self.total_losses = []
        self.learning_rates = []
        self.model_instance = model_instance
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model_instance.optimizer.learning_rate
        self.total_losses.append(logs.get('loss'))
        self.empirical_losses.append(float(empirical_loss_value.numpy()))
        self.physical_losses.append(float(physics_loss_value.numpy()))
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model_instance.optimizer.iterations)
        self.learning_rates.append(float(tf.keras.backend.get_value(lr)))


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def root_mean_squared_error(y_true, y_pred):
    """Root mean squared error with additional term."""
    return K.sqrt(K.mean(K.square(y_pred[0] - y_true))) + K.abs(1 / K.mean(.2 + y_pred[1]))


def pure_rmse(y_true, y_pred):
    """Pure root mean squared error."""
    y_true_flat = tf.reshape(y_true, [-1])
    return K.sqrt(K.mean(K.square(y_pred - y_true_flat)))


def physical_consistency_loss(y_true, y_pred, physics_info):
    """Physics-based consistency loss function."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    dG_pred = y_pred
    y_true = tf.reshape(y_true, (-1, 1))

    host = tf.gather(physics_info, [0, 3, 6, 9, 12], axis=1)
    guest = tf.gather(physics_info, [1, 4, 7, 10, 13], axis=1)
    complex_ = tf.gather(physics_info, [2, 5, 8, 11, 14], axis=1)

    dG_physics = tf.reduce_sum(complex_, axis=1, keepdims=True) - (
        tf.reduce_sum(host, axis=1, keepdims=True) + 
        tf.reduce_sum(guest, axis=1, keepdims=True)
    )
    
    phy_loss = K.sqrt(K.mean(K.square(dG_pred - dG_physics)))
    return phy_loss


def combined_loss(physics_hyperparam=0.0003):
    def loss_function(y_true, y_pred):
        prediction = y_pred[:, 0]
        physics_info = y_pred[:, 1:16]
        
        empirical_loss = pure_rmse(y_true, prediction)
        physics_loss = physical_consistency_loss(y_true, prediction, physics_info)
        
        total_loss = empirical_loss + (physics_hyperparam * physics_loss)
        empirical_loss_value.assign(empirical_loss)
        physics_loss_value.assign(physics_loss)

        return total_loss
    
    return loss_function


# ============================================================================
# SCALABLE TRAINING UTILITIES
# ============================================================================

def pad_sequences_adaptive(X, config, max_atoms_actual):
    """Adaptive padding based on actual data and memory constraints."""
    estimated_memory_gb = (max_atoms_actual * 53 * 4 * len(X) * config.batch_size) / (1024**3)
    
    if estimated_memory_gb < config.memory_limit_gb * 0.7:
        max_length = max_atoms_actual
        print(f"Using actual max atoms: {max_length}")
    else:
        max_length = min(max_atoms_actual, config.max_padding)
        print(f"Memory constraint active: using {max_length} atoms")
    
    current_memory = get_memory_usage()
    if current_memory > config.memory_limit_gb * 0.6:
        max_length = min(max_length, config.max_padding // 2)
        print(f"CRITICAL: Memory usage high, reducing to {max_length} atoms")
    
    for i in range(len(X)):
        if X[i].shape[0] < max_length:
            padding_size = max_length - X[i].shape[0]
            padding = np.zeros([padding_size, X[i].shape[1]], dtype=np.float32)
            X[i] = np.concatenate([X[i], padding], 0).astype(np.float32)
        elif X[i].shape[0] > max_length:
            X[i] = X[i][:max_length].astype(np.float32)
    
    return np.array(X, dtype=np.float32)


def main():
    """Main training function with all fixes."""
    print("=" * 70)
    print("Starting PGGCN Scalable Training Script")
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
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    
    # Clear session
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Load data
    df_final, filtered_PDBs = load_data(config)
    info = extract_physics_info(df_final, filtered_PDBs)
    
    # Prepare features
    X, y, max_atoms_actual = prepare_features_chunked(df_final, filtered_PDBs, info, config)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    print(f"Data split: {len(X_train)} training, {len(X_test)} testing")
    
    # Physics hyperparameters
    physics_hyperparam = [1e-5, 1e-6]
    
    # Initialize result tracking
    all_results = []
    start = time.time()
    
    # Training loop
    for physics_weight in physics_hyperparam:
        print(f"\n{'='*50}")
        print(f"Training with physics_weight: {physics_weight}")
        print(f"{'='*50}")
        
        # Clear session
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Initialize model
        m = PGGCN_Hybrid()
        m.addRule("sum", 0, 32)
        m.addRule("multiply", 32, 33)
        m.addRule("distance", 33, 36)
        
        # Set up optimizer
        lr_schedule_value = 1e-4
        tf.random.set_seed(int(physics_weight * 10000))
        base_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule_value)
        opt = PCGrad(base_opt)
        
        # Prepare training data
        print("Preparing training data...")
        input_shapes = [X_train[i].shape[0] for i in range(len(X_train))]
        m.set_input_shapes(input_shapes)
        
        X_train_padded = pad_sequences_adaptive(copy.deepcopy(X_train), config, max_atoms_actual)
        y_train_array = np.array(y_train, dtype=np.float32)
        
        print(f"Training data shape: {X_train_padded.shape}")
        
        # Training metrics
        total_losses = []
        empirical_losses = []
        physics_losses = []
        
        # Early stopping
        best_train_loss = float("inf")
        patience = 15
        patience_counter = 0
        min_delta = 0.001
        best_weights = None
        
        epochs = config.epochs
        
        for ep in range(epochs):
            try:
                with tf.GradientTape(persistent=True) as tape:
                    predictions = m(X_train_padded, training=True)
                    
                    emp_loss = pure_rmse(y_train_array, predictions[:, 0])
                    phy_loss = physical_consistency_loss(
                        y_train_array, 
                        predictions[:, 0], 
                        predictions[:, 1:16]
                    )
                    total_loss = emp_loss + physics_weight * phy_loss
                
                weights_vec = [1.0, float(physics_weight)]
                grads_and_vars = opt.compute_gradients(
                    [emp_loss, phy_loss], 
                    tape, 
                    m.trainable_variables, 
                    weights=weights_vec
                )
                
                opt.apply_gradients(grads_and_vars)
                del tape
                
                current_total_loss = float(total_loss.numpy())
                total_losses.append(current_total_loss)
                empirical_losses.append(float(emp_loss.numpy()))
                physics_losses.append(float(phy_loss.numpy()))
                
                if current_total_loss + min_delta < best_train_loss:
                    best_train_loss = current_total_loss
                    best_weights = m.get_weights()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {ep + 1}")
                        break
                
                if (ep + 1) % 10 == 0 or ep == 0:
                    print(f"Epoch {ep + 1}/{epochs} - Total: {current_total_loss:.4f}, "
                          f"Emp: {emp_loss.numpy():.4f}, Phy: {phy_loss.numpy():.4f}")
                
                if (ep + 1) % 20 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"Error at epoch {ep + 1}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Restore best weights
        if best_weights is not None:
            m.set_weights(best_weights)
            print("Restored best weights from training")
        
        if not total_losses:
            print("Training failed - no epochs completed")
            continue
        
        # Prepare test data
        print("\nPreparing test data...")
        X_test_padded = pad_sequences_adaptive(copy.deepcopy(X_test), config, max_atoms_actual)
        y_test_array = np.array(y_test, dtype=np.float32)

        print("Making predictions on test set...")
        test_input_shapes = [X_test[i].shape[0] for i in range(len(X_test))]
        m.set_input_shapes(test_input_shapes)

        y_pred_full = m(X_test_padded, training=False)
        y_pred_test = y_pred_full[:, 0]
        phys_info = y_pred_full[:, 1:16]

        y_pred_test_np = y_pred_test.numpy() if hasattr(y_pred_test, "numpy") else np.array(y_pred_test)

        test_mae = float(np.mean(np.abs(y_test_array - y_pred_test_np)))
        test_rmse = float(np.sqrt(np.mean((y_test_array - y_pred_test_np) ** 2)))

        emp = pure_rmse(y_test_array, y_pred_test)
        phy = physical_consistency_loss(y_test_array, y_pred_test, phys_info)
        test_loss_value = float((emp + physics_weight * phy).numpy())

        print(f"\nResults for physics_weight={physics_weight}:")
        print(f"  Final training loss: {total_losses[-1]:.6f}")
        print(f"  Test MAE: {test_mae:.6f}")
        print(f"  Test RMSE: {test_rmse:.6f}")
        print(f"  Test loss: {test_loss_value:.6f}")
        print(f"  Epochs trained: {len(total_losses)}")

        results = {
            'physics_weight': physics_weight,
            'final_train_loss': total_losses[-1],
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_loss': test_loss_value,
            'epochs_trained': len(total_losses),
            'y_true_test': y_test_array,
            'y_pred_test': y_pred_test_np,
            'training_history': {
                'total_losses': total_losses,
                'empirical_losses': empirical_losses,
                'physics_losses': physics_losses
            }
        }
        all_results.append(results)
        
        del X_test_padded, m
        gc.collect()
    
    end = time.time()
    runtime_minutes = (end - start) / 60
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Dataset size: {config.dataset_size} structures")
    print(f"Total runtime: {runtime_minutes:.2f} minutes")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Results'
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, f'PGGCN_results_{config.dataset_size}structures_{timestamp}.pkl')
    
    runs = [r for r in all_results if isinstance(r, dict)]
    
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
        'all_results': runs,
        'summary_metrics': {}
    }
    
    if runs:
        best_by_rmse = min(runs, key=lambda r: r['test_rmse'])
        best_by_mae = min(runs, key=lambda r: r['test_mae'])
        best_by_train = min(runs, key=lambda r: r['final_train_loss'])

        results_data['summary_metrics'] = {
            'best_physics_weight_by_rmse': best_by_rmse['physics_weight'],
            'best_test_rmse': float(best_by_rmse['test_rmse']),
            'best_physics_weight_by_mae': best_by_mae['physics_weight'],
            'best_test_mae': float(best_by_mae['test_mae']),
            'best_train_loss': float(best_by_train['final_train_loss'])
        }
    
    try:
        with open(full_path, 'wb') as f:
            pickle.dump(results_data, f)
        print(f"\nResults saved to: {full_path}")
    except Exception as e:
        print(f"Error saving pickle file: {e}")
    
    print(f"\nFinal Results Summary:")
    for r in runs:
        print(f"  Physics weight {r['physics_weight']}: "
              f"Train {r['final_train_loss']:.6f}, "
              f"Test RMSE {r['test_rmse']:.6f}, "
              f"Test MAE {r['test_mae']:.6f}")


if __name__ == "__main__":
    main()