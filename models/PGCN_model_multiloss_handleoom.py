from tensorflow.keras.callbacks import Callback
import importlib
import keras.backend as K
import numpy as np
import copy
import tensorflow as tf
# import tensorflow_addons as tfa
from models.dcFeaturizer import atom_features as get_atom_features

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import regularizers, constraints, callbacks
import sys
from tensorflow.keras.callbacks import EarlyStopping
import rdkit
import pickle
from . import layers_update_mobley as layers
import importlib
importlib.reload(layers)



# Enable memory growth to avoid OOM errors
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled on {len(gpus)} GPUs")
except Exception as e:
    print(f"Warning: GPU configuration failed: {e}")
    print("Continuing with default settings")

# Enable mixed precision for faster computations
try:
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")
except Exception as e:
    print(f"Mixed precision not available: {e}")
    print("Continuing with default precision")

# Required imports for RuleGraphConvLayer and ConvLayer
# You'll need to ensure these are properly imported in your environment
try:
    # This is a placeholder - in your code, you'd import your actual layers
    from layers_update_mobley import RuleGraphConvLayer, ConvLayer
    layers_imported = True
except Exception as e:
    print(f"Warning: Could not import custom layers: {e}")
    print("Defining placeholder classes for documentation purposes only")
    layers_imported = False
    
    # Placeholder classes if imports fail
    class RuleGraphConvLayer(tf.keras.layers.Layer):
        def __init__(self, out_channel, num_features=81, num_bond=22):
            super().__init__()
            self.out_channel = out_channel
            self.num_features = num_features
            self.num_bond = num_bond
            self.combination_rules = []
            
        def addRule(self, rule, start_index, end_index=None):
            pass
            
        def call(self, inputs):
            return inputs
    
    class ConvLayer(tf.keras.Model):
        def __init__(self, out_channel, num_features=20):
            super().__init__()
            self.out_channel = out_channel
            self.num_features = num_features
            
        def call(self, inputs):
            return tf.zeros([len(inputs), self.out_channel])


class PGGCNModel(tf.keras.Model):
    def __init__(self, num_atom_features=36, r_out_channel=20, c_out_channel=1024, l2=1e-5, dropout_rate=0.4, maxnorm=3.0):
        super().__init__()
        self.ruleGraphConvLayer = RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
        self.ruleGraphConvLayer.combination_rules = []
        self.conv = ConvLayer(c_out_channel, r_out_channel)
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
                                           kernel_initializer=tf.keras.initializers.Constant([0.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]),
                                           bias_initializer=tf.keras.initializers.Zeros(),
                                           kernel_regularizer=regularizers.l2(l2), 
                                           bias_regularizer=regularizers.l2(l2), 
                                           kernel_constraint=constraints.MaxNorm(maxnorm))
        # Initialize attributes
        self.max_num_atoms = 0
        self.is_first_call = True
        self.i_s = []

    def addRule(self, rule, start_index, end_index=None):
        """Add a combination rule to the RuleGraphConvLayer"""
        self.ruleGraphConvLayer.addRule(rule, start_index, end_index)
        
    def set_input_shapes(self, i_s):
        """Set the input shapes for each molecule"""
        self.i_s = i_s
        # Pre-calculate max atom count to optimize memory usage
        self.max_atoms = max(i_s) if i_s else 0
        self.max_num_atoms = self.max_atoms
        
    def call(self, inputs, training=False):
        """Forward pass of the model"""
        # Print only on first call for efficiency
        if self.is_first_call:
            print("Inside call")
            self.is_first_call = False
            
        # Extract physics info (common across all samples)
        physics_info = inputs[:, 0, 38:] 
        
        # Get batch size from inputs
        batch_size = tf.shape(inputs)[0]
        all_outputs = []
        
        # Process each sample in the batch
        for b in range(batch_size):
            try:
                # Initialize container for molecule data
                x_a = []
                
                # Get data for current sample
                sample = inputs[b]
                
                # Process molecule data
                for i in range(len(self.i_s)):
                    if i < len(self.i_s):  # Safety check
                        # Get atom features up to the actual number of atoms
                        sample_atoms = sample[:self.i_s[i], :38]
                        x_a.append(sample_atoms)
                
                # Forward pass through model layers
                x = self.ruleGraphConvLayer(x_a)
                x = self.conv(x)
                x = self.dense1(x)
                x = self.dropout1(x, training=training)
                x = self.dense5(x)
                x = self.dropout2(x, training=training)
                model_var = self.dense6(x)
                
                # Get physics info for this sample
                sample_physics = physics_info[b:b+1]
                
                # Combine model output with physics info
                merged = tf.concat([model_var, sample_physics], axis=1)
                out = self.dense7(merged)
                all_outputs.append(tf.concat([out, sample_physics], axis=1))
                
            except Exception as e:
                # Handle exceptions gracefully
                print(f"Error processing sample {b}: {e}")
                # Create a placeholder output with zeros
                placeholder = tf.zeros([1, 16])  # Adjust size if needed
                all_outputs.append(placeholder)
        
        # Stack all outputs into a single batch tensor
        return tf.concat(all_outputs, axis=0)


class MemoryEfficientBatchGenerator(tf.keras.utils.Sequence):
    """Custom batch generator that efficiently manages memory"""
    
    def __init__(self, X, y, batch_size, max_num_atoms, n_features=53, shuffle=True):
        """Initialize the batch generator
        
        Args:
            X: List of molecule feature arrays
            y: List of target values
            batch_size: Number of molecules per batch
            max_num_atoms: Maximum number of atoms to pad to
            n_features: Number of features per atom
            shuffle: Whether to shuffle the data after each epoch
        """
        self.X = X
        self.y = y
        self.batch_size = min(batch_size, len(X))  # Ensure batch_size doesn't exceed dataset size
        self.max_num_atoms = max_num_atoms
        self.n_features = n_features
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        
        # Shuffle initially if requested
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        """Return the number of batches per epoch"""
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get a batch of data"""
        # Get indices for this batch
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.X))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Prepare batch data
        batch_X = []
        batch_y = []
        
        for i in batch_indices:
            try:
                # Get original molecule data
                mol_X = self.X[i]
                
                # Pad molecule if needed
                if mol_X.shape[0] < self.max_num_atoms:
                    padding = np.zeros([self.max_num_atoms - mol_X.shape[0], self.n_features])
                    mol_X = np.concatenate([mol_X, padding], axis=0)
                elif mol_X.shape[0] > self.max_num_atoms:
                    # Truncate if molecule is larger than max_num_atoms
                    mol_X = mol_X[:self.max_num_atoms, :]
                
                # Add to batch
                batch_X.append(mol_X)
                batch_y.append(self.y[i])
                
            except Exception as e:
                print(f"Error preparing molecule {i}: {e}")
                # Skip this molecule or create a placeholder
                continue
        
        # Convert to numpy arrays
        if not batch_X:  # Handle empty batch case
            # Create a small placeholder batch to avoid errors
            return np.zeros((1, self.max_num_atoms, self.n_features)), np.zeros(1)
            
        return np.array(batch_X), np.array(batch_y)
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


class AdvancedLossTracker(Callback):
    """Advanced callback for tracking losses and implementing early stopping"""
    
    def __init__(self, model_instance, patience=10, min_delta=0.001, verbose=1):
        """Initialize the loss tracker
        
        Args:
            model_instance: The model being trained
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        super().__init__()
        self.total_losses = []
        self.learning_rates = []
        self.model = model_instance
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.best_weights = None
        self.verbose = verbose
        self.epoch_times = []
        self.start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of each epoch"""
        import time
        self.start_time = time.time()
        
        # Print progress periodically
        if self.verbose > 0 and epoch % 10 == 0:
            print(f"\nStarting epoch {epoch}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        import time
        logs = logs or {}
        
        # Calculate epoch duration
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.epoch_times.append(duration)
            if self.verbose > 0 and (epoch % 10 == 0 or epoch < 10):
                print(f"Epoch {epoch} took {duration:.2f} seconds")
        
        # Get current loss
        current_loss = logs.get('loss')
        
        # Store metrics
        if current_loss is not None:
            self.total_losses.append(current_loss)
            
            if self.verbose > 1:
                print(f"Epoch {epoch} loss: {current_loss:.6f}")
        
        # Get learning rate
        try:
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, '__call__'):
                lr = lr(self.model.optimizer.iterations)
            lr_value = float(tf.keras.backend.get_value(lr))
            self.learning_rates.append(lr_value)
            
            if self.verbose > 1:
                print(f"Learning rate: {lr_value:.6f}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not get learning rate: {e}")
        
        # Early stopping logic
        if current_loss is None:
            return
            
        if current_loss < self.best_loss - self.min_delta:
            if self.verbose > 0:
                print(f"Loss improved from {self.best_loss:.6f} to {current_loss:.6f}")
            self.best_loss = current_loss
            self.wait = 0
            # Save weights
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.verbose > 0 and self.wait > 0:
                print(f"Loss did not improve from {self.best_loss:.6f}, wait: {self.wait}/{self.patience}")
                
            if self.wait >= self.patience:
                self.model.stop_training = True
                if self.verbose > 0:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                # Restore best weights
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    if self.verbose > 0:
                        print("Restored best weights")


def root_mean_squared_error(y_true, y_pred):
    """Calculate RMSE loss with additional term"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Extract main prediction and additional term
    pred_val = y_pred[:, 0]
    additional_term = y_pred[:, 1]
    
    # Calculate RMSE
    rmse = tf.sqrt(tf.reduce_mean(tf.square(pred_val - y_true)))
    
    # Add regularization term
    reg_term = tf.abs(1.0 / tf.reduce_mean(0.2 + additional_term))
    
    return rmse + reg_term


def pure_rmse(y_true, y_pred):
    """Calculate pure RMSE without additional terms"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


def physical_consistency_loss(y_true, y_pred, physics_info):
    """Calculate physical consistency loss based on physics principles"""
    # Ensure all inputs have consistent types
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    physics_info = tf.cast(physics_info, tf.float32)
    
    # Reshape true values if needed
    y_true = tf.reshape(y_true, (-1, 1))

    try:
        # Extract energy components from physics info
        host = tf.gather(physics_info, [0, 3, 6, 9, 12], axis=1)
        guest = tf.gather(physics_info, [1, 4, 7, 10, 13], axis=1)
        complex_ = tf.gather(physics_info, [2, 5, 8, 11, 14], axis=1)
        
        # Calculate physics-based ΔG
        dG_physics = tf.reduce_sum(complex_, axis=1, keepdims=True) - (
            tf.reduce_sum(host, axis=1, keepdims=True) + 
            tf.reduce_sum(guest, axis=1, keepdims=True)
        )
        
        # Calculate difference between predicted and physics-based values
        phy_loss = tf.sqrt(tf.reduce_mean(tf.square(y_pred - dG_physics)))
        
        return phy_loss
        
    except Exception as e:
        print(f"Error in physical_consistency_loss: {e}")
        # Return a small constant value in case of error
        return tf.constant(0.01, dtype=tf.float32)


def combined_loss(physics_hyperparam=0.0003):
    """Create a combined loss function with empirical and physics components
    
    Args:
        physics_hyperparam: Weight for the physics-based loss component
    
    Returns:
        A loss function that can be used in model.compile()
    """
    def loss_function(y_true, y_pred):
        try:
            # Cast inputs to ensure consistent types
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            # Extract prediction and physics info
            prediction = y_pred[:, 0]
            physics_info = y_pred[:, 1:16]  # Assuming 15 physical features
            
            # Calculate individual loss components
            empirical_loss = pure_rmse(y_true, prediction)
            physics_loss = physical_consistency_loss(y_true, prediction, physics_info)
            
            # Combine with weighting
            total_loss = empirical_loss + (physics_hyperparam * physics_loss)
            
            return total_loss
            
        except Exception as e:
            print(f"Error in combined_loss: {e}")
            # Return a default loss in case of error to avoid training failure
            return tf.reduce_mean(tf.square(y_pred[:, 0] - y_true)) * 10.0
    
    return loss_function


def get_trained_model(X, y, epochs=1, physics_weight=0.00005, batch_size=8, max_num_atoms=None, n_features=53, verbose=1):
    """Train a PGGCN model on the provided data
    
    Args:
        X: List of molecule feature arrays
        y: List of target values
        epochs: Number of training epochs
        physics_weight: Weight for physics-based loss
        batch_size: Number of molecules per batch
        max_num_atoms: Maximum number of atoms to pad to (calculated if None)
        n_features: Number of features per atom
        verbose: Verbosity level
        
    Returns:
        Tuple of (loss_history, trained_model, processed_X)
    """
    # Print training parameters
    if verbose > 0:
        print(f"Training model with:")
        print(f"  - {len(X)} molecules")
        print(f"  - {epochs} epochs")
        print(f"  - {batch_size} batch size")
        print(f"  - {physics_weight} physics weight")
    
    # Determine optimal padding size
    if max_num_atoms is None:
        # Find sizes of all molecules
        sizes = [x.shape[0] for x in X]
        
        if len(sizes) > 1:
            # Sort sizes and use second largest to avoid excessive padding
            sizes.sort(reverse=True)
            max_num_atoms = sizes[1]
        else:
            # If only one molecule, use its size
            max_num_atoms = sizes[0]
            
        # Round up to nearest 1000 for better memory alignment
        max_num_atoms = ((max_num_atoms + 999) // 1000) * 1000
        
        if verbose > 0:
            print(f"Using optimal max_num_atoms: {max_num_atoms}")
    
    try:
        # Create model
        m = PGGCNModel()
        
        # Add rules
        m.addRule("sum", 0, 32)
        m.addRule("multiply", 32, 33)
        m.addRule("distance", 33, 36)
        
        # Set learning rate schedule
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.005,
            decay_steps=5000,
            decay_rate=0.9,
            staircase=True
        )

        # Create optimizer with gradient clipping
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0  # Clip gradients to prevent explosions
        )
        
        # Compile model
        m.compile(loss=combined_loss(physics_weight), optimizer=opt)
        
        # Set input shapes
        input_shapes = [x.shape[0] for x in X]
        m.set_input_shapes(input_shapes)
        
        # Set up advanced loss tracker with early stopping
        loss_tracker = AdvancedLossTracker(m, patience=15, min_delta=0.001, verbose=verbose)
        
        # Create memory-efficient data generator
        train_gen = MemoryEfficientBatchGenerator(X, y, batch_size, max_num_atoms, n_features)
        
        # Train model
        if verbose > 0:
            print("Starting training...")
            
        hist = m.fit(
            train_gen,
            epochs=epochs,
            verbose=max(0, verbose-1),  # Reduce keras verbosity
            callbacks=[loss_tracker]
        )
        
        if verbose > 0:
            print("Training completed")
            
        # Return history and model
        return loss_tracker.total_losses, m, X
        
    except Exception as e:
        print(f"Error during model training: {e}")
        # Return empty results in case of failure
        return [], None, X


def test_model(X_test, y_test, m, batch_size=8, max_num_atoms=None, n_features=53, verbose=1):
    """Test a trained model on test data
    
    Args:
        X_test: List of molecule feature arrays for testing
        y_test: List of target values for testing
        m: Trained PGGCNModel
        batch_size: Number of molecules per batch
        max_num_atoms: Maximum number of atoms to pad to
        n_features: Number of features per atom
        verbose: Verbosity level
        
    Returns:
        Tuple of (evaluation_result, y_difference, processed_X_test)
    """
    if m is None:
        print("Error: No model provided")
        return None, None, X_test
        
    try:
        # Use model's max_num_atoms if not specified
        if max_num_atoms is None:
            if hasattr(m, 'max_num_atoms'):
                max_num_atoms = m.max_num_atoms
            else:
                # Calculate from data
                sizes = [x.shape[0] for x in X_test]
                max_num_atoms = max(sizes)
                max_num_atoms = ((max_num_atoms + 999) // 1000) * 1000
        
        # Set input shapes
        input_shapes = [x.shape[0] for x in X_test]
        m.set_input_shapes(input_shapes)
        
        # Create test data generator
        test_gen = MemoryEfficientBatchGenerator(
            X_test, y_test, batch_size, max_num_atoms, n_features, shuffle=False
        )
        
        if verbose > 0:
            print(f"Evaluating model on {len(X_test)} test molecules...")
            
        # Evaluate model
        eval_result = m.evaluate(test_gen, verbose=max(0, verbose-1))
        
        # Get predictions batch by batch
        if verbose > 0:
            print("Generating predictions...")
            
        y_preds = []
        for i in range(len(test_gen)):
            try:
                batch_x, _ = test_gen[i]
                batch_pred = m.predict(batch_x, verbose=0)
                batch_preds = batch_pred[:, 0].numpy()  # Extract predictions
                y_preds.extend(batch_preds)
            except Exception as e:
                if verbose > 0:
                    print(f"Error predicting batch {i}: {e}")
                # Add placeholders for failed predictions
                batch_size_actual = min(batch_size, len(X_test) - i*batch_size)
                y_preds.extend([0.0] * batch_size_actual)
        
        # Ensure predictions match test data length
        y_pred_test = np.array(y_preds[:len(y_test)])
        
        # Calculate mean absolute difference
        y_difference = np.mean(np.abs(np.abs(y_test) - np.abs(y_pred_test)))
        
        if verbose > 0:
            print(f"Test evaluation complete")
            print(f"Mean absolute difference: {y_difference:.6f}")
            
        return eval_result, y_difference, X_test
        
    except Exception as e:
        print(f"Error during model testing: {e}")
        return None, None, X_test
