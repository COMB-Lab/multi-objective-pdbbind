#!/usr/bin/env python3
"""
GPU Utilization Diagnostic and Fixes for PGGCN Training

This script identifies why GPU isn't being used and provides fixes.
"""

import os
import sys
import tensorflow as tf
import numpy as np

# ============================================================================
# STEP 1: DIAGNOSTIC SCRIPT - Run this first to identify the problem
# ============================================================================

def diagnose_gpu_setup():
    """Comprehensive GPU diagnostic."""
    print("="*70)
    print("GPU DIAGNOSTIC REPORT")
    print("="*70)
    
    # Check 1: TensorFlow GPU detection
    print("\n1. TensorFlow GPU Detection:")
    print(f"   TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"   Physical GPUs detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"     GPU {i}: {gpu}")
    
    # Check 2: CUDA availability
    print("\n2. CUDA Availability:")
    print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"   GPU available: {tf.test.is_gpu_available()}")
    
    # Check 3: Device placement
    print("\n3. Current Device Placement:")
    with tf.device('/GPU:0'):
        try:
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            c = tf.matmul(a, b)
            print(f"   Test operation device: {c.device}")
            print(f"   ✓ GPU operations working")
        except Exception as e:
            print(f"   ✗ GPU operation failed: {e}")
    
    # Check 4: Memory growth
    print("\n4. GPU Memory Configuration:")
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_memory_info(gpu.name.replace('physical_', ''))
            print(f"   {gpu.name}:")
            print(f"     Current memory: {details['current'] / 1e9:.2f} GB")
            print(f"     Peak memory: {details['peak'] / 1e9:.2f} GB")
        except:
            print(f"   Memory info not available for {gpu.name}")
    
    # Check 5: Tensor placement test
    print("\n5. Tensor Placement Test:")
    x = tf.random.normal([1000, 1000])
    print(f"   Random tensor device: {x.device}")
    
    # Check 6: Operation logs
    print("\n6. Enabling device placement logging...")
    tf.debugging.set_log_device_placement(True)
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0]])
        b = tf.constant([[3.0], [4.0]])
        c = tf.matmul(a, b)
    tf.debugging.set_log_device_placement(False)
    
    print("\n" + "="*70)
    return len(gpus) > 0


# ============================================================================
# STEP 2: FIXES FOR COMMON GPU ISSUES
# ============================================================================

# FIX 1: Proper GPU Configuration (Replace your GPU config section)
def configure_gpu_properly():
    """Proper GPU configuration that ensures utilization."""
    
    # CRITICAL: Set before importing other TF components
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '2'
    
    print("Configuring TensorFlow for GPU usage...")
    
    # List all devices
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print(f"Available GPUs: {len(gpus)}")
    print(f"Available CPUs: {len(cpus)}")
    
    if gpus:
        try:
            # Set memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set visible devices explicitly
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            # Get logical devices
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
            
            # Test GPU with actual computation
            with tf.device('/GPU:0'):
                test_tensor = tf.random.normal([1000, 1000])
                result = tf.matmul(test_tensor, test_tensor)
                print(f"GPU test successful. Result device: {result.device}")
            
            return True
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("No GPUs found!")
        return False


# FIX 2: Force GPU execution for model (add to PGGCN_Hybrid class)
class PGGCN_Hybrid_GPU(tf.keras.Model):
    """GPU-optimized version with explicit device placement."""
    
    def __init__(self, num_atom_features=36, r_out_channel=20, c_out_channel=128, 
                 l2=1e-4, dropout_rate=0.2, maxnorm=3.0):
        super().__init__()
        
        # Force layer creation on GPU
        with tf.device('/GPU:0'):
            from models import layers_update_mobley as layers
            
            self.ruleGraphConvLayer = layers.RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
            self.ruleGraphConvLayer.combination_rules = []
            self.conv = layers.ConvLayer(c_out_channel, r_out_channel)
            
            self.dense1 = tf.keras.layers.Dense(
                32, activation='relu', name='dense1',
                kernel_regularizer=tf.keras.regularizers.l2(l2),
                bias_regularizer=tf.keras.regularizers.l2(l2),
                kernel_constraint=tf.keras.constraints.MaxNorm(maxnorm)
            )
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
            
            self.dense5 = tf.keras.layers.Dense(
                16, activation='relu', name='dense2',
                kernel_regularizer=tf.keras.regularizers.l2(l2),
                bias_regularizer=tf.keras.regularizers.l2(l2),
                kernel_constraint=tf.keras.constraints.MaxNorm(maxnorm)
            )
            self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
            
            self.dense6 = tf.keras.layers.Dense(
                1, name='dense6',
                kernel_regularizer=tf.keras.regularizers.l2(l2),
                bias_regularizer=tf.keras.regularizers.l2(l2),
                kernel_constraint=tf.keras.constraints.MaxNorm(maxnorm)
            )
            
            self.dense7 = tf.keras.layers.Dense(
                1, name='dense7',
                kernel_initializer=tf.keras.initializers.Constant(
                    [.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]
                ),
                bias_initializer=tf.keras.initializers.Zeros(),
                kernel_regularizer=tf.keras.regularizers.l2(l2),
                bias_regularizer=tf.keras.regularizers.l2(l2),
                kernel_constraint=tf.keras.constraints.MaxNorm(maxnorm)
            )
        
        self.i_s = None

    def addRule(self, rule, start_index, end_index=None):
        self.ruleGraphConvLayer.addRule(rule, start_index, end_index)
    
    def set_input_shapes(self, i_s):
        self.i_s = i_s

    @tf.function(jit_compile=False)  # Disable XLA initially for debugging
    def call(self, inputs, training=True):
        """GPU-optimized call with explicit device placement."""
        
        # Ensure computation happens on GPU
        with tf.device('/GPU:0'):
            batch_size = tf.shape(inputs)[0]
            
            # Extract physics info
            physics_info = inputs[:, 0, 38:]
            
            # Pre-allocate output list
            conv_outputs = []
            
            # Process each sample
            for i in tf.range(batch_size):
                sample = inputs[i]
                atom_features = sample[:, :38]
                
                # Determine number of atoms
                if self.i_s is not None and i < len(self.i_s):
                    num_atoms = self.i_s[i]
                else:
                    atom_mask = tf.reduce_sum(tf.abs(atom_features), axis=1) > 0
                    num_atoms = tf.reduce_sum(tf.cast(atom_mask, tf.int32))
                
                # Slice to actual atoms
                atom_features_actual = atom_features[:num_atoms]
                
                # Graph convolution
                x_list = self.ruleGraphConvLayer([atom_features_actual])
                x = self.conv(x_list)
                
                conv_outputs.append(x)
            
            # Stack outputs
            x = tf.concat(conv_outputs, axis=0)
            
            # Dense layers
            x = self.dense1(x)
            x = self.dropout1(x, training=training)
            x = self.dense5(x)
            x = self.dropout2(x, training=training)
            model_var = self.dense6(x)
            
            # Merge and final output
            merged = tf.concat([model_var, physics_info], axis=1)
            out = self.dense7(merged)
            out = -tf.exp(out)
            
            return tf.concat([out, physics_info], axis=1)


# FIX 3: Vectorized batch processing (more GPU-efficient)
class PGGCN_Hybrid_Vectorized(tf.keras.Model):
    """Fully vectorized version that processes entire batch at once."""
    
    def __init__(self, num_atom_features=36, r_out_channel=20, c_out_channel=128,
                 l2=1e-4, dropout_rate=0.2, maxnorm=3.0):
        super().__init__()
        
        with tf.device('/GPU:0'):
            from models import layers_update_mobley as layers
            
            self.ruleGraphConvLayer = layers.RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
            self.conv = layers.ConvLayer(c_out_channel, r_out_channel)
            
            self.dense1 = tf.keras.layers.Dense(32, activation='relu')
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
            self.dense5 = tf.keras.layers.Dense(16, activation='relu')
            self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
            self.dense6 = tf.keras.layers.Dense(1)
            self.dense7 = tf.keras.layers.Dense(1)

    @tf.function(jit_compile=True)  # Enable XLA for full vectorization
    def call(self, inputs, training=True):
        """Vectorized processing of entire batch."""
        with tf.device('/GPU:0'):
            # Extract components
            atom_features = inputs[:, :, :38]  # [batch, atoms, 38]
            physics_info = inputs[:, 0, 38:]    # [batch, 15]
            
            # Create mask for valid atoms
            atom_mask = tf.reduce_sum(tf.abs(atom_features), axis=-1) > 0  # [batch, atoms]
            
            # Reshape for batch processing
            batch_size = tf.shape(inputs)[0]
            max_atoms = tf.shape(inputs)[1]
            
            # Flatten batch dimension for graph conv
            flat_features = tf.reshape(atom_features, [-1, 38])
            flat_mask = tf.reshape(atom_mask, [-1])
            
            # Process through graph conv (you may need to modify layers to handle batching)
            # This is a simplified version - actual implementation depends on your layers
            x = self.dense1(flat_features)
            x = self.dropout1(x, training=training)
            
            # Reshape back
            x = tf.reshape(x, [batch_size, max_atoms, -1])
            
            # Masked pooling
            x = tf.where(
                tf.expand_dims(atom_mask, -1),
                x,
                tf.zeros_like(x)
            )
            x = tf.reduce_sum(x, axis=1)  # [batch, features]
            
            # Rest of network
            x = self.dense5(x)
            x = self.dropout2(x, training=training)
            model_var = self.dense6(x)
            
            merged = tf.concat([model_var, physics_info], axis=1)
            out = self.dense7(merged)
            out = -tf.exp(out)
            
            return tf.concat([out, physics_info], axis=1)


# FIX 4: Updated training loop with GPU monitoring
def train_with_gpu_monitoring(model, X_train, y_train, config, physics_weight, opt):
    """Training loop with GPU utilization monitoring."""
    
    print(f"\nStarting training with GPU monitoring...")
    print(f"Input shape: {X_train.shape}")
    print(f"Model device: {model.device if hasattr(model, 'device') else 'Not set'}")
    
    # Convert to TF dataset for better GPU utilization
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(config.batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Enable profiling for first few steps
    tf.profiler.experimental.start('/tmp/tensorboard_logs')
    
    for epoch in range(config.epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_dataset):
            # Explicitly move to GPU
            with tf.device('/GPU:0'):
                with tf.GradientTape(persistent=True) as tape:
                    # Forward pass
                    predictions = model(X_batch, training=True)
                    
                    # Compute losses
                    emp_loss = pure_rmse(y_batch, predictions[:, 0])
                    phy_loss = physical_consistency_loss(
                        y_batch, predictions[:, 0], predictions[:, 1:16]
                    )
                    total_loss = emp_loss + physics_weight * phy_loss
                
                # Compute gradients
                grads_and_vars = opt.compute_gradients(
                    [emp_loss, phy_loss],
                    tape,
                    model.trainable_variables,
                    weights=[1.0, float(physics_weight)]
                )
                
                # Apply gradients
                opt.apply_gradients(grads_and_vars)
                
                del tape
                
                epoch_loss += float(total_loss.numpy())
                num_batches += 1
            
            # Stop profiling after first epoch
            if epoch == 0 and batch_idx == 10:
                tf.profiler.experimental.stop()
                print("Profiling stopped. Check /tmp/tensorboard_logs")
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}")
            
            # Check GPU memory
            for gpu in tf.config.list_physical_devices('GPU'):
                try:
                    details = tf.config.experimental.get_memory_info(
                        gpu.name.replace('physical_', '')
                    )
                    print(f"  GPU memory: {details['current'] / 1e9:.2f} GB")
                except:
                    pass


# ============================================================================
# COMPLETE REPLACEMENT MAIN FUNCTION
# ============================================================================

def main_with_gpu_fixes():
    """Main function with all GPU fixes applied."""
    
    # Step 1: Run diagnostics
    print("Running GPU diagnostics...")
    gpu_available = diagnose_gpu_setup()
    
    if not gpu_available:
        print("\n❌ No GPU detected! Check your CUDA installation.")
        return
    
    # Step 2: Configure GPU properly
    configure_gpu_properly()
    
    # Step 3: Load your data (use your existing code)
    print("\nLoading data...")
    # [Your existing data loading code here]
    
    # Step 4: Create model with explicit GPU placement
    print("\nCreating GPU-optimized model...")
    with tf.device('/GPU:0'):
        model = PGGCN_Hybrid_GPU()
        model.addRule("sum", 0, 32)
        model.addRule("multiply", 32, 33)
        model.addRule("distance", 33, 36)
    
    print(f"Model created on: {model.device if hasattr(model, 'device') else 'default device'}")
    
    # Step 5: Verify GPU is being used
    print("\nVerifying GPU usage with test forward pass...")
    test_input = tf.random.normal([2, 100, 53])  # Small test batch
    
    with tf.device('/GPU:0'):
        test_output = model(test_input, training=False)
        print(f"Test output device: {test_output.device}")
        print(f"✓ Model successfully runs on GPU")
    
    # Step 6: Continue with training
    print("\nStarting training...")
    # [Your existing training code here, but use train_with_gpu_monitoring]


if __name__ == "__main__":
    print("GPU Utilization Diagnostic and Fix Script")
    print("="*70)
    
    # Run diagnostic first
    diagnose_gpu_setup()
    
    print("\n" + "="*70)
    print("Apply the fixes above to your training script")
    print("="*70)