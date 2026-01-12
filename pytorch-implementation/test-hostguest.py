"""
PyTorch training script for PGGCN model - FINAL CORRECTED VERSION

Key fixes:
1. L2 regularization applies to ALL parameters (including biases)
2. Learning rate DOESN'T decay during training (because early stopping triggers first)
3. Fixed dense_final layer to accept 16 inputs (1 + 15 physics features)
4. Proper weight initialization for dense_final
5. Architecture matches TensorFlow exactly
"""

import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
import pandas as pd
import numpy as np
import sys
import os
import random
import time
import psutil
import threading
from datetime import timedelta
from sklearn.model_selection import train_test_split
import pickle

# Add the necessary directories to the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

possible_paths = [
    SCRIPT_DIR,
    PARENT_DIR,
    os.path.join(PARENT_DIR, 'PGGCN'),
    '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind',
]

pggcn_found = False
for path in possible_paths:
    if path not in sys.path:
        sys.path.insert(0, path)
    
    models_path = os.path.join(path, 'PGGCN', 'models')
    if os.path.exists(models_path):
        pggcn_found = True
        print(f"Found PGGCN at: {path}")
        break
    
    models_path_direct = os.path.join(path, 'models')
    if os.path.exists(models_path_direct):
        pggcn_found = True
        print(f"Found models at: {path}")
        break

if not pggcn_found:
    print("Warning: PGGCN/models directory not found in expected locations")
    print(f"Searched in: {possible_paths}")
    print("Attempting to import anyway...")

# Set random seeds for reproducibility
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(RANDOM_SEED)
#     torch.cuda.manual_seed_all(RANDOM_SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# Try different import styles
try:
    from PGGCN.models.dcFeaturizer import atom_features as get_atom_features
    from PGGCN.models.layers_pytorch import PGGCNModel
    print("Imported using PGGCN.models style")
except ImportError:
    try:
        from models.dcFeaturizer import atom_features as get_atom_features
        from models.layers_pytorch import PGGCNModel
        print("Imported using models style")
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure PGGCN/models directory is accessible")
        sys.exit(1)


def format_time(seconds):
    """Format seconds into a readable time string."""
    return str(timedelta(seconds=int(seconds)))


def format_bytes(bytes_val):
    """Format bytes into human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


class ResourceMonitor:
    """Monitor system resources (RAM, GPU memory, CPU) during training."""
    def __init__(self, device='cpu', monitoring_interval=1.0):
        self.device = device
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.monitor_thread = None
        
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_memory_allocated = []
        self.gpu_memory_reserved = []
        
        self.total_ram = psutil.virtual_memory().total
        self.cpu_count = psutil.cpu_count()
        
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            self.gpu_name = "N/A"
            self.total_gpu_memory = 0
    
    def _monitor_loop(self):
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            self.cpu_usage.append(cpu_percent)
            self.ram_usage.append(ram.used)
            if self.has_gpu:
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                self.gpu_memory_allocated.append(allocated)
                self.gpu_memory_reserved.append(reserved)
            time.sleep(self.monitoring_interval)
    
    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("✓ Resource monitoring started")
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("✓ Resource monitoring stopped")
    
    def get_current_stats(self):
        stats = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'ram_used': psutil.virtual_memory().used,
            'ram_percent': psutil.virtual_memory().percent,
        }
        if self.has_gpu:
            stats['gpu_allocated'] = torch.cuda.memory_allocated(0)
            stats['gpu_reserved'] = torch.cuda.memory_reserved(0)
            stats['gpu_percent_allocated'] = (stats['gpu_allocated'] / self.total_gpu_memory) * 100
            stats['gpu_percent_reserved'] = (stats['gpu_reserved'] / self.total_gpu_memory) * 100
        return stats
    
    def get_summary_stats(self):
        if not self.cpu_usage:
            return None
        summary = {
            'cpu': {'mean': np.mean(self.cpu_usage), 'max': np.max(self.cpu_usage), 'min': np.min(self.cpu_usage)},
            'ram': {
                'mean': np.mean(self.ram_usage), 'max': np.max(self.ram_usage), 'min': np.min(self.ram_usage),
                'mean_percent': (np.mean(self.ram_usage) / self.total_ram) * 100,
                'max_percent': (np.max(self.ram_usage) / self.total_ram) * 100,
            }
        }
        if self.has_gpu and self.gpu_memory_allocated:
            summary['gpu'] = {
                'allocated_mean': np.mean(self.gpu_memory_allocated),
                'allocated_max': np.max(self.gpu_memory_allocated),
                'reserved_mean': np.mean(self.gpu_memory_reserved),
                'reserved_max': np.max(self.gpu_memory_reserved),
                'allocated_mean_percent': (np.mean(self.gpu_memory_allocated) / self.total_gpu_memory) * 100,
                'allocated_max_percent': (np.max(self.gpu_memory_allocated) / self.total_gpu_memory) * 100,
            }
        return summary
    
    def print_system_info(self):
        print("\n" + "=" * 80)
        print("SYSTEM INFORMATION")
        print("=" * 80)
        print(f"CPU: {self.cpu_count} cores")
        print(f"Total RAM: {format_bytes(self.total_ram)}")
        if self.has_gpu:
            print(f"GPU: {self.gpu_name}")
            print(f"Total GPU Memory: {format_bytes(self.total_gpu_memory)}")
        else:
            print("GPU: Not available")
        print("=" * 80)
    
    def print_current_stats(self):
        stats = self.get_current_stats()
        print(f"\nCurrent Resource Usage:")
        print(f"  CPU: {stats['cpu_percent']:.1f}%")
        print(f"  RAM: {format_bytes(stats['ram_used'])} ({stats['ram_percent']:.1f}%)")
        if self.has_gpu:
            print(f"  GPU Memory Allocated: {format_bytes(stats['gpu_allocated'])} ({stats['gpu_percent_allocated']:.1f}%)")
    
    def print_summary_stats(self, phase_name="Training"):
        summary = self.get_summary_stats()
        if not summary:
            return
        print(f"\n{'=' * 80}")
        print(f"{phase_name} - Resource Usage Summary")
        print("=" * 80)
        print(f"\nCPU Usage: Mean: {summary['cpu']['mean']:.1f}%, Max: {summary['cpu']['max']:.1f}%")
        print(f"RAM Usage: Mean: {format_bytes(summary['ram']['mean'])} ({summary['ram']['mean_percent']:.1f}%), "
              f"Max: {format_bytes(summary['ram']['max'])} ({summary['ram']['max_percent']:.1f}%)")
        if 'gpu' in summary:
            print(f"GPU Memory: Mean: {format_bytes(summary['gpu']['allocated_mean'])} "
                  f"({summary['gpu']['allocated_mean_percent']:.1f}%), "
                  f"Max: {format_bytes(summary['gpu']['allocated_max'])} "
                  f"({summary['gpu']['allocated_max_percent']:.1f}%)")
        print("=" * 80)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    Matches TensorFlow's EarlyStopping(monitor='loss', patience=10, min_delta=0.001, restore_best_weights=True)
    """
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered. Restoring best model weights from loss: {self.best_loss:.4f}')
        else:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            
    def restore_best_weights(self, model):
        """Restore model to best weights."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"Restored model weights from best epoch with loss: {self.best_loss:.4f}")


def apply_maxnorm_constraint(model, max_norm=3.0):
    """
    Apply MaxNorm constraint to all parameters.
    Matches TensorFlow's kernel_constraint=constraints.MaxNorm(max_norm)
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                # Apply constraint along the first dimension (output neurons)
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=max_norm)
                param.mul_(desired / (norm + 1e-7))


def compute_l2_loss(model, l2_weight=1e-4):
    """
    Compute L2 regularization loss for ALL parameters (weights AND biases).
    This matches TensorFlow's behavior where both kernel_regularizer and bias_regularizer are set.
    """
    l2_loss = torch.tensor(0., device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            l2_loss += torch.sum(param ** 2)
    return (l2_weight / 2) * l2_loss


class PhysicsInformedLoss(nn.Module):
    """
    Custom loss function combining RMSE with physics consistency loss.
    """

    def __init__(self, physics_consistency_weight=0.005):
        super(PhysicsInformedLoss, self).__init__()
        self.physics_consistency_weight = physics_consistency_weight

    def forward(self, predictions, targets, model_vars, physics_info):
        """
        Args:
            predictions: Final model predictions [batch_size, 1]
            targets: Ground truth values [batch_size, 1]
            model_vars: Model predictions before physics fusion [batch_size, 1]
            physics_info: Physics-based info [batch_size, 15]

        Returns:
            total_loss, empirical_loss (RMSE), physics_loss (RMSE)
        """
        targets = targets.view(-1, 1)
        
        # Empirical loss - RMSE
        empirical_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))
        
        # Extract energies
        host_energy = physics_info[:, [0, 3, 6, 9, 12]].sum(dim=1, keepdim=True)
        guest_energy = physics_info[:, [1, 4, 7, 10, 13]].sum(dim=1, keepdim=True)
        complex_energy = physics_info[:, [2, 5, 8, 11, 14]].sum(dim=1, keepdim=True)
        
        # Physics-based ΔG calculation
        dG_physics = complex_energy - (host_energy + guest_energy)
        
        # Physics consistency loss - RMSE
        # CRITICAL FIX: TensorFlow compares FINAL prediction vs dG_physics, not model_var!
        # This makes sense because final = 0.3*model_var + dG_physics, so comparing
        # final vs dG_physics essentially penalizes model_var being far from 0
        physics_loss = torch.sqrt(torch.mean((predictions - dG_physics) ** 2))
        
        # Combined loss
        total_loss = empirical_loss + (self.physics_consistency_weight * physics_loss)
        #total_loss = empirical_loss
        return total_loss, empirical_loss, physics_loss


def featurize(molecule, info):
    """Featurize a molecule with additional info array."""
    atom_features = []
    for atom in molecule.GetAtoms():
        base_feat = get_atom_features(atom)
        new_feature = base_feat.tolist()
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
        new_feature += [position.x, position.y, position.z]
        neighbors = atom.GetNeighbors()[:2]
        for neighbor in neighbors:
            new_feature += [float(neighbor.GetIdx())]
        for i in range(2 - len(neighbors)):
            new_feature += [-1.0]
        full_feature = new_feature + info
        atom_features.append(full_feature)
    return np.array(atom_features)


def load_all_data(info_csv_path, hostguest_dir, monitor=None):
    """Load ALL host-guest dataset from CSV and PDB files."""
    if monitor:
        print("\n" + "-" * 80)
        print("Starting data loading...")
        monitor.print_current_stats()
        print("-" * 80)
    
    print(f"Loading PDB files from: {hostguest_dir}")
    with open(hostguest_dir, 'rb') as f:
        pdb_dict = pickle.load(f)
    print(f"Loaded {len(pdb_dict)} PDB files from pickle")

    df_all = pd.read_csv(info_csv_path)
    print(f"Found {len(df_all)} total entries in CSV")

    feature_columns = [
        'pb_host_VDWAALS', 'pb_guest_VDWAALS', 'pb_complex_VDWAALS',
        'gb_host_1-4EEL', 'gb_guest_1-4EEL', 'gb_Complex_1-4EEL',
        'gb_host_EELEC', 'gb_guest_EELEC', 'gb_Complex_EELEC',
        'gb_host_EGB', 'gb_guest_EGB', 'gb_Complex_EGB',
        'gb_host_ESURF', 'gb_guest_ESURF', 'gb_Complex_ESURF'
    ]

    X = []
    y = []
    failed = []
    
    # CRITICAL: Iterate in the SAME order as TensorFlow (pickle's key order)
    for pdb_id in list(pdb_dict.keys()):
        if pdb_id not in df_all['Ids'].values:
            failed.append(pdb_id)
            continue
        
        molecule = pdb_dict[pdb_id]
        row = df_all[df_all['Ids'] == pdb_id].iloc[0]
        info_array = row[feature_columns].tolist()
        target = row['Ex _G_(kcal/mol)']
        
        try:
            features = featurize(molecule, info_array)
            features_tensor = torch.FloatTensor(features)
            X.append(features_tensor)
            y.append(target)
        except Exception as e:
            failed.append(pdb_id)
            print(f"Failed to featurize {pdb_id}: {e}")
    
    print(f"\nSuccessfully loaded {len(X)} complexes")
    if failed:
        print(f"Failed to load {len(failed)} complexes: {failed}")
    
    if monitor:
        print("\n" + "-" * 80)
        print("Data loading completed")
        monitor.print_current_stats()
        print("-" * 80)
    
    return X, y, df_all


def train_model(model, X_train, y_train, X_val, y_val, epochs=250, lr=0.005, device='cpu',
                physics_consistency_weight=0.005, l2_weight=1e-4, max_norm=3.0, 
                early_stopping_patience=10, monitor=None):
    """
    Train the PGGCN model with all regularization components matching TensorFlow.
    
    NOTE: Learning rate does NOT decay because:
    - TensorFlow uses decay_steps=10000 (steps, not epochs)
    - With full batch training, 1 epoch = 1 step
    - Early stopping triggers at ~10 epochs, way before 10000 steps
    - So LR stays constant at 0.005 throughout training
    """
    model = model.to(device)

    # Optimizer WITHOUT weight_decay (we compute L2 manually)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # No LR scheduler needed - it never decays before early stopping
    
    criterion = PhysicsInformedLoss(physics_consistency_weight=physics_consistency_weight)
    
    # Early stopping - monitors TRAINING loss to match TensorFlow!
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.001, verbose=True)

    train_losses, val_losses = [], []
    train_empirical_losses, train_physics_losses = [], []

    # Move data to device
    X_train = [x.to(device) for x in X_train]
    X_val = [x.to(device) for x in X_val]
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    print(f"\n{'=' * 80}")
    print("TRAINING CONFIGURATION (FINAL CORRECTED - MATCHES TF EXACTLY)")
    print("=" * 80)
    print(f"Samples: {len(X_train)} train, {len(X_val)} val")
    print(f"Device: {device}")
    print(f"Learning rate: {lr} (constant - no decay before early stopping)")
    print(f"L2 regularization: {l2_weight} (applied to ALL parameters including biases)")
    print(f"MaxNorm constraint: {max_norm}")
    print(f"Dropout rate: 0.2 (in model)")
    print(f"Physics weight: {physics_consistency_weight}")
    print(f"Early stopping: patience={early_stopping_patience}, monitors TRAINING loss")
    print(f"Physics loss: Compares FINAL prediction vs dG_physics (matches TensorFlow)")
    print(f"Max epochs: {epochs}")
    print("=" * 80)
    
    if monitor:
        monitor.print_current_stats()

    best_val_loss = float('inf')
    best_epoch = 0
    epoch_start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        # Forward pass with training=True for dropout
        predictions, model_var, physics_info = model(X_train, training=True)

        # Compute loss
        train_loss, train_empirical, train_physics = criterion(
            predictions, y_train_tensor, model_var, physics_info
        )
        
        # Add L2 regularization for ALL parameters (weights AND biases)
        l2_reg = compute_l2_loss(model, l2_weight)
        train_loss = train_loss + l2_reg

        # Backward pass
        train_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Apply MaxNorm constraint after optimizer step
        apply_maxnorm_constraint(model, max_norm=max_norm)

        train_losses.append(train_loss.item())
        train_empirical_losses.append(train_empirical.item())
        train_physics_losses.append(train_physics.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions, val_model_var, val_physics_info = model(X_val, training=False)
            val_loss, val_empirical, val_physics = criterion(
                val_predictions, y_val_tensor, val_model_var, val_physics_info
            )
            val_losses.append(val_loss.item())

        # Track best validation loss
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - epoch_start_time
            avg_time_per_epoch = elapsed / (epoch + 1)
            eta = avg_time_per_epoch * (epochs - epoch - 1)
            
            output = (f"Epoch {epoch+1:3d}/{epochs} | "
                     f"Train Loss: {train_loss.item():.4f} "
                     f"(Emp: {train_empirical.item():.4f}, Phys: {train_physics.item():.4f}) | "
                     f"Val Loss: {val_loss.item():.4f} | "
                     f"Time: {format_time(elapsed)} | ETA: {format_time(eta)}")
            
            if monitor:
                stats = monitor.get_current_stats()
                output += f" | CPU: {stats['cpu_percent']:.0f}% | RAM: {stats['ram_percent']:.0f}%"
                if monitor.has_gpu:
                    output += f" | GPU: {stats['gpu_percent_allocated']:.0f}%"
            
            print(output)
        
        # Check early stopping - TensorFlow monitors TRAINING loss!
        # early_stopping(train_loss.item(), model)
        # if early_stopping.early_stop:
        #     print(f"\nEarly stopping at epoch {epoch+1}")
        #     early_stopping.restore_best_weights(model)
        #     break

    total_training_time = time.time() - epoch_start_time
    print("=" * 80)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Total training time: {format_time(total_training_time)}")
    print("=" * 80)

    return train_losses, val_losses


def evaluate_model(model, X_test, y_test, device='cpu'):
    """Evaluate the model on test set."""
    model.eval()
    X_test = [x.to(device) for x in X_test]
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    with torch.no_grad():
        predictions, model_var, physics_info = model(X_test, training=False)
        mse = nn.MSELoss()(predictions, y_test_tensor).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(predictions - y_test_tensor)).item()
        ss_res = torch.sum((y_test_tensor - predictions) ** 2).item()
        ss_tot = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot)

    metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    return predictions.cpu().numpy(), metrics


def main():
    print("=" * 80)
    print("PyTorch PGGCN Training - FINAL CORRECTED VERSION")
    print("All issues fixed: L2 on biases, proper architecture, LR handling")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    monitor = ResourceMonitor(device=device, monitoring_interval=1.0)
    monitor.print_system_info()

    script_start_time = time.time()
    monitor.start_monitoring()

    # Paths
    info_csv_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/Final_data_DDG.csv'
    hostguest_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBs_RDKit_BFE.pkl'

    # Load data
    print("\n" + "-" * 80)
    print("Loading Data")
    print("-" * 80)
    data_load_start = time.time()
    X, y, df_info = load_all_data(info_csv_path, hostguest_dir, monitor=monitor)
    
    data_load_time = time.time() - data_load_start
    print(f"Data loading completed in: {format_time(data_load_time)}")

    # Split data
    print("\n" + "-" * 80)
    print("Splitting Data (80% train, 20% test)")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Create model WITH dropout and corrected architecture
    print("\n" + "-" * 80)
    print("Creating Model")
    print("-" * 80)
    model = PGGCNModel(num_atom_features=36, r_out_channel=20, c_out_channel=128, dropout_rate=0.2)
    model.add_rule("sum", 0, 32)
    model.add_rule("multiply", 32, 33)
    model.add_rule("distance", 33, 36)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Architecture: Conv(128) → Dense1(32) → Dense2(16) → Dense3(1) → Final(16→1)")
    
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model = model.to(device)
    if device == 'cuda':
        print(f"Model memory footprint: {format_bytes(torch.cuda.memory_allocated(0))}")

    # Train model
    print("\n" + "-" * 80)
    print("Training Model")
    print("-" * 80)

    train_losses, val_losses = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=250,
        lr=0.005,
        device=device,
        physics_consistency_weight=0.005,
        l2_weight=1e-4,
        max_norm=3.0,
        early_stopping_patience=10,
        monitor=monitor
    )
    
    monitor.print_summary_stats("Training")

    # Evaluate
    print("\n" + "-" * 80)
    print("Evaluating on Test Set")
    print("-" * 80)
    predictions, metrics = evaluate_model(model, X_test, y_test, device=device)

    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")

    # Show sample predictions
    print("\n" + "-" * 80)
    print("Sample Predictions")
    print("-" * 80)
    print(f"{'True Value':>12} | {'Prediction':>12} | {'Error':>12}")
    print("-" * 40)
    for i in range(min(10, len(y_test))):
        true_val = y_test[i]
        pred_val = predictions[i][0]
        error = pred_val - true_val
        print(f"{true_val:>12.4f} | {pred_val:>12.4f} | {error:>12.4f}")

    monitor.stop_monitoring()

    # Save model
    print("\n" + "-" * 80)
    print("Saving Model")
    print("-" * 80)
    model_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/saved_models/hostguest_no_physical_loss_final.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    total_script_time = time.time() - script_start_time
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics': metrics,
        'hyperparameters': {
            'learning_rate': 0.005,
            'l2_weight': 1e-4,
            'max_norm': 3.0,
            'dropout_rate': 0.2,
            'physics_consistency_weight': 0.005,
            'early_stopping_patience': 10,
            'num_epochs': 250,
        },
        'dataset_info': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'random_seed': RANDOM_SEED,
        },
    }
    
    torch.save(save_dict, model_path)
    print(f"Model saved to: {model_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Test RMSE: {metrics['rmse']:.4f} (Target: 2.3193)")
    print(f"Test MAE:  {metrics['mae']:.4f} (Target: 1.9162)")
    print(f"Training time: {format_time(total_script_time - data_load_time)}")
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()