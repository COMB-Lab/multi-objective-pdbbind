"""
PyTorch training script for PGGCN model on Host-Guest dataset.

This script:
1. Reads the host-guest dataset from Final_data_DDG.csv
2. Loads PDB structures from pickle file
3. Trains the PGGCN model with 80/20 train/test split
4. Evaluates the model on test set
5. Monitors system resources (RAM, GPU memory, CPU usage)
6. Saves comprehensive statistics with the model
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

# Add the necessary directories to the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# Try to find PGGCN module - check multiple possible locations
possible_paths = [
    SCRIPT_DIR,  # Same directory as script
    PARENT_DIR,  # Parent directory
    os.path.join(PARENT_DIR, 'PGGCN'),  # PGGCN subdirectory in parent
    '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind',  # Absolute path
]

pggcn_found = False
for path in possible_paths:
    if path not in sys.path:
        sys.path.insert(0, path)
    
    # Check if we can find the models directory
    models_path = os.path.join(path, 'PGGCN', 'models')
    if os.path.exists(models_path):
        pggcn_found = True
        print(f"Found PGGCN at: {path}")
        break
    
    # Also check for models directory directly in path
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
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def print_saved_model_info(model_path):
    """
    Load and print comprehensive information from a saved model checkpoint.
    
    Args:
        model_path: Path to the saved .pth file
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("=" * 80)
    print("SAVED MODEL INFORMATION")
    print("=" * 80)
    
    # Hyperparameters
    if 'hyperparameters' in checkpoint:
        print("\nHyperparameters:")
        for key, value in checkpoint['hyperparameters'].items():
            print(f"  {key}: {value}")
    
    # Dataset info
    if 'dataset_info' in checkpoint:
        print("\nDataset Information:")
        for key, value in checkpoint['dataset_info'].items():
            print(f"  {key}: {value}")
    
    # Metrics
    if 'metrics' in checkpoint:
        print("\nTest Metrics:")
        for key, value in checkpoint['metrics'].items():
            if key == 'sign_accuracy':
                print(f"  {key}: {value*100:.1f}%")
            else:
                print(f"  {key}: {value:.4f}")
    
    # Timing
    if 'timing' in checkpoint:
        print("\nTiming:")
        timing = checkpoint['timing']
        print(f"  Data Loading: {timing.get('data_load_time_formatted', 'N/A')}")
        print(f"  Training: {timing.get('training_time_formatted', 'N/A')}")
        print(f"  Total: {timing.get('total_time_formatted', 'N/A')}")
    
    # Resource usage
    if 'resource_usage' in checkpoint:
        print("\nResource Usage:")
        res = checkpoint['resource_usage']
        
        if 'cpu' in res:
            print(f"  CPU ({res['cpu']['cores']} cores):")
            print(f"    Mean: {res['cpu']['mean_percent']:.1f}%")
            print(f"    Max:  {res['cpu']['max_percent']:.1f}%")
        
        if 'ram' in res:
            print(f"  RAM:")
            print(f"    Mean: {res['ram']['mean_formatted']} ({res['ram']['mean_percent']:.1f}%)")
            print(f"    Max:  {res['ram']['max_formatted']} ({res['ram']['max_percent']:.1f}%)")
            print(f"    Total: {res['ram']['total_formatted']}")
        
        if 'gpu' in res:
            print(f"  GPU ({res['gpu']['gpu_name']}):")
            print(f"    Mean: {res['gpu']['allocated_mean_formatted']} ({res['gpu']['allocated_mean_percent']:.1f}%)")
            print(f"    Max:  {res['gpu']['allocated_max_formatted']} ({res['gpu']['allocated_max_percent']:.1f}%)")
            print(f"    Total: {res['gpu']['total_formatted']}")
    
    # System info
    if 'system_info' in checkpoint:
        print("\nSystem Information:")
        sys_info = checkpoint['system_info']
        print(f"  Device: {sys_info.get('device', 'N/A')}")
        print(f"  CPU Cores: {sys_info.get('cpu_count', 'N/A')}")
        print(f"  Total RAM: {sys_info.get('total_ram_formatted', 'N/A')}")
        if sys_info.get('has_gpu', False):
            print(f"  GPU: {sys_info.get('gpu_name', 'N/A')}")
            print(f"  GPU Memory: {sys_info.get('total_gpu_memory_formatted', 'N/A')}")
    
    print("=" * 80)


class ResourceMonitor:
    """
    Monitor system resources (RAM, GPU memory, CPU) during training.
    """
    def __init__(self, device='cpu', monitoring_interval=1.0):
        self.device = device
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Storage for metrics
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_memory_allocated = []
        self.gpu_memory_reserved = []
        
        # Get system info
        self.total_ram = psutil.virtual_memory().total
        self.cpu_count = psutil.cpu_count()
        
        # Check GPU availability
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            self.gpu_name = "N/A"
            self.total_gpu_memory = 0
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            # CPU and RAM
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            
            self.cpu_usage.append(cpu_percent)
            self.ram_usage.append(ram.used)
            
            # GPU memory (if available)
            if self.has_gpu:
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                self.gpu_memory_allocated.append(allocated)
                self.gpu_memory_reserved.append(reserved)
            
            time.sleep(self.monitoring_interval)
    
    def start_monitoring(self):
        """Start background monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("✓ Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("✓ Resource monitoring stopped")
    
    def get_current_stats(self):
        """Get current resource usage statistics."""
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
        """Get summary statistics from monitoring period."""
        if not self.cpu_usage:
            return None
        
        summary = {
            'cpu': {
                'mean': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage),
                'min': np.min(self.cpu_usage),
            },
            'ram': {
                'mean': np.mean(self.ram_usage),
                'max': np.max(self.ram_usage),
                'min': np.min(self.ram_usage),
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
        """Print system information."""
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
        """Print current resource usage."""
        stats = self.get_current_stats()
        print(f"\nCurrent Resource Usage:")
        print(f"  CPU: {stats['cpu_percent']:.1f}%")
        print(f"  RAM: {format_bytes(stats['ram_used'])} ({stats['ram_percent']:.1f}%)")
        if self.has_gpu:
            print(f"  GPU Memory Allocated: {format_bytes(stats['gpu_allocated'])} "
                  f"({stats['gpu_percent_allocated']:.1f}%)")
            print(f"  GPU Memory Reserved: {format_bytes(stats['gpu_reserved'])} "
                  f"({stats['gpu_percent_reserved']:.1f}%)")
    
    def print_summary_stats(self, phase_name="Training"):
        """Print summary statistics."""
        summary = self.get_summary_stats()
        if not summary:
            print("No monitoring data available")
            return
        
        print(f"\n{'=' * 80}")
        print(f"{phase_name} - Resource Usage Summary")
        print("=" * 80)
        
        print(f"\nCPU Usage:")
        print(f"  Mean: {summary['cpu']['mean']:.1f}%")
        print(f"  Max:  {summary['cpu']['max']:.1f}%")
        print(f"  Min:  {summary['cpu']['min']:.1f}%")
        
        print(f"\nRAM Usage:")
        print(f"  Mean: {format_bytes(summary['ram']['mean'])} ({summary['ram']['mean_percent']:.1f}%)")
        print(f"  Max:  {format_bytes(summary['ram']['max'])} ({summary['ram']['max_percent']:.1f}%)")
        print(f"  Min:  {format_bytes(summary['ram']['min'])}")
        
        if 'gpu' in summary:
            print(f"\nGPU Memory (Allocated):")
            print(f"  Mean: {format_bytes(summary['gpu']['allocated_mean'])} "
                  f"({summary['gpu']['allocated_mean_percent']:.1f}%)")
            print(f"  Max:  {format_bytes(summary['gpu']['allocated_max'])} "
                  f"({summary['gpu']['allocated_max_percent']:.1f}%)")
        
        print("=" * 80)
    
    def reset_stats(self):
        """Reset monitoring statistics."""
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_memory_allocated = []
        self.gpu_memory_reserved = []


class PhysicsInformedLoss(nn.Module):
    """
    Custom loss function that combines MSE with a sign penalty.

    The loss penalizes when model_var has a different sign than (host_energy - guest_energy).
    This enforces physics-based constraints on the model's predictions.

    Loss = MSE(prediction, target) + sign_penalty_weight * sign_loss
    """

    def __init__(self, sign_penalty_weight=1.0):
        super(PhysicsInformedLoss, self).__init__()
        self.sign_penalty_weight = sign_penalty_weight
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets, model_vars, physics_info):
        """
        Args:
            predictions: Final model predictions [batch_size, 1]
            targets: Ground truth values [batch_size, 1]
            model_vars: Model predictions before physics fusion [batch_size, 1]
            physics_info: Physics-based info [batch_size, 2] where
                         [:, 0] = guest_energy
                         [:, 1] = host_energy

        Returns:
            Combined loss value
        """
        # Standard MSE loss
        mse_loss = self.mse(predictions, targets)

        # Calculate physics-based sign
        # host_energy - guest_energy
        physics_diff = physics_info[:, 1] - physics_info[:, 0]  # [batch_size]
        physics_sign = torch.sign(physics_diff)  # [batch_size]

        # Model prediction sign
        model_sign = torch.sign(model_vars.squeeze(-1))  # [batch_size]

        # Sign mismatch penalty
        # If signs match: sign_product = 1 (positive)
        # If signs mismatch: sign_product = -1 (negative)
        sign_product = physics_sign * model_sign  # [batch_size]

        # Penalize when sign_product < 0 (signs don't match)
        # Use ReLU on negative sign_product to create penalty
        sign_penalty = torch.mean(torch.relu(-sign_product))

        # Combined loss
        total_loss = mse_loss + self.sign_penalty_weight * sign_penalty

        return total_loss, mse_loss, sign_penalty


def featurize(molecule, info):
    """
    Featurize a molecule with additional info array.

    Args:
        molecule: RDKit molecule object
        info: List of additional information (e.g., [guest_energy, host_energy])

    Returns:
        numpy array of atom features
    """
    atom_features = []
    for atom in molecule.GetAtoms():
        # Base features from DeepChem (32 features)
        base_feat = get_atom_features(atom)
        new_feature = base_feat.tolist()

        # Add position information
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
        new_feature += [position.x, position.y, position.z]

        # At this point we have 32 + 3 + 3 = 38 features

        # Add neighbor indices (up to 2 neighbors)
        neighbors = atom.GetNeighbors()[:2]
        for neighbor in neighbors:
            neighbor_idx = neighbor.GetIdx()
            new_feature += [float(neighbor_idx)]

        # Pad if less than 2 neighbors
        for i in range(2 - len(neighbors)):
            new_feature += [0.0]

        # Now we have 38 + 2 = 40 features

        # Concatenate info array
        full_feature = new_feature + info
        atom_features.append(full_feature)

    return np.array(atom_features)


def load_all_data(info_csv_path, hostguest_dir, monitor=None):
    """
    Load ALL host-guest dataset from CSV and PDB files.
    """
    if monitor:
        print("\n" + "-" * 80)
        print("Starting data loading - monitoring resources...")
        monitor.print_current_stats()
        print("-" * 80)
    
    # Load PDB files from pickle
    print(f"Loading PDB files from: {hostguest_dir}")
    with open(hostguest_dir, 'rb') as f:
        pdb_dict = pd.read_pickle(f)
    print(f"Loaded {len(pdb_dict)} PDB files from pickle")

    # Read CSV
    df = pd.read_csv(info_csv_path)
    
    # Extract dataset and guest info from Ids column
    df['dataset'] = df['Ids'].str.split('--').str[0]
    df['guest_name'] = df['Ids'].str.split('--').str[1]
    
    # DON'T filter - use all rows
    df_all = df.copy()
    print(f"Found {len(df_all)} total entries")
    print(f"Datasets: {df_all['dataset'].unique()}")
    
    # Calculate energy totals from GB components
    df_all['guest_energy'] = (
        df_all['gb_guest_EELEC'] + 
        df_all['gb_guest_EGB'] + 
        df_all['gb_guest_ESURF']
    )
    df_all['host_energy'] = (
        df_all['gb_host_EELEC'] + 
        df_all['gb_host_EGB'] + 
        df_all['gb_host_ESURF']
    )

    # Define all features to include
    feature_columns = [
        'pb_host_VDWAALS', 'pb_guest_VDWAALS', 'pb_complex_VDWAALS',
        'gb_host_1-4EEL', 'gb_guest_1-4EEL', 'gb_Complex_1-4EEL',
        'gb_host_EELEC', 'gb_guest_EELEC', 'gb_Complex_EELEC',
        'gb_host_EGB', 'gb_guest_EGB', 'gb_Complex_EGB',
        'gb_host_ESURF', 'gb_guest_ESURF', 'gb_Complex_ESURF'
    ]
    
    print(f"\nUsing {len(feature_columns)} additional features:")
    for col in feature_columns:
        print(f"  - {col}")

    X = []
    y = []
    failed = []
    
    # Process each entry
    for idx, row in df_all.iterrows():
        pdb_id = row['Ids']
        target = row['Ex _G_(kcal/mol)']
        
        # Check if PDB exists in dictionary
        if pdb_id not in pdb_dict:
            if len(failed) < 10:
                print(f"Warning: PDB {pdb_id} not found in pickle file")
            failed.append(pdb_id)
            continue
        
        # Get molecule from pickle
        molecule = pdb_dict[pdb_id]
        
        # Create info array with all features
        info_array = row[feature_columns].tolist()
        
        # Featurize
        try:
            features = featurize(molecule, info_array)
            features_tensor = torch.FloatTensor(features)
            X.append(features_tensor)
            y.append(target)
        except Exception as e:
            if len(failed) < 10:
                print(f"Warning: Failed to featurize {pdb_id}: {e}")
            failed.append(pdb_id)
    
    print(f"\nSuccessfully loaded {len(X)} complexes")
    if failed:
        print(f"Failed to load {len(failed)} complexes")
    
    if monitor:
        print("\n" + "-" * 80)
        print("Data loading completed - resource usage:")
        monitor.print_current_stats()
        print("-" * 80)
    
    return X, y, df_all


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, device='cpu',
                sign_penalty_weight=1.0, monitor=None):
    """
    Train the PGGCN model with physics-informed loss.

    Args:
        model: PGGCNModel instance
        X_train: Training features (list of tensors)
        y_train: Training targets (list)
        X_val: Validation features (list of tensors)
        y_val: Validation targets (list)
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        sign_penalty_weight: Weight for sign penalty in loss function
        monitor: ResourceMonitor instance

    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = PhysicsInformedLoss(sign_penalty_weight=sign_penalty_weight)

    train_losses = []
    val_losses = []
    train_mse_losses = []
    train_sign_penalties = []

    # Move data to device
    X_train = [x.to(device) for x in X_train]
    X_val = [x.to(device) for x in X_val]
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    print(f"\nTraining on {len(X_train)} samples, validating on {len(X_val)} samples")
    print(f"Device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Sign penalty weight: {sign_penalty_weight}")
    print(f"Epochs: {epochs}")
    print("=" * 80)
    
    if monitor:
        monitor.print_current_stats()

    best_val_loss = float('inf')
    epoch_start_time = time.time()

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        # Forward pass - now returns (predictions, model_var, physics_info)
        predictions, model_var, physics_info = model(X_train)

        # Compute loss with physics-informed penalty
        train_loss, train_mse, train_sign_penalty = criterion(
            predictions, y_train_tensor, model_var, physics_info
        )

        # Backward pass
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())
        train_mse_losses.append(train_mse.item())
        train_sign_penalties.append(train_sign_penalty.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions, val_model_var, val_physics_info = model(X_val)
            val_loss, val_mse, val_sign_penalty = criterion(
                val_predictions, y_val_tensor, val_model_var, val_physics_info
            )
            val_losses.append(val_loss.item())

            # Calculate RMSE (from MSE component only)
            train_rmse = torch.sqrt(train_mse).item()
            val_rmse = torch.sqrt(val_mse).item()

        # Print progress with timing and resource usage
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - epoch_start_time
            epochs_done = epoch + 1
            avg_time_per_epoch = elapsed / epochs_done
            eta = avg_time_per_epoch * (epochs - epochs_done)
            
            output = (f"Epoch {epoch+1:3d}/{epochs} | "
                     f"Train Loss: {train_loss.item():.4f} (MSE: {train_mse.item():.4f}, "
                     f"Sign: {train_sign_penalty.item():.4f}) | "
                     f"Val Loss: {val_loss.item():.4f} (RMSE: {val_rmse:.4f}) | "
                     f"Time: {format_time(elapsed)} | ETA: {format_time(eta)}")
            
            if monitor:
                stats = monitor.get_current_stats()
                output += f" | CPU: {stats['cpu_percent']:.0f}% | RAM: {stats['ram_percent']:.0f}%"
                if monitor.has_gpu:
                    output += f" | GPU: {stats['gpu_percent_allocated']:.0f}%"
            
            print(output)

        # Save best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1

    total_training_time = time.time() - epoch_start_time
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Total training time: {format_time(total_training_time)}")

    return train_losses, val_losses


def evaluate_model(model, X_test, y_test, device='cpu'):
    """
    Evaluate the model on test set.

    Args:
        model: Trained PGGCNModel
        X_test: Test features (list of tensors)
        y_test: Test targets (list)
        device: Device to evaluate on

    Returns:
        predictions: Model predictions
        metrics: Dictionary of evaluation metrics
    """
    model.eval()

    # Move data to device
    X_test = [x.to(device) for x in X_test]
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    with torch.no_grad():
        predictions, model_var, physics_info = model(X_test)

        # Calculate metrics
        mse = nn.MSELoss()(predictions, y_test_tensor).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(predictions - y_test_tensor)).item()

        # R^2 score
        ss_res = torch.sum((y_test_tensor - predictions) ** 2).item()
        ss_tot = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot)

        # Calculate sign accuracy
        physics_diff = physics_info[:, 1] - physics_info[:, 0]
        physics_sign = torch.sign(physics_diff)
        model_sign = torch.sign(model_var.squeeze(-1))
        sign_matches = (physics_sign == model_sign).float()
        sign_accuracy = torch.mean(sign_matches).item()

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'sign_accuracy': sign_accuracy
    }

    return predictions.cpu().numpy(), metrics


def main():
    print("=" * 80)
    print("PyTorch PGGCN Training on Host-Guest Dataset")
    print("=" * 80)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Initialize resource monitor
    monitor = ResourceMonitor(device=device, monitoring_interval=1.0)
    monitor.print_system_info()

    # Start total timing
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

    # Split data (80/20)
    print("\n" + "-" * 80)
    print("Splitting Data (80% train, 20% test)")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Create model
    print("\n" + "-" * 80)
    print("Creating Model")
    print("-" * 80)
    model = PGGCNModel(num_atom_features=36, r_out_channel=20, c_out_channel=1024)

    # Add rules
    model.add_rule("sum", 0, 30)
    model.add_rule("multiply", 30, 31)
    model.add_rule("distance", 33, 36)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Rule-based graph convolution rules: {len(model.rule_graph_conv.combination_rules)}")
    
    # Show model memory footprint
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
    model = model.to(device)
        
    if device == 'cuda':
        model_memory = torch.cuda.memory_allocated(0)
        print(f"\nModel memory footprint: {format_bytes(model_memory)}")

    # Train model
    print("\n" + "-" * 80)
    print("Training Model")
    print("-" * 80)

    # Hyperparameters
    learning_rate = 0.001
    sign_penalty_weight = 1.0
    num_epochs = 250

    train_losses, val_losses = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=num_epochs,
        lr=learning_rate,
        device=device,
        sign_penalty_weight=sign_penalty_weight,
        monitor=monitor
    )
    
    # Print resource summary for training
    monitor.print_summary_stats("Training")

    # Evaluate on test set
    print("\n" + "-" * 80)
    print("Evaluating on Test Set")
    print("-" * 80)
    predictions, metrics = evaluate_model(model, X_test, y_test, device=device)

    print(f"\nTest Set Metrics:")
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  Sign Accuracy: {metrics['sign_accuracy']*100:.1f}%")

    # Show some predictions
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

    # Stop monitoring
    monitor.stop_monitoring()

    # Save model with comprehensive statistics
    print("\n" + "-" * 80)
    print("Saving Model")
    print("-" * 80)
    model_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/saved_models/hostguest_model.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Get resource summary
    total_script_time = time.time() - script_start_time
    resource_summary = monitor.get_summary_stats()
    
    # Prepare save dictionary with all information
    save_dict = {
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics': metrics,
        'hyperparameters': {
            'learning_rate': learning_rate,
            'sign_penalty_weight': sign_penalty_weight,
            'num_epochs': num_epochs,
            'num_atom_features': 36,
            'r_out_channel': 20,
            'c_out_channel': 1024,
        },
        'dataset_info': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_test_split': 0.2,
            'random_seed': RANDOM_SEED,
            'dataset_name': 'Host-Guest Complexes',
        },
        'timing': {
            'data_load_time_seconds': data_load_time,
            'total_training_time_seconds': total_script_time - data_load_time,
            'total_script_time_seconds': total_script_time,
            'data_load_time_formatted': format_time(data_load_time),
            'training_time_formatted': format_time(total_script_time - data_load_time),
            'total_time_formatted': format_time(total_script_time),
        },
    }
    
    # Add resource usage statistics if available
    if resource_summary:
        save_dict['resource_usage'] = {
            'cpu': {
                'mean_percent': resource_summary['cpu']['mean'],
                'max_percent': resource_summary['cpu']['max'],
                'min_percent': resource_summary['cpu']['min'],
                'cores': monitor.cpu_count,
            },
            'ram': {
                'mean_bytes': resource_summary['ram']['mean'],
                'max_bytes': resource_summary['ram']['max'],
                'min_bytes': resource_summary['ram']['min'],
                'mean_percent': resource_summary['ram']['mean_percent'],
                'max_percent': resource_summary['ram']['max_percent'],
                'total_bytes': monitor.total_ram,
                'mean_formatted': format_bytes(resource_summary['ram']['mean']),
                'max_formatted': format_bytes(resource_summary['ram']['max']),
                'total_formatted': format_bytes(monitor.total_ram),
            },
        }
        
        if 'gpu' in resource_summary:
            save_dict['resource_usage']['gpu'] = {
                'allocated_mean_bytes': resource_summary['gpu']['allocated_mean'],
                'allocated_max_bytes': resource_summary['gpu']['allocated_max'],
                'reserved_mean_bytes': resource_summary['gpu']['reserved_mean'],
                'reserved_max_bytes': resource_summary['gpu']['reserved_max'],
                'allocated_mean_percent': resource_summary['gpu']['allocated_mean_percent'],
                'allocated_max_percent': resource_summary['gpu']['allocated_max_percent'],
                'total_bytes': monitor.total_gpu_memory,
                'gpu_name': monitor.gpu_name,
                'allocated_mean_formatted': format_bytes(resource_summary['gpu']['allocated_mean']),
                'allocated_max_formatted': format_bytes(resource_summary['gpu']['allocated_max']),
                'total_formatted': format_bytes(monitor.total_gpu_memory),
            }
    
    # Add system information
    save_dict['system_info'] = {
        'device': device,
        'cpu_count': monitor.cpu_count,
        'total_ram_bytes': monitor.total_ram,
        'total_ram_formatted': format_bytes(monitor.total_ram),
        'has_gpu': monitor.has_gpu,
    }
    
    if monitor.has_gpu:
        save_dict['system_info']['gpu_name'] = monitor.gpu_name
        save_dict['system_info']['total_gpu_memory_bytes'] = monitor.total_gpu_memory
        save_dict['system_info']['total_gpu_memory_formatted'] = format_bytes(monitor.total_gpu_memory)
    
    # Save everything
    torch.save(save_dict, model_path)
    print(f"Model saved to: {model_path}")
    print(f"Saved information includes:")
    print(f"  - Model state and architecture")
    print(f"  - Training/validation losses")
    print(f"  - Test metrics")
    print(f"  - Hyperparameters")
    print(f"  - Dataset information")
    print(f"  - Timing statistics")
    print(f"  - Resource usage (CPU, RAM{', GPU' if monitor.has_gpu else ''})")
    print(f"  - System information")

    # Print final summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nHyperparameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Sign Penalty Weight: {sign_penalty_weight}")
    print(f"  Epochs: {num_epochs}")
    
    print(f"\nFinal Metrics:")
    print(f"  Test RMSE: {metrics['rmse']:.4f}")
    print(f"  Test R²: {metrics['r2']:.4f}")
    print(f"  Sign Accuracy: {metrics['sign_accuracy']*100:.1f}%")
    
    print(f"\nResource Usage (Peak):")
    if resource_summary:
        print(f"  RAM: {format_bytes(resource_summary['ram']['max'])} "
              f"({resource_summary['ram']['max_percent']:.1f}% of {format_bytes(monitor.total_ram)})")
        if 'gpu' in resource_summary:
            print(f"  GPU Memory: {format_bytes(resource_summary['gpu']['allocated_max'])} "
                  f"({resource_summary['gpu']['allocated_max_percent']:.1f}% of {format_bytes(monitor.total_gpu_memory)})")
    
    print(f"\nTiming:")
    print(f"  Data Loading: {format_time(data_load_time)}")
    print(f"  Training: {format_time(total_script_time - data_load_time)}")
    print(f"  Total: {format_time(total_script_time)}")
    
    # Estimate scalability
    if resource_summary:
        print(f"\n{'=' * 80}")
        print("SCALABILITY ESTIMATE")
        print("=" * 80)
        print(f"Current dataset: {len(X)} samples")
        
        avg_samples = (len(X_train) + len(X_test)) / 2
        ram_max = resource_summary['ram']['max']
        samples_per_gb_ram = avg_samples / (ram_max / (1024**3))
        
        print(f"Approximate samples per GB RAM: {samples_per_gb_ram:.0f}")
        
        available_ram = monitor.total_ram - ram_max
        estimated_additional_samples = (available_ram / (1024**3)) * samples_per_gb_ram
        print(f"Available RAM: {format_bytes(available_ram)}")
        print(f"Estimated max additional samples: {estimated_additional_samples:.0f}")
        print(f"Estimated total capacity: {len(X) + estimated_additional_samples:.0f} samples")
        
        if monitor.has_gpu and 'gpu' in resource_summary:
            gpu_max = resource_summary['gpu']['allocated_max']
            available_gpu = monitor.total_gpu_memory - gpu_max
            print(f"\nAvailable GPU Memory: {format_bytes(available_gpu)}")
            samples_per_gb_gpu = avg_samples / (gpu_max / (1024**3))
            estimated_gpu_samples = (available_gpu / (1024**3)) * samples_per_gb_gpu
            print(f"Estimated max additional samples (GPU constrained): {estimated_gpu_samples:.0f}")
    
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nModel and statistics saved to: {model_path}")
    print("\nTo view saved statistics later, run:")
    print(f"  python view_model_stats.py '{model_path}'")


if __name__ == "__main__":
    main()