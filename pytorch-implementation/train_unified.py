"""
Unified PyTorch training script for PGGCN model.

Works with both:
1. Host-Guest dataset (Final_data_DDG.csv + PDBs_RDKit_BFE.pkl)
2. PDBBind dataset (pdbbind_100.csv + PDBBind_100.pkl)

Usage:
    python train_unified.py --dataset hostguest
    python train_unified.py --dataset pdbbind
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
import argparse
from datetime import timedelta
from sklearn.model_selection import train_test_split

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

# Set random seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        sys.exit(1)


# Dataset configurations
DATASET_CONFIGS = {
    'hostguest': {
        'csv_path': '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/Final_data_DDG.csv',
        'pkl_path': '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBs_RDKit_BFE.pkl',
        'id_column': 'Ids',
        'target_column': 'Ex _G_(kcal/mol)',
        'physics_columns': [
            'pb_host_VDWAALS', 'pb_guest_VDWAALS', 'pb_complex_VDWAALS',
            'gb_host_1-4EEL', 'gb_guest_1-4EEL', 'gb_Complex_1-4EEL',
            'gb_host_EELEC', 'gb_guest_EELEC', 'gb_Complex_EELEC',
            'gb_host_EGB', 'gb_guest_EGB', 'gb_Complex_EGB',
            'gb_host_ESURF', 'gb_guest_ESURF', 'gb_Complex_ESURF'
        ],
        'host_indices': [0, 3, 6, 9, 12],  # host/protein energy indices
        'guest_indices': [1, 4, 7, 10, 13],  # guest/ligand energy indices
        'complex_indices': [2, 5, 8, 11, 14],  # complex energy indices
        'name': 'Host-Guest Complexes'
    },
    'pdbbind': {
        'csv_path': '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind_100.csv',
        'pkl_path': '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_100.pkl',
        'id_column': 'complex-name',
        'target_column': 'ddg',
        'physics_columns': [
            'pb-protein-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals',
            'gb-protein-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
            'gb-protein-eelect', 'gb-ligand-eelec', 'gb-complex-eelec',
            'gb-protein-egb', 'gb-ligand-egb', 'gb-complex-egb',
            'gb-protein-esurf', 'gb-ligand-esurf', 'gb-complex-esurf'
        ],
        'host_indices': [0, 3, 6, 9, 12],  # protein energy indices
        'guest_indices': [1, 4, 7, 10, 13],  # ligand energy indices
        'complex_indices': [2, 5, 8, 11, 14],  # complex energy indices
        'name': 'PDBBind Protein-Ligand Complexes'
    }
}


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
    
    def get_summary_stats(self):
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
                'allocated_mean_percent': (np.mean(self.gpu_memory_allocated) / self.total_gpu_memory) * 100,
                'allocated_max_percent': (np.max(self.gpu_memory_allocated) / self.total_gpu_memory) * 100,
            }
        
        return summary


class PhysicsInformedLoss(nn.Module):
    """Custom loss function combining MSE with physics consistency loss."""

    def __init__(self, physics_consistency_weight=0.005):
        super(PhysicsInformedLoss, self).__init__()
        self.physics_consistency_weight = physics_consistency_weight
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets, model_vars, physics_info, config):
        """
        Args:
            predictions: Final model predictions [batch_size, 1]
            targets: Ground truth values [batch_size, 1]
            model_vars: Model predictions before physics fusion [batch_size, 1]
            physics_info: Physics-based info [batch_size, 15]
            config: Dataset configuration dict with energy indices

        Returns:
            total_loss, mse_loss, physics_loss
        """
        mse_loss = self.mse(predictions, targets)

        # Extract energies using dataset-specific indices
        guest_energy = physics_info[:, config['guest_indices']].sum(dim=1, keepdim=True)
        host_energy = physics_info[:, config['host_indices']].sum(dim=1, keepdim=True)
        complex_energy = physics_info[:, config['complex_indices']].sum(dim=1, keepdim=True)
        
        # Physics-based ΔG calculation
        dG_physics = complex_energy - (host_energy + guest_energy)
        
        # Physics consistency loss
        physics_loss = torch.sqrt(torch.mean((model_vars - dG_physics) ** 2))
        
        # Combined loss
        total_loss = mse_loss + self.physics_consistency_weight * physics_loss
        
        return total_loss, mse_loss, physics_loss


def featurize(molecule, info):
    """Featurize a molecule with additional info array."""
    atom_features = []
    for atom in molecule.GetAtoms():
        base_feat = get_atom_features(atom)
        new_feature = base_feat.tolist()
        
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
        new_feature += [position.x, position.y, position.z]
        
        # Add neighbor indices (up to 2 neighbors)
        neighbors = atom.GetNeighbors()[:2]
        for neighbor in neighbors:
            neighbor_idx = neighbor.GetIdx()
            new_feature += [float(neighbor_idx)]
        
        # Pad if less than 2 neighbors
        for i in range(2 - len(neighbors)):
            new_feature += [0.0]
        
        # Concatenate info array
        full_feature = new_feature + info
        atom_features.append(full_feature)
    
    return np.array(atom_features)


def load_data(dataset_name, monitor=None):
    """
    Load dataset based on configuration.
    
    Args:
        dataset_name: 'hostguest' or 'pdbbind'
        monitor: ResourceMonitor instance
        
    Returns:
        X, y, df, config
    """
    config = DATASET_CONFIGS[dataset_name]
    
    print(f"\n{'=' * 80}")
    print(f"Loading {config['name']} Dataset")
    print(f"{'=' * 80}")
    
    # Load PDB files from pickle
    print(f"Loading PDB files from: {config['pkl_path']}")
    with open(config['pkl_path'], 'rb') as f:
        pdb_dict = pd.read_pickle(f)
    print(f"Loaded {len(pdb_dict)} PDB files from pickle")
    
    # Read CSV
    print(f"Loading CSV from: {config['csv_path']}")
    df = pd.read_csv(config['csv_path'])
    print(f"Loaded {len(df)} entries from CSV")
    
    # Clean data based on dataset type
    if dataset_name == 'hostguest':
        # Extract dataset and guest info
        df['dataset'] = df[config['id_column']].str.split('--').str[0]
        df['guest_name'] = df[config['id_column']].str.split('--').str[1]
        print(f"Datasets: {df['dataset'].unique()}")
    elif dataset_name == 'pdbbind':
        # Remove entries with NaN in target column
        initial_size = len(df)
        df = df.dropna(subset=[config['target_column']])
        print(f"Removed {initial_size - len(df)} entries with NaN in {config['target_column']}")
        
        # Remove problematic entries with 'E+' in complex names
        valid_mask = ~df[config['id_column']].astype(str).str.contains('E\\+')
        invalid_count = (~valid_mask).sum()
        df = df[valid_mask]
        print(f"Removed {invalid_count} entries with 'E+' in complex names")
    
    # Filter to only entries with matching PDB structures
    pdb_keys = set(pdb_dict.keys())
    df = df[df[config['id_column']].isin(pdb_keys)]
    print(f"Filtered to {len(df)} entries with matching PDB structures")
    
    # Check which physics columns exist
    available_physics_cols = [col for col in config['physics_columns'] if col in df.columns]
    missing_cols = set(config['physics_columns']) - set(available_physics_cols)
    
    if missing_cols:
        print(f"\nWarning: Missing {len(missing_cols)} physics columns:")
        for col in list(missing_cols)[:5]:  # Show first 5
            print(f"  - {col}")
        if len(missing_cols) > 5:
            print(f"  ... and {len(missing_cols) - 5} more")
    
    print(f"\nUsing {len(available_physics_cols)} physics features")
    
    # Featurize all molecules
    print("\nFeaturizing molecules...")
    X = []
    y = []
    failed = []
    
    for idx, row in df.iterrows():
        complex_id = row[config['id_column']]
        target = row[config['target_column']]
        
        if complex_id not in pdb_dict:
            failed.append(complex_id)
            continue
        
        molecule = pdb_dict[complex_id]
        
        # Create info array with physics features
        info_array = row[available_physics_cols].fillna(0.0).tolist()
        
        # Pad to 15 features if necessary
        while len(info_array) < 15:
            info_array.append(0.0)
        info_array = info_array[:15]  # Ensure exactly 15 features
        
        try:
            features = featurize(molecule, info_array)
            
            # Debug first sample
            if len(X) == 0:
                print(f"\nFirst sample (ID: {complex_id}):")
                print(f"  Atoms: {molecule.GetNumAtoms()}")
                print(f"  Feature shape: {features.shape}")
                print(f"  Target: {target:.4f}")
            
            features_tensor = torch.FloatTensor(features)
            X.append(features_tensor)
            y.append(target)
            
        except Exception as e:
            if len(failed) < 10:
                print(f"Warning: Failed to featurize {complex_id}: {e}")
            failed.append(complex_id)
    
    print(f"\nSuccessfully loaded {len(X)} complexes")
    if failed:
        print(f"Failed to load {len(failed)} complexes")
    
    return X, y, df, config


def train_model(model, X_train, y_train, X_val, y_val, config, 
                epochs=250, lr=0.005, physics_weight=0.005, device='cpu', monitor=None):
    """Train the PGGCN model."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = PhysicsInformedLoss(physics_consistency_weight=physics_weight)
    
    # Move data to device
    X_train = [x.to(device) for x in X_train]
    X_val = [x.to(device) for x in X_val]
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    print(f"\nTraining on {len(X_train)} samples, validating on {len(X_val)} samples")
    print(f"Device: {device}, LR: {lr}, Physics weight: {physics_weight}, Epochs: {epochs}")
    print("=" * 80)
    
    best_val_loss = float('inf')
    epoch_start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        predictions, model_var, physics_info = model(X_train)
        train_loss, train_mse, train_physics_loss = criterion(
            predictions, y_train_tensor, model_var, physics_info, config
        )
        
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions, val_model_var, val_physics_info = model(X_val)
            val_loss, val_mse, val_physics_loss = criterion(
                val_predictions, y_val_tensor, val_model_var, val_physics_info, config
            )
            
            train_rmse = torch.sqrt(train_mse).item()
            val_rmse = torch.sqrt(val_mse).item()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - epoch_start_time
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss.item():.4f} (MSE: {train_mse.item():.4f}, Phys: {train_physics_loss.item():.4f}) | "
                  f"Val: {val_loss.item():.4f} (RMSE: {val_rmse:.4f}) | "
                  f"Time: {format_time(elapsed)} | ETA: {format_time(eta)}")
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1
    
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Total training time: {format_time(time.time() - epoch_start_time)}")


def evaluate_model(model, X_test, y_test, device='cpu'):
    """Evaluate the model on test set."""
    model.eval()
    
    X_test = [x.to(device) for x in X_test]
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    with torch.no_grad():
        predictions, _, _ = model(X_test)
        
        mse = nn.MSELoss()(predictions, y_test_tensor).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(predictions - y_test_tensor)).item()
        
        ss_res = torch.sum((y_test_tensor - predictions) ** 2).item()
        ss_tot = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return predictions.cpu().numpy(), metrics


def main():
    parser = argparse.ArgumentParser(description='Train PGGCN on molecular datasets')
    parser.add_argument('--dataset', type=str, choices=['hostguest', 'pdbbind'], 
                       default='hostguest', help='Dataset to use')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--physics_weight', type=float, default=0.005, 
                       help='Physics consistency loss weight')
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--c_out', type=int, default=128, help='Conv layer output channels')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"PyTorch PGGCN Training - {args.dataset.upper()} Dataset")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    monitor = ResourceMonitor(device=device)
    monitor.start_monitoring()
    
    # Load data
    X, y, df, config = load_data(args.dataset, monitor=monitor)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Create model
    model = PGGCNModel(num_atom_features=36, r_out_channel=20, c_out_channel=args.c_out)
    model.add_rule("sum", 0, 32)
    model.add_rule("multiply", 32, 33)
    model.add_rule("distance", 33, 36)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train
    train_model(model, X_train, y_train, X_test, y_test, config,
               epochs=args.epochs, lr=args.lr, physics_weight=args.physics_weight,
               device=device, monitor=monitor)
    
    # Evaluate
    predictions, metrics = evaluate_model(model, X_test, y_test, device=device)
    
    print(f"\n{'=' * 80}")
    print("TEST METRICS")
    print(f"{'=' * 80}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    
    # Sample predictions
    print(f"\nSample Predictions:")
    print(f"{'True':>10} | {'Pred':>10} | {'Error':>10}")
    print("-" * 35)
    for i in range(min(10, len(y_test))):
        print(f"{y_test[i]:>10.4f} | {predictions[i][0]:>10.4f} | {predictions[i][0] - y_test[i]:>10.4f}")
    
    monitor.stop_monitoring()
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()