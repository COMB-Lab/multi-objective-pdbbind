"""
PyTorch PGGCN Training Script - Unified Version

Features:
- Support for both host-guest (DDG) and PDBBind datasets
- Option to use PCGrad or standard additive loss
- Batching support (essential for PDBBind)
- Matches TensorFlow implementation exactly
- Physics sign fix applied
- Predefined coefficients for dense_final layer
- Proper L2 regularization via weight_decay
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
from datetime import timedelta
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse

# ============================================================================
# SETUP & IMPORTS
# ============================================================================

# Add the necessary directories to the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

possible_paths = [
    SCRIPT_DIR,
    PARENT_DIR,
    os.path.join(PARENT_DIR, 'PGGCN'),
    '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind',
]

for path in possible_paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import model modules
try:
    from PGGCN.models.dcFeaturizer import atom_features as get_atom_features
    from PGGCN.models.layers_pytorch import PGGCNModel
    print("✓ Imported using PGGCN.models style")
except ImportError:
    try:
        from models.dcFeaturizer import atom_features as get_atom_features
        from models.layers_pytorch import PGGCNModel
        print("✓ Imported using models style")
    except ImportError as e:
        print(f"✗ Error importing modules: {e}")
        sys.exit(1)

# Try to import PCGrad (optional)
try:
    from pcgrad_pytorch import PCGrad
    PCGRAD_AVAILABLE = True
    print("✓ PCGrad optimizer available")
except ImportError:
    PCGRAD_AVAILABLE = False
    print("✗ PCGrad not available (will use standard optimizer)")


# ============================================================================
# CONFIGURATION
# ============================================================================

class HostGuestConfig:
    """Training configuration for host-guest dataset"""
    # Data paths
    CSV_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/Final_data_DDG.csv'
    PDB_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBs_RDKit_BFE.pkl'
    
    # Model architecture
    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL = 20
    C_OUT_CHANNEL = 1024
    DROPOUT_RATE = 0.2
    
    # Training hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    L2_WEIGHT = 1e-2
    MAX_NORM = 3.0
    PHYSICS_WEIGHT = 0.02
    EARLY_STOPPING_PATIENCE = 10
    
    # Data split
    TEST_SIZE = 0.2
    RANDOM_SEED = 50


class PDBBindConfig:
    """Training configuration for PDBBind dataset"""
    # Data paths
    CSV_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind_100.csv'
    PKL_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_100.pkl'
    CSV_PATH_ALT = '/home/exouser/multiloss-bfe/multiloss_pdbbind/Datasets/pdbbind_100.csv'
    PKL_PATH_ALT = '/home/exouser/multiloss-bfe/multiloss_pdbbind/Datasets/PDBBind_100.pkl'
    
    # Model architecture
    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL = 20
    C_OUT_CHANNEL = 1024  # Different from host-guest!
    DROPOUT_RATE = 0.2
    
    # Training hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 8  # Essential for PDBBind!
    LEARNING_RATE = 1e-5
    L2_WEIGHT = 1e-2
    MAX_NORM = 3.0
    PHYSICS_WEIGHT = 0.02
    EARLY_STOPPING_PATIENCE = 10
    
    # Data split
    TEST_SIZE = 0.2
    RANDOM_SEED = 50


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds):
    """Format seconds into readable time string"""
    return str(timedelta(seconds=int(seconds)))


def format_bytes(bytes_val):
    """Format bytes into human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


# ============================================================================
# DATASET CLASS FOR BATCHING
# ============================================================================

class MoleculeDataset(Dataset):
    """Dataset wrapper for batching molecules"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_molecules(batch):
    """
    Custom collate function for variable-sized molecules
    Returns lists instead of stacked tensors
    """
    X_batch = [item[0] for item in batch]
    y_batch = [item[1] for item in batch]
    return X_batch, torch.FloatTensor(y_batch)


# ============================================================================
# DATA LOADING
# ============================================================================

def featurize(molecule, info):
    """Featurize a molecule with additional physics info"""
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


def load_hostguest_data(csv_path, pdb_path):
    """Load host-guest dataset from CSV and PDB files"""
    print(f"Loading host-guest PDB files from: {pdb_path}")
    with open(pdb_path, 'rb') as f:
        pdb_dict = pickle.load(f)
    print(f"✓ Loaded {len(pdb_dict)} PDB files")

    df_all = pd.read_csv(csv_path)
    print(f"✓ Found {len(df_all)} entries in CSV")

    feature_columns = [
        'pb_host_VDWAALS', 'pb_guest_VDWAALS', 'pb_complex_VDWAALS',
        'gb_host_1-4EEL', 'gb_guest_1-4EEL', 'gb_Complex_1-4EEL',
        'gb_host_EELEC', 'gb_guest_EELEC', 'gb_Complex_EELEC',
        'gb_host_EGB', 'gb_guest_EGB', 'gb_Complex_EGB',
        'gb_host_ESURF', 'gb_guest_ESURF', 'gb_Complex_ESURF'
    ]

    X, y = [], []
    failed = []
    
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
            print(f"  Warning: Failed to featurize {pdb_id}")
    
    print(f"✓ Successfully loaded {len(X)} complexes")
    if failed:
        print(f"  Warning: Failed to load {len(failed)} complexes")
    
    return X, y


def load_pdbbind_data(config):
    """Load PDBBind dataset"""
    print("Loading PDBBind dataset...")
    
    for csv_path, pkl_path in [
        (config.CSV_PATH, config.PKL_PATH),
        (config.CSV_PATH_ALT, config.PKL_PATH_ALT)
    ]:
        if os.path.exists(csv_path) and os.path.exists(pkl_path):
            print(f"✓ Found files at: {os.path.dirname(csv_path)}")
            break
    else:
        raise FileNotFoundError("Could not find PDBBind data!")
    
    df = pd.read_csv(csv_path)
    with open(pkl_path, 'rb') as f:
        pdb_dict = pickle.load(f)
    
    print(f"✓ Loaded {len(df)} CSV entries, {len(pdb_dict)} PDB structures")
    
    df = df.dropna(subset=['ddg'])
    df = df[df['complex-name'].apply(lambda x: 'E+' not in str(x))]
    
    physics_columns = [
        'pb-protein-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals',
        'gb-protein-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
        'gb-protein-eelect', 'gb-ligand-eelec', 'gb-complex-eelec',
        'gb-protein-egb', 'gb-ligand-egb', 'gb-complex-egb',
        'gb-protein-esurf', 'gb-ligand-esurf', 'gb-complex-esurf'
    ]
    
    common_keys = set(df['complex-name']) & set(pdb_dict.keys())
    df = df[df['complex-name'].isin(common_keys)]
    print(f"✓ Final dataset: {len(df)} structures")
    
    X, y = [], []
    for pdb_id in df['complex-name']:
        row = df[df['complex-name'] == pdb_id].iloc[0]
        info_array = row[physics_columns].tolist()
        target = row['ddg']
        
        try:
            features = featurize(pdb_dict[pdb_id], info_array)
            X.append(torch.FloatTensor(features))
            y.append(target)
        except Exception:
            pass
    
    print(f"✓ Successfully featurized {len(X)} structures")
    return X, y


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def apply_maxnorm_constraint(model, max_norm=3.0):
    """Apply MaxNorm constraint to parameters (matches TensorFlow)"""
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.dim() >= 2:
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=max_norm)
                param.mul_(desired / (norm + 1e-7))


def compute_task_losses(predictions, targets, physics_info, physics_weight):
    """
    Compute empirical and physics losses
    
    PAPER: Equation 4 specifies ΔG = complex - (host + guest)
    This creates gradient conflict with empirical loss, which PCGrad resolves
    """
    targets = targets.view(-1, 1)
    
    # Empirical loss (RMSE)
    empirical_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))
    
    # Extract energy components
    host_energy = physics_info[:, [0, 3, 6, 9, 12]].sum(dim=1, keepdim=True)
    guest_energy = physics_info[:, [1, 4, 7, 10, 13]].sum(dim=1, keepdim=True)
    complex_energy = physics_info[:, [2, 5, 8, 11, 14]].sum(dim=1, keepdim=True)
    
    # PAPER Eq. 4: ΔG = ΔGcomplex - (ΔGhost + ΔGguest)
    dG_physics = complex_energy - (host_energy + guest_energy)
    
    # Physics consistency loss (RMSE)
    raw_physics_loss = torch.sqrt(torch.mean((predictions - dG_physics) ** 2))
    weighted_physics_loss = physics_weight * raw_physics_loss
    
    return empirical_loss, weighted_physics_loss, raw_physics_loss


class EarlyStopping:
    """Early stopping (matches TensorFlow behavior)"""
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
            if self.verbose and self.counter >= self.patience:
                print(f'Early stopping triggered at counter {self.counter}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            
    def restore_best_weights(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"✓ Restored best weights (loss: {self.best_loss:.4f})")


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model_full_batch(model, X_train, y_train, X_val, y_val, 
                          use_pcgrad=False, config=None, device='cpu'):
    """
    Train the PGGCN model with full-batch updates (for host-guest)
    
    Args:
        use_pcgrad: If True, use PCGrad optimizer. If False, use standard additive loss.
    """
    if config is None:
        config = HostGuestConfig()
    
    model = model.to(device)
    
    # Setup optimizer
    if use_pcgrad and PCGRAD_AVAILABLE:
        base_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0)
        optimizer = PCGrad(base_optimizer)
        opt_name = "PCGrad + Adam"
    elif use_pcgrad and not PCGRAD_AVAILABLE:
        print("⚠️  PCGrad requested but not available. Using standard Adam.")
        base_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                                    weight_decay=config.L2_WEIGHT)
        optimizer = base_optimizer
        opt_name = "Adam (PCGrad unavailable)"
    else:
        base_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                                    weight_decay=config.L2_WEIGHT)
        optimizer = base_optimizer
        opt_name = "Adam"
    
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, 
                                   min_delta=0.001, verbose=True)
    
    # Move data to device
    X_train = [x.to(device) for x in X_train]
    X_val = [x.to(device) for x in X_val]
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # Print configuration
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION (FULL BATCH)")
    print("=" * 80)
    print(f"Optimizer: {opt_name}")
    print(f"Samples: {len(X_train)} train, {len(X_val)} val")
    print(f"Device: {device}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    if use_pcgrad and PCGRAD_AVAILABLE:
        print(f"L2 regularization: {config.L2_WEIGHT}")
    else:
        print(f"L2 regularization: {config.L2_WEIGHT} (via weight_decay)")
    print(f"MaxNorm constraint: {config.MAX_NORM}")
    print(f"Dropout rate: {config.DROPOUT_RATE}")
    print(f"Physics weight: {config.PHYSICS_WEIGHT}")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"Max epochs: {config.EPOCHS}")
    print("=" * 80)
    
    # Training history
    train_losses, val_losses = [], []
    train_empirical_losses, train_physics_losses = [], []
    val_empirical_losses, val_physics_losses = [], []
    
    best_val_loss = float('inf')
    best_epoch = 0
    start_time = time.time()
    
    # Training loop
    for epoch in range(config.EPOCHS):
        # Training phase
        model.train()
        
        # Forward pass
        predictions, model_var, physics_info = model(X_train, training=True)
        
        # Compute losses
        train_empirical, train_weighted_physics, train_raw_physics = compute_task_losses(
            predictions, y_train_tensor, physics_info, config.PHYSICS_WEIGHT
        )
        
        # Backward pass
        if use_pcgrad and PCGRAD_AVAILABLE:
            optimizer.pc_backward([train_empirical, train_weighted_physics])
            optimizer.step()
        else:
            optimizer.zero_grad()
            combined_loss = train_empirical + train_weighted_physics
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Apply MaxNorm constraint
        apply_maxnorm_constraint(model, max_norm=config.MAX_NORM)
        
        # Record training losses
        train_loss_total = train_empirical.item() + train_weighted_physics.item()
        train_losses.append(train_loss_total)
        train_empirical_losses.append(train_empirical.item())
        train_physics_losses.append(train_raw_physics.item())
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_predictions, val_model_var, val_physics_info = model(X_val, training=False)
            val_empirical, val_weighted_physics, val_raw_physics = compute_task_losses(
                val_predictions, y_val_tensor, val_physics_info, config.PHYSICS_WEIGHT
            )
            val_loss_total = val_empirical.item() + val_weighted_physics.item()
            val_losses.append(val_loss_total)
            val_empirical_losses.append(val_empirical.item())
            val_physics_losses.append(val_raw_physics.item())
        
        # Track best
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            best_epoch = epoch + 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (config.EPOCHS - epoch - 1)
            
            print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train: {train_loss_total:.4f} "
                  f"(Emp: {train_empirical.item():.4f}, Phys: {train_raw_physics.item():.4f}) | "
                  f"Val: {val_loss_total:.4f} "
                  f"(Emp: {val_empirical.item():.4f}, Phys: {val_raw_physics.item():.4f}) | "
                  f"Time: {format_time(elapsed)} | ETA: {format_time(eta)}")
        
        # Check early stopping (optional)
        # early_stopping(val_loss_total, model)
        # if early_stopping.early_stop:
        #     print(f"\nEarly stopping at epoch {epoch+1}")
        #     early_stopping.restore_best_weights(model)
        #     break
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"Training completed in {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print("=" * 80)
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_empirical': train_empirical_losses,
        'train_physics': train_physics_losses,
        'val_empirical': val_empirical_losses,
        'val_physics': val_physics_losses
    }
    
    return history


def train_model_batched(model, train_loader, val_loader, 
                       use_pcgrad=False, config=None, device='cpu'):
    """
    Train the PGGCN model with mini-batches (for PDBBind)
    
    Args:
        use_pcgrad: If True, use PCGrad optimizer. If False, use standard additive loss.
    """
    if config is None:
        config = PDBBindConfig()
    
    model = model.to(device)
    
    # Setup optimizer
    if use_pcgrad and PCGRAD_AVAILABLE:
        base_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0)
        optimizer = PCGrad(base_optimizer)
        opt_name = "PCGrad + Adam"
    elif use_pcgrad and not PCGRAD_AVAILABLE:
        print("⚠️  PCGrad requested but not available. Using standard Adam.")
        base_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                                    weight_decay=config.L2_WEIGHT)
        optimizer = base_optimizer
        opt_name = "Adam (PCGrad unavailable)"
    else:
        base_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                                    weight_decay=config.L2_WEIGHT)
        optimizer = base_optimizer
        opt_name = "Adam"
    
    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION (BATCHED)")
    print("="*80)
    print(f"Optimizer: {opt_name}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Batches per epoch: ~{len(train_loader)}")
    print(f"Weight updates per epoch: {len(train_loader)}")
    print(f"Total updates over {config.EPOCHS} epochs: {len(train_loader) * config.EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    if use_pcgrad and PCGRAD_AVAILABLE:
        print(f"L2 regularization: {config.L2_WEIGHT} (explicit 3rd objective for PCGrad)")
    else:
        print(f"L2 regularization: {config.L2_WEIGHT} (via weight_decay)")
    print(f"Physics weight: {config.PHYSICS_WEIGHT}")
    print("="*80)
    
    history = {
        'train_losses': [], 'val_losses': [],
        'train_empirical': [], 'train_physics': [],
        'val_empirical': [], 'val_physics': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        epoch_train_emp = []
        epoch_train_phys = []
        epoch_train_total = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Move batch to device
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            
            # Forward pass
            predictions, _, physics_info = model(X_batch, training=True)
            
            # Compute losses
            train_emp, train_phys_w, train_phys_r = compute_task_losses(
                predictions, y_batch, physics_info, config.PHYSICS_WEIGHT)
            
            # Backward pass
            if use_pcgrad and PCGRAD_AVAILABLE:
                optimizer.pc_backward([train_emp, train_phys_w])
                optimizer.step()
            else:
                optimizer.zero_grad()
                total_loss = train_emp + train_phys_w
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            apply_maxnorm_constraint(model, config.MAX_NORM)
            
            # Track metrics for this batch
            epoch_train_emp.append(train_emp.item())
            epoch_train_phys.append(train_phys_r.item())
            epoch_train_total.append((train_emp + train_phys_w).item())
        
        # Average across batches
        avg_train_emp = np.mean(epoch_train_emp)
        avg_train_phys = np.mean(epoch_train_phys)
        avg_train_total = np.mean(epoch_train_total)
        
        history['train_empirical'].append(avg_train_emp)
        history['train_physics'].append(avg_train_phys)
        history['train_losses'].append(avg_train_total)
        
        # Validation (full dataset)
        model.eval()
        with torch.no_grad():
            all_val_preds = []
            all_val_targets = []
            all_val_phys = []
            
            for X_batch, y_batch in val_loader:
                X_batch = [x.to(device) for x in X_batch]
                y_batch = y_batch.unsqueeze(1).to(device)
                
                val_pred, _, val_phys = model(X_batch, training=False)
                
                all_val_preds.append(val_pred)
                all_val_targets.append(y_batch)
                all_val_phys.append(val_phys)
            
            val_predictions = torch.cat(all_val_preds, dim=0)
            val_targets = torch.cat(all_val_targets, dim=0)
            val_physics = torch.cat(all_val_phys, dim=0)
            
            # Compute validation losses
            val_emp, val_phys_w, val_phys_r = compute_task_losses(
                val_predictions, val_targets, val_physics, config.PHYSICS_WEIGHT)
            
            val_total_loss = val_emp.item() + val_phys_w.item()
            
            history['val_empirical'].append(val_emp.item())
            history['val_physics'].append(val_phys_r.item())
            history['val_losses'].append(val_total_loss)
        
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_epoch = epoch + 1
        
        # Progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (config.EPOCHS - epoch - 1)
            
            print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train: {avg_train_total:.4f} "
                  f"(Emp: {avg_train_emp:.4f}, Phys: {avg_train_phys:.4f}) | "
                  f"Val: {val_total_loss:.4f} "
                  f"(Emp: {val_emp.item():.4f}, Phys: {val_phys_r.item():.4f}) | "
                  f"Time: {format_time(elapsed)} | ETA: {format_time(eta)}")
    
    print(f"\nCompleted {config.EPOCHS} epochs in {format_time(time.time() - start_time)}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    
    return history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model_full_batch(model, X_test, y_test, device='cpu'):
    """Evaluate model on test set (full batch)"""
    model.eval()
    X_test = [x.to(device) for x in X_test]
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    with torch.no_grad():
        predictions, model_var, physics_info = model(X_test, training=False)
        
        # Compute metrics
        mse = nn.MSELoss()(predictions, y_test_tensor).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(predictions - y_test_tensor)).item()
        
        # R²
        ss_res = torch.sum((y_test_tensor - predictions) ** 2).item()
        ss_tot = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot)
        
        # Loss breakdown
        _, _, phys_loss = compute_task_losses(predictions, y_test_tensor, 
                                             physics_info, 0.02)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'physics_loss': phys_loss.item()
    }
    
    return predictions.cpu().numpy(), metrics


def evaluate_model_batched(model, test_loader, device='cpu'):
    """Evaluate model on test set (batched)"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            
            predictions, _, _ = model(X_batch, training=False)
            all_preds.append(predictions)
            all_targets.append(y_batch)
    
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    mse = torch.mean((predictions - targets) ** 2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    ss_res = torch.sum((targets - predictions) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return predictions.cpu().numpy(), metrics


def print_sample_predictions(predictions, y_test, n=10):
    """Print sample predictions"""
    print("\n" + "-" * 80)
    print("Sample Predictions")
    print("-" * 80)
    print(f"{'True Value':>12} | {'Prediction':>12} | {'Error':>12}")
    print("-" * 40)
    for i in range(min(n, len(y_test))):
        true_val = y_test[i]
        pred_val = predictions[i][0]
        error = pred_val - true_val
        print(f"{true_val:>12.4f} | {pred_val:>12.4f} | {error:>12.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train PGGCN model')
    parser.add_argument('--dataset', choices=['hostguest', 'pdbbind'], required=True,
                       help='Which dataset to use')
    parser.add_argument('--use-pcgrad', action='store_true',
                       help='Use PCGrad optimizer (default: standard additive loss)')
    parser.add_argument('--run-both', action='store_true',
                       help='Run both PCGrad and no-PCGrad for comparison')
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"PYTORCH PGGCN TRAINING - {args.dataset.upper()} DATASET")
    print("=" * 80)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Select config
    if args.dataset == 'hostguest':
        config = HostGuestConfig()
    else:
        config = PDBBindConfig()
    
    set_random_seeds(config.RANDOM_SEED)
    
    # Load data
    print("\n" + "-" * 80)
    print("Loading Data")
    print("-" * 80)
    
    if args.dataset == 'hostguest':
        X, y = load_hostguest_data(config.CSV_PATH, config.PDB_PATH)
    else:
        X, y = load_pdbbind_data(config)
    
    # Split data
    print("\n" + "-" * 80)
    print("Splitting Data")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    print(f"✓ Training: {len(X_train)} samples")
    print(f"✓ Test: {len(X_test)} samples")
    print(f"Target stats: mean={np.mean(y):.2f}, std={np.std(y):.2f}")
    
    # Determine which runs to do
    if args.run_both:
        runs = [('No PCGrad', False), ('PCGrad', True)]
    else:
        runs = [('PCGrad' if args.use_pcgrad else 'No PCGrad', args.use_pcgrad)]
    
    results = {}
    
    # Run training
    for run_name, use_pcgrad in runs:
        print("\n" + "=" * 80)
        print(f"TRAINING: {run_name.upper()}")
        print("=" * 80)
        
        # Create fresh model
        print("\n" + "-" * 80)
        print("Creating Model")
        print("-" * 80)
        model = PGGCNModel(
            num_atom_features=config.NUM_ATOM_FEATURES,
            r_out_channel=config.R_OUT_CHANNEL,
            c_out_channel=config.C_OUT_CHANNEL,
            dropout_rate=config.DROPOUT_RATE
        )
        model.add_rule("sum", 0, 32)
        model.add_rule("multiply", 32, 33)
        model.add_rule("distance", 33, 36)
        
        print(f"Model created ({sum(p.numel() for p in model.parameters())} parameters)")
        
        # Verify dense_final initialization
        expected = torch.tensor([0.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1])
        if torch.allclose(model.dense_final.weight.data.flatten(), expected, atol=0.01):
            print("✓ dense_final weights correctly initialized [0.3, 1, 1, -1, ...]")
        else:
            print("⚠️  WARNING: dense_final weights not correctly initialized!")
        
        model = model.to(device)
        
        # Train
        print("\n" + "-" * 80)
        print("Training")
        print("-" * 80)
        
        if args.dataset == 'hostguest':
            # Full-batch training
            history = train_model_full_batch(model, X_train, y_train, X_test, y_test,
                                            use_pcgrad=use_pcgrad, config=config, device=device)
            
            # Evaluate
            print("\n" + "-" * 80)
            print("Evaluation")
            print("-" * 80)
            predictions, metrics = evaluate_model_full_batch(model, X_test, y_test, device=device)
            
        else:
            # Batched training (PDBBind)
            train_dataset = MoleculeDataset(X_train, y_train)
            test_dataset = MoleculeDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                     shuffle=True, collate_fn=collate_molecules)
            test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                                    shuffle=False, collate_fn=collate_molecules)
            
            history = train_model_batched(model, train_loader, test_loader,
                                         use_pcgrad=use_pcgrad, config=config, device=device)
            
            # Evaluate
            print("\n" + "-" * 80)
            print("Evaluation")
            print("-" * 80)
            predictions, metrics = evaluate_model_batched(model, test_loader, device=device)
        
        print(f"\nTest Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        if 'physics_loss' in metrics:
            print(f"  Physics Loss: {metrics['physics_loss']:.4f}")
        
        print_sample_predictions(predictions, y_test)
        
        # Store results
        results[run_name] = {
            'model': model,
            'history': history,
            'metrics': metrics,
            'predictions': predictions
        }
        
        # Save model
        save_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/saved_models'
        os.makedirs(save_dir, exist_ok=True)
        
        pcgrad_suffix = 'pcgrad' if use_pcgrad else 'no_pcgrad'
        filename = f"pggcn_{args.dataset}_{pcgrad_suffix}_final_0.02.pth"
        save_path = os.path.join(save_dir, filename)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'metrics': metrics,
            'config': {
                'dataset': args.dataset,
                'learning_rate': config.LEARNING_RATE,
                'l2_weight': config.L2_WEIGHT,
                'max_norm': config.MAX_NORM,
                'dropout_rate': config.DROPOUT_RATE,
                'physics_weight': config.PHYSICS_WEIGHT,
                'batch_size': config.BATCH_SIZE,
                'use_pcgrad': use_pcgrad,
            }
        }, save_path)
        
        print(f"\n✓ Model saved to: {save_path}")
    
    # Comparison if both were run
    if args.run_both:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"\n{'Metric':<20} | {'No PCGrad':>12} | {'PCGrad':>12} | {'Difference':>12}")
        print("-" * 60)
        
        no_pcgrad = results['No PCGrad']['metrics']
        pcgrad = results['PCGrad']['metrics']
        
        for metric in ['rmse', 'mae', 'r2']:
            if metric in no_pcgrad and metric in pcgrad:
                no_pc_val = no_pcgrad[metric]
                pc_val = pcgrad[metric]
                diff = pc_val - no_pc_val
                print(f"{metric.upper():<20} | {no_pc_val:>12.4f} | {pc_val:>12.4f} | {diff:>+12.4f}")
        
        if args.dataset == 'hostguest':
            print("\nTensorFlow Targets (for reference):")
            print("  Without PCGrad: MAE 3.06, Emp 3.69, Phys 12.78")
            print("  With PCGrad:    MAE 2.94, Emp 3.58, Phys 12.90")
        else:
            print("\nTensorFlow Baseline: MAE 9.24")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()