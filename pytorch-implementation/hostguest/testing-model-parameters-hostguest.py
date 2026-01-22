"""
PyTorch PGGCN Training Script - Testing Model Parameters for HostGuest Dataset
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
import argparse
from datetime import timedelta
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

# ============================================================================
# SETUP & IMPORTS
# ============================================================================

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
    print("✓ Imported using PGGCN.models style")
except ImportError:
    try:
        from models.dcFeaturizer import atom_features as get_atom_features
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
    print("✗ PCGrad not available")


# ============================================================================
# IMPROVED MODEL ARCHITECTURE
# ============================================================================

# Import layers - try multiple paths
try:
    from PGGCN.models.layers_pytorch import RuleGraphConvLayer, ConvLayer
    print("✓ Imported layers using PGGCN.models style")
except ImportError:
    try:
        from models.layers_pytorch import RuleGraphConvLayer, ConvLayer
        print("✓ Imported layers using models style")
    except ImportError:
        try:
            from layers_pytorch import RuleGraphConvLayer, ConvLayer
            print("✓ Imported layers directly")
        except ImportError as e:
            print(f"✗ Error importing layer modules: {e}")
            print("Please ensure layers_pytorch.py is in the same directory or PYTHONPATH")
            sys.exit(1)

class ImprovedPGGCNModel(nn.Module):
    """
    Improved PGGCN model with flexible architecture.
    
    Changes from original:
    - Configurable dense layer sizes
    - Larger conv capacity
    - Higher dropout options
    - More parameters for complex datasets
    """
    
    def __init__(self, 
                 num_atom_features=36,
                 r_out_channel=20,
                 c_out_channel=256,  # INCREASED from 128
                 dense1_size=32,
                 dense2_size=32,     # INCREASED from 16
                 dense3_size=32,     # NEW layer option
                 dropout_rate=0.3):  # INCREASED from 0.2
        super(ImprovedPGGCNModel, self).__init__()

        self.num_atom_features = num_atom_features
        self.num_physics_features = 15
        
        # Graph convolution layers
        self.rule_graph_conv = RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
        self.conv = ConvLayer(c_out_channel, r_out_channel)
        
        # Flexible dense architecture
        self.dense1 = nn.Linear(c_out_channel, dense1_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.dense2 = nn.Linear(dense1_size, dense2_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Optional third dense layer for more capacity
        self.use_dense3 = dense3_size > 0
        if self.use_dense3:
            self.dense3 = nn.Linear(dense2_size, dense3_size)
            self.dropout3 = nn.Dropout(dropout_rate)
            final_dense_input = dense3_size
        else:
            final_dense_input = dense2_size
        
        # Model variance output
        self.model_var_layer = nn.Linear(final_dense_input, 1)
        
        # Final prediction layer (combines model_var + physics)
        self.dense_final = nn.Linear(16, 1)
        
        # Initialize final layer weights to match TensorFlow
        with torch.no_grad():
            init_weights = torch.tensor([
                0.3,   # model_var weight
                1.0, 1.0, -1.0,  # VDW
                1.0, 1.0, -1.0,  # 1-4EEL
                1.0, 1.0, -1.0,  # EELEC
                1.0, 1.0, -1.0,  # EGB
                1.0, 1.0, -1.0   # ESURF
            ]).reshape(1, 16)
            self.dense_final.weight.copy_(init_weights)
            self.dense_final.bias.zero_()
        
        self.relu = nn.ReLU()
        
        # Print architecture
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print("MODEL ARCHITECTURE")
        print(f"{'='*60}")
        print(f"Graph Conv: {num_atom_features} → {r_out_channel}")
        print(f"Mol Conv:   {r_out_channel} → {c_out_channel}")
        print(f"Dense 1:    {c_out_channel} → {dense1_size}")
        print(f"Dense 2:    {dense1_size} → {dense2_size}")
        if self.use_dense3:
            print(f"Dense 3:    {dense2_size} → {dense3_size}")
            print(f"Model Var:  {dense3_size} → 1")
        else:
            print(f"Model Var:  {dense2_size} → 1")
        print(f"Final:      16 → 1 (model_var + physics)")
        print(f"Dropout:    {dropout_rate}")
        print(f"Total Parameters: {total_params:,}")
        print(f"{'='*60}\n")

    def add_rule(self, rule, start_index, end_index=None):
        """Add a combination rule to the RuleGraphConvLayer."""
        self.rule_graph_conv.add_rule(rule, start_index, end_index)

    def forward(self, batch_molecules, training=True):
        """Forward pass with flexible architecture."""
        # Extract atom features and physics info
        atom_features_batch = []
        physics_info_batch = []

        for mol in batch_molecules:
            atom_feat = mol[:, :self.num_atom_features + 2]
            atom_features_batch.append(atom_feat)
            physics_info = mol[0, -self.num_physics_features:]
            physics_info_batch.append(physics_info)

        physics_info_tensor = torch.stack(physics_info_batch)

        # Graph convolutions
        x = self.rule_graph_conv(atom_features_batch)
        x = self.conv(x)

        # Dense layers with dropout
        x = self.dense1(x)
        x = self.relu(x)
        if training:
            x = self.dropout1(x)
        
        x = self.dense2(x)
        x = self.relu(x)
        if training:
            x = self.dropout2(x)
        
        if self.use_dense3:
            x = self.dense3(x)
            x = self.relu(x)
            if training:
                x = self.dropout3(x)
        
        # Model variance
        model_var = self.model_var_layer(x)

        # Combine with physics and predict
        merged = torch.cat([model_var, physics_info_tensor], dim=1)
        out = self.dense_final(merged)

        return out, model_var, physics_info_tensor


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration with flexible options"""
    # Data paths (defaults for HostGuest dataset)
    CSV_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/Final_data_DDG.csv'
    PDB_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBs_RDKit_BFE.pkl'
    TARGET_COL = 'Ex _G_(kcal/mol)'  # Default target column
    
    # Model architecture (IMPROVED DEFAULTS)
    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL = 20
    C_OUT_CHANNEL = 256      # INCREASED from 128
    DENSE1_SIZE = 32
    DENSE2_SIZE = 32         # INCREASED from 16
    DENSE3_SIZE = 0          # 0 = disabled, >0 = add extra layer
    DROPOUT_RATE = 0.3       # INCREASED from 0.2
    
    # Training hyperparameters
    EPOCHS = 500             # INCREASED from 250
    LEARNING_RATE = 1e-4
    L2_WEIGHT = 1e-4
    MAX_NORM = 3.0
    PHYSICS_WEIGHT = 0.58
    EARLY_STOPPING_PATIENCE = 30  # INCREASED from 10
    
    # Data options
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    SUBSET_SIZE = None       # None = use all data, or specify number for quick tests
    
    # Save options
    SAVE_DIR = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/saved_models'
    PLOT_DIR = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/plots'


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


def load_data(csv_path, pdb_path, target_col='Ex _G_(kcal/mol)', subset_size=None):
    """Load dataset with optional subset for quick testing"""
    print(f"Loading PDB files from: {pdb_path}")
    with open(pdb_path, 'rb') as f:
        pdb_dict = pickle.load(f)
    print(f"✓ Loaded {len(pdb_dict)} PDB files")

    df_all = pd.read_csv(csv_path)
    print(f"✓ Found {len(df_all)} entries in CSV")
    print(f"✓ Using target column: '{target_col}'")

    # Determine feature columns based on CSV structure
    # For HostGuest: specific column names
    # For PDBBind: may have different column structure
    
    # Try to auto-detect feature columns
    if 'pb_host_VDWAALS' in df_all.columns:
        # HostGuest format
        feature_columns = [
            'pb_host_VDWAALS', 'pb_guest_VDWAALS', 'pb_complex_VDWAALS',
            'gb_host_1-4EEL', 'gb_guest_1-4EEL', 'gb_Complex_1-4EEL',
            'gb_host_EELEC', 'gb_guest_EELEC', 'gb_Complex_EELEC',
            'gb_host_EGB', 'gb_guest_EGB', 'gb_Complex_EGB',
            'gb_host_ESURF', 'gb_guest_ESURF', 'gb_Complex_ESURF'
        ]
        id_column = 'Ids'
    elif 'gb-host-1-4-eel' in df_all.columns or 'gb_protein_1_4_eel' in df_all.columns:
        # PDBBind format - try different naming conventions
        # Check for hyphen vs underscore format
        if 'gb-host-1-4-eel' in df_all.columns:
            feature_columns = [
                'pb-host-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals',
                'gb-host-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
                'gb-host-eelect', 'gb-ligand-eelec', 'gb-complex-eelect',
                'gb-host-egb', 'gb-ligand-egb', 'gb-complex-egb',
                'gb-host-esurf', 'gb-ligand-esurf', 'gb-complex-esurf'
            ]
        else:
            # Try protein/ligand naming
            feature_columns = [
                'pb_protein_vdwaals', 'pb_ligand_vdwaals', 'pb_complex_vdwaals',
                'gb_protein_1_4_eel', 'gb_ligand_1_4_eel', 'gb_complex_1_4_eel',
                'gb_protein_eelect', 'gb_ligand_eelec', 'gb_complex_eelect',
                'gb_protein_egb', 'gb_ligand_egb', 'gb_complex_egb',
                'gb_protein_esurf', 'gb_ligand_esurf', 'gb_complex_esurf'
            ]
        id_column = 'complex-name'
    else:
        raise ValueError(f"Could not auto-detect feature columns. Available columns: {df_all.columns.tolist()}")
    
    print(f"✓ Using feature columns format: {feature_columns[0].split('_')[0]}...")
    print(f"✓ Using ID column: '{id_column}'")

    X, y, pdb_ids = [], [], []
    
    # Optionally limit to subset for quick testing
    pdb_keys = list(pdb_dict.keys())
    if subset_size is not None and subset_size < len(pdb_keys):
        print(f"⚠️  Using subset of {subset_size} complexes for quick testing")
        pdb_keys = pdb_keys[:subset_size]
    
    failed_count = 0
    for pdb_id in pdb_keys:
        if pdb_id not in df_all[id_column].values:
            failed_count += 1
            continue
        
        molecule = pdb_dict[pdb_id]
        row = df_all[df_all[id_column] == pdb_id].iloc[0]
        
        # Get feature array
        try:
            info_array = row[feature_columns].tolist()
        except KeyError as e:
            print(f"  Warning: Missing columns for {pdb_id}: {e}")
            failed_count += 1
            continue
            
        target = row[target_col]
        
        try:
            features = featurize(molecule, info_array)
            features_tensor = torch.FloatTensor(features)
            X.append(features_tensor)
            y.append(target)
            pdb_ids.append(pdb_id)
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # Only print first few errors
                print(f"  Warning: Failed to featurize {pdb_id}: {e}")
    
    if failed_count > 5:
        print(f"  Warning: Failed to load {failed_count} complexes total")
    
    print(f"✓ Successfully loaded {len(X)} complexes")
    return X, y, pdb_ids


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def apply_maxnorm_constraint(model, max_norm=3.0):
    """Apply MaxNorm constraint to parameters"""
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.dim() >= 2:
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=max_norm)
                param.mul_(desired / (norm + 1e-7))


def compute_l2_loss(model):
    """Compute L2 regularization loss explicitly for PCGrad"""
    l2_loss = torch.tensor(0., device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.sum(param ** 2)
    return l2_loss


def compute_task_losses(predictions, targets, physics_info, physics_weight):
    """Compute empirical and physics losses"""
    targets = targets.view(-1, 1)
    
    # Empirical loss (RMSE)
    empirical_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))
    
    # Extract energy components
    host_energy = physics_info[:, [0, 3, 6, 9, 12]].sum(dim=1, keepdim=True)
    guest_energy = physics_info[:, [1, 4, 7, 10, 13]].sum(dim=1, keepdim=True)
    complex_energy = physics_info[:, [2, 5, 8, 11, 14]].sum(dim=1, keepdim=True)
    
    # Physics consistency: ΔG = complex - (host + guest)
    dG_physics = complex_energy - (host_energy + guest_energy)
    
    # Physics loss (RMSE)
    raw_physics_loss = torch.sqrt(torch.mean((predictions - dG_physics) ** 2))
    weighted_physics_loss = physics_weight * raw_physics_loss
    
    return empirical_loss, weighted_physics_loss, raw_physics_loss


class EarlyStopping:
    """Early stopping with validation loss monitoring"""
    def __init__(self, patience=30, min_delta=0.001, verbose=True):
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
            if self.verbose and self.counter % 5 == 0:
                print(f'  Early stopping counter: {self.counter}/{self.patience}')
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
# TRAINING FUNCTION
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, 
                use_pcgrad=False, config=None, device='cpu'):
    """Train the improved PGGCN model"""
    if config is None:
        config = Config()
    
    model = model.to(device)
    
    # Setup optimizer
    if use_pcgrad and PCGRAD_AVAILABLE:
        base_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0)
        optimizer = PCGrad(base_optimizer)
        opt_name = "PCGrad + Adam"
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
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Optimizer: {opt_name}")
    print(f"Samples: {len(X_train)} train, {len(X_val)} val")
    print(f"Device: {device}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"L2 weight: {config.L2_WEIGHT}")
    print(f"MaxNorm: {config.MAX_NORM}")
    print(f"Dropout: {config.DROPOUT_RATE}")
    print(f"Physics weight: {config.PHYSICS_WEIGHT}")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"Max epochs: {config.EPOCHS}")
    print("=" * 80)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_empirical': [],
        'train_physics': [],
        'val_empirical': [],
        'val_physics': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    start_time = time.time()
    
    # Training loop
    for epoch in range(config.EPOCHS):
        # Training phase
        model.train()
        
        predictions, model_var, physics_info = model(X_train, training=True)
        
        train_empirical, train_weighted_physics, train_raw_physics = compute_task_losses(
            predictions, y_train_tensor, physics_info, config.PHYSICS_WEIGHT
        )
        
        # Backward pass
        if use_pcgrad and PCGRAD_AVAILABLE:
            l2_loss = compute_l2_loss(model)
            l2_weighted = config.L2_WEIGHT * l2_loss
            optimizer.pc_backward([train_empirical, l2_weighted, train_weighted_physics])
            optimizer.step()
        else:
            optimizer.zero_grad()
            combined_loss = train_empirical + train_weighted_physics
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        apply_maxnorm_constraint(model, max_norm=config.MAX_NORM)
        
        # Record training losses
        train_loss_total = train_empirical.item() + train_weighted_physics.item()
        history['train_loss'].append(train_loss_total)
        history['train_empirical'].append(train_empirical.item())
        history['train_physics'].append(train_raw_physics.item())
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_predictions, val_model_var, val_physics_info = model(X_val, training=False)
            val_empirical, val_weighted_physics, val_raw_physics = compute_task_losses(
                val_predictions, y_val_tensor, val_physics_info, config.PHYSICS_WEIGHT
            )
            val_loss_total = val_empirical.item() + val_weighted_physics.item()
            history['val_loss'].append(val_loss_total)
            history['val_empirical'].append(val_empirical.item())
            history['val_physics'].append(val_raw_physics.item())
        
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
                  f"(E: {train_empirical.item():.4f}, P: {train_raw_physics.item():.4f}) | "
                  f"Val: {val_loss_total:.4f} "
                  f"(E: {val_empirical.item():.4f}, P: {val_raw_physics.item():.4f}) | "
                  f"Best: {best_val_loss:.4f} @ {best_epoch} | "
                  f"ETA: {format_time(eta)}")
        
        # Check early stopping
        early_stopping(val_loss_total, model)
        if early_stopping.early_stop:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            early_stopping.restore_best_weights(model)
            break
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"Training completed in {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print("=" * 80)
    
    return history


# ============================================================================
# EVALUATION & VISUALIZATION
# ============================================================================

def evaluate_model(model, X_test, y_test, device='cpu'):
    """Evaluate model on test set"""
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
        
        _, _, phys_loss = compute_task_losses(predictions, y_test_tensor, 
                                             physics_info, 0.58)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'physics_loss': phys_loss.item()
    }
    
    return predictions.cpu().numpy(), metrics


def plot_training_history(history, save_path=None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.7)
    axes[0, 0].plot(history['val_loss'], label='Validation', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Empirical loss
    axes[0, 1].plot(history['train_empirical'], label='Train', alpha=0.7)
    axes[0, 1].plot(history['val_empirical'], label='Validation', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Empirical Loss (RMSE)')
    axes[0, 1].set_title('Empirical Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Physics loss
    axes[1, 0].plot(history['train_physics'], label='Train', alpha=0.7)
    axes[1, 0].plot(history['val_physics'], label='Validation', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Physics Loss (RMSE)')
    axes[1, 0].set_title('Physics Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss comparison
    epochs = range(len(history['train_loss']))
    axes[1, 1].plot(epochs, history['train_empirical'], label='Train Empirical', alpha=0.7)
    axes[1, 1].plot(epochs, history['val_empirical'], label='Val Empirical', alpha=0.7)
    axes[1, 1].plot(epochs, history['train_physics'], label='Train Physics', alpha=0.7, linestyle='--')
    axes[1, 1].plot(epochs, history['val_physics'], label='Val Physics', alpha=0.7, linestyle='--')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Loss Components')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train improved PGGCN model')
    
    # Dataset options
    parser.add_argument('--csv-path', type=str, default=None,
                       help='Path to CSV file with features and targets')
    parser.add_argument('--pkl-path', type=str, default=None,
                       help='Path to PKL file with molecular structures')
    parser.add_argument('--target-col', type=str, default=None,
                       help='Target column name (e.g., "Ex _G_(kcal/mol)" or "ddg")')
    parser.add_argument('--dataset-name', type=str, default='hostguest',
                       help='Dataset name for labeling outputs (e.g., "hostguest", "pdbbind")')
    
    # Training options
    parser.add_argument('--use-pcgrad', action='store_true',
                       help='Use PCGrad optimizer')
    parser.add_argument('--subset', type=int, default=None,
                       help='Use subset of data for quick testing (e.g., 20)')
    
    # Architecture options
    parser.add_argument('--c-out', type=int, default=256,
                       help='Conv layer output channels (default: 256)')
    parser.add_argument('--dense1', type=int, default=32,
                       help='Dense layer 1 size (default: 32)')
    parser.add_argument('--dense2', type=int, default=32,
                       help='Dense layer 2 size (default: 32)')
    parser.add_argument('--dense3', type=int, default=0,
                       help='Dense layer 3 size, 0=disabled (default: 0)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of epochs (default: 500)')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PYTORCH PGGCN TRAINING - IMPROVED VERSION")
    print("=" * 80)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    config = Config()
    
    # Update paths if provided
    if args.csv_path:
        config.CSV_PATH = args.csv_path
    if args.pkl_path:
        config.PDB_PATH = args.pkl_path
    if args.target_col:
        config.TARGET_COL = args.target_col
    
    # Update config with args
    config.C_OUT_CHANNEL = args.c_out
    config.DENSE1_SIZE = args.dense1
    config.DENSE2_SIZE = args.dense2
    config.DENSE3_SIZE = args.dense3
    config.DROPOUT_RATE = args.dropout
    config.EPOCHS = args.epochs
    config.EARLY_STOPPING_PATIENCE = args.patience
    config.LEARNING_RATE = args.lr
    config.SUBSET_SIZE = args.subset
    
    set_random_seeds(config.RANDOM_SEED)
    
    # Load data
    print("\n" + "-" * 80)
    print(f"Loading Data: {args.dataset_name.upper()}")
    print("-" * 80)
    print(f"CSV: {config.CSV_PATH}")
    print(f"PKL: {config.PDB_PATH}")
    print(f"Target: {config.TARGET_COL}")
    
    X, y, pdb_ids = load_data(config.CSV_PATH, config.PDB_PATH, 
                              target_col=config.TARGET_COL,
                              subset_size=config.SUBSET_SIZE)
    
    # Split data
    print("\n" + "-" * 80)
    print("Splitting Data")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    print(f"✓ Training: {len(X_train)} samples")
    print(f"✓ Test: {len(X_test)} samples")
    
    # Create model
    print("\n" + "-" * 80)
    print("Creating Model")
    print("-" * 80)
    model = ImprovedPGGCNModel(
        num_atom_features=config.NUM_ATOM_FEATURES,
        r_out_channel=config.R_OUT_CHANNEL,
        c_out_channel=config.C_OUT_CHANNEL,
        dense1_size=config.DENSE1_SIZE,
        dense2_size=config.DENSE2_SIZE,
        dense3_size=config.DENSE3_SIZE,
        dropout_rate=config.DROPOUT_RATE
    )
    
    model.add_rule("sum", 0, 32)
    model.add_rule("multiply", 32, 33)
    model.add_rule("distance", 33, 36)
    
    # Verify dense_final initialization
    expected = torch.tensor([0.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1])
    if torch.allclose(model.dense_final.weight.data.flatten(), expected, atol=0.01):
        print("✓ dense_final weights correctly initialized")
    
    model = model.to(device)
    
    # Train
    print("\n" + "-" * 80)
    print("Training")
    print("-" * 80)
    history = train_model(model, X_train, y_train, X_test, y_test,
                        use_pcgrad=args.use_pcgrad, config=config, device=device)
    
    # Evaluate
    print("\n" + "-" * 80)
    print("Evaluation")
    print("-" * 80)
    predictions, metrics = evaluate_model(model, X_test, y_test, device=device)
    
    print(f"\nTest Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  Physics Loss: {metrics['physics_loss']:.4f}")
    
    # Save model
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = f"pggcn_{args.dataset_name}_{'pcgrad' if args.use_pcgrad else 'standard'}"
    model_name += f"_c{args.c_out}_d{args.dense1}_{args.dense2}"
    if args.dense3 > 0:
        model_name += f"_{args.dense3}"
    model_name += f"_drop{args.dropout}_{timestamp}"
    
    save_path = os.path.join(config.SAVE_DIR, f"{model_name}.pth")
    plot_path = os.path.join(config.PLOT_DIR, f"{model_name}.png")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'metrics': metrics,
        'config': vars(config),
        'args': vars(args),
        'dataset_name': args.dataset_name
    }, save_path)
    
    print(f"\n✓ Model saved to: {save_path}")
    
    # Plot training curves
    plot_training_history(history, save_path=plot_path)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()