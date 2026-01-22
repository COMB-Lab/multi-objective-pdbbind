"""
IMPORTANT: PDBbind physics features have different scale than host-guest!
- Host-guest: Physics weight 0.58 works well
- PDBbind: Need 1e-5 to 1e-6 (physics loss is ~46 vs ~13)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import os
import pickle
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split

# Setup paths
sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind')

try:
    from PGGCN.models.dcFeaturizer import atom_features as get_atom_features
    from PGGCN.models.layers_pytorch import PGGCNModel
except ImportError:
    from models.dcFeaturizer import atom_features as get_atom_features
    from models.layers_pytorch import PGGCNModel


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data paths
    CSV_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind_100.csv'
    PKL_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_100.pkl'
    
    # Model architecture
    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL = 20
    C_OUT_CHANNEL = 128
    DROPOUT_RATE = 0.2
    
    # Training hyperparameters
    EPOCHS = 250
    LEARNING_RATE = 1e-4
    L2_WEIGHT = 1e-4
    MAX_NORM = 3.0
    
    # Physics weight (might need to use different weights for PDBbind)
    PHYSICS_WEIGHT = 1e-5
    # Dataset
    MAX_SAMPLES = 50  # Starting at 50 to test
    TEST_SIZE = 0.2
    RANDOM_SEED = 42


# ============================================================================
# UTILITIES
# ============================================================================

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def featurize(molecule, info):
    """Featurize molecule with physics info"""
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


def load_data(config):
    """Load PDBbind data"""
    print(f"Loading PDBbind-100 dataset...")
    
    # Load files
    df = pd.read_csv(config.CSV_PATH)
    with open(config.PKL_PATH, 'rb') as f:
        pdb_dict = pickle.load(f)
    
    print(f"Loaded {len(df)} CSV entries, {len(pdb_dict)} PDB structures")
    
    # Clean data
    df = df.dropna(subset=['ddg'])
    df = df[df['complex-name'].apply(lambda x: 'E+' not in str(x))]
    
    # Limit samples
    if config.MAX_SAMPLES:
        df = df.head(config.MAX_SAMPLES)
        print(f"Limited to {config.MAX_SAMPLES} samples")
    
    # Physics columns
    physics_columns = [
        'pb-protein-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals',
        'gb-protein-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
        'gb-protein-eelect', 'gb-ligand-eelec', 'gb-complex-eelec',
        'gb-protein-egb', 'gb-ligand-egb', 'gb-complex-egb',
        'gb-protein-esurf', 'gb-ligand-esurf', 'gb-complex-esurf'
    ]
    
    # Match PDB structures
    common_keys = set(df['complex-name']) & set(pdb_dict.keys())
    df = df[df['complex-name'].isin(common_keys)]
    
    print(f"Final dataset: {len(df)} structures")
    
    # Featurize
    X, y = [], []
    for pdb_id in df['complex-name']:
        row = df[df['complex-name'] == pdb_id].iloc[0]
        info_array = row[physics_columns].tolist()
        target = row['ddg']
        
        try:
            features = featurize(pdb_dict[pdb_id], info_array)
            X.append(torch.FloatTensor(features))
            y.append(target)
        except Exception as e:
            print(f"Failed to featurize {pdb_id}: {e}")
    
    print(f"Successfully featurized {len(X)} structures")
    return X, y


def compute_task_losses(predictions, targets, physics_info, physics_weight):
    """Compute empirical and physics losses"""
    targets = targets.view(-1, 1)
    
    # Empirical loss
    empirical_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))
    
    # Physics calculation
    host_energy = physics_info[:, [0, 3, 6, 9, 12]].sum(dim=1, keepdim=True)
    guest_energy = physics_info[:, [1, 4, 7, 10, 13]].sum(dim=1, keepdim=True)
    complex_energy = physics_info[:, [2, 5, 8, 11, 14]].sum(dim=1, keepdim=True)
    
    dG_physics = complex_energy - (host_energy + guest_energy)
    
    # Physics loss
    raw_physics_loss = torch.sqrt(torch.mean((predictions - dG_physics) ** 2))
    weighted_physics_loss = physics_weight * raw_physics_loss
    
    return empirical_loss, weighted_physics_loss, raw_physics_loss


def compute_l2_loss(model, l2_weight):
    """L2 regularization"""
    l2_loss = torch.tensor(0., device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.sum(param ** 2)
    return (l2_weight / 2) * l2_loss


def apply_maxnorm_constraint(model, max_norm=3.0):
    """MaxNorm constraint"""
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.dim() >= 2:
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=max_norm)
                param.mul_(desired / (norm + 1e-7))


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_test, y_test, config, device):
    """Train model with corrected hyperparameters"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    X_train = [x.to(device) for x in X_train]
    X_test = [x.to(device) for x in X_test]
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Samples: {len(X_train)} train, {len(X_test)} test")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Physics weight: {config.PHYSICS_WEIGHT}")
    print(f"Architecture: c_out={config.C_OUT_CHANNEL}")
    print(f"L2 weight: {config.L2_WEIGHT}")
    print(f"Max epochs: {config.EPOCHS}")
    print("="*80)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        optimizer.zero_grad()
        
        predictions, model_var, physics_info = model(X_train, training=True)
        
        train_empirical, train_weighted_physics, train_raw_physics = compute_task_losses(
            predictions, y_train_tensor, physics_info, config.PHYSICS_WEIGHT)
        
        l2_reg = compute_l2_loss(model, config.L2_WEIGHT)
        total_loss = train_empirical + train_weighted_physics + l2_reg
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        apply_maxnorm_constraint(model, max_norm=config.MAX_NORM)
        
        train_losses.append(total_loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions, model_var, val_physics_info = model(X_test, training=False)
            val_empirical, val_weighted_physics, val_raw_physics = compute_task_losses(
                val_predictions, y_test_tensor, val_physics_info, config.PHYSICS_WEIGHT)
            
            val_total = val_empirical + val_weighted_physics
            val_losses.append(val_total.item())
            
            if val_total.item() < best_val_loss:
                best_val_loss = val_total.item()
        
        # Progress
        if (epoch + 1) % 25 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (config.EPOCHS - epoch - 1)
            
            print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train: {train_empirical.item():.2f} "
                  f"(Phys: {train_raw_physics.item():.2f}) | "
                  f"Val: {val_empirical.item():.2f} | "
                  f"ETA: {format_time(eta)}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {format_time(training_time)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses


def evaluate_model(model, X_test, y_test, device):
    """Evaluate model"""
    model.eval()
    X_test = [x.to(device) for x in X_test]
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    with torch.no_grad():
        predictions, model_var, val_physics_info = model(X_test, training=False)
        
        rmse = torch.sqrt(torch.mean((predictions - y_test_tensor) ** 2)).item()
        mae = torch.mean(torch.abs(predictions - y_test_tensor)).item()
        
        ss_res = torch.sum((y_test_tensor - predictions) ** 2).item()
        ss_tot = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("CORRECTED PDBBIND-100 TRAINING")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    config = Config()
    
    # Load data
    X, y = load_data(config)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Create model with CORRECT architecture
    model = PGGCNModel(
        num_atom_features=config.NUM_ATOM_FEATURES,
        r_out_channel=config.R_OUT_CHANNEL,
        c_out_channel=config.C_OUT_CHANNEL,
        dropout_rate=config.DROPOUT_RATE
    )
    model.add_rule("sum", 0, 32)
    model.add_rule("multiply", 32, 33)
    model.add_rule("distance", 33, 36)
    
    # Train
    train_losses, val_losses = train_model(
        model, X_train, y_train, X_test, y_test, config, device)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, device)
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print("="*80)
    print(f"Test RMSE: {metrics['rmse']:.4f} kcal/mol")
    print(f"Test MAE:  {metrics['mae']:.4f} kcal/mol")
    print(f"Test R²:   {metrics['r2']:.4f}")
    
    print(f"\nCOMPARISON TO OLD MODEL:")
    print(f"  Old RMSE: 300.92 → New RMSE: {metrics['rmse']:.2f}")
    print(f"  Old MAE:  175.55 → New MAE:  {metrics['mae']:.2f}")
    print(f"  Old R²:   -10506 → New R²:   {metrics['r2']:.2f}")
    
    # Save
    save_path = f'/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/saved_models/pdbbind_corrected_{config.MAX_SAMPLES}samples.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics': metrics,
        'config': {
            'learning_rate': config.LEARNING_RATE,
            'physics_weight': config.PHYSICS_WEIGHT,
            'l2_weight': config.L2_WEIGHT,
            'c_out_channel': config.C_OUT_CHANNEL,
            'max_samples': config.MAX_SAMPLES
        }
    }, save_path)
    
    print(f"\n✓ Model saved to: {save_path}")
    print("="*80)


if __name__ == "__main__":
    main()