"""
PDBBind Training - Three-Way Comparison
Tests: PCGrad+Physics, Adam+Physics, Empirical-Only

Creates table like:
model,rmse,mae,r2,physics_loss
ΔG with PCGrad + Multi-loss,...
ΔG with Adam + Multi-loss,...
ΔG without Multi-loss,...
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
import csv
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind')
sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind')

from models.dcFeaturizer import atom_features as get_atom_features
from models.layers_pytorch_pdbbind import PGGCNModel
from models.pcgrad_pytorch import PCGrad
# Clustered data based on size
from train_split_data import load_data_with_saved_split

PCGRAD_AVAILABLE = True


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    CSV_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind.csv'
    PKL_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_full.pkl'
    
    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL = 20
    C_OUT_CHANNEL = 1024
    DROPOUT_RATE = 0.2
    
    EPOCHS = 250
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    L2_WEIGHT = 1e-2
    MAX_NORM = 3.0
    PHYSICS_WEIGHT = 0.73  # For physics-enabled runs
    
    TEST_SIZE = 0.2
    RANDOM_SEED = 50


# ============================================================================
# DATASET
# ============================================================================

class MoleculeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_molecules(batch):
    X_batch = [item[0] for item in batch]
    y_batch = [item[1] for item in batch]
    return X_batch, torch.FloatTensor(y_batch)


# ============================================================================
# DATA LOADING
# ============================================================================

def featurize(molecule, info):
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
    print("Loading PDBBind dataset...")
    
    csv_path = config.CSV_PATH
    pkl_path = config.PKL_PATH
    
    if not os.path.exists(csv_path) or not os.path.exists(pkl_path):
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
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.dim() >= 2:
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=max_norm)
                param.mul_(desired / (norm + 1e-7))


def compute_task_losses(predictions, targets, physics_info, physics_weight):
    targets = targets.view(-1, 1)
    
    # Empirical loss (RMSE)
    empirical_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))
    
    # Extract energy components
    host_energy = physics_info[:, [0, 3, 6, 9, 12]].sum(dim=1, keepdim=True)
    guest_energy = physics_info[:, [1, 4, 7, 10, 13]].sum(dim=1, keepdim=True)
    complex_energy = physics_info[:, [2, 5, 8, 11, 14]].sum(dim=1, keepdim=True)
    dG_physics = complex_energy - (host_energy + guest_energy)
    
    raw_physics_loss = torch.sqrt(torch.mean((predictions - dG_physics) ** 2))
    weighted_physics_loss = physics_weight * raw_physics_loss
    mae = torch.mean(torch.abs(predictions - targets))
    
    return empirical_loss, weighted_physics_loss, raw_physics_loss, mae


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, use_pcgrad, use_physics, config, device):
    """
    Train model with specified configuration
    
    Args:
        use_pcgrad: True to use PCGrad, False for standard Adam
        use_physics: True to include physics loss, False for empirical only
    """
    model = model.to(device)
    
    # Determine physics weight
    physics_weight = config.PHYSICS_WEIGHT if use_physics else 0.0
    
    # Setup optimizer
    if use_pcgrad and PCGRAD_AVAILABLE and use_physics:
        base_optimizer = optim.Adam(model.parameters(), 
                                    lr=config.LEARNING_RATE, 
                                    weight_decay=config.L2_WEIGHT)
        optimizer = PCGrad(base_optimizer)
        opt_name = "PCGrad + Adam"
    else:
        optimizer = optim.Adam(model.parameters(), 
                              lr=config.LEARNING_RATE, 
                              weight_decay=config.L2_WEIGHT)
        opt_name = "Adam"
    
    print(f"\n{'='*80}")
    print(f"Optimizer: {opt_name}")
    print(f"Physics loss: {'ENABLED (weight={physics_weight})' if use_physics else 'DISABLED'}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            
            predictions, _, physics_info = model(X_batch, training=True)
            train_emp, train_phys_w, _, _ = compute_task_losses(
                predictions, y_batch, physics_info, physics_weight)
            
            if use_pcgrad and PCGRAD_AVAILABLE and use_physics:
                optimizer.pc_backward([train_emp, train_phys_w])
                optimizer.step()
            else:
                optimizer.zero_grad()
                if use_physics:
                    total_loss = train_emp + train_phys_w
                else:
                    total_loss = train_emp
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            apply_maxnorm_constraint(model, config.MAX_NORM)
        
        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (config.EPOCHS - epoch - 1)
            print(f"  Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Time: {str(timedelta(seconds=int(elapsed)))} | "
                  f"ETA: {str(timedelta(seconds=int(eta)))}")
    
    total_time = time.time() - start_time
    print(f"Completed in {str(timedelta(seconds=int(total_time)))}")
    
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model and return all metrics"""
    model.eval()
    
    all_preds, all_targets, all_physics = [], [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            
            predictions, _, physics_info = model(X_batch, training=False)
            all_preds.append(predictions)
            all_targets.append(y_batch)
            all_physics.append(physics_info)
    
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    physics_info = torch.cat(all_physics, dim=0)
    
    # Compute all metrics
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    ss_res = torch.sum((targets - predictions) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot)
    
    # Always compute physics loss (for comparison even if not used in training)
    _, _, raw_physics_loss, _ = compute_task_losses(
        predictions, targets, physics_info, physics_weight=1.0)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'physics_loss': raw_physics_loss.item()
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Set seeds
    import random
    torch.manual_seed(50)
    torch.cuda.manual_seed_all(50)
    np.random.seed(50)
    random.seed(50)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    print("="*80)
    print("PDBBIND THREE-WAY COMPARISON")
    print("PCGrad+Physics vs Adam+Physics vs Empirical-Only")
    print("="*80)
    print(f"Timestamp: {timestamp}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    config = Config()
    
    # Load data
    #X, y = load_data(config)
    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    
    # Load clustered data
    split_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind_stratified_split.pkl'
    X_train, X_test, y_train, y_test = load_data_with_saved_split(config, split_path)

    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples\n")
    
    # Create dataloaders
    train_dataset = MoleculeDataset(X_train, y_train)
    test_dataset = MoleculeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, collate_fn=collate_molecules)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=False, collate_fn=collate_molecules)
    
    # Define three configurations to test
    configs = [
        ('ΔG with PCGrad + Multi-loss', True, True),   # PCGrad + Physics
        ('ΔG with Adam + Multi-loss', False, True),    # Adam + Physics
        ('ΔG without Multi-loss', False, False),       # Empirical only
    ]
    
    results = []
    
    # Test each configuration
    for run_name, use_pcgrad, use_physics in configs:
        print(f"\n{'='*80}")
        print(f"TRAINING: {run_name}")
        print(f"{'='*80}")
        
        # Create fresh model
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
        model = train_model(model, train_loader, test_loader, 
                           use_pcgrad, use_physics, config, device)
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        
        print(f"\nResults:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  Physics Loss: {metrics['physics_loss']:.4f}")
        
        # Store results
        results.append({
            'model': run_name,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'physics_loss': metrics['physics_loss']
        })
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<35} | {'RMSE':<10} | {'MAE':<10} | {'R²':<10} | {'Physics Loss':<12}")
    print("-" * 90)
    for r in results:
        print(f"{r['model']:<35} | "
              f"{r['rmse']:<10.4f} | "
              f"{r['mae']:<10.4f} | "
              f"{r['r2']:<10.4f} | "
              f"{r['physics_loss']:<12.4f}")
    
    # Save to CSV
    save_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind/comparison'
    os.makedirs(save_dir, exist_ok=True)
    
    csv_path = os.path.join(save_dir, f'pdbbind_comparison_{timestamp}.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'rmse', 'mae', 'r2', 'physics_loss'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    pcgrad = results[0]
    adam = results[1]
    empirical = results[2]
    
    print("\n1. Empirical Performance (RMSE):")
    print(f"   Best: {empirical['model']} = {empirical['rmse']:.4f}")
    print(f"   PCGrad+Physics: {pcgrad['rmse']:.4f} ({100*(pcgrad['rmse']/empirical['rmse']-1):+.1f}%)")
    print(f"   Adam+Physics: {adam['rmse']:.4f} ({100*(adam['rmse']/empirical['rmse']-1):+.1f}%)")
    
    print("\n2. Physics Consistency:")
    print(f"   Best: {min(results, key=lambda x: x['physics_loss'])['model']}")
    print(f"   PCGrad+Physics: {pcgrad['physics_loss']:.4f}")
    print(f"   Adam+Physics: {adam['physics_loss']:.4f}")
    print(f"   Empirical-only: {empirical['physics_loss']:.4f}")
    
    print("\n3. Key Observations:")
    if empirical['rmse'] < pcgrad['rmse']:
        print("   ⚠️  Empirical-only has LOWER RMSE than physics-informed models")
        print("      This suggests:")
        print("      - Physics loss may be hurting empirical performance")
        print("      - Physics weight may need tuning")
        print("      - Physics assumptions may not match PDBBind data")
    else:
        print("   ✅ Physics-informed models achieve better empirical performance")
    
    if pcgrad['physics_loss'] < empirical['physics_loss']:
        print("\n   ✅ PCGrad improves physics consistency vs empirical-only")
    else:
        print("\n   ⚠️  PCGrad does not improve physics consistency")
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()