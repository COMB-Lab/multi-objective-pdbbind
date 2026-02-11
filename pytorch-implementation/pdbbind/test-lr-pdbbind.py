"""
PCGrad PDBBind Training - Learning Rate Sweep

This script tests multiple learning rates for the PGGCN model on PDBBind dataset
and saves results for comparison.
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
import argparse
import json
from datetime import timedelta
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind')

from models.dcFeaturizer import atom_features as get_atom_features
from models.layers_pytorch_pdbbind import PGGCNModel
from models.pcgrad_pytorch import PCGrad

PCGRAD_AVAILABLE = True


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    CSV_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind_100.csv'
    PKL_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_100.pkl'
    
    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL = 20
    #C_OUT_CHANNEL = 1024
    C_OUT_CHANNEL = 128
    DROPOUT_RATE = 0.2
    
    EPOCHS = 100
    BATCH_SIZE = 8
    #LEARNING_RATES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]  # Test these learning rates
    LEARNING_RATES = [5e-5]
    L2_WEIGHT = 1e-2
    MAX_NORM = 3.0
    
    PHYSICS_WEIGHT = 1e-6
    
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
    
    for csv_path, pkl_path in [
        (config.CSV_PATH, config.PKL_PATH),
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
    
    # MAE for monitoring
    mae = torch.mean(torch.abs(predictions - targets))
    
    return empirical_loss, weighted_physics_loss, raw_physics_loss, mae


# ============================================================================
# TRAINING
# ============================================================================

def train_model_batched(model, train_loader, val_loader, learning_rate, config, device, use_pcgrad=True):
    """
    Train with batching (for PDBBind)
    
    PAPER CORRECTION: PCGrad uses only 2 objectives [empirical, physics]
    L2 regularization is handled via weight_decay in Adam optimizer
    """
    
    model = model.to(device)
    
    # Setup optimizer
    if use_pcgrad and PCGRAD_AVAILABLE:
        base_optimizer = optim.Adam(model.parameters(), 
                                    lr=learning_rate, 
                                    weight_decay=config.L2_WEIGHT)
        optimizer = PCGrad(base_optimizer)
        opt_name = "PCGrad + Adam"
    else:
        base_optimizer = optim.Adam(model.parameters(), 
                                    lr=learning_rate, 
                                    weight_decay=config.L2_WEIGHT)
        optimizer = base_optimizer
        opt_name = "Adam (standard)"
    
    print(f"\n{'='*80}")
    print(f"TRAINING WITH LEARNING RATE: {learning_rate}")
    print("="*80)
    print(f"Optimizer: {opt_name}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"L2 regularization: {config.L2_WEIGHT}")
    print(f"Physics weight: {config.PHYSICS_WEIGHT}")
    print(f"Max epochs: {config.EPOCHS}")
    print("="*80)
    
    history = {
        'train_losses': [], 'val_losses': [],
        'train_empirical': [], 'train_physics': [],
        'val_empirical': [], 'val_physics': [],
        'train_mae': [], 'val_mae': []
    }
    
    best_val_mae = float('inf')
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        epoch_metrics = {'emp': [], 'phys': [], 'mae': []}
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            
            # Forward
            predictions, _, physics_info = model(X_batch, training=True)
            
            # Compute losses
            train_emp, train_phys_w, train_phys_r, train_mae = compute_task_losses(
                predictions, y_batch, physics_info, config.PHYSICS_WEIGHT)
            
            # Backward
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
            
            epoch_metrics['emp'].append(train_emp.item())
            epoch_metrics['phys'].append(train_phys_r.item())
            epoch_metrics['mae'].append(train_mae.item())
        
        # Average training metrics
        avg_train_emp = np.mean(epoch_metrics['emp'])
        avg_train_phys = np.mean(epoch_metrics['phys'])
        avg_train_mae = np.mean(epoch_metrics['mae'])
        
        history['train_empirical'].append(avg_train_emp)
        history['train_physics'].append(avg_train_phys)
        history['train_mae'].append(avg_train_mae)
        
        # Validation
        model.eval()
        with torch.no_grad():
            all_val_preds, all_val_targets, all_val_phys = [], [], []
            
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
            
            val_emp, val_phys_w, val_phys_r, val_mae = compute_task_losses(
                val_predictions, val_targets, val_physics, config.PHYSICS_WEIGHT)
            
            history['val_empirical'].append(val_emp.item())
            history['val_physics'].append(val_phys_r.item())
            history['val_mae'].append(val_mae.item())
        
        if val_mae.item() < best_val_mae:
            best_val_mae = val_mae.item()
            best_epoch = epoch + 1
        
        # Progress (every 10 epochs)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (config.EPOCHS - epoch - 1)
            
            print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train MAE: {avg_train_mae:6.2f} | "
                  f"Val MAE: {val_mae.item():6.2f} | "
                  f"Best: {best_val_mae:6.2f} @ {best_epoch} | "
                  f"ETA: {str(timedelta(seconds=int(eta)))}")
    
    total_time = time.time() - start_time
    print(f"Completed in {str(timedelta(seconds=int(total_time)))}")
    print(f"Best validation MAE: {best_val_mae:.2f} at epoch {best_epoch}")
    
    return history, best_val_mae, best_epoch


def evaluate_model(model, test_loader, device):
    model.eval()
    
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            
            predictions, _, _ = model(X_batch, training=False)
            all_preds.append(predictions)
            all_targets.append(y_batch)
    
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    ss_res = torch.sum((targets - predictions) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot)
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


# ============================================================================
# MAIN - LEARNING RATE SWEEP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test multiple learning rates for PGGCN')
    parser.add_argument('--physics-weight', type=float, default=1e-6,
                       help='Physics loss weight (default: 1e-6)')
    parser.add_argument('--no-pcgrad', action='store_true',
                       help='Use standard optimizer instead of PCGrad')
    args = parser.parse_args()
    
    # Set random seeds
    import random
    torch.manual_seed(50)
    torch.cuda.manual_seed_all(50)
    np.random.seed(50)
    random.seed(50)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print("="*80)
    print("LEARNING RATE SWEEP - PGGCN PDBBIND")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    config = Config()
    config.PHYSICS_WEIGHT = args.physics_weight
    
    # Use learning rates from config
    learning_rates = config.LEARNING_RATES
    print(f"\nTesting {len(learning_rates)} learning rates: {learning_rates}")
    
    # Load data once
    X, y = load_data(config)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    
    print(f"\n✓ Training: {len(X_train)} samples")
    print(f"✓ Test: {len(X_test)} samples")
    
    # Create dataloaders
    train_dataset = MoleculeDataset(X_train, y_train)
    test_dataset = MoleculeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, collate_fn=collate_molecules)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=False, collate_fn=collate_molecules)
    
    # Store results
    all_results = []
    use_pcgrad = not args.no_pcgrad
    
    # Test each learning rate
    for lr_idx, lr in enumerate(learning_rates):
        print(f"\n{'#'*80}")
        print(f"TESTING LEARNING RATE {lr_idx + 1}/{len(learning_rates)}: {lr}")
        print(f"{'#'*80}")
        
        # Create fresh model for each LR
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
        history, best_val_mae, best_epoch = train_model_batched(
            model, train_loader, test_loader, lr, config, device, use_pcgrad=use_pcgrad
        )
        
        # Evaluate on test set
        metrics = evaluate_model(model, test_loader, device)
        
        print(f"\nResults for LR={lr}:")
        print(f"  Test RMSE: {metrics['rmse']:.2f} kcal/mol")
        print(f"  Test MAE:  {metrics['mae']:.2f} kcal/mol")
        print(f"  Test R²:   {metrics['r2']:.4f}")
        print(f"  Best Val MAE: {best_val_mae:.2f} @ epoch {best_epoch}")
        
        # Store results
        result = {
            'learning_rate': lr,
            'test_rmse': metrics['rmse'],
            'test_mae': metrics['mae'],
            'test_r2': metrics['r2'],
            'best_val_mae': best_val_mae,
            'best_epoch': best_epoch,
            'history': history
        }
        all_results.append(result)
        
        # Save individual model
        save_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind/saved_models/learning-rate-tests/'
        os.makedirs(save_dir, exist_ok=True)
        
        method = 'pcgrad' if use_pcgrad else 'standard'
        filename = f"pggcn_pdbbind_{method}_lr{lr}_{timestamp}.pth"
        save_path = os.path.join(save_dir, filename)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,  # Full training history saved
            'metrics': metrics,
            'learning_rate': lr,
            'best_val_mae': best_val_mae,
            'best_epoch': best_epoch,
            'timestamp': timestamp,
            'config': {
                'physics_weight': config.PHYSICS_WEIGHT,
                'learning_rate': lr,
                'l2_weight': config.L2_WEIGHT,
                'batch_size': config.BATCH_SIZE,
                'epochs': config.EPOCHS,
                'use_pcgrad': use_pcgrad,
            }
        }, save_path)
        print(f"✓ Saved model to: {save_path}")
    
    # ========================================================================
    # SUMMARY OF ALL RESULTS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("LEARNING RATE SWEEP SUMMARY")
    print("="*80)
    print(f"{'LR':<12} | {'Test MAE':<10} | {'Test RMSE':<10} | {'Test R²':<10} | {'Best Val MAE':<12} | {'Best Epoch':<10}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['learning_rate']:<12.2e} | "
              f"{result['test_mae']:<10.2f} | "
              f"{result['test_rmse']:<10.2f} | "
              f"{result['test_r2']:<10.4f} | "
              f"{result['best_val_mae']:<12.2f} | "
              f"{result['best_epoch']:<10}")
    
    # Find best learning rate
    best_result = min(all_results, key=lambda x: x['test_mae'])
    print("\n" + "="*80)
    print(f"BEST LEARNING RATE: {best_result['learning_rate']:.2e}")
    print(f"  Test MAE:  {best_result['test_mae']:.2f} kcal/mol")
    print(f"  Test RMSE: {best_result['test_rmse']:.2f} kcal/mol")
    print(f"  Test R²:   {best_result['test_r2']:.4f}")
    print("="*80)
    
    # Save summary
    summary_path = os.path.join(save_dir, f'lr_sweep_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        # Convert history to serializable format
        summary = []
        for r in all_results:
            summary.append({
                'learning_rate': r['learning_rate'],
                'test_rmse': r['test_rmse'],
                'test_mae': r['test_mae'],
                'test_r2': r['test_r2'],
                'best_val_mae': r['best_val_mae'],
                'best_epoch': r['best_epoch']
            })
        json.dump({
            'timestamp': timestamp,
            'method': method,
            'results': summary,
            'best_lr': best_result['learning_rate']
        }, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()