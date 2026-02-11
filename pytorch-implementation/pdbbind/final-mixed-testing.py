"""
PyTorch PGGCN Training Script - PDBBind Dataset with Enhanced Reporting

This script implements training of a PGGCN model on the PDBBind dataset
with options to compare PCGrad vs standard Adam optimizer.
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
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind')

from models.dcFeaturizer import atom_features as get_atom_features
from models.layers_pytorch_pdbbind import PGGCNModel

# Try to import PCGrad
try:
    from models.pcgrad_pytorch import PCGrad
    PCGRAD_AVAILABLE = True
    print("✓ PCGrad optimizer available")
except ImportError:
    PCGRAD_AVAILABLE = False
    print("✗ PCGrad not available (will use standard optimizer)")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    # Data paths
    CSV_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind_100.csv'
    PKL_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_100.pkl'
    
    # Model architecture
    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL = 20
    C_OUT_CHANNEL = 1024
    DROPOUT_RATE = 0.2
    
    # Training hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5  # Best from sweep
    L2_WEIGHT = 1e-2
    MAX_NORM = 3.0
    PHYSICS_WEIGHT = 1e-6
    
    # Data split
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
# UTILITY FUNCTIONS
# ============================================================================

def set_random_seeds(seed=50):
    """Set random seeds for reproducibility"""
    import random
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


def load_data(config):
    """Load PDBBind dataset from CSV and PKL files"""
    print("\n" + "-" * 80)
    print("Loading Data")
    print("-" * 80)
    
    csv_path = config.CSV_PATH
    pkl_path = config.PKL_PATH
    
    if not os.path.exists(csv_path) or not os.path.exists(pkl_path):
        raise FileNotFoundError("Could not find PDBBind data!")
    
    print(f"✓ Found files at: {os.path.dirname(csv_path)}")
    
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
    """Apply MaxNorm constraint to parameters"""
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
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, val_loader, use_pcgrad, config, device):
    """
    Train the PGGCN model
    
    Args:
        use_pcgrad: If True, use PCGrad optimizer. If False, use standard Adam.
    """
    model = model.to(device)
    
    # Setup optimizer
    if use_pcgrad and PCGRAD_AVAILABLE:
        base_optimizer = optim.Adam(model.parameters(), 
                                    lr=config.LEARNING_RATE, 
                                    weight_decay=config.L2_WEIGHT)
        optimizer = PCGrad(base_optimizer)
        opt_name = "PCGrad + Adam"
    else:
        base_optimizer = optim.Adam(model.parameters(), 
                                    lr=config.LEARNING_RATE, 
                                    weight_decay=config.L2_WEIGHT)
        optimizer = base_optimizer
        opt_name = "Adam (standard)"
    
    # Print configuration
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Optimizer: {opt_name}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Batches per epoch: ~{len(train_loader)}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"L2 regularization: {config.L2_WEIGHT} (via weight_decay)")
    print(f"Physics weight: {config.PHYSICS_WEIGHT}")
    print(f"MaxNorm constraint: {config.MAX_NORM}")
    print(f"Dropout rate: {config.DROPOUT_RATE}")
    print(f"Physics formula: complex - (host + guest)")
    if use_pcgrad and PCGRAD_AVAILABLE:
        print(f"PCGrad objectives: [empirical, physics]")
    print(f"Max epochs: {config.EPOCHS}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Training history
    history = {
        'train_losses': [], 'val_losses': [],
        'train_empirical': [], 'train_physics': [],
        'val_empirical': [], 'val_physics': [],
        'train_mae': [], 'val_mae': []
    }
    
    best_val_mae = float('inf')
    best_epoch = 0
    start_time = time.time()
    
    # Training loop
    for epoch in range(config.EPOCHS):
        # Training phase
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
        history['train_losses'].append(avg_train_emp + config.PHYSICS_WEIGHT * avg_train_phys)
        
        # Validation phase
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
            history['val_losses'].append(val_emp.item() + val_phys_w.item())
        
        if val_mae.item() < best_val_mae:
            best_val_mae = val_mae.item()
            best_epoch = epoch + 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (config.EPOCHS - epoch - 1)
            
            print(f"Epoch {epoch+1:3d}/{config.EPOCHS}")
            print(f"  Train -> Total: {history['train_losses'][-1]:.4f}, RMSE: {avg_train_emp:.4f}, Phys: {avg_train_phys:.4f}, MAE: {avg_train_mae:.4f}")
            print(f"  Val   -> Total: {history['val_losses'][-1]:.4f}, RMSE: {val_emp.item():.4f}, Phys: {val_phys_r.item():.4f}, MAE: {val_mae.item():.4f}")
            print(f"  Time: {format_time(elapsed)} | ETA: {format_time(eta)}")
            print("-" * 80)
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"Training completed in {format_time(total_time)}")
    print(f"Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}")
    print("=" * 80)
    
    return history, best_val_mae, best_epoch


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
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
    
    # Compute metrics
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    ss_res = torch.sum((targets - predictions) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot)
    
    return predictions.cpu().numpy(), {'rmse': rmse, 'mae': mae, 'r2': r2}


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


def print_comparison_table(results):
    """
    Print a comparison table similar to the paper
    
    Args:
        results: Dictionary with keys like 'PCGrad', 'Adam'
                 Each containing 'metrics' and 'history' dictionaries
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    
    # Calculate statistics for each model
    table_data = []
    
    for model_name, data in results.items():
        history = data['history']
        metrics = data['metrics']
        
        # Get final losses (mean of last 10 epochs for stability)
        train_losses = history['train_empirical'][-10:]
        test_losses = history['val_empirical'][-10:]
        
        mean_train = np.mean(train_losses)
        std_train = np.std(train_losses)
        mean_test = np.mean(test_losses)
        std_test = np.std(test_losses)
        mae = metrics['mae']
        rmse = metrics['rmse']
        r2 = metrics['r2']
        
        table_data.append({
            'model': model_name,
            'mean_train': mean_train,
            'std_train': std_train,
            'mean_test': mean_test,
            'std_test': std_test,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
    
    # Print ASCII table
    print(f"\n{'Model Type':<30} | {'Mean Train Loss':<20} | {'Mean Test Loss':<20} | {'Test MAE':<10} | {'Test RMSE':<10} | {'R²':<10}")
    print("-" * 120)
    
    for row in table_data:
        print(f"{row['model']:<30} | "
              f"{row['mean_train']:.2f}±{row['std_train']:.2f}{'':<12} | "
              f"{row['mean_test']:.2f}±{row['std_test']:.2f}{'':<12} | "
              f"{row['mae']:.2f}{'':<4} | "
              f"{row['rmse']:.2f}{'':<4} | "
              f"{row['r2']:.4f}")
    
    print("\n" + "=" * 80)
    
    # Print LaTeX version
    print("\nLaTeX Table Format:")
    print("-" * 80)
    print(r"\begin{table}[ht!]")
    print(r"  \caption{Performance comparison for binding affinity ($\Delta G$) prediction on PDBBind (kcal/mol).}")
    print(r"  \label{tab:pdbbind_comparison}")
    print(r"  \centering")
    print(r"  \small")
    print(r"  \begin{tabular}{p{4.0cm} p{2.0cm} p{2.0cm} p{1.5cm} p{1.5cm}}")
    print(r"    \toprule")
    print(r"    \textbf{Model Type} & \textbf{Mean Train Loss} & \textbf{Mean Test Loss} & \textbf{MAE} & \textbf{RMSE} \\")
    print(r"    \midrule")
    
    for row in table_data:
        print(f"    {row['model']} & "
              f"{row['mean_train']:.2f}$\\pm${row['std_train']:.2f} & "
              f"{row['mean_test']:.2f}$\\pm${row['std_test']:.2f} & "
              f"{row['mae']:.2f} & "
              f"{row['rmse']:.2f} \\\\")
    
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train PGGCN on PDBBind dataset')
    parser.add_argument('--use-pcgrad', action='store_true',
                       help='Use PCGrad optimizer (default: standard Adam)')
    parser.add_argument('--run-all', action='store_true',
                       help='Run both PCGrad and standard Adam for comparison')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print("=" * 80)
    print("PYTORCH PGGCN TRAINING - PDBBIND DATASET")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    config = Config()
    set_random_seeds(config.RANDOM_SEED)
    
    # Load data
    X, y = load_data(config)
    
    # Print target statistics
    print(f"\nTarget statistics:")
    print(f"  Mean: {np.mean(y):.2f} kcal/mol")
    print(f"  Std:  {np.std(y):.2f} kcal/mol")
    print(f"  Min:  {np.min(y):.2f} kcal/mol")
    print(f"  Max:  {np.max(y):.2f} kcal/mol")
    
    # Split data
    print("\n" + "-" * 80)
    print("Splitting Data")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    print(f"✓ Training: {len(X_train)} samples")
    print(f"✓ Test: {len(X_test)} samples")
    
    # Create dataloaders
    train_dataset = MoleculeDataset(X_train, y_train)
    test_dataset = MoleculeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, collate_fn=collate_molecules)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=False, collate_fn=collate_molecules)
    
    # Determine which runs to do
    if args.run_all:
        runs = [
            ('ΔG with PCGrad + Multi-loss', True),
            ('ΔG with Adam + Multi-loss', False)
        ]
    else:
        runs = [('PCGrad' if args.use_pcgrad else 'Adam', args.use_pcgrad)]
    
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
        
        print(f"✓ Model created ({sum(p.numel() for p in model.parameters())} parameters)")
        
        model = model.to(device)
        
        # Train
        print("\n" + "-" * 80)
        print("Training")
        print("-" * 80)
        history, best_val_mae, best_epoch = train_model(
            model, train_loader, test_loader, use_pcgrad, config, device
        )
        
        # Evaluate
        print("\n" + "-" * 80)
        print("Evaluation")
        print("-" * 80)
        predictions, metrics = evaluate_model(model, test_loader, device)
        
        print(f"\nTest Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f} kcal/mol")
        print(f"  MAE:  {metrics['mae']:.4f} kcal/mol")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  Best Val MAE: {best_val_mae:.4f} @ epoch {best_epoch}")
        
        print_sample_predictions(predictions, y_test)
        
        # Store results
        results[run_name] = {
            'model': model,
            'history': history,
            'metrics': metrics,
            'predictions': predictions,
            'best_val_mae': best_val_mae,
            'best_epoch': best_epoch
        }
        
        # Save model
        save_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind/saved_models'
        os.makedirs(save_dir, exist_ok=True)
        
        suffix = 'pcgrad' if use_pcgrad else 'adam'
        filename = f"pggcn_pdbbind_{suffix}_lr{config.LEARNING_RATE}_pw{config.PHYSICS_WEIGHT}_{timestamp}.pth"
        save_path = os.path.join(save_dir, filename)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'metrics': metrics,
            'best_val_mae': best_val_mae,
            'best_epoch': best_epoch,
            'timestamp': timestamp,
            'config': {
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
    
    # Print comparison table if multiple runs
    if len(results) > 1:
        print_comparison_table(results)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()