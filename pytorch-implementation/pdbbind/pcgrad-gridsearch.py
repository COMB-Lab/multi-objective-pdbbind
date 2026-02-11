"""
Grid Search for Physics Weight with PCGrad - PDBBind Dataset
CORRECTED VERSION: Uses physics weights from 1e-7 to 1e-5 (matching TensorFlow)
Creates Pareto front similar to Paper Figure 2(a)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import os
import random
import time
import pickle
import json
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# Add paths
sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind')
sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind')
from models.dcFeaturizer import atom_features as get_atom_features
from models.layers_pytorch_pdbbind import PGGCNModel
from train_split_data import load_data_with_saved_split

try:
    from models.pcgrad_pytorch import PCGrad
    PCGRAD_AVAILABLE = True
except ImportError:
    PCGRAD_AVAILABLE = False
    print("ERROR: PCGrad not available! Install pcgrad_pytorch")
    sys.exit(1)

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    # Data paths
    CSV_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind.csv'
    PKL_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_full.pkl'
    
    # Model architecture
    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL = 20
    C_OUT_CHANNEL = 1024
    DROPOUT_RATE = 0.2
    
    # Training hyperparameters
    EPOCHS = 250
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    L2_WEIGHT = 1e-2
    MAX_NORM = 3.0
    
    # TensorFlow used: [1e-5, 1e-6, 1e-7, 2e-6, 5e-6]
    #PHYSICS_WEIGHTS = np.logspace(-8, -4, 41)  # 41 points from 1e-8 to 1e-4
    #PHYSICS_WEIGHTS = [1e-6]
    PHYSICS_WEIGHTS = np.arange(0.0, 1.01, 0.01)  # 101 weights from 0.0 to 1.0
    
    # Data split
    TEST_SIZE = 0.2
    RANDOM_SEED = 50
    EARLY_STOPPING_PATIENCE = 10
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_GAMMA = 0.85

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
# UTILITIES
# ============================================================================
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


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


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


def load_data(csv_path, pkl_path):
    """Load PDBBind dataset"""
    print("Loading PDBBind dataset...")
    
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
        except:
            pass
    
    print(f"✓ Successfully featurized {len(X)} structures")
    return X, y


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

def train_with_weight(model, train_loader, test_loader, physics_weight, config, device):
    """Train model with specific physics weight using batched data"""
    model = model.to(device)
    
    # PCGrad optimizer
    base_optimizer = optim.Adam(model.parameters(), 
                                lr=config.LEARNING_RATE, 
                                weight_decay=config.L2_WEIGHT)
    optimizer = PCGrad(base_optimizer)
    #early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, min_delta=0.001, verbose=True)
    #if config.USE_LR_SCHEDULER:
    #    scheduler = StepLR(base_optimizer, step_size=10, gamma=0.85)
    # Training loop
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_train_empirical = 0
        epoch_train_physics = 0
        epoch_train_total = 0
        n_train_batches = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            
            predictions, _, physics_info = model(X_batch, training=True)
            
            # FIX 2: Capture train_raw_physics
            train_empirical, train_weighted_physics, train_raw_physics, _ = compute_task_losses(
                predictions, y_batch, physics_info, physics_weight)
            
            optimizer.pc_backward([train_empirical, train_weighted_physics])
            optimizer.step()
            apply_maxnorm_constraint(model, max_norm=config.MAX_NORM)
            
            # Accumulate losses
            epoch_train_empirical += train_empirical.item()
            epoch_train_physics += train_raw_physics.item()  # FIX 2: Use raw, not weighted
            epoch_train_total += (train_empirical + train_weighted_physics).item()
            n_train_batches += 1
        
        # Calculate averages
        avg_train_empirical = epoch_train_empirical / n_train_batches
        avg_train_physics = epoch_train_physics / n_train_batches
        avg_train_total = epoch_train_total / n_train_batches
        
        # FIX 3: Print average loss
        if epoch == 0 or (epoch + 1) % 20 == 0:
            current_lr = base_optimizer.param_groups[0]['lr']
            print(f"      Epoch {epoch+1}/{config.EPOCHS} | LR: {current_lr:.2e}")
            print(f"        Train -> Total: {avg_train_total:.4f}, "
                f"Emp: {avg_train_empirical:.4f}, Phys: {avg_train_physics:.4f}")
        
        # Step scheduler FIRST
        #if config.USE_LR_SCHEDULER:
        #    scheduler.step()
        
        # FIX 1: Use average loss for early stopping
        #early_stopping(avg_train_total, model)
        #if early_stopping.early_stop:
        #    print(f"\nEarly stopping at epoch {epoch+1}")
        #    early_stopping.restore_best_weights(model)
        #    break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Train metrics
        all_train_preds, all_train_targets, all_train_phys = [], [], []
        for X_batch, y_batch in train_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            
            train_pred, _, train_phys = model(X_batch, training=False)
            all_train_preds.append(train_pred)
            all_train_targets.append(y_batch)
            all_train_phys.append(train_phys)
        
        train_predictions = torch.cat(all_train_preds, dim=0)
        train_targets = torch.cat(all_train_targets, dim=0)
        train_physics = torch.cat(all_train_phys, dim=0)
        
        train_empirical, _, train_physics_loss, _ = compute_task_losses(
            train_predictions, train_targets, train_physics, physics_weight)
        
        # Test metrics
        all_test_preds, all_test_targets, all_test_phys = [], [], []
        for X_batch, y_batch in test_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            
            test_pred, _, test_phys = model(X_batch, training=False)
            all_test_preds.append(test_pred)
            all_test_targets.append(y_batch)
            all_test_phys.append(test_phys)
        
        test_predictions = torch.cat(all_test_preds, dim=0)
        test_targets = torch.cat(all_test_targets, dim=0)
        test_physics = torch.cat(all_test_phys, dim=0)
        
        test_empirical, _, test_physics_loss, test_mae = compute_task_losses(
            test_predictions, test_targets, test_physics, physics_weight)
    
    return {
        'physics_weight': physics_weight,
        'train_empirical': train_empirical.item(),
        'train_physics': train_physics_loss.item(),
        'test_empirical': test_empirical.item(),
        'test_physics': test_physics_loss.item(),
        'test_mae': test_mae.item()
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    print("="*80)
    print("GRID SEARCH: PHYSICS WEIGHT (CORRECTED RANGE)")
    print("PDBBind Dataset - Creating Pareto Front")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Weight range: 1e-8 to 1e-4 (matching TensorFlow scale)")
    print(f"Number of weights: {len(Config.PHYSICS_WEIGHTS)}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Running on CPU. This will be VERY SLOW!")
        print("         Estimated total time: Many hours")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Expected time: Several hours for {len(Config.PHYSICS_WEIGHTS)} weights\n")
    
    config = Config()
    set_random_seeds(config.RANDOM_SEED)
    
    # Print weight range
    print("Physics weights to test:")
    print(f"  Range: {config.PHYSICS_WEIGHTS[0]:.2e} to {config.PHYSICS_WEIGHTS[-1]:.2e}")
    print(f"  Sample weights: {config.PHYSICS_WEIGHTS[::10]}")  # Show every 10th
    print()
    
    # Load data once
    print("Loading data...")
    #X, y = load_data(config.CSV_PATH, config.PKL_PATH)
        # Load clustered data
    # Change when you need to switch between number of structures
    split_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/subsets/pdbbind_subset_2660.pkl'
    
    # print(f"\nTarget statistics:")
    # print(f"  Mean: {np.mean(y):.2f} kcal/mol")
    # print(f"  Std:  {np.std(y):.2f} kcal/mol")
    # print(f"  Min:  {np.min(y):.2f} kcal/mol")
    # print(f"  Max:  {np.max(y):.2f} kcal/mol")
    
    # Split once (same split for all weights)
    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)    
    X_train, X_test, y_train, y_test = load_data_with_saved_split(config, split_path)

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Create dataloaders
    train_dataset = MoleculeDataset(X_train, y_train)
    test_dataset = MoleculeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, collate_fn=collate_molecules)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=False, collate_fn=collate_molecules)
    
    # Grid search
    n_weights = len(config.PHYSICS_WEIGHTS)
    print(f"\nStarting grid search over {n_weights} physics weights...")
    print(f"Progress will be saved every 10 weights.\n")
    
    results = []
    start_time = time.time()
    
    for i, weight in enumerate(config.PHYSICS_WEIGHTS):
        iter_start = time.time()
        
        print(f"\n[{i+1:3d}/{n_weights}] Training with λ={weight:.2e}")
        
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
        
        # Train with this weight
        result = train_with_weight(model, train_loader, test_loader,
                                   weight, config, device)
        results.append(result)
        
        iter_time = time.time() - iter_start
        elapsed_total = time.time() - start_time
        avg_time = elapsed_total / (i + 1)
        eta = avg_time * (n_weights - i - 1)
        
        # Progress update
        print(f"   ✓ Completed in {format_time(iter_time)}")
        print(f"   Results: Test MAE={result['test_mae']:.4f}, "
              f"Emp={result['test_empirical']:.4f}, "
              f"Phys={result['test_physics']:.4f}")
        print(f"   Overall progress: {format_time(elapsed_total)} elapsed, "
              f"{format_time(eta)} remaining")
        
        # Save intermediate results every 10 iterations
        if (i + 1) % 10 == 0:
            save_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind/grid_search/full_structures'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'pdbbind_grid_search_full_structures_{timestamp}.json')
            
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Progress saved to: {save_path}")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    print(f"Total time: {format_time(total_time)}")
    print(f"Average per weight: {format_time(total_time / n_weights)}\n")
    
    # Find best weights
    best_mae = min(results, key=lambda x: x['test_mae'])
    best_empirical = min(results, key=lambda x: x['test_empirical'])
    best_physics = min(results, key=lambda x: x['test_physics'])
    
    print("Best Results:")
    print(f"  Best MAE: {best_mae['test_mae']:.4f} at λ={best_mae['physics_weight']:.2e}")
    print(f"  Best Empirical: {best_empirical['test_empirical']:.4f} at λ={best_empirical['physics_weight']:.2e}")
    print(f"  Best Physics: {best_physics['test_physics']:.4f} at λ={best_physics['physics_weight']:.2e}")
    
    # Compare to TensorFlow results
    print("\n" + "-"*80)
    print("Comparison to TensorFlow Results:")
    print("-"*80)
    print("TensorFlow best MAE: ~8.85-8.94")
    print(f"PyTorch best MAE:    {best_mae['test_mae']:.2f}")
    print()
    
    # Save final results
    save_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind/grid_search/full_structures'
    os.makedirs(save_dir, exist_ok=True)
    
    json_path = os.path.join(save_dir, f'pdbbind_grid_search_full_structures_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {json_path}")
    
    # Create CSV for easy plotting
    csv_path = json_path.replace('.json', '.csv')
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved to: {csv_path}")
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'total_time_seconds': total_time,
        'avg_time_per_weight': total_time / n_weights,
        'n_weights_tested': n_weights,
        'weight_range': {
            'min': float(config.PHYSICS_WEIGHTS[0]),
            'max': float(config.PHYSICS_WEIGHTS[-1]),
            'scale': 'logarithmic'
        },
        'best_mae': {
            'value': best_mae['test_mae'],
            'lambda': best_mae['physics_weight']
        },
        'best_empirical': {
            'value': best_empirical['test_empirical'],
            'lambda': best_empirical['physics_weight']
        },
        'best_physics': {
            'value': best_physics['test_physics'],
            'lambda': best_physics['physics_weight']
        },
        'config': {
            'learning_rate': config.LEARNING_RATE,
            'l2_weight': config.L2_WEIGHT,
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'random_seed': config.RANDOM_SEED
        },
    }
    
    summary_path = os.path.join(save_dir, f'pdbbind_grid_search_summary_full_data_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("All files saved successfully!")
    print("Use the CSV file to create Pareto front plots.")
    print("="*80)


if __name__ == "__main__":
    main()