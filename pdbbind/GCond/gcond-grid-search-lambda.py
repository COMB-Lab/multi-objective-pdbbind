"""
PDBBind GCond Physics-Weight Grid Search
Sweeps physics_weight for GCond
hyperparameters: theta_main = -0.5, theta_crit = -0.8

Usage:
    python pdbbind_gcond_gridsearch.py
"""

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import os
import pickle
import time
import csv
import argparse
from datetime import timedelta, datetime
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind')
sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind')
sys.path.insert(0, '/home/exouser/GCond')  # for GCond import

from models.dcFeaturizer import atom_features as get_atom_features
from models.layers_pytorch_pdbbind import PGGCNModel
from train_split_data import load_data_with_saved_split

# GCond import
from GCond.grad_conductor import GradientConductor


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    # Data — uses subset pkl for consistent split, full CSV/PKL for featurization
    CSV_PATH  = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind.csv'
    PKL_PATH  = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_full_noH.pkl'
    PKL_PATH_FALLBACK = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_full.pkl'
    # Change as needed for subset size
    SPLIT_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/subsets/pdbbind_subset_2000.pkl'

    # Model architecture
    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL     = 20
    C_OUT_CHANNEL     = 1024
    DROPOUT_RATE      = 0.05

    EPOCHS        = 300
    BATCH_SIZE    = 8
    LEARNING_RATE = 5e-4
    L2_WEIGHT     = 1e-4
    MAX_NORM      = 3.0

    # GCond conflict-threshold hyperparameters
    THETA_MAIN = -0.5
    THETA_CRIT = -0.8
    THETA_WEAK =  0.0
    
     # Grid search axis
    PHYSICS_WEIGHTS = [round(i * 0.01, 6) for i in range(101)]
    
    RANDOM_SEED = 50
    
    # Save directories
    SAVE_DIR = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind/GCond-implementation/comparison'
    RESULTS_DIR = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind/GCond-implementation/results'


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
    return [item[0] for item in batch], torch.FloatTensor([item[1] for item in batch])


# ============================================================================
# UTILITIES
# ============================================================================

def set_random_seeds(seed=50):
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
    return str(timedelta(seconds=int(seconds)))


def apply_maxnorm_constraint(model, max_norm=3.0):
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.dim() >= 2:
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=max_norm)
                param.mul_(desired / (norm + 1e-7))


def compute_task_losses(predictions, targets, physics_info, physics_weight):
    targets = targets.view(-1, 1)

    empirical_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))

    host_energy    = physics_info[:, [0, 3, 6, 9, 12]].sum(dim=1, keepdim=True)
    guest_energy   = physics_info[:, [1, 4, 7, 10, 13]].sum(dim=1, keepdim=True)
    complex_energy = physics_info[:, [2, 5, 8, 11, 14]].sum(dim=1, keepdim=True)
    dG_physics = complex_energy - (host_energy + guest_energy)

    raw_physics_loss      = torch.sqrt(torch.mean((predictions - dG_physics) ** 2))
    weighted_physics_loss = physics_weight * raw_physics_loss
    mae = torch.mean(torch.abs(predictions - targets))

    return empirical_loss, weighted_physics_loss, raw_physics_loss, mae


def normalize_physics(mol_list, mean, std):
    normalized = []
    for mol in mol_list:
        mol = mol.clone()
        mol[:, -15:] = (mol[:, -15:] - mean) / std
        normalized.append(mol)
    return normalized


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(config):
    print("\n" + "-" * 80)
    print("Loading Data")
    print("-" * 80)

    if os.path.exists(config.PKL_PATH):
        print(f"Using preprocessed (no-H) PKL")
        remove_hs = False
        pkl_path  = config.PKL_PATH
    else:
        print(f"  noH PKL not found — using full PKL with RemoveHs at runtime")
        remove_hs = True
        pkl_path  = config.PKL_PATH_FALLBACK

    class _Cfg:
        CSV_PATH = config.CSV_PATH
        PKL_PATH = pkl_path

    if remove_hs:
        X_train, X_test, y_train, y_test = _load_with_remove_hs(config, pkl_path)
    else:
        X_train, X_test, y_train, y_test = load_data_with_saved_split(_Cfg(), config.SPLIT_PATH)

    print(f"\n✓ Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"\nTarget statistics (train):")
    print(f"  Mean: {np.mean(y_train):.2f} kcal/mol")
    print(f"  Std:  {np.std(y_train):.2f} kcal/mol")
    print(f"  Min:  {np.min(y_train):.2f} kcal/mol")
    print(f"  Max:  {np.max(y_train):.2f} kcal/mol")

    return X_train, X_test, y_train, y_test


def _load_with_remove_hs(config, pkl_path):
    """Fallback loader that applies RemoveHs at runtime."""
    physics_columns = [
        'pb-protein-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals',
        'gb-protein-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
        'gb-protein-eelect',  'gb-ligand-eelec',   'gb-complex-eelec',
        'gb-protein-egb',     'gb-ligand-egb',      'gb-complex-egb',
        'gb-protein-esurf',   'gb-ligand-esurf',    'gb-complex-esurf'
    ]

    with open(config.SPLIT_PATH, 'rb') as f:
        split_data = pickle.load(f)
    with open(pkl_path, 'rb') as f:
        pdb_dict = pickle.load(f)

    df = pd.read_csv(config.CSV_PATH)
    df = df.dropna(subset=['ddg'])
    df = df[df['complex-name'].apply(lambda x: 'E+' not in str(x))]
    df = df.set_index('complex-name')

    def featurize_split(names):
        X, y = [], []
        skipped = 0
        for pdb_id in names:
            if pdb_id not in df.index or pdb_id not in pdb_dict:
                skipped += 1
                continue
            row = df.loc[pdb_id]
            info_array = row[physics_columns].tolist()
            target = row['ddg']
            try:
                mol = Chem.RemoveHs(pdb_dict[pdb_id])
                feat = []
                for atom in mol.GetAtoms():
                    base_feat = get_atom_features(atom)
                    new_feature = base_feat.tolist()
                    position = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                    new_feature += [atom.GetMass(), atom.GetAtomicNum(), atom.GetFormalCharge()]
                    new_feature += [position.x, position.y, position.z]
                    neighbors = atom.GetNeighbors()[:2]
                    for neighbor in neighbors:
                        new_feature += [float(neighbor.GetIdx())]
                    for _ in range(2 - len(neighbors)):
                        new_feature += [-1.0]
                    feat.append(new_feature + info_array)
                X.append(torch.FloatTensor(np.array(feat)))
                y.append(float(target))
            except Exception:
                skipped += 1
        if skipped:
            print(f"  Warning: skipped {skipped} structures")
        return X, y

    train_names = [n for n in split_data['train_names'] if n in set(df.index)]
    test_names  = [n for n in split_data['test_names']  if n in set(df.index)]

    print("  Featurizing training set...")
    X_train, y_train = featurize_split(train_names)
    print("  Featurizing test set...")
    X_test,  y_test  = featurize_split(test_names)

    train_physics = torch.stack([mol[0, -15:] for mol in X_train])
    phys_mean = train_physics.mean(dim=0)
    phys_std  = train_physics.std(dim=0).clamp(min=1e-8)
    X_train = normalize_physics(X_train, phys_mean, phys_std)
    X_test  = normalize_physics(X_test,  phys_mean, phys_std)

    return X_train, X_test, y_train, y_test


# GCond loss functions
def empirical_loss_fun(output, y):
    predictions, _, physics_info = output
    emp, _, _, _ = compute_task_losses(predictions, y, physics_info, physics_weight=0.0)
    return emp

def physics_loss_fun(output, y):
    predictions, _, physics_info = output
    _, weighted_phys, raw_phys, _ = compute_task_losses(predictions, y, physics_info, physics_weight=1.0)
    return raw_phys  # GCond applies its own weighting via lambdas


# ============================================================================
# TRAINING (GCond only)
# ============================================================================

def train_gcond_model(model, train_loader, test_loader, physics_weight, config, device, run_name):
    model = model.to(device)

    loss_fns = {
        "empirical_loss": empirical_loss_fun,
        "physics_loss": physics_loss_fun
    }
    lambdas = {"empirical_loss": 1.0, "physics_loss": physics_weight}

    base_optimizer = optim.Adam(model.parameters(),
                                lr=config.LEARNING_RATE,
                                weight_decay=config.L2_WEIGHT)

    # conflict_thresholds is ordered (theta_crit, theta_main, theta_weak)
    thresholds = (config.THETA_CRIT, config.THETA_MAIN, config.THETA_WEAK)

    conductor = GradientConductor(
        model=model,
        loss_fns=loss_fns,
        lambdas=lambdas,
        accumulation_steps=2,
        freeze_bn=False,
        conflict_thresholds=thresholds,
    )

    train_iter = iter(train_loader)
    def data_provider():
        nonlocal train_iter
        try:
            X_batch, y_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X_batch, y_batch = next(train_iter)
        X_batch = [x.to(device) for x in X_batch]
        y_batch = y_batch.unsqueeze(1).to(device)
        return {'args': (X_batch,), 'kwargs': {'training': True}}, y_batch

    print(f"\n{'='*80}")
    print(f"Training: {run_name}")
    print(f"{'='*80}")
    print(f"  Optimizer:      GCond + Adam")
    print(f"  physics_weight: {physics_weight}")
    print(f"  theta_crit:     {config.THETA_CRIT}")
    print(f"  theta_main:     {config.THETA_MAIN}")
    print(f"  theta_weak:     {config.THETA_WEAK}")
    print(f"  Epochs:         {config.EPOCHS}")
    print(f"  Learning rate:  {config.LEARNING_RATE}")
    print(f"  L2 weight:      {config.L2_WEIGHT}")
    print(f"  Dropout:        {config.DROPOUT_RATE}")

    best_val_mae     = float('inf')
    best_model_state = None
    start_time       = time.time()

    for epoch in range(config.EPOCHS):
        model.train()
        for _ in range(len(train_loader)):
            conductor.step(data_provider)
            base_optimizer.step()
            apply_maxnorm_constraint(model, config.MAX_NORM)
            base_optimizer.zero_grad()

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            all_preds, all_targets, all_phys = [], [], []
            for X_batch, y_batch in test_loader:
                X_batch = [x.to(device) for x in X_batch]
                y_batch = y_batch.unsqueeze(1).to(device)
                preds, _, phys = model(X_batch, training=False)
                all_preds.append(preds)
                all_targets.append(y_batch)
                all_phys.append(phys)

            val_preds   = torch.cat(all_preds)
            val_targets = torch.cat(all_targets)
            val_phys    = torch.cat(all_phys)
            _, _, _, val_mae = compute_task_losses(
                val_preds, val_targets, val_phys, physics_weight)

        if val_mae.item() < best_val_mae:
            best_val_mae     = val_mae.item()
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta     = (elapsed / (epoch + 1)) * (config.EPOCHS - epoch - 1)
            print(f"  Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Val MAE: {val_mae.item():.4f} | "
                  f"Best: {best_val_mae:.4f} | "
                  f"Time: {format_time(elapsed)} | ETA: {format_time(eta)}")

    print(f"\nRestoring best model (val MAE: {best_val_mae:.4f})")
    model.load_state_dict(best_model_state)
    print(f"Completed in {format_time(time.time() - start_time)}")

    return model, best_val_mae


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_targets, all_phys = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            preds, _, phys = model(X_batch, training=False)
            all_preds.append(preds)
            all_targets.append(y_batch)
            all_phys.append(phys)

    predictions = torch.cat(all_preds)
    targets     = torch.cat(all_targets)
    physics     = torch.cat(all_phys)

    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    mae  = torch.mean(torch.abs(predictions - targets)).item()

    ss_res = torch.sum((targets - predictions) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2     = 1 - (ss_res / ss_tot)

    _, _, raw_physics_loss, _ = compute_task_losses(
        predictions, targets, physics, physics_weight=1.0)

    return {
        'rmse':         rmse,
        'mae':          mae,
        'r2':           r2,
        'physics_loss': raw_physics_loss.item(),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GCond physics_weight grid search (theta_main=-0.5, theta_crit=-0.8 fixed)')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                         help='Override the physics_weight grid, e.g. --weights 0.1 0.3 0.51 0.7 1.0')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config    = Config()
    if args.weights is not None:
        config.PHYSICS_WEIGHTS = args.weights

    print("=" * 80)
    print("PDBBIND GCOND PHYSICS-WEIGHT GRID SEARCH — FULL DATASET")
    print("=" * 80)
    print(f"Timestamp:  {timestamp}")
    print(f"theta_main: {config.THETA_MAIN}")
    print(f"theta_crit: {config.THETA_CRIT}")
    print(f"Grid:       {config.PHYSICS_WEIGHTS}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device:     {device}")

    set_random_seeds(config.RANDOM_SEED)

    # Load data once — all runs reuse the same split
    X_train, X_test, y_train, y_test = load_data(config)

    train_loader = DataLoader(
        MoleculeDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_molecules)
    test_loader = DataLoader(
        MoleculeDataset(X_test, y_test),
        batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_molecules)

    results = []

    for physics_weight in config.PHYSICS_WEIGHTS:
        run_name = f"GCond physics_weight={physics_weight} (theta_main={config.THETA_MAIN}, theta_crit={config.THETA_CRIT})"

        # Fresh model + fresh seed per run for reproducibility across the grid
        set_random_seeds(config.RANDOM_SEED)
        model = PGGCNModel(
            num_atom_features=config.NUM_ATOM_FEATURES,
            r_out_channel=config.R_OUT_CHANNEL,
            c_out_channel=config.C_OUT_CHANNEL,
            dropout_rate=config.DROPOUT_RATE
        )
        model.add_rule("sum", 0, 32)
        model.add_rule("multiply", 32, 33)
        model.add_rule("distance", 33, 36)

        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

        model, best_val_mae = train_gcond_model(
            model, train_loader, test_loader, physics_weight, config, device, run_name)

        metrics = evaluate_model(model, test_loader, device)

        print(f"\nResults — {run_name}:")
        print(f"  RMSE:          {metrics['rmse']:.4f} kcal/mol")
        print(f"  MAE:           {metrics['mae']:.4f} kcal/mol")
        print(f"  R²:            {metrics['r2']:.4f}")
        print(f"  Physics loss:  {metrics['physics_loss']:.4f}")
        print(f"  Best val MAE:  {best_val_mae:.4f}")

        results.append({
            'physics_weight': physics_weight,
            'theta_crit':     config.THETA_CRIT,
            'theta_main':     config.THETA_MAIN,
            'theta_weak':     config.THETA_WEAK,
            'rmse':           metrics['rmse'],
            'mae':            metrics['mae'],
            'r2':             metrics['r2'],
            'physics_loss':   metrics['physics_loss'],
            'best_val_mae':   best_val_mae,
        })

        # Save model checkpoint
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        model_path = os.path.join(
            config.SAVE_DIR,
            f'pdbbind_full_gcond_pw{physics_weight}_'
            f'tc{config.THETA_CRIT}_tm{config.THETA_MAIN}_tw{config.THETA_WEAK}_{timestamp}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics':          metrics,
            'best_val_mae':     best_val_mae,
            'timestamp':        timestamp,
            'config': {
                'learning_rate':  config.LEARNING_RATE,
                'l2_weight':      config.L2_WEIGHT,
                'dropout_rate':   config.DROPOUT_RATE,
                'physics_weight': physics_weight,
                'theta_crit':     config.THETA_CRIT,
                'theta_main':     config.THETA_MAIN,
                'theta_weak':     config.THETA_WEAK,
                'batch_size':     config.BATCH_SIZE,
                'epochs':         config.EPOCHS,
            }
        }, model_path)
        print(f"Model saved to: {model_path}")

    # ── Final comparison table ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("GRID SEARCH RESULTS — GCond, theta_main=-0.5, theta_crit=-0.8")
    print(f"{'='*80}")
    print(f"\n{'physics_weight':>14} | {'RMSE':>8} | {'MAE':>8} | {'R²':>8} | {'PhysLoss':>10} | {'BestValMAE':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['physics_weight']:>14} | "
              f"{r['rmse']:>8.4f} | "
              f"{r['mae']:>8.4f} | "
              f"{r['r2']:>8.4f} | "
              f"{r['physics_loss']:>10.4f} | "
              f"{r['best_val_mae']:>10.4f}")

    # Save results CSV
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR,
                            f'pdbbind_gcond_gridsearch_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {csv_path}")

    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()