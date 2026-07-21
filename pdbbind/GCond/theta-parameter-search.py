# Small test to see the difference in results when using different Θ_main based on the second quadrant of the unit circle.

"""
PDBBind GCond Threshold Search
Grid search over theta_main and theta_crit with theta_weak=0.0 fixed.

Usage:
    python pdbbind_gcond_threshold_search.py
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
from datetime import timedelta, datetime
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind')
sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind')
sys.path.insert(0, '/home/exouser/GCond')

from models.dcFeaturizer import atom_features as get_atom_features
from models.layers_pytorch_pdbbind import PGGCNModel
from train_split_data import load_data_with_saved_split
from GCond.grad_conductor import GradientConductor


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    CSV_PATH       = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind.csv'
    PKL_PATH       = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_full_noH.pkl'
    PKL_PATH_FALLBACK = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_full.pkl'
    # Change number to change subset sizes (100, 250, 500, 1000, 2000, 2660)
    SPLIT_PATH     = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/subsets/pdbbind_subset_250.pkl'

    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL     = 20
    C_OUT_CHANNEL     = 1024
    DROPOUT_RATE      = 0.05

    # Keep at 100 for now since there are 12 combinatons total. Observe how long it takes to run and ajust to 300 or remove candidates and then adjust.
    EPOCHS        = 300
    BATCH_SIZE    = 8
    LEARNING_RATE = 5e-4
    L2_WEIGHT     = 1e-4
    MAX_NORM      = 3.0
    # 100 = 0.99, 250 = 0.50, 500 = 0.51, 1000 = 0.73, 2000 = 0.63, 2660 = 0.53
    PHYSICS_WEIGHT = 0.50
    RANDOM_SEED    = 50

    # Fixed threshold
    THETA_WEAK =  0.0

    # Grid search candidates for theta_main
    THETA_MAIN_CANDIDATES = [-0.6, -0.5, -0.4, -0.3]

    # Grid search candidates for theta_crit. These have to be lower than theta_main candidates
    THETA_CRIT_CANDIDATES = [-0.9, -0.8, -0.7]

    RESULTS_DIR = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind/results/GCond'


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
    dG_physics     = complex_energy - (host_energy + guest_energy)
 
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
# LOSS FUNCTIONS FOR GCOND
# ============================================================================
 
def empirical_loss_fn(output, y):
    predictions, _, physics_info = output
    emp, _, _, _ = compute_task_losses(predictions, y, physics_info, physics_weight=0.0)
    return emp
 
 
def physics_loss_fn(output, y):
    predictions, _, physics_info = output
    _, _, raw_phys, _ = compute_task_losses(predictions, y, physics_info, physics_weight=1.0)
    return raw_phys
 
 
# ============================================================================
# DATA LOADING
# ============================================================================
 
def load_data(config):
    print("\n" + "-" * 80)
    print("Loading Data")
    print("-" * 80)
 
    if os.path.exists(config.PKL_PATH):
        print("Using preprocessed (no-H) PKL")
        remove_hs = False
        pkl_path  = config.PKL_PATH
    else:
        print("  noH PKL not found — using full PKL with RemoveHs at runtime")
        remove_hs = True
        pkl_path  = config.PKL_PATH_FALLBACK
 
    class _Cfg:
        CSV_PATH = config.CSV_PATH
        PKL_PATH = pkl_path
 
    if remove_hs:
        X_train, X_test, y_train, y_test = _load_with_remove_hs(config, pkl_path)
    else:
        X_train, X_test, y_train, y_test = load_data_with_saved_split(_Cfg(), config.SPLIT_PATH)
 
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test
 
 
def _load_with_remove_hs(config, pkl_path):
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
 
 
# ============================================================================
# MODEL BUILDER
# ============================================================================
 
def build_model(config):
    model = PGGCNModel(
        num_atom_features=config.NUM_ATOM_FEATURES,
        r_out_channel=config.R_OUT_CHANNEL,
        c_out_channel=config.C_OUT_CHANNEL,
        dropout_rate=config.DROPOUT_RATE
    )
    model.add_rule("sum",      0, 32)
    model.add_rule("multiply", 32, 33)
    model.add_rule("distance", 33, 36)
    return model
 
 
# ============================================================================
# TRAINING
# ============================================================================
 
def train_gcond(model, train_loader, test_loader, config, device, theta_crit, theta_main):
    model = model.to(device)
 
    thresholds = (theta_crit, theta_main, config.THETA_WEAK)
 
    loss_fns = {
        "empirical": empirical_loss_fn,
        "physics":   physics_loss_fn,
    }
    lambdas = {
        "empirical": 1.0,
        "physics":   config.PHYSICS_WEIGHT,
    }
 
    conductor = GradientConductor(
        model=model,
        loss_fns=loss_fns,
        lambdas=lambdas,
        accumulation_steps=2,
        freeze_bn=False,
        conflict_thresholds=thresholds,
    )
 
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.L2_WEIGHT,
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
 
    best_val_mae     = float('inf')
    best_model_state = None
    start_time       = time.time()
 
    for epoch in range(config.EPOCHS):
        model.train()
        for _ in range(len(train_loader)):
            conductor.step(data_provider)
            optimizer.step()
            apply_maxnorm_constraint(model, config.MAX_NORM)
            optimizer.zero_grad()
 
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
                val_preds, val_targets, val_phys, config.PHYSICS_WEIGHT)
 
        if val_mae.item() < best_val_mae:
            best_val_mae     = val_mae.item()
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
 
        if (epoch + 1) % 25 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta     = (elapsed / (epoch + 1)) * (config.EPOCHS - epoch - 1)
            print(f"    Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Val MAE: {val_mae.item():.4f} | "
                  f"Best: {best_val_mae:.4f} | "
                  f"ETA: {format_time(eta)}")
 
    model.load_state_dict(best_model_state)
    return model, best_val_mae
 
 
# ============================================================================
# EVALUATION
# ============================================================================
 
def evaluate_model(model, test_loader, config, device):
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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config    = Config()
 
    print("=" * 80)
    print("GCOND THRESHOLD GRID SEARCH")
    print(f"Fixed:   theta_weak={config.THETA_WEAK}")
    print(f"Search:  theta_crit in {config.THETA_CRIT_CANDIDATES}")
    print(f"         theta_main in {config.THETA_MAIN_CANDIDATES}")
    print(f"Total runs: {len(config.THETA_CRIT_CANDIDATES) * len(config.THETA_MAIN_CANDIDATES)}")
    print(f"Epochs per run: {config.EPOCHS}")
    print("=" * 80)
 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
 
    set_random_seeds(config.RANDOM_SEED)
 
    X_train, X_test, y_train, y_test = load_data(config)
 
    train_loader = DataLoader(
        MoleculeDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_molecules)
    test_loader = DataLoader(
        MoleculeDataset(X_test, y_test),
        batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_molecules)
 
    results = []
 
    for theta_crit in config.THETA_CRIT_CANDIDATES:
        for theta_main in config.THETA_MAIN_CANDIDATES:
 
            # Skip invalid combinations — crit must be strictly less than main
            if theta_crit >= theta_main:
                print(f"\n  Skipping theta_crit={theta_crit}, theta_main={theta_main} "
                      f"— crit must be < main")
                continue
 
            thresholds_str = f"({theta_crit}, {theta_main}, {config.THETA_WEAK})"
            print(f"\n{'='*80}")
            print(f"  theta_crit={theta_crit}  theta_main={theta_main}  "
                  f"thresholds={thresholds_str}")
            print(f"{'='*80}")
 
            set_random_seeds(config.RANDOM_SEED)
            model = build_model(config)
            print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
 
            model, best_val_mae = train_gcond(
                model, train_loader, test_loader,
                config, device, theta_crit, theta_main)
 
            metrics = evaluate_model(model, test_loader, config, device)
 
            print(f"\n  Results (theta_crit={theta_crit}, theta_main={theta_main}):")
            print(f"    RMSE:         {metrics['rmse']:.4f} kcal/mol")
            print(f"    MAE:          {metrics['mae']:.4f} kcal/mol")
            print(f"    R²:           {metrics['r2']:.4f}")
            print(f"    Physics loss: {metrics['physics_loss']:.4f}")
            print(f"    Best val MAE: {best_val_mae:.4f}")
 
            results.append({
                'theta_crit':   theta_crit,
                'theta_main':   theta_main,
                'theta_weak':   config.THETA_WEAK,
                'thresholds':   thresholds_str,
                'rmse':         metrics['rmse'],
                'mae':          metrics['mae'],
                'r2':           metrics['r2'],
                'physics_loss': metrics['physics_loss'],
                'best_val_mae': best_val_mae,
            })
 
    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("GRID SEARCH SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'theta_crit':>12} | {'theta_main':>12} | {'RMSE':>8} | {'MAE':>8} | {'R²':>8} | {'PhysLoss':>10} | {'BestValMAE':>12}")
    print("-" * 85)
    for r in results:
        print(f"{r['theta_crit']:>12} | "
              f"{r['theta_main']:>12} | "
              f"{r['rmse']:>8.4f} | "
              f"{r['mae']:>8.4f} | "
              f"{r['r2']:>8.4f} | "
              f"{r['physics_loss']:>10.4f} | "
              f"{r['best_val_mae']:>12.4f}")
 
    best = min(results, key=lambda x: x['mae'])
    print(f"\nBest: theta_crit={best['theta_crit']}, theta_main={best['theta_main']}  "
          f"(MAE={best['mae']:.4f})")
 
    # Save CSV
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR,
                            f'gcond_threshold_search_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    main()
