"""
Host-Guest Grid Search - 0.01 Step (101 weights)

Changes from original:
- Physics normalization using training set statistics only
- Early stopping (patience=20) to avoid running past optimal
- Epochs=300
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
import json
import pickle
from datetime import timedelta
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Add paths
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

try:
    from PGGCN.models.dcFeaturizer import atom_features as get_atom_features
    from PGGCN.models.layers_pytorch import PGGCNModel
except ImportError:
    try:
        from models.dcFeaturizer import atom_features as get_atom_features
        from models.layers_pytorch import PGGCNModel
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)

from pcgrad_pytorch import PCGrad
print("PCGrad optimizer available")


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

def apply_maxnorm_constraint(model, max_norm=3.0):
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.dim() >= 2:
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=max_norm)
                param.mul_(desired / (norm + 1e-7))


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


def load_all_data(info_csv_path, hostguest_dir):
    print(f"Loading PDB files from: {hostguest_dir}")
    with open(hostguest_dir, 'rb') as f:
        pdb_dict = pickle.load(f)
    print(f"Loaded {len(pdb_dict)} PDB files")

    df_all = pd.read_csv(info_csv_path)
    print(f"Loaded {len(df_all)} CSV entries")

    feature_columns = [
        'pb_host_VDWAALS', 'pb_guest_VDWAALS', 'pb_complex_VDWAALS',
        'gb_host_1-4EEL', 'gb_guest_1-4EEL', 'gb_Complex_1-4EEL',
        'gb_host_EELEC', 'gb_guest_EELEC', 'gb_Complex_EELEC',
        'gb_host_EGB', 'gb_guest_EGB', 'gb_Complex_EGB',
        'gb_host_ESURF', 'gb_guest_ESURF', 'gb_Complex_ESURF'
    ]

    X, y = [], []

    for pdb_id in list(pdb_dict.keys()):
        if pdb_id not in df_all['Ids'].values:
            continue

        molecule = pdb_dict[pdb_id]
        row = df_all[df_all['Ids'] == pdb_id].iloc[0]
        info_array = row[feature_columns].tolist()
        target = row['Ex _G_(kcal/mol)']

        try:
            features = featurize(molecule, info_array)
            X.append(torch.FloatTensor(features))
            y.append(target)
        except Exception:
            pass

    print(f"Successfully loaded {len(X)} complexes")
    return X, y


def normalize_physics(mol_list, mean, std):
    normalized = []
    for mol in mol_list:
        mol = mol.clone()
        mol[:, -15:] = (mol[:, -15:] - mean) / std
        normalized.append(mol)
    return normalized


def compute_task_losses(predictions, targets, model_vars, physics_info, physics_weight):
    targets = targets.view(-1, 1)
    empirical_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))

    host_energy    = physics_info[:, [0, 3, 6, 9, 12]].sum(dim=1, keepdim=True)
    guest_energy   = physics_info[:, [1, 4, 7, 10, 13]].sum(dim=1, keepdim=True)
    complex_energy = physics_info[:, [2, 5, 8, 11, 14]].sum(dim=1, keepdim=True)
    dG_physics = complex_energy - (host_energy + guest_energy)

    physics_loss          = torch.sqrt(torch.mean((predictions - dG_physics) ** 2))
    weighted_physics_loss = physics_weight * physics_loss

    return empirical_loss, weighted_physics_loss, physics_loss


# ============================================================================
# SINGLE TRAINING RUN — BATCHED
# ============================================================================

def train_single_config(X_train, y_train, X_val, y_val, physics_weight, device,
                        epochs=300, lr=0.005, batch_size=8,
                        patience=20, verbose=True):
    """
    Train a single configuration using DataLoader batching and early stopping.
    Much faster than unbatched version due to better GPU utilization.
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = PGGCNModel(num_atom_features=36, r_out_channel=20,
                       c_out_channel=128, dropout_rate=0.2)
    model.add_rule("sum", 0, 32)
    model.add_rule("multiply", 32, 33)
    model.add_rule("distance", 33, 36)
    model = model.to(device)

    base_optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer      = PCGrad(base_optimizer)

    # DataLoaders
    train_loader = DataLoader(
        MoleculeDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, collate_fn=collate_molecules)
    val_loader = DataLoader(
        MoleculeDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False, collate_fn=collate_molecules)

    best_val_loss      = float('inf')
    best_val_empirical = float('inf')
    best_val_physics   = float('inf')
    best_val_rmse      = float('inf')
    best_train_rmse    = float('inf')
    epochs_no_improve  = 0
    stopped_epoch      = epochs

    train_rmse_history      = []
    val_rmse_history        = []
    train_empirical_history = []
    val_empirical_history   = []
    train_physics_history   = []
    val_physics_history     = []

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        batch_emp, batch_phys, batch_rmse = [], [], []

        for X_batch, y_batch in train_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)

            predictions, model_var, physics_info = model(X_batch, training=True)
            train_empirical, train_weighted_physics, train_raw_physics = \
                compute_task_losses(predictions, y_batch, model_var,
                                    physics_info, physics_weight)

            optimizer.pc_backward([train_empirical, train_weighted_physics])
            optimizer.step()
            apply_maxnorm_constraint(model, max_norm=3.0)

            batch_emp.append(train_empirical.item())
            batch_phys.append(train_raw_physics.item())
            batch_rmse.append(
                torch.sqrt(nn.MSELoss()(predictions, y_batch)).item())

        train_rmse     = np.mean(batch_rmse)
        avg_train_emp  = np.mean(batch_emp)
        avg_train_phys = np.mean(batch_phys)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_batch_emp, val_batch_phys, val_batch_rmse = [], [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = [x.to(device) for x in X_batch]
                y_batch = y_batch.unsqueeze(1).to(device)

                val_pred, val_var, val_phys = model(X_batch, training=False)
                val_empirical, val_weighted_physics, val_raw_physics = \
                    compute_task_losses(val_pred, y_batch, val_var,
                                        val_phys, physics_weight)

                val_batch_emp.append(val_empirical.item())
                val_batch_phys.append(val_raw_physics.item())
                val_batch_rmse.append(
                    torch.sqrt(nn.MSELoss()(val_pred, y_batch)).item())

        val_rmse     = np.mean(val_batch_rmse)
        avg_val_emp  = np.mean(val_batch_emp)
        avg_val_phys = np.mean(val_batch_phys)
        val_total    = avg_val_emp + physics_weight * avg_val_phys

        # ── History ───────────────────────────────────────────────────────────
        train_rmse_history.append(train_rmse)
        val_rmse_history.append(val_rmse)
        train_empirical_history.append(avg_train_emp)
        val_empirical_history.append(avg_val_emp)
        train_physics_history.append(avg_train_phys)
        val_physics_history.append(avg_val_phys)

        # ── Checkpoint & early stopping ────────────────────────────────────────
        if val_total < best_val_loss:
            best_val_loss      = val_total
            best_val_empirical = avg_val_emp
            best_val_physics   = avg_val_phys
            best_val_rmse      = val_rmse
            best_train_rmse    = train_rmse
            epochs_no_improve  = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                stopped_epoch = epoch + 1
                if verbose:
                    print(f"      Early stopping at epoch {epoch+1} "
                          f"(no improvement for {patience} epochs)")
                break

        # ── Logging ───────────────────────────────────────────────────────────
        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(f"      Epoch {epoch+1:3d}/{epochs} | "
                  f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | "
                  f"Train Emp: {avg_train_emp:.4f} | Val Emp: {avg_val_emp:.4f} | "
                  f"Train Phys: {avg_train_phys:.4f} | Val Phys: {avg_val_phys:.4f}")

    # ── Final evaluation ───────────────────────────────────────────────────────
    model.eval()
    all_val_preds, all_val_targets = [], []
    all_train_preds, all_train_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            val_pred, _, _ = model(X_batch, training=False)
            all_val_preds.append(val_pred)
            all_val_targets.append(y_batch)

        for X_batch, y_batch in train_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            train_pred, _, _ = model(X_batch, training=False)
            all_train_preds.append(train_pred)
            all_train_targets.append(y_batch)

    val_preds_cat   = torch.cat(all_val_preds)
    val_targets_cat = torch.cat(all_val_targets)
    train_preds_cat   = torch.cat(all_train_preds)
    train_targets_cat = torch.cat(all_train_targets)

    final_val_rmse   = torch.sqrt(nn.MSELoss()(val_preds_cat, val_targets_cat)).item()
    val_mae          = torch.mean(torch.abs(val_preds_cat - val_targets_cat)).item()
    final_train_rmse = torch.sqrt(nn.MSELoss()(train_preds_cat, train_targets_cat)).item()
    train_mae        = torch.mean(torch.abs(train_preds_cat - train_targets_cat)).item()

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'physics_weight':          physics_weight,
        'stopped_epoch':           stopped_epoch,
        'val_rmse':                final_val_rmse,
        'val_mae':                 val_mae,
        'train_rmse':              final_train_rmse,
        'train_mae':               train_mae,
        'best_val_total_loss':     best_val_loss,
        'best_val_empirical_loss': best_val_empirical,
        'best_val_physics_loss':   best_val_physics,
        'best_val_rmse':           best_val_rmse,
        'best_train_rmse':         best_train_rmse,
        'train_rmse_history':      train_rmse_history,
        'val_rmse_history':        val_rmse_history,
        'train_empirical_history': train_empirical_history,
        'val_empirical_history':   val_empirical_history,
        'train_physics_history':   train_physics_history,
        'val_physics_history':     val_physics_history,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("Host-Guest Grid Search")
    print("0.01 Step (101 weights) | Batching + Early Stopping")
    print("=" * 80)

    physics_weights = [round(i * 0.01, 6) for i in range(101)]

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs     = 300
    lr         = 0.005
    batch_size = 8
    patience   = 20

    print(f"\nConfiguration:")
    print(f"  Device:           {device}")
    print(f"  Optimizer:        PCGrad + Adam")
    print(f"  Learning rate:    {lr}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Max epochs:       {epochs}")
    print(f"  Early stopping:   patience={patience}")
    print(f"  Weights:          {len(physics_weights)} (0.00 → 1.00, step 0.01)")

    info_csv_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/Final_data_DDG.csv'
    hostguest_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBs_RDKit_BFE.pkl'
    output_dir    = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/grid_search_results/hostguest'

    print("\n" + "=" * 80)
    print("Loading Data...")
    print("=" * 80)
    X, y = load_all_data(info_csv_path, hostguest_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"Training: {len(X_train)} | Validation: {len(X_val)}")

    # Normalize physics using train stats only
    train_physics = torch.stack([mol[0, -15:] for mol in X_train])
    phys_mean = train_physics.mean(dim=0)
    phys_std  = train_physics.std(dim=0).clamp(min=1e-8)
    X_train = normalize_physics(X_train, phys_mean, phys_std)
    X_val   = normalize_physics(X_val,   phys_mean, phys_std)

    normalized_check = torch.stack([m[0, -15:] for m in X_train])
    print(f"Physics normalized:")
    print(f"  Max abs: {normalized_check.abs().max().item():.4f}")
    print(f"  Mean:    {normalized_check.mean().item():.4f}")
    print(f"  Std:     {normalized_check.std().item():.4f}")

    os.makedirs(output_dir, exist_ok=True)

    results     = []
    total_start = time.time()

    print("\n" + "=" * 80)
    print("Starting Grid Search")
    print("=" * 80)

    for i, weight in enumerate(physics_weights, 1):
        iter_start = time.time()

        print(f"\n[{i}/{len(physics_weights)}] Physics Weight = {weight:.6f}")
        print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            result = train_single_config(
                X_train, y_train, X_val, y_val,
                weight, device,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                patience=patience,
                verbose=True
            )

            results.append(result)

            iter_time = time.time() - iter_start
            print(f"\n Completed in {timedelta(seconds=int(iter_time))} "
                  f"(stopped at epoch {result['stopped_epoch']})")
            print(f"  ┌─ Final Metrics ─────────────────────────")
            print(f"  │ Train RMSE: {result['train_rmse']:.4f}")
            print(f"  │ Val RMSE:   {result['val_rmse']:.4f}")
            print(f"  │ Train MAE:  {result['train_mae']:.4f}")
            print(f"  │ Val MAE:    {result['val_mae']:.4f}")
            print(f"  ├─ Best During Training ──────────────────")
            print(f"  │ Best Train RMSE: {result['best_train_rmse']:.4f}")
            print(f"  │ Best Val RMSE:   {result['best_val_rmse']:.4f}")
            print(f"  │ Best Empirical:  {result['best_val_empirical_loss']:.4f}")
            print(f"  │ Best Physics:    {result['best_val_physics_loss']:.4f}")
            print(f"  └─────────────────────────────────────────")

            elapsed   = time.time() - total_start
            avg_time  = elapsed / i
            remaining = avg_time * (len(physics_weights) - i)

            print(f"\n  Progress: {i/len(physics_weights)*100:.1f}% ({i}/{len(physics_weights)})")
            print(f"  Elapsed:   {timedelta(seconds=int(elapsed))}")
            print(f"  Remaining: {timedelta(seconds=int(remaining))}")
            print(f"  Est. finish: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + remaining))}")

            if i % 10 == 0 or i == len(physics_weights):
                checkpoint_path = os.path.join(
                    output_dir,
                    f'hostguest_grid_search_batched_checkpoint_{i}.json')
                with open(checkpoint_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    print(f"Total time:      {timedelta(seconds=int(total_time))}")
    print(f"Successful runs: {len(results)}/{len(physics_weights)}")
    print(f"Finished:        {time.strftime('%Y-%m-%d %H:%M:%S')}")

    final_path = os.path.join(output_dir, 'hostguest_grid_search_batched_FINAL.json')
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal results saved: {final_path}")

    if results:
        sorted_results = sorted(results, key=lambda x: x['best_val_rmse'])

        print("\n" + "=" * 80)
        print("TOP 10 CONFIGURATIONS BY BEST VAL RMSE")
        print("=" * 80)
        print(f"{'Rank':<6} {'Weight':<10} {'Best Val RMSE':<15} {'Best Emp':<12} "
              f"{'Best Phys':<12} {'Stopped Epoch'}")
        print("-" * 75)

        for rank, result in enumerate(sorted_results[:10], 1):
            print(f"{rank:<6} {result['physics_weight']:<10.4f} "
                  f"{result['best_val_rmse']:<15.4f} "
                  f"{result['best_val_empirical_loss']:<12.4f} "
                  f"{result['best_val_physics_loss']:<12.4f} "
                  f"{result['stopped_epoch']}")

        best = sorted_results[0]
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION")
        print("=" * 80)
        print(f"Physics Weight:   {best['physics_weight']:.6f}")
        print(f"Best Val RMSE:    {best['best_val_rmse']:.4f}")
        print(f"Best Val MAE:     {best['val_mae']:.4f}")
        print(f"Best Empirical:   {best['best_val_empirical_loss']:.4f}")
        print(f"Best Physics:     {best['best_val_physics_loss']:.4f}")
        print(f"Stopped at epoch: {best['stopped_epoch']}")
        print("=" * 80)


if __name__ == "__main__":
    main()