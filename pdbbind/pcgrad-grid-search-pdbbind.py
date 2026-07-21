"""
Grid Search for Physics Weight with PCGrad - PDBBind Dataset
Finds the best physics weight across 101 values (0.0 to 1.0).
Data loads once, all 101 runs reuse the same split.

To switch dataset size, change SUBSET_SIZE in Config.
"""

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import os
import random
import time
import json
from datetime import timedelta, datetime
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind')
sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind')

from models.layers_pytorch_pdbbind import PGGCNModel
# Also removed hydrogen atoms from the dataset
from train_split_data import load_data_with_saved_split

try:
    from models.pcgrad_pytorch import PCGrad
    PCGRAD_AVAILABLE = True
except ImportError:
    PCGRAD_AVAILABLE = False
    print("ERROR: PCGrad not available!")
    sys.exit(1)


# ============================================================================
# CONFIG — change SUBSET_SIZE to switch dataset size
# ============================================================================

class Config:
    # -------------------------------------------------------------------------
    # CHANGE THIS to switch dataset size: 100, 250, 500, 1000, 1500, 2000, 2660
    SUBSET_SIZE = 100
    # -------------------------------------------------------------------------

    SUBSET_DIR = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/subsets'
    CSV_PATH   = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind.csv'
    PKL_PATH   = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_full.pkl'

    # Model architecture — matches train_subset.py
    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL     = 20
    C_OUT_CHANNEL     = 1024
    DROPOUT_RATE      = 0.05   # tuned from sweep

    # Training hyperparameters — tuned from sweep
    EPOCHS      = 300
    BATCH_SIZE  = 8
    LEARNING_RATE = 5e-4
    L2_WEIGHT   = 1e-4
    MAX_NORM    = 3.0

    # Physics weights to search — 101 values from 0.0 to 1.0
    PHYSICS_WEIGHTS = np.arange(0.0, 1.01, 0.01)

    RANDOM_SEED = 50

    @property
    def split_path(self):
        return os.path.join(self.SUBSET_DIR, f'pdbbind_subset_{self.SUBSET_SIZE}.pkl')

    @property
    def save_dir(self):
        return (
            '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/'
            f'pytorch-implementation/pdbbind/grid_search/subset_{self.SUBSET_SIZE}'
        )


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


# ============================================================================
# SINGLE TRAINING RUN
# ============================================================================

def train_with_weight(model, train_loader, test_loader, physics_weight, config, device):
    """Train model with PCGrad for a specific physics weight. Returns metrics dict."""
    model = model.to(device)

    base_optimizer = optim.Adam(model.parameters(),
                                lr=config.LEARNING_RATE,
                                weight_decay=config.L2_WEIGHT)
    optimizer = PCGrad(base_optimizer)

    best_val_mae     = float('inf')
    best_model_state = None

    for epoch in range(config.EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)

            predictions, _, physics_info = model(X_batch, training=True)
            emp, phys_w, _, _ = compute_task_losses(
                predictions, y_batch, physics_info, physics_weight)

            optimizer.pc_backward([emp, phys_w])
            optimizer.step()
            apply_maxnorm_constraint(model, config.MAX_NORM)

        # ── Validate ──────────────────────────────────────────────────────────
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
            best_val_mae    = val_mae.item()
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best and evaluate
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        all_train_preds, all_train_targets, all_train_phys = [], [], []
        for X_batch, y_batch in train_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            preds, _, phys = model(X_batch, training=False)
            all_train_preds.append(preds)
            all_train_targets.append(y_batch)
            all_train_phys.append(phys)

        all_test_preds, all_test_targets, all_test_phys = [], [], []
        for X_batch, y_batch in test_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            preds, _, phys = model(X_batch, training=False)
            all_test_preds.append(preds)
            all_test_targets.append(y_batch)
            all_test_phys.append(phys)

        train_preds   = torch.cat(all_train_preds)
        train_targets = torch.cat(all_train_targets)
        train_phys    = torch.cat(all_train_phys)
        test_preds    = torch.cat(all_test_preds)
        test_targets  = torch.cat(all_test_targets)
        test_phys     = torch.cat(all_test_phys)

        train_emp, _, train_phys_loss, _ = compute_task_losses(
            train_preds, train_targets, train_phys, physics_weight)
        test_emp, _, test_phys_loss, test_mae = compute_task_losses(
            test_preds, test_targets, test_phys, physics_weight)

        # R²
        ss_res = torch.sum((test_targets - test_preds) ** 2).item()
        ss_tot = torch.sum((test_targets - torch.mean(test_targets)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot)

    return {
        'physics_weight':   float(physics_weight),
        'train_empirical':  train_emp.item(),
        'train_physics':    train_phys_loss.item(),
        'test_empirical':   test_emp.item(),
        'test_physics':     test_phys_loss.item(),
        'test_mae':         test_mae.item(),
        'test_r2':          r2,
        'best_val_mae':     best_val_mae,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config    = Config()

    print("=" * 80)
    print(f"GRID SEARCH: PHYSICS WEIGHT — Subset {config.SUBSET_SIZE}")
    print("=" * 80)
    print(f"Timestamp:      {timestamp}")
    print(f"Subset size:    {config.SUBSET_SIZE}")
    print(f"Weights to test: {len(config.PHYSICS_WEIGHTS)} (0.0 → 1.0)")
    print(f"Epochs per run:  {config.EPOCHS}")
    print(f"Learning rate:   {config.LEARNING_RATE}")
    print(f"L2 weight:       {config.L2_WEIGHT}")
    print(f"Dropout:         {config.DROPOUT_RATE}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device:          {device}\n")

    if not os.path.exists(config.split_path):
        raise FileNotFoundError(
            f"Subset pkl not found: {config.split_path}\n"
            f"Available sizes: 100, 250, 500, 1000, 2000, 2660"
        )

    set_random_seeds(config.RANDOM_SEED)

    # Load data ONCE — all 101 runs reuse this
    print("Loading data (this happens once)...")
    X_train, X_test, y_train, y_test = load_data_with_saved_split(config, config.split_path)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    train_loader = DataLoader(
        MoleculeDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_molecules
    )
    test_loader = DataLoader(
        MoleculeDataset(X_test, y_test),
        batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_molecules
    )

    os.makedirs(config.save_dir, exist_ok=True)
    save_path = os.path.join(config.save_dir, f'grid_search_{config.SUBSET_SIZE}_{timestamp}.json')
    csv_path  = save_path.replace('.json', '.csv')

    n_weights  = len(config.PHYSICS_WEIGHTS)
    results    = []
    start_time = time.time()

    print(f"Starting grid search over {n_weights} physics weights...\n")

    for i, weight in enumerate(config.PHYSICS_WEIGHTS):
        iter_start = time.time()
        print(f"[{i+1:3d}/{n_weights}] λ={weight:.2f}", end="  ", flush=True)

        # Fresh model for each weight
        model = PGGCNModel(
            num_atom_features=config.NUM_ATOM_FEATURES,
            r_out_channel=config.R_OUT_CHANNEL,
            c_out_channel=config.C_OUT_CHANNEL,
            dropout_rate=config.DROPOUT_RATE
        )
        model.add_rule("sum", 0, 32)
        model.add_rule("multiply", 32, 33)
        model.add_rule("distance", 33, 36)

        result = train_with_weight(
            model, train_loader, test_loader, weight, config, device)
        results.append(result)

        iter_time    = time.time() - iter_start
        elapsed      = time.time() - start_time
        eta          = (elapsed / (i + 1)) * (n_weights - i - 1)

        print(f"RMSE: {result['test_empirical']:.4f}  "
              f"MAE: {result['test_mae']:.4f}  "
              f"R²: {result['test_r2']:.4f}  "
              f"PhysLoss: {result['test_physics']:.4f}  "
              f"[{format_time(iter_time)}]  ETA: {format_time(eta)}")

        # Save incrementally every 10 runs
        if (i + 1) % 10 == 0:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            pd.DataFrame(results).to_csv(csv_path, index=False)
            print(f"Progress saved ({i+1}/{n_weights})")

    total_time = time.time() - start_time

    # Final save
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(csv_path, index=False)

    # Summary
    best_rmse    = min(results, key=lambda x: x['test_empirical'])
    best_mae     = min(results, key=lambda x: x['test_mae'])
    best_physics = min(results, key=lambda x: x['test_physics'])

    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    print(f"Total time: {format_time(total_time)}")
    print(f"Avg per weight: {format_time(total_time / n_weights)}\n")
    print(f"Best RMSE:         {best_rmse['test_empirical']:.4f} at λ={best_rmse['physics_weight']:.2f}")
    print(f"Best MAE:          {best_mae['test_mae']:.4f} at λ={best_mae['physics_weight']:.2f}")
    print(f"Best physics loss: {best_physics['test_physics']:.4f} at λ={best_physics['physics_weight']:.2f}")
    print(f"\n✓ Results saved to: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
