"""
PyTorch PGGCN Training Script - Host-Guest Final Version

Usage:
    python train_hostguest.py                  # Adam+Physics and RMSE-Only (default)
    python train_hostguest.py --no-pcgrad      # same as default
    python train_hostguest.py --pcgrad         # all three: PCGrad+Physics, Adam+Physics, RMSE-Only

Changes from original:
- Physics normalization using training set statistics only (no leakage)
- Best model checkpoint restoration before evaluation
- Three-way comparison via --pcgrad / --no-pcgrad flags
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
import argparse
from datetime import timedelta
from sklearn.model_selection import train_test_split

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

try:
    from PGGCN.models.dcFeaturizer import atom_features as get_atom_features
    from PGGCN.models.layers_pytorch import PGGCNModel
    print("✓ Imported using PGGCN.models style")
except ImportError:
    try:
        from models.dcFeaturizer import atom_features as get_atom_features
        from models.layers_pytorch import PGGCNModel
        print("✓ Imported using models style")
    except ImportError as e:
        print(f"✗ Error importing modules: {e}")
        sys.exit(1)

try:
    from pcgrad_pytorch import PCGrad
    PCGRAD_AVAILABLE = True
    print("✓ PCGrad optimizer available")
except ImportError:
    PCGRAD_AVAILABLE = False
    print("✗ PCGrad not available (will use standard optimizer)")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    CSV_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/Final_data_DDG.csv'
    PDB_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBs_RDKit_BFE.pkl'

    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL     = 20
    C_OUT_CHANNEL     = 128
    DROPOUT_RATE      = 0.2

    EPOCHS         = 250
    LEARNING_RATE  = 1e-4
    L2_WEIGHT      = 1e-4
    MAX_NORM       = 3.0
    PHYSICS_WEIGHT = 0.58   # best from grid search — update after new grid search

    TEST_SIZE   = 0.2
    RANDOM_SEED = 42

    SAVE_DIR = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/hostguest/saved_models'


# ============================================================================
# UTILITIES
# ============================================================================

def set_random_seeds(seed=42):
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


def normalize_physics(mol_list, mean, std):
    """Normalize last 15 columns (physics features) using provided mean/std."""
    normalized = []
    for mol in mol_list:
        mol = mol.clone()
        mol[:, -15:] = (mol[:, -15:] - mean) / std
        normalized.append(mol)
    return normalized


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


def load_data(csv_path, pdb_path):
    print(f"Loading PDB files from: {pdb_path}")
    with open(pdb_path, 'rb') as f:
        pdb_dict = pickle.load(f)
    print(f"✓ Loaded {len(pdb_dict)} PDB files")

    df_all = pd.read_csv(csv_path)
    print(f"✓ Found {len(df_all)} entries in CSV")

    feature_columns = [
        'pb_host_VDWAALS', 'pb_guest_VDWAALS', 'pb_complex_VDWAALS',
        'gb_host_1-4EEL', 'gb_guest_1-4EEL', 'gb_Complex_1-4EEL',
        'gb_host_EELEC', 'gb_guest_EELEC', 'gb_Complex_EELEC',
        'gb_host_EGB', 'gb_guest_EGB', 'gb_Complex_EGB',
        'gb_host_ESURF', 'gb_guest_ESURF', 'gb_Complex_ESURF'
    ]

    X, y = [], []
    failed = []

    for pdb_id in list(pdb_dict.keys()):
        if pdb_id not in df_all['Ids'].values:
            failed.append(pdb_id)
            continue

        molecule = pdb_dict[pdb_id]
        row = df_all[df_all['Ids'] == pdb_id].iloc[0]
        info_array = row[feature_columns].tolist()
        target = row['Ex _G_(kcal/mol)']

        try:
            features = featurize(molecule, info_array)
            X.append(torch.FloatTensor(features))
            y.append(float(target))
        except Exception as e:
            failed.append(pdb_id)
            print(f"  Warning: Failed to featurize {pdb_id}")

    print(f"✓ Successfully loaded {len(X)} complexes")
    if failed:
        print(f"  Warning: Failed to load {len(failed)} complexes")

    return X, y


# ============================================================================
# LOSS COMPUTATION
# ============================================================================

def compute_task_losses(predictions, targets, physics_info, physics_weight):
    targets = targets.view(-1, 1)

    empirical_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))

    host_energy    = physics_info[:, [0, 3, 6, 9, 12]].sum(dim=1, keepdim=True)
    guest_energy   = physics_info[:, [1, 4, 7, 10, 13]].sum(dim=1, keepdim=True)
    complex_energy = physics_info[:, [2, 5, 8, 11, 14]].sum(dim=1, keepdim=True)
    dG_physics = complex_energy - (host_energy + guest_energy)

    raw_physics_loss      = torch.sqrt(torch.mean((predictions - dG_physics) ** 2))
    weighted_physics_loss = physics_weight * raw_physics_loss

    return empirical_loss, weighted_physics_loss, raw_physics_loss


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val,
                use_pcgrad, use_physics, config, device, run_name):
    model = model.to(device)

    physics_weight = config.PHYSICS_WEIGHT if use_physics else 0.0

    if use_pcgrad and PCGRAD_AVAILABLE and use_physics:
        base_optimizer = optim.Adam(model.parameters(),
                                    lr=config.LEARNING_RATE, weight_decay=0)
        optimizer = PCGrad(base_optimizer)
        opt_name  = "PCGrad + Adam"
    else:
        base_optimizer = optim.Adam(model.parameters(),
                                    lr=config.LEARNING_RATE,
                                    weight_decay=config.L2_WEIGHT)
        optimizer = base_optimizer
        opt_name  = "Adam"

    X_train_dev    = [x.to(device) for x in X_train]
    X_val_dev      = [x.to(device) for x in X_val]
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_val_tensor   = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    print(f"\n{'='*80}")
    print(f"Training: {run_name}")
    print(f"{'='*80}")
    print(f"  Optimizer:      {opt_name}")
    print(f"  Physics loss:   {'ENABLED (λ=' + str(physics_weight) + ')' if use_physics else 'DISABLED'}")
    print(f"  Epochs:         {config.EPOCHS}")
    print(f"  Learning rate:  {config.LEARNING_RATE}")
    print(f"  L2 weight:      {config.L2_WEIGHT}")
    print(f"  Dropout:        {config.DROPOUT_RATE}")
    print(f"  Samples:        {len(X_train)} train | {len(X_val)} val")

    train_losses, val_losses           = [], []
    train_empirical_losses             = []
    train_physics_losses               = []
    val_empirical_losses               = []
    val_physics_losses                 = []

    best_val_loss    = float('inf')
    best_epoch       = 0
    best_model_state = None
    start_time       = time.time()

    for epoch in range(config.EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        predictions, model_var, physics_info = model(X_train_dev, training=True)

        train_empirical, train_weighted_physics, train_raw_physics = compute_task_losses(
            predictions, y_train_tensor, physics_info, physics_weight)

        if use_pcgrad and PCGRAD_AVAILABLE and use_physics:
            optimizer.pc_backward([train_empirical, train_weighted_physics])
            optimizer.step()
        else:
            optimizer.zero_grad()
            total_loss = train_empirical + train_weighted_physics if use_physics else train_empirical
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        apply_maxnorm_constraint(model, config.MAX_NORM)

        train_loss_total = train_empirical.item() + train_weighted_physics.item()
        train_losses.append(train_loss_total)
        train_empirical_losses.append(train_empirical.item())
        train_physics_losses.append(train_raw_physics.item())

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_predictions, _, val_physics_info = model(X_val_dev, training=False)
            val_empirical, val_weighted_physics, val_raw_physics = compute_task_losses(
                val_predictions, y_val_tensor, val_physics_info, physics_weight)
            val_loss_total = val_empirical.item() + val_weighted_physics.item()

        val_losses.append(val_loss_total)
        val_empirical_losses.append(val_empirical.item())
        val_physics_losses.append(val_raw_physics.item())

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_loss_total < best_val_loss:
            best_val_loss    = val_loss_total
            best_epoch       = epoch + 1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # ── Logging ───────────────────────────────────────────────────────────
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta     = (elapsed / (epoch + 1)) * (config.EPOCHS - epoch - 1)
            print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train: {train_loss_total:.4f} "
                  f"(Emp: {train_empirical.item():.4f}, Phys: {train_raw_physics.item():.4f}) | "
                  f"Val: {val_loss_total:.4f} "
                  f"(Emp: {val_empirical.item():.4f}, Phys: {val_raw_physics.item():.4f}) | "
                  f"Time: {format_time(elapsed)} | ETA: {format_time(eta)}")

    # Restore best checkpoint
    if best_model_state is not None:
        print(f"\nRestoring best model from epoch {best_epoch} "
              f"(val loss: {best_val_loss:.4f})")
        model.load_state_dict(best_model_state)

    total_time = time.time() - start_time
    print(f"Completed in {format_time(total_time)}")

    history = {
        'train_losses':    train_losses,
        'val_losses':      val_losses,
        'train_empirical': train_empirical_losses,
        'train_physics':   train_physics_losses,
        'val_empirical':   val_empirical_losses,
        'val_physics':     val_physics_losses,
    }

    return history, best_val_loss, best_epoch


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, config, device):
    model.eval()
    X_test_dev    = [x.to(device) for x in X_test]
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    with torch.no_grad():
        predictions, _, physics_info = model(X_test_dev, training=False)

        rmse = torch.sqrt(torch.mean((predictions - y_test_tensor) ** 2)).item()
        mae  = torch.mean(torch.abs(predictions - y_test_tensor)).item()

        ss_res = torch.sum((y_test_tensor - predictions) ** 2).item()
        ss_tot = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2).item()
        r2     = 1 - (ss_res / ss_tot)

        _, _, phys_loss = compute_task_losses(
            predictions, y_test_tensor, physics_info, config.PHYSICS_WEIGHT)

    return predictions.cpu().numpy(), {
        'rmse':         rmse,
        'mae':          mae,
        'r2':           r2,
        'physics_loss': phys_loss.item(),
    }


def print_sample_predictions(predictions, y_test, n=10):
    print("\n" + "-" * 60)
    print("Sample Predictions")
    print("-" * 60)
    print(f"{'True Value':>12} | {'Prediction':>12} | {'Error':>12}")
    print("-" * 42)
    for i in range(min(n, len(y_test))):
        true_val = y_test[i]
        pred_val = predictions[i][0]
        print(f"{true_val:>12.4f} | {pred_val:>12.4f} | {pred_val - true_val:>12.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Host-Guest PGGCN Training')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--no-pcgrad', action='store_true',
                       help='Adam+Physics and RMSE-Only (default)')
    group.add_argument('--pcgrad', action='store_true',
                       help='All three: PCGrad+Physics, Adam+Physics, RMSE-Only')
    args = parser.parse_args()

    print("=" * 80)
    print("HOST-GUEST PGGCN TRAINING")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    config = Config()
    set_random_seeds(config.RANDOM_SEED)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("Loading Data")
    print("-" * 80)
    X, y = load_data(config.CSV_PATH, config.PDB_PATH)

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("Splitting Data")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    print(f"✓ Training: {len(X_train)} samples")
    print(f"✓ Test:     {len(X_test)} samples")

    # ── Physics normalization (train stats only) ───────────────────────────────
    train_physics = torch.stack([mol[0, -15:] for mol in X_train])
    phys_mean = train_physics.mean(dim=0)
    phys_std  = train_physics.std(dim=0).clamp(min=1e-8)

    X_train = normalize_physics(X_train, phys_mean, phys_std)
    X_test  = normalize_physics(X_test,  phys_mean, phys_std)

    normalized_check = torch.stack([m[0, -15:] for m in X_train])
    print(f"\n✓ Physics normalized:")
    print(f"  Max abs: {normalized_check.abs().max().item():.4f}")
    print(f"  Mean:    {normalized_check.mean().item():.4f}")
    print(f"  Std:     {normalized_check.std().item():.4f}")

    print(f"\nTarget statistics (train):")
    print(f"  Mean: {np.mean(y_train):.2f} kcal/mol")
    print(f"  Std:  {np.std(y_train):.2f} kcal/mol")
    print(f"  Min:  {np.min(y_train):.2f} kcal/mol")
    print(f"  Max:  {np.max(y_train):.2f} kcal/mol")

    # ── Define runs ───────────────────────────────────────────────────────────
    if args.pcgrad:
        runs = [
            ('ΔΔG with PCGrad + Multi-loss', True,  True),
            ('ΔΔG with Adam + Multi-loss',   False, True),
            ('ΔΔG without Multi-loss',        False, False),
        ]
    else:
        # default / --no-pcgrad
        runs = [
            ('ΔΔG with Adam + Multi-loss', False, True),
            ('ΔΔG without Multi-loss',      False, False),
        ]

    results = {}

    # ── Training loop ─────────────────────────────────────────────────────────
    for run_name, use_pcgrad, use_physics in runs:
        model = PGGCNModel(
            num_atom_features=config.NUM_ATOM_FEATURES,
            r_out_channel=config.R_OUT_CHANNEL,
            c_out_channel=config.C_OUT_CHANNEL,
            dropout_rate=config.DROPOUT_RATE
        )
        model.add_rule("sum", 0, 32)
        model.add_rule("multiply", 32, 33)
        model.add_rule("distance", 33, 36)

        # Verify dense_final initialization
        expected = torch.tensor([0.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1])
        if torch.allclose(model.dense_final.weight.data.flatten(), expected, atol=0.01):
            print(f"\n dense_final weights correctly initialized")
        else:
            print(f"\n WARNING: dense_final weights not correctly initialized!")

        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        history, best_val_loss, best_epoch = train_model(
            model, X_train, y_train, X_test, y_test,
            use_pcgrad, use_physics, config, device, run_name)

        predictions, metrics = evaluate_model(model, X_test, y_test, config, device)

        print(f"\nResults — {run_name}:")
        print(f"  RMSE:          {metrics['rmse']:.4f} kcal/mol")
        print(f"  MAE:           {metrics['mae']:.4f} kcal/mol")
        print(f"  R²:            {metrics['r2']:.4f}")
        print(f"  Physics loss:  {metrics['physics_loss']:.4f}")
        print(f"  Best val loss: {best_val_loss:.4f} @ epoch {best_epoch}")

        print_sample_predictions(predictions, y_test)

        results[run_name] = {
            'history':      history,
            'metrics':      metrics,
            'predictions':  predictions,
            'best_val_loss': best_val_loss,
            'best_epoch':   best_epoch,
        }

        # Save model
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        suffix = 'pcgrad' if use_pcgrad else ('adam' if use_physics else 'rmse_only')
        save_path = os.path.join(
            config.SAVE_DIR,
            f"pggcn_hostguest_{suffix}_pw{config.PHYSICS_WEIGHT}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'history':          history,
            'metrics':          metrics,
            'best_val_loss':    best_val_loss,
            'best_epoch':       best_epoch,
            'config': {
                'learning_rate':  config.LEARNING_RATE,
                'l2_weight':      config.L2_WEIGHT,
                'dropout_rate':   config.DROPOUT_RATE,
                'physics_weight': config.PHYSICS_WEIGHT,
                'use_pcgrad':     use_pcgrad,
                'use_physics':    use_physics,
            }
        }, save_path)
        print(f"\nModel saved to: {save_path}")

    # ── Final comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"\n{'Model':<35} | {'RMSE':>8} | {'MAE':>8} | {'R²':>8} | {'PhysLoss':>10}")
    print("-" * 78)
    for name, data in results.items():
        m = data['metrics']
        print(f"{name:<35} | "
              f"{m['rmse']:>8.4f} | "
              f"{m['mae']:>8.4f} | "
              f"{m['r2']:>8.4f} | "
              f"{m['physics_loss']:>10.4f}")

if __name__ == "__main__":
    main()