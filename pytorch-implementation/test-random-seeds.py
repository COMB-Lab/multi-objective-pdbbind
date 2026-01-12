"""
Test PyTorch PGGCN with multiple random seeds to find which gives results closest to TensorFlow.
This will help us understand if the 0.5 RMSE difference is due to initialization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
import pandas as pd
import numpy as np
import sys
import os
import random
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
import pickle
import json

# Add the necessary directories to the path
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

# Try different import styles
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


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_l2_loss(model, l2_weight=1e-4):
    """Compute L2 regularization for all parameters."""
    l2_loss = torch.tensor(0., device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.sum(param ** 2)
    return (l2_weight / 2) * l2_loss


def apply_maxnorm_constraint(model, max_norm=3.0):
    """Apply MaxNorm constraint to all parameters."""
    with torch.no_grad():
        for param in model.named_parameters():
            if param[1].requires_grad and param[1].dim() >= 2:
                norm = param[1].norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=max_norm)
                param[1].mul_(desired / (norm + 1e-7))


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss function."""
    def __init__(self, physics_consistency_weight=0.005):
        super(PhysicsInformedLoss, self).__init__()
        self.physics_consistency_weight = physics_consistency_weight

    def forward(self, predictions, targets, model_vars, physics_info):
        targets = targets.view(-1, 1)
        
        # Empirical loss
        empirical_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))
        
        # Physics loss
        host_energy = physics_info[:, [0, 3, 6, 9, 12]].sum(dim=1, keepdim=True)
        guest_energy = physics_info[:, [1, 4, 7, 10, 13]].sum(dim=1, keepdim=True)
        complex_energy = physics_info[:, [2, 5, 8, 11, 14]].sum(dim=1, keepdim=True)
        dG_physics = complex_energy - (host_energy + guest_energy)
        physics_loss = torch.sqrt(torch.mean((predictions - dG_physics) ** 2))
        
        total_loss = empirical_loss + (self.physics_consistency_weight * physics_loss)
        return total_loss, empirical_loss, physics_loss


def featurize(molecule, info):
    """Featurize a molecule."""
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


def load_data(info_csv_path, hostguest_dir):
    """Load dataset."""
    with open(hostguest_dir, 'rb') as f:
        pdb_dict = pickle.load(f)
    
    df_all = pd.read_csv(info_csv_path)
    
    feature_columns = [
        'pb_host_VDWAALS', 'pb_guest_VDWAALS', 'pb_complex_VDWAALS',
        'gb_host_1-4EEL', 'gb_guest_1-4EEL', 'gb_Complex_1-4EEL',
        'gb_host_EELEC', 'gb_guest_EELEC', 'gb_Complex_EELEC',
        'gb_host_EGB', 'gb_guest_EGB', 'gb_Complex_EGB',
        'gb_host_ESURF', 'gb_guest_ESURF', 'gb_Complex_ESURF'
    ]
    
    X = []
    y = []
    
    for pdb_id in list(pdb_dict.keys()):
        if pdb_id not in df_all['Ids'].values:
            continue
        
        molecule = pdb_dict[pdb_id]
        row = df_all[df_all['Ids'] == pdb_id].iloc[0]
        info_array = row[feature_columns].tolist()
        target = row['Ex _G_(kcal/mol)']
        
        try:
            features = featurize(molecule, info_array)
            features_tensor = torch.FloatTensor(features)
            X.append(features_tensor)
            y.append(target)
        except Exception:
            continue
    
    return X, y


def train_one_seed(seed, X_train, y_train, X_test, y_test, device, epochs=250, verbose=False):
    """Train model with one specific seed."""
    # Set seed
    if seed is not None:
        set_seed(seed)
    
    # Create model
    model = PGGCNModel(num_atom_features=36, r_out_channel=20, c_out_channel=128, dropout_rate=0.2)
    model.add_rule("sum", 0, 32)
    model.add_rule("multiply", 32, 33)
    model.add_rule("distance", 33, 36)
    model = model.to(device)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = PhysicsInformedLoss(physics_consistency_weight=0.005)
    
    # Move data to device
    X_train_dev = [x.to(device) for x in X_train]
    X_test_dev = [x.to(device) for x in X_test]
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    # Track first epoch loss for comparison
    first_epoch_loss = None
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions, model_var, physics_info = model(X_train_dev, training=True)
        train_loss, train_emp, train_phys = criterion(predictions, y_train_tensor, model_var, physics_info)
        
        l2_reg = compute_l2_loss(model, 1e-4)
        train_loss = train_loss + l2_reg
        
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        apply_maxnorm_constraint(model, max_norm=3.0)
        
        if epoch == 0:
            first_epoch_loss = train_loss.item()
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_loss.item():.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        predictions, model_var, physics_info = model(X_test_dev, training=False)
        mse = nn.MSELoss()(predictions, y_test_tensor).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(predictions - y_test_tensor)).item()
    
    return rmse, mae, first_epoch_loss, predictions.cpu().numpy()


def main():
    print("=" * 80)
    print("TESTING MULTIPLE RANDOM SEEDS")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Paths
    info_csv_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/Final_data_DDG.csv'
    hostguest_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBs_RDKit_BFE.pkl'
    
    # Load data
    print("Loading data...")
    X, y = load_data(info_csv_path, hostguest_dir)
    print(f"Loaded {len(X)} samples\n")
    
    # Split (using same random_state as always)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different seeds
    seeds_to_test = [
        None,  # No seed (system random)
        1, 10, 42, 100, 123, 
        456, 789, 999, 1234, 2024,
        # Add more if needed
    ]
    
    results = []
    
    print("=" * 80)
    print("TRAINING WITH DIFFERENT SEEDS")
    print("=" * 80)
    print(f"{'Seed':<10} | {'RMSE':<8} | {'MAE':<8} | {'1st Epoch Loss':<15} | {'vs TF (2.32)':<12}")
    print("-" * 80)
    
    target_rmse = 2.3193  # TensorFlow result
    
    for i, seed in enumerate(seeds_to_test):
        start_time = time.time()
        
        seed_str = "None" if seed is None else str(seed)
        print(f"Testing seed {seed_str}...", end=" ", flush=True)
        
        rmse, mae, first_loss, predictions = train_one_seed(
            seed, X_train, y_train, X_test, y_test, device, 
            epochs=250, verbose=False
        )
        
        elapsed = time.time() - start_time
        diff_from_tf = rmse - target_rmse
        
        results.append({
            'seed': seed,
            'rmse': rmse,
            'mae': mae,
            'first_epoch_loss': first_loss,
            'diff_from_tf': diff_from_tf,
            'predictions': predictions.tolist(),
            'training_time': elapsed
        })
        
        print(f"Done in {elapsed:.0f}s")
        print(f"{seed_str:<10} | {rmse:<8.4f} | {mae:<8.4f} | {first_loss:<15.4f} | {diff_from_tf:+.4f}")
    
    print("=" * 80)
    
    # Find best seed
    best_result = min(results, key=lambda x: abs(x['diff_from_tf']))
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTarget (TensorFlow): RMSE = 2.3193, MAE = 1.9162")
    print(f"\nBest seed: {best_result['seed']}")
    print(f"  RMSE: {best_result['rmse']:.4f} (diff: {best_result['diff_from_tf']:+.4f})")
    print(f"  MAE:  {best_result['mae']:.4f}")
    print(f"  First epoch loss: {best_result['first_epoch_loss']:.4f}")
    
    # Show top 5 closest to TensorFlow
    print("\nTop 5 seeds closest to TensorFlow:")
    sorted_results = sorted(results, key=lambda x: abs(x['diff_from_tf']))
    for i, result in enumerate(sorted_results[:5], 1):
        seed_str = "None" if result['seed'] is None else str(result['seed'])
        print(f"  {i}. Seed {seed_str:<6}: RMSE = {result['rmse']:.4f} (diff: {result['diff_from_tf']:+.4f})")
    
    # Statistical summary
    rmses = [r['rmse'] for r in results]
    print(f"\nAcross all seeds:")
    print(f"  Mean RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"  Min RMSE:  {np.min(rmses):.4f}")
    print(f"  Max RMSE:  {np.max(rmses):.4f}")
    print(f"  Range:     {np.max(rmses) - np.min(rmses):.4f}")
    
    # Save results
    output_file = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/seed_test_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert predictions to list for JSON serialization
    results_to_save = []
    for r in results:
        r_copy = r.copy()
        r_copy['predictions'] = r_copy['predictions']  # Already converted to list
        results_to_save.append(r_copy)
    
    with open(output_file, 'w') as f:
        json.dump({
            'target_tensorflow': {'rmse': 2.3193, 'mae': 1.9162},
            'results': results_to_save,
            'summary': {
                'mean_rmse': float(np.mean(rmses)),
                'std_rmse': float(np.std(rmses)),
                'min_rmse': float(np.min(rmses)),
                'max_rmse': float(np.max(rmses)),
                'best_seed': best_result['seed']
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()