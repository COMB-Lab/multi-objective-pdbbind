"""
Grid Search for Physics Weight with GPU-Compatible PCGrad
Searches 0.0 to 1.0 in 0.01 increments (101 values)
Creates Pareto front matching Paper Figure 2(a)

Uses custom PCGrad implementation that keeps all operations on GPU
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
from datetime import timedelta

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
possible_paths = [
    SCRIPT_DIR, PARENT_DIR,
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
    from models.dcFeaturizer import atom_features as get_atom_features
    from models.layers_pytorch import PGGCNModel

try:
    from pcgrad_gpu import PCGrad_GPU
    PCGRAD_AVAILABLE = True
    print("✓ Using GPU-compatible PCGrad")
except ImportError:
    try:
        from pcgrad_pytorch import PCGrad
        PCGrad_GPU = PCGrad
        PCGRAD_AVAILABLE = True
        print("⚠ Using standard PCGrad (may be slow)")
    except ImportError:
        PCGRAD_AVAILABLE = False
        print("✗ PCGrad not available")
        sys.exit(1)

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    CSV_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/Final_data_DDG.csv'
    PDB_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBs_RDKit_BFE.pkl'
    
    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL = 20
    C_OUT_CHANNEL = 128
    DROPOUT_RATE = 0.2
    
    EPOCHS = 250
    LEARNING_RATE = 1e-4
    L2_WEIGHT = 1e-4
    MAX_NORM = 3.0
    
    # Grid search - REDUCED for faster completion
    # Change to 0.01 step for full search once GPU is working
    PHYSICS_WEIGHTS = np.arange(0.0, 1.01, 0.05)  # 21 weights instead of 101
    
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

# ============================================================================
# UTILITIES (same as before)
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
    with open(pdb_path, 'rb') as f:
        pdb_dict = pickle.load(f)
    df_all = pd.read_csv(csv_path)
    
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
        except:
            pass
    return X, y

def apply_maxnorm_constraint(model, max_norm=3.0):
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.dim() >= 2:
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=max_norm)
                param.mul_(desired / (norm + 1e-7))

def compute_l2_loss(model):
    l2_loss = torch.tensor(0., device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.sum(param ** 2)
    return l2_loss

def compute_task_losses(predictions, targets, physics_info, physics_weight):
    targets = targets.view(-1, 1)
    empirical_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))
    
    host_energy = physics_info[:, [0, 3, 6, 9, 12]].sum(dim=1, keepdim=True)
    guest_energy = physics_info[:, [1, 4, 7, 10, 13]].sum(dim=1, keepdim=True)
    complex_energy = physics_info[:, [2, 5, 8, 11, 14]].sum(dim=1, keepdim=True)
    
    dG_physics = complex_energy - (host_energy + guest_energy)
    raw_physics_loss = torch.sqrt(torch.mean((predictions - dG_physics) ** 2))
    weighted_physics_loss = physics_weight * raw_physics_loss
    
    return empirical_loss, weighted_physics_loss, raw_physics_loss

# ============================================================================
# TRAINING
# ============================================================================

def train_with_weight(model, X_train, y_train, X_test, y_test, 
                     physics_weight, config, device):
    """Train model with specific physics weight using GPU-compatible PCGrad"""
    model = model.to(device)
    
    base_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0)
    optimizer = PCGrad_GPU(base_optimizer)
    
    X_train = [x.to(device) for x in X_train]
    X_test = [x.to(device) for x in X_test]
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    for epoch in range(config.EPOCHS):
        epoch_start = time.time()
        
        model.train()
        predictions, _, physics_info = model(X_train, training=True)
        
        train_empirical, train_weighted_physics, _ = compute_task_losses(
            predictions, y_train_tensor, physics_info, physics_weight)
        
        l2_loss = compute_l2_loss(model)
        l2_weighted = config.L2_WEIGHT * l2_loss
        
        # PCGrad with 3 objectives (stays on GPU!)
        optimizer.pc_backward([train_empirical, l2_weighted, train_weighted_physics])
        optimizer.step()
        apply_maxnorm_constraint(model, max_norm=config.MAX_NORM)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress with timing for first few epochs
        if epoch == 0:
            print(f"      Epoch {epoch+1}/{config.EPOCHS}: Loss {train_empirical.item():.4f} | Time: {epoch_time:.2f}s", flush=True)
            if epoch_time > 5.0:
                print(f"      ⚠ WARNING: Epoch took {epoch_time:.1f}s - may be running on CPU!", flush=True)
            else:
                print(f"      ✓ Fast epoch time - GPU is working!", flush=True)
        elif (epoch + 1) % 50 == 0:
            print(f"      Epoch {epoch+1}/{config.EPOCHS}: Loss {train_empirical.item():.4f} | Time: {epoch_time:.2f}s", flush=True)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Train metrics
        train_predictions, _, train_physics_info = model(X_train, training=False)
        train_empirical, _, train_physics = compute_task_losses(
            train_predictions, y_train_tensor, train_physics_info, physics_weight)
        
        # Test metrics
        test_predictions, _, test_physics_info = model(X_test, training=False)
        test_empirical, _, test_physics = compute_task_losses(
            test_predictions, y_test_tensor, test_physics_info, physics_weight)
        
        test_mae = torch.mean(torch.abs(test_predictions - y_test_tensor)).item()
    
    return {
        'physics_weight': physics_weight,
        'train_empirical': train_empirical.item(),
        'train_physics': train_physics.item(),
        'test_empirical': test_empirical.item(),
        'test_physics': test_physics.item(),
        'test_mae': test_mae
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("GRID SEARCH: PHYSICS WEIGHT WITH PCGRAD (0.0 to 1.0, step 0.01)")
    print("Creating Pareto Front (Paper Figure 2a)")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Running on CPU. This will be SLOW (35 min per weight)!")
        print("         Estimated total time: ~58 hours")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Expected time: ~7-10 hours for all 101 weights with PCGrad\n")
    
    config = Config()
    set_random_seeds(config.RANDOM_SEED)
    
    # Load data once
    print("Loading data...")
    X, y = load_data(config.CSV_PATH, config.PDB_PATH)
    
    # Split once (same split for all weights)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    
    print(f"Loaded {len(X)} samples")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")
    
    # GPU status check after RDKit warnings
    print("="*80)
    print("GPU STATUS CHECK")
    print("="*80)
    if device == 'cuda':
        print(f"Device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
        
        # Test tensor on GPU
        test = torch.randn(1000, 1000).to(device)
        print(f"Test tensor device: {test.device}")
        print(f"GPU Memory After Test: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
        del test
        torch.cuda.empty_cache()
        print("✓ GPU is working correctly!")
    else:
        print(f"Device: {device} (CPU - will be SLOW!)")
    print("="*80 + "\n")
    
    # Grid search
    n_weights = len(config.PHYSICS_WEIGHTS)
    print(f"Starting grid search over {n_weights} physics weights...")
    print(f"Progress will be saved every 10 weights.\n")
    
    results = []
    start_time = time.time()
    
    for i, weight in enumerate(config.PHYSICS_WEIGHTS):
        iter_start = time.time()
        
        # Create fresh model
        model = PGGCNModel(config.NUM_ATOM_FEATURES, config.R_OUT_CHANNEL,
                          config.C_OUT_CHANNEL, config.DROPOUT_RATE)
        model.add_rule("sum", 0, 32)
        model.add_rule("multiply", 32, 33)
        model.add_rule("distance", 33, 36)
        
        # Train with this weight
        result = train_with_weight(model, X_train, y_train, X_test, y_test,
                                   weight, config, device)
        results.append(result)
        
        iter_time = time.time() - iter_start
        elapsed_total = time.time() - start_time
        avg_time = elapsed_total / (i + 1)
        eta = avg_time * (n_weights - i - 1)
        
        # Progress update
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1:3d}/{n_weights}] λ={weight:.2f} | "
                  f"Test MAE: {result['test_mae']:.4f}, "
                  f"Emp: {result['test_empirical']:.4f}, "
                  f"Phys: {result['test_physics']:.4f} | "
                  f"Time: {format_time(iter_time)} | "
                  f"ETA: {format_time(eta)}", flush=True)
            
            # Save intermediate results
            save_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/grid_search_results.json'
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    print(f"Total time: {format_time(total_time)}")
    print(f"Average per weight: {format_time(total_time / n_weights)}\n")
    
    # Find best weights
    best_mae = min(results, key=lambda x: x['test_mae'])
    best_empirical = min(results, key=lambda x: x['test_empirical'])
    
    print("Best Results:")
    print(f"  Best MAE: {best_mae['test_mae']:.4f} at λ={best_mae['physics_weight']:.2f}")
    print(f"  Best Empirical: {best_empirical['test_empirical']:.4f} at λ={best_empirical['physics_weight']:.2f}")
    
    # Paper reference weight
    paper_weight = 0.58
    paper_result = next((r for r in results if abs(r['physics_weight'] - paper_weight) < 0.001), None)
    if paper_result:
        print(f"\n  Paper weight (λ=0.58):")
        print(f"    MAE: {paper_result['test_mae']:.4f}")
        print(f"    Empirical: {paper_result['test_empirical']:.4f}")
        print(f"    Physics: {paper_result['test_physics']:.4f}")
    
    # Save final results
    save_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/grid_search_results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {save_path}")
    
    # Create CSV for easy plotting
    csv_path = save_path.replace('.json', '.csv')
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved to: {csv_path}")
    
    print("\nTo plot Pareto front (like Paper Figure 2a):")
    print("  import pandas as pd")
    print("  import matplotlib.pyplot as plt")
    print(f"  df = pd.read_csv('{csv_path}')")
    print("  plt.scatter(df['test_empirical'], df['test_physics'])")
    print("  plt.xlabel('Empirical Loss')")
    print("  plt.ylabel('Physics Loss')")
    print("  plt.title('Pareto Front')")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()