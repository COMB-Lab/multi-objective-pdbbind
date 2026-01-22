"""
PDBBind Training with BATCHING - Testing Model Parameters
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
from datetime import timedelta
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind')

from models.dcFeaturizer import atom_features as get_atom_features
from models.layers_pytorch import RuleGraphConvLayer, ConvLayer

# ============================================================================
# MODEL (Same as before)
# ============================================================================

class PGGCNModel(nn.Module):
    def __init__(self, num_atom_features=36, r_out_channel=20, c_out_channel=1024, dropout_rate=0.2):
        super(PGGCNModel, self).__init__()
        self.num_atom_features = num_atom_features
        self.num_physics_features = 15
        
        self.rule_graph_conv = RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
        self.conv = ConvLayer(c_out_channel, r_out_channel)
        
        self.dense1 = nn.Linear(c_out_channel, 32)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.model_var_layer = nn.Linear(16, 1)
        self.dense_final = nn.Linear(16, 1)
        
        with torch.no_grad():
            init_weights = torch.tensor([
                0.3, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0
            ]).reshape(1, 16)
            self.dense_final.weight.copy_(init_weights)
            self.dense_final.bias.zero_()
        
        self.relu = nn.ReLU()
        print(f"Model: c_out={c_out_channel}, dense=32→16, params={sum(p.numel() for p in self.parameters()):,}")

    def add_rule(self, rule, start_index, end_index=None):
        self.rule_graph_conv.add_rule(rule, start_index, end_index)

    def forward(self, batch_molecules, training=True):
        atom_features_batch = []
        physics_info_batch = []
        
        for mol in batch_molecules:
            atom_feat = mol[:, :self.num_atom_features + 2]
            atom_features_batch.append(atom_feat)
            physics_info = mol[0, -self.num_physics_features:]
            physics_info_batch.append(physics_info)
        
        physics_info_tensor = torch.stack(physics_info_batch)
        
        x = self.rule_graph_conv(atom_features_batch)
        x = self.conv(x)
        x = self.dense1(x)
        x = self.relu(x)
        if training:
            x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu(x)
        if training:
            x = self.dropout2(x)
        model_var = self.model_var_layer(x)
        merged = torch.cat([model_var, physics_info_tensor], dim=1)
        out = self.dense_final(merged)
        
        return out, model_var, physics_info_tensor


# ============================================================================
# DATASET CLASS FOR BATCHING
# ============================================================================

class MoleculeDataset(Dataset):
    """Dataset wrapper for batching molecules"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_molecules(batch):
    """
    Custom collate function for variable-sized molecules
    Returns lists instead of stacked tensors
    """
    X_batch = [item[0] for item in batch]
    y_batch = [item[1] for item in batch]
    return X_batch, torch.FloatTensor(y_batch)


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    CSV_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind_100.csv'
    PKL_PATH = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_100.pkl'
    CSV_PATH_ALT = '/home/exouser/multiloss-bfe/multiloss_pdbbind/Datasets/pdbbind_100.csv'
    PKL_PATH_ALT = '/home/exouser/multiloss-bfe/multiloss_pdbbind/Datasets/PDBBind_100.pkl'
    
    NUM_ATOM_FEATURES = 36
    R_OUT_CHANNEL = 20
    C_OUT_CHANNEL = 1024
    DROPOUT_RATE = 0.2
    
    EPOCHS = 100  # Match TensorFlow
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    L2_WEIGHT = 1e-2
    MAX_NORM = 3.0
    PHYSICS_WEIGHT = 1e-6
    
    TEST_SIZE = 0.2
    RANDOM_SEED = 50  # Match TensorFlow


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


def load_data(config):
    print("Loading PDBBind dataset...")
    
    for csv_path, pkl_path in [
        (config.CSV_PATH, config.PKL_PATH),
        (config.CSV_PATH_ALT, config.PKL_PATH_ALT)
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


def apply_maxnorm_constraint(model, max_norm=3.0):
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.dim() >= 2:
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=max_norm)
                param.mul_(desired / (norm + 1e-7))


# ============================================================================
# TRAINING WITH BATCHING
# ============================================================================

def train_model_batched(model, train_loader, val_loader, config, device):
    """Train with mini-batches like TensorFlow"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.L2_WEIGHT)
    
    print(f"\n{'='*80}")
    print("TRAINING WITH BATCHING (MATCHES TENSORFLOW)")
    print("="*80)
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Batches per epoch: ~{len(train_loader)}")
    print(f"Weight updates per epoch: {len(train_loader)}")
    print(f"Total updates over {config.EPOCHS} epochs: {len(train_loader) * config.EPOCHS}")
    print(f"LR: {config.LEARNING_RATE}, Physics: {config.PHYSICS_WEIGHT}")
    print("="*80)
    
    history = {'train_mae': [], 'val_mae': []}
    best_val_mae = float('inf')
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        epoch_train_emp = []      # Track empirical losses
        epoch_train_phys = []     # Track physics losses
        epoch_train_total = []    # Track total losses
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Move batch to device
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions, model_var, physics_info = model(X_batch, training=True)
            
            # Compute loss
            train_emp, train_phys_w, train_phys_r = compute_task_losses(
                predictions, y_batch, physics_info, config.PHYSICS_WEIGHT)
            
            total_loss = train_emp + train_phys_w
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            apply_maxnorm_constraint(model, config.MAX_NORM)
            
            # Track metrics for this batch
            epoch_train_emp.append(train_emp.item())
            epoch_train_phys.append(train_phys_r.item())
            epoch_train_total.append(total_loss.item())
        
        # Average across batches
        avg_train_emp = np.mean(epoch_train_emp)
        avg_train_phys = np.mean(epoch_train_phys)
        avg_train_total = np.mean(epoch_train_total)
        
        # Validation (full dataset)
        model.eval()
        with torch.no_grad():
            all_val_preds = []
            all_val_targets = []
            all_val_phys = []
            
            for X_batch, y_batch in val_loader:
                X_batch = [x.to(device) for x in X_batch]
                y_batch = y_batch.unsqueeze(1).to(device)
                
                val_pred, val_model_var, val_phys = model(X_batch, training=False)
                
                all_val_preds.append(val_pred)
                all_val_targets.append(y_batch)
                all_val_phys.append(val_phys)
            
            val_predictions = torch.cat(all_val_preds, dim=0)
            val_targets = torch.cat(all_val_targets, dim=0)
            val_physics = torch.cat(all_val_phys, dim=0)
            
            # Compute validation losses
            val_emp, val_phys_w, val_phys_r = compute_task_losses(
                val_predictions, val_targets, val_physics, config.PHYSICS_WEIGHT)
            
            val_total_loss = val_emp.item() + val_phys_w.item()
            val_mae = torch.mean(torch.abs(val_predictions - val_targets)).item()
        
        history['train_mae'].append(val_mae)
        history['val_mae'].append(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
        
        # Progress - UPDATED FORMAT
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (config.EPOCHS - epoch - 1)
            
            print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train: {avg_train_total:.4f} "
                  f"(Emp: {avg_train_emp:.4f}, Phys: {avg_train_phys:.4f}) | "
                  f"Val: {val_total_loss:.4f} "
                  f"(Emp: {val_emp.item():.4f}, Phys: {val_phys_r.item():.4f}) | "
                  f"Time: {format_time(elapsed)} | ETA: {format_time(eta)}")
    
    print(f"\nCompleted {config.EPOCHS} epochs in {format_time(time.time() - start_time)}")
    print(f"Best validation MAE: {best_val_mae:.2f} at epoch {best_epoch}")
    
    return history


def evaluate_model(model, test_loader, device):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_batch = y_batch.unsqueeze(1).to(device)
            
            predictions, model_var, physics_info = model(X_batch, training=False)
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


def main():
    print("="*80)
    print("PDBBIND WITH BATCHING - MATCHES TENSORFLOW")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    config = Config()
    
    # Load data
    X, y = load_data(config)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    
    print(f"\nTarget stats: mean={np.mean(y):.2f}, std={np.std(y):.2f}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Create datasets and dataloaders
    train_dataset = MoleculeDataset(X_train, y_train)
    test_dataset = MoleculeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, collate_fn=collate_molecules)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=False, collate_fn=collate_molecules)
    
    # Create model
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
    history = train_model_batched(model, train_loader, test_loader, config, device)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device)
    # Save
    save_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/pdbbind/saved_models'
    os.makedirs(save_dir, exist_ok=True)

    filename = f"pdbbind_baseline_pw{config.PHYSICS_WEIGHT}.pth"
    save_path = os.path.join(save_dir, filename)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'metrics': metrics,
        'config': {
            'physics_weight': config.PHYSICS_WEIGHT,
            'learning_rate': config.LEARNING_RATE,
            'l2_weight': config.L2_WEIGHT,
            'batch_size': config.BATCH_SIZE
        }
    }, save_path)
    print(f"\n✓ Saved to: {save_path}")
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print("="*80)
    print(f"Test RMSE: {metrics['rmse']:.2f} kcal/mol")
    print(f"Test MAE:  {metrics['mae']:.2f} kcal/mol")
    print(f"Test R²:   {metrics['r2']:.4f}")
    print(f"\nTensorFlow Baseline: MAE 9.24")
    print(f"PyTorch Result:      MAE {metrics['mae']:.2f}")
    
    if metrics['mae'] < 10.0:
        print("\nSUCCESS! Matched TensorFlow!")
    else:
        print(f"\nMAE {metrics['mae']:.2f} - needs improvement")
    
    print("="*80)


if __name__ == "__main__":
    main()