"""
Ensure Consistent Train/Test Splits Across Experiments

This creates a saved split file so all experiments use the exact same data split
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def create_and_save_split(csv_path, pkl_path, save_path, test_size=0.2, random_seed=50):
    """
    Create train/test split and save it for reuse.
    
    Args:
        csv_path: Path to CSV file
        pkl_path: Path to PKL file  
        save_path: Where to save the split indices
        test_size: Fraction for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with train/test complex names and indices
    """
    # Load data
    df = pd.read_csv(csv_path)
    with open(pkl_path, 'rb') as f:
        pdb_dict = pickle.load(f)
    
    # Clean data (same as in training script)
    df = df.dropna(subset=['ddg'])
    df = df[~df['complex-name'].astype(str).str.contains('E\+', na=False)]
    
    # Get common keys between CSV and PKL
    common_keys = set(df['complex-name']) & set(pdb_dict.keys())
    df = df[df['complex-name'].isin(common_keys)]
    
    # Get complex names
    complex_names = df['complex-name'].values
    
    # Create indices
    indices = np.arange(len(complex_names))
    
    # Split
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_seed
    )
    
    # Create split dictionary
    split_data = {
        'train_indices': train_idx,
        'test_indices': test_idx,
        'train_names': complex_names[train_idx].tolist(),
        'test_names': complex_names[test_idx].tolist(),
        'all_names': complex_names.tolist(),
        'test_size': test_size,
        'random_seed': random_seed,
        'total_samples': len(complex_names)
    }
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(split_data, f)
    
    print(f"✓ Split saved to: {save_path}")
    print(f"  Total: {len(complex_names)}")
    print(f"  Train: {len(train_idx)} ({len(train_idx)/len(complex_names)*100:.1f}%)")
    print(f"  Test: {len(test_idx)} ({len(test_idx)/len(complex_names)*100:.1f}%)")
    
    return split_data


def load_split(split_path):
    """Load saved train/test split."""
    with open(split_path, 'rb') as f:
        split_data = pickle.load(f)
    
    print(f"✓ Loaded split from: {split_path}")
    print(f"  Total: {split_data['total_samples']}")
    print(f"  Train: {len(split_data['train_indices'])}")
    print(f"  Test: {len(split_data['test_indices'])}")
    
    return split_data


def load_data_with_saved_split(config, split_path):
    """
    Load data using a saved train/test split.
    
    This ensures all experiments use the EXACT same data split.
    """
    import torch
    from models.dcFeaturizer import atom_features as get_atom_features
    
    print("Loading PDBBind dataset with saved split...")
    
    # Load split
    split_data = load_split(split_path)
    
    # Load data
    df = pd.read_csv(config.CSV_PATH)
    with open(config.PKL_PATH, 'rb') as f:
        pdb_dict = pickle.load(f)
    
    print(f"✓ Loaded {len(df)} CSV entries, {len(pdb_dict)} PDB structures")
    
    # Clean data (same way as when creating split)
    df = df.dropna(subset=['ddg'])
    df = df[~df['complex-name'].astype(str).str.contains('E\+', na=False)]
    
    physics_columns = [
        'pb-protein-vdwaals', 'pb-ligand-vdwaals', 'pb-complex-vdwaals',
        'gb-protein-1-4-eel', 'gb-ligand-1-4-eel', 'gb-complex-1-4-eel',
        'gb-protein-eelect', 'gb-ligand-eelec', 'gb-complex-eelec',
        'gb-protein-egb', 'gb-ligand-egb', 'gb-complex-egb',
        'gb-protein-esurf', 'gb-ligand-esurf', 'gb-complex-esurf'
    ]
    
    common_keys = set(df['complex-name']) & set(pdb_dict.keys())
    df = df[df['complex-name'].isin(common_keys)]
    
    # Use saved split to get train/test names
    train_names = split_data['train_names']
    test_names = split_data['test_names']
    
    # Verify all names still exist
    available_names = set(df['complex-name'])
    train_names = [n for n in train_names if n in available_names]
    test_names = [n for n in test_names if n in available_names]
    
    print(f"✓ Using saved split: {len(train_names)} train, {len(test_names)} test")
    
    # Featurize
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
    
    # Process train set
    X_train, y_train = [], []
    for pdb_id in train_names:
        row = df[df['complex-name'] == pdb_id].iloc[0]
        info_array = row[physics_columns].tolist()
        target = row['ddg']
        
        try:
            features = featurize(pdb_dict[pdb_id], info_array)
            X_train.append(torch.FloatTensor(features))
            y_train.append(target)
        except Exception:
            pass
    
    # Process test set
    X_test, y_test = [], []
    for pdb_id in test_names:
        row = df[df['complex-name'] == pdb_id].iloc[0]
        info_array = row[physics_columns].tolist()
        target = row['ddg']
        
        try:
            features = featurize(pdb_dict[pdb_id], info_array)
            X_test.append(torch.FloatTensor(features))
            y_test.append(target)
        except Exception:
            pass
    
    print(f"✓ Featurized {len(X_train)} train, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# USAGE IN YOUR TRAINING SCRIPT
# ============================================================================

"""
# Step 1: Create and save split (do this ONCE)
split_path = 'data_splits/pdbbind_split.pkl'
split_data = create_and_save_split(
    csv_path='/path/to/pdbbind.csv',
    pkl_path='/path/to/PDBBind_full.pkl',
    save_path=split_path,
    test_size=0.2,
    random_seed=50
)

# Step 2: In your training script, use the saved split
X_train, X_test, y_train, y_test = load_data_with_saved_split(config, split_path)

# Now all experiments use the EXACT same train/test split!
"""


if __name__ == "__main__":
    # Example usage
    csv_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind.csv'
    pkl_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_full.pkl'
    split_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind_stratified_split.pkl'
    
    print("Creating consistent train/test split...")
    split_data = create_and_save_split(
        csv_path=csv_path,
        pkl_path=pkl_path,
        save_path=split_path,
        test_size=0.2,
        random_seed=50
    )
    
    print("\n✓ Split created and saved!")
    print("\nTo use in your training script:")
    print(f"  X_train, X_test, y_train, y_test = load_data_with_saved_split(config, '{split_path}')")