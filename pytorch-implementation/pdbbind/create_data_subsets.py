"""
Create Multiple Data Subsets for Scaling Analysis

Creates 6 different dataset sizes to analyze how optimal physics weight
changes with amount of training data.

Sizes: 100, 250, 500, 1000, 2000, 2660 (full)
"""

import pickle
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def create_data_subsets(csv_path, pkl_path, output_dir, random_seed=50):
    """
    Create multiple dataset size subsets for scaling analysis.
    
    Args:
        csv_path: Path to CSV file
        pkl_path: Path to PKL file
        output_dir: Directory to save subset split files
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary of {size: split_file_path}
    """
    print("="*80)
    print("Creating Data Subsets for Scaling Analysis")
    print("="*80)
    
    # Load data
    print("\nLoading full dataset...")
    df = pd.read_csv(csv_path)
    with open(pkl_path, 'rb') as f:
        pdb_dict = pickle.load(f)
    
    # Clean data
    df = df.dropna(subset=['ddg'])
    df = df[~df['complex-name'].astype(str).str.contains('E\+', na=False)]
    
    # Get common keys
    common_keys = set(df['complex-name']) & set(pdb_dict.keys())
    df = df[df['complex-name'].isin(common_keys)]
    
    all_names = df['complex-name'].values
    print(f"✓ Total available structures: {len(all_names)}")
    
    # Define subset sizes
    # Suggested: 6 data points spanning the range
    subset_sizes = [100, 250, 500, 1000, 2000, len(all_names)]
    
    print(f"\nCreating {len(subset_sizes)} subsets:")
    for size in subset_sizes:
        print(f"  - {size} structures")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subsets
    subset_files = {}
    
    for size in subset_sizes:
        print(f"\n{'='*60}")
        print(f"Creating subset: {size} structures")
        print(f"{'='*60}")
        
        # Sample structures
        if size >= len(all_names):
            subset_names = all_names
            print(f"Using all {len(all_names)} structures (full dataset)")
        else:
            np.random.seed(random_seed)
            indices = np.random.choice(len(all_names), size=size, replace=False)
            subset_names = all_names[indices]
            print(f"Sampled {size} structures randomly")
        
        # Create train/test split (80/20)
        indices = np.arange(len(subset_names))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=random_seed
        )
        
        # Create split data
        split_data = {
            'train_indices': train_idx,
            'test_indices': test_idx,
            'train_names': subset_names[train_idx].tolist(),
            'test_names': subset_names[test_idx].tolist(),
            'all_names': subset_names.tolist(),
            'subset_size': size,
            'test_size': 0.2,
            'random_seed': random_seed,
            'total_samples': len(subset_names)
        }
        
        # Save
        filename = f'pdbbind_subset_{size}.pkl'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(split_data, f)
        
        subset_files[size] = filepath
        
        print(f"✓ Saved to: {filepath}")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Test: {len(test_idx)} samples")
    
    # Save manifest
    manifest = {
        'subset_sizes': subset_sizes,
        'subset_files': subset_files,
        'random_seed': random_seed,
        'csv_path': csv_path,
        'pkl_path': pkl_path,
        'created': pd.Timestamp.now().isoformat()
    }
    
    manifest_path = os.path.join(output_dir, 'subsets_manifest.pkl')
    with open(manifest_path, 'wb') as f:
        pickle.dump(manifest, f)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Created {len(subset_sizes)} data subsets")
    print(f"Manifest saved to: {manifest_path}")
    print("\nSubset files:")
    for size, path in subset_files.items():
        print(f"  {size:5d} structures → {path}")
    
    return subset_files, manifest_path


def main():
    # Paths
    csv_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind.csv'
    pkl_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_full.pkl'
    output_dir = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/subsets'
    
    # Create subsets
    subset_files, manifest_path = create_data_subsets(
        csv_path=csv_path,
        pkl_path=pkl_path,
        output_dir=output_dir,
        random_seed=50
    )
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Run grid search for each subset:")
    print("   python run_grid_search_scaling.py")
    print("\n2. This will run 6 grid searches:")
    for size in sorted(subset_files.keys()):
        print(f"   - {size} structures")
    print("\n3. Results will be saved separately for each size")
    print("\n4. Use plot_scaling_analysis.py to visualize results")
    print("="*80)


if __name__ == "__main__":
    main()