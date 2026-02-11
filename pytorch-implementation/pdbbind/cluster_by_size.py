"""
Cluster Structures by Size and Create Stratified Splits

This ensures train/test sets have balanced distribution of structure sizes
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import os


def analyze_structure_sizes(pkl_path, csv_path):
    """
    Analyze distribution of structure sizes.
    
    Returns:
        DataFrame with complex names, sizes, and clustering info
    """
    print("Analyzing structure sizes...")
    
    # Load PKL
    with open(pkl_path, 'rb') as f:
        pdb_dict = pickle.load(f)
    
    # Get sizes
    sizes = []
    names = []
    
    for name, mol in pdb_dict.items():
        num_atoms = mol.GetNumAtoms()
        sizes.append(num_atoms)
        names.append(name)
    
    # Create DataFrame
    df = pd.DataFrame({
        'complex-name': names,
        'num_atoms': sizes
    })
    
    # Statistics
    print(f"\nStructure size statistics:")
    print(f"  Total structures: {len(df)}")
    print(f"  Min atoms: {df['num_atoms'].min()}")
    print(f"  Max atoms: {df['num_atoms'].max()}")
    print(f"  Mean atoms: {df['num_atoms'].mean():.1f}")
    print(f"  Median atoms: {df['num_atoms'].median():.1f}")
    print(f"  Std atoms: {df['num_atoms'].std():.1f}")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = np.percentile(df['num_atoms'], p)
        print(f"  {p}th: {val:.0f} atoms")
    
    return df


def cluster_by_size(df, n_clusters=5):
    """
    Cluster structures into size bins using K-means.
    
    Args:
        df: DataFrame with 'num_atoms' column
        n_clusters: Number of size clusters
    
    Returns:
        DataFrame with added 'size_cluster' column
    """
    print(f"\nClustering into {n_clusters} size groups...")
    
    # Reshape for sklearn
    X = df['num_atoms'].values.reshape(-1, 1)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['size_cluster'] = kmeans.fit_predict(X)
    
    # Get cluster centers
    centers = kmeans.cluster_centers_.flatten()
    sorted_centers = np.sort(centers)
    
    # Reassign clusters so 0=smallest, n_clusters-1=largest
    cluster_mapping = {}
    for i, center in enumerate(kmeans.cluster_centers_.flatten()):
        cluster_mapping[i] = np.where(sorted_centers == center)[0][0]
    
    df['size_cluster'] = df['size_cluster'].map(cluster_mapping)
    
    # Print cluster info
    print("\nCluster distribution:")
    for cluster_id in range(n_clusters):
        cluster_data = df[df['size_cluster'] == cluster_id]
        print(f"  Cluster {cluster_id}: {len(cluster_data)} structures")
        print(f"    Size range: {cluster_data['num_atoms'].min()}-{cluster_data['num_atoms'].max()} atoms")
        print(f"    Mean: {cluster_data['num_atoms'].mean():.1f} atoms")
    
    return df


def create_stratified_split(df, test_size=0.2, random_seed=50):
    """
    Create train/test split stratified by size cluster.
    
    This ensures both train and test sets have balanced size distributions.
    """
    print(f"\nCreating stratified split (test_size={test_size})...")
    
    # Stratified split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        stratify=df['size_cluster'].values,
        random_state=random_seed
    )
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Check cluster distribution in train/test
    print("\nCluster distribution in train set:")
    print(train_df['size_cluster'].value_counts().sort_index())
    
    print("\nCluster distribution in test set:")
    print(test_df['size_cluster'].value_counts().sort_index())
    
    # Verify proportions are similar
    print("\nProportion of each cluster:")
    print("Cluster | Train % | Test %")
    print("-" * 30)
    for cluster_id in sorted(df['size_cluster'].unique()):
        train_pct = (train_df['size_cluster'] == cluster_id).sum() / len(train_df) * 100
        test_pct = (test_df['size_cluster'] == cluster_id).sum() / len(test_df) * 100
        print(f"   {cluster_id}    | {train_pct:6.2f} | {test_pct:6.2f}")
    
    return train_idx, test_idx, train_df, test_df


def plot_size_distribution(df, train_df, test_df, save_path=None):
    """Plot size distribution across train/test sets."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall distribution
    axes[0, 0].hist(df['num_atoms'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Number of Atoms')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Overall Size Distribution')
    axes[0, 0].axvline(df['num_atoms'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # Train vs Test
    axes[0, 1].hist([train_df['num_atoms'], test_df['num_atoms']], 
                    bins=50, alpha=0.6, label=['Train', 'Test'], edgecolor='black')
    axes[0, 1].set_xlabel('Number of Atoms')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Train vs Test Distribution')
    axes[0, 1].legend()
    
    # Cluster distribution
    cluster_counts = df['size_cluster'].value_counts().sort_index()
    axes[1, 0].bar(cluster_counts.index, cluster_counts.values, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Size Cluster')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Structures per Cluster')
    
    # Box plot by cluster
    data_by_cluster = [df[df['size_cluster'] == i]['num_atoms'].values 
                       for i in sorted(df['size_cluster'].unique())]
    axes[1, 1].boxplot(data_by_cluster, labels=sorted(df['size_cluster'].unique()))
    axes[1, 1].set_xlabel('Size Cluster')
    axes[1, 1].set_ylabel('Number of Atoms')
    axes[1, 1].set_title('Size Distribution by Cluster')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {save_path}")
    
    return fig


def save_stratified_split(train_idx, test_idx, df, save_path):
    """Save stratified split for reuse."""
    
    split_data = {
        'train_indices': train_idx,
        'test_indices': test_idx,
        'train_names': df.iloc[train_idx]['complex-name'].tolist(),
        'test_names': df.iloc[test_idx]['complex-name'].tolist(),
        'all_names': df['complex-name'].tolist(),
        'size_clusters': df['size_cluster'].tolist(),
        'num_atoms': df['num_atoms'].tolist(),
        'n_clusters': len(df['size_cluster'].unique()),
        'stratified': True
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(split_data, f)
    
    print(f"\n✓ Stratified split saved to: {save_path}")
    
    return split_data


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Complete workflow for size-based clustering and stratified splitting."""
    
    # Paths
    csv_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind.csv'
    pkl_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/PDBBind_full.pkl'
    split_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets/pdbbind_stratified_split.pkl'
    plot_path = '/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/pytorch-implementation/comparison_plots/size_distribution.png'
    
    print("="*80)
    print("PDBBind Structure Size Analysis and Stratified Splitting")
    print("="*80)
    
    # Step 1: Analyze sizes
    df = analyze_structure_sizes(pkl_path, csv_path)
    
    # Step 2: Cluster by size
    n_clusters = 5  # You can adjust this
    df = cluster_by_size(df, n_clusters=n_clusters)
    
    # Step 3: Create stratified split
    train_idx, test_idx, train_df, test_df = create_stratified_split(
        df, test_size=0.2, random_seed=50
    )
    
    # Step 4: Visualize
    fig = plot_size_distribution(df, train_df, test_df, save_path=plot_path)
    
    # Step 5: Save split
    split_data = save_stratified_split(train_idx, test_idx, df, split_path)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nUse this split in your training script:")
    print(f"  X_train, X_test, y_train, y_test = load_data_with_saved_split(config, '{split_path}')")

if __name__ == "__main__":
    main()