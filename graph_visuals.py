#!/usr/bin/env python3
# analysis_pdbbind.py
# End-to-end CSV + PKL visualization & profiling for your PDBbind slice.

import os
import argparse
import pickle
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Use non-interactive backend so this works on headless machines
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: comment seaborn out if not installed
import seaborn as sns

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze PDBbind CSV/PKL features, visualize, and profile tensors.")
    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing CSV and PKL files.")
    p.add_argument("--csv", type=str, default="pdbbind_100.csv", help="CSV filename.")
    p.add_argument("--pkl", type=str, default="PDBBind_100.pkl", help="PKL filename.")
    p.add_argument("--out", type=Path, default=Path("analysis_outputs"), help="Output directory for plots/reports.")
    p.add_argument("--sample", type=int, default=10, help="Number of complexes to sample for quicklook.")
    p.add_argument("--quiet-tf", action="store_true", help="Silence TF GPU warnings (CPU-only).")
    p.add_argument("--distance-max_atoms", type=int, default=400, help="Cap for distance heatmap (avoid giant images).")
    return p.parse_args()

# ---------- Utils ----------

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def obj_size_bytes(x) -> int:
    try:
        if isinstance(x, np.ndarray):
            return x.nbytes
        return len(pickle.dumps(x, protocol=4))
    except Exception:
        return 0

# ---------- Section A: CSV analysis ----------

def analyze_csv(csv_path: Path, out_dir: Path) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    print("CSV Columns:", list(df.columns))
    print(df.head(3))

    # Identify ID and common target columns
    cand_id = [c for c in df.columns if c.lower() in {"pdb_id", "pdbid", "complex-name", "id"}]
    id_col = cand_id[0] if cand_id else df.columns[0]
    print("Using ID column:", id_col)

    # Quick sample table
    cols_exist = [c for c in ["ddg", "enthalpy-gb", "entropy"] if c in df.columns]
    sample_cols = [id_col] + cols_exist
    sample_df = df.sample(min(len(df), 10), random_state=0)[sample_cols]
    print("\nSample rows:")
    print(sample_df)

    # Save sample table
    sample_df.to_csv(out_dir / "csv_sample_rows.csv", index=False)

    # Correlation over all numeric columns
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 1:
        plt.figure(figsize=(12, 9))
        sns.heatmap(num_df.corr(), cmap="coolwarm", center=0)
        plt.title("Correlation among numeric energy terms")
        savefig(out_dir / "csv_correlation_full.png")

    # Focused correlation if the key columns exist
    focus_cols = [c for c in ["ddg", "enthalpy-gb", "entropy"] if c in df.columns]
    if len(focus_cols) >= 2:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[focus_cols].corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation among selected energy terms")
        savefig(out_dir / "csv_correlation_selected.png")

    # Scatter ddg vs enthalpy-gb if both exist
    if {"ddg", "enthalpy-gb"}.issubset(df.columns):
        plt.figure(figsize=(7, 5))
        plt.scatter(df["enthalpy-gb"], df["ddg"], alpha=0.7)
        plt.xlabel("Enthalpy (GB)")
        plt.ylabel("ΔΔG (ddg)")
        plt.title("ΔΔG vs Enthalpy (GB)")
        plt.grid(True)
        savefig(out_dir / "csv_scatter_enthalpy_vs_ddg.png")

    # Distributions for a few columns if present
    for c in ["ddg", "enthalpy-gb", "entropy"]:
        if c in df.columns:
            plt.figure(figsize=(7, 4))
            sns.histplot(df[c].dropna(), kde=True)
            plt.title(f"Distribution of {c}")
            savefig(out_dir / f"csv_dist_{c}.png")

    return {"df": df, "id_col": id_col}

# ---------- Section B: PKL profiling ----------

def profile_pkl(pkl_path: Path, out_dir: Path, max_rows:int=50) -> Dict[str, Any]:
    if not pkl_path.exists():
        print(f"PKL not found at {pkl_path}. Skipping PKL profiling.")
        return {"blob": None, "profile": None}

    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)

    print("\nPKL root type:", type(blob))
    if not isinstance(blob, dict):
        print("Unexpected PKL structure; expected dict keyed by pdb_id. Skipping deep profile.")
        return {"blob": blob, "profile": None}

    keys = list(blob.keys())
    print("Total PKL entries:", len(keys))
    print("First 10 keys:", keys[:10])

    rows = []
    for k in keys[:max_rows]:
        v = blob[k]
        if isinstance(v, dict):
            for sk, sv in v.items():
                shape = getattr(sv, "shape", None)
                dtype = getattr(sv, "dtype", None)
                rows.append({
                    "pdb_id": k,
                    "field": sk,
                    "type": type(sv).__name__,
                    "shape": tuple(shape) if shape is not None else None,
                    "dtype": str(dtype) if dtype is not None else None,
                    "bytes": obj_size_bytes(sv)
                })
        else:
            rows.append({
                "pdb_id": k, "field": "<root>", "type": type(v).__name__,
                "shape": getattr(v, "shape", None),
                "dtype": str(getattr(v, "dtype", None)),
                "bytes": obj_size_bytes(v)
            })

    prof = pd.DataFrame(rows).sort_values(["pdb_id", "field"])
    prof_path = out_dir / "pkl_quick_profile.csv"
    prof.to_csv(prof_path, index=False)
    print(f"Saved PKL quick profile → {prof_path}")

    # Aggregate common shapes per field
    if "field" in prof and "shape" in prof:
        grouped = (prof.dropna(subset=["shape"])
                        .groupby("field")["shape"]
                        .agg(lambda s: pd.Series(s).value_counts().head(3).to_dict()))
        print("\nMost common shapes per field (top 3):")
        for fld, stats in grouped.items():
            print(f"- {fld}: {stats}")

    # Try some high-value visuals if we can guess fields
    # 1) Node feature heatmap for a random entry (cap size)
    rng = random.Random(0)
    try:
        sample_key = rng.choice(keys)
        entry = blob[sample_key]
        if isinstance(entry, dict):
            # Node features
            if "node_features" in entry and isinstance(entry["node_features"], np.ndarray):
                nf = entry["node_features"]
                nrows = min(nf.shape[0], 400)   # cap rows to keep image readable
                ncols = min(nf.shape[1], 128)   # cap columns for fast plot
                plt.figure(figsize=(8, 6))
                sns.heatmap(nf[:nrows, :ncols], cmap="viridis")
                plt.title(f"Node feature heatmap ({sample_key}) [{nrows}x{ncols}]")
                savefig(out_dir / f"pkl_node_features_heatmap_{sample_key}.png")

            # Coordinates → distance matrix heatmap (cap atoms)
            if "coords" in entry and isinstance(entry["coords"], np.ndarray) and entry["coords"].ndim == 2:
                coords = entry["coords"]
                max_atoms = min(coords.shape[0], 400)
                if max_atoms >= 5:
                    from scipy.spatial.distance import pdist, squareform
                    dist_mat = squareform(pdist(coords[:max_atoms]))
                    plt.figure(figsize=(7, 6))
                    sns.heatmap(dist_mat, cmap="mako")
                    plt.title(f"Interatomic distances (Å) ({sample_key}) [{max_atoms} atoms]")
                    savefig(out_dir / f"pkl_distance_matrix_{sample_key}.png")

            # Size histogram across many entries if node_features present
            sizes = []
            for k in keys[:min(len(keys), 200)]:
                v = blob[k]
                if isinstance(v, dict) and "node_features" in v and isinstance(v["node_features"], np.ndarray):
                    sizes.append(v["node_features"].shape[0])
            if sizes:
                plt.figure(figsize=(7, 4))
                sns.histplot(sizes, kde=True)
                plt.title("Distribution of atom counts per complex (sampled)")
                plt.xlabel("# atoms")
                savefig(out_dir / "pkl_atom_count_dist.png")

    except Exception as e:
        print("PKL visualization step failed:", e)

    return {"blob": blob, "profile": prof}

# ---------- Section C: Optional RDKit grid (only if SMILES exist) ----------

def try_molecule_grid(df: pd.DataFrame, id_col: str, out_dir: Path, sample: int = 10):
    smiles_cols = [c for c in df.columns if "smiles" in c.lower()]
    if not smiles_cols:
        print("No SMILES column detected; skipping molecule grid.")
        return
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except Exception:
        print("RDKit not available; skipping molecule grid.")
        return

    sc = smiles_cols[0]
    sub = df[[id_col, sc]].dropna().head(sample)
    mols = [Chem.MolFromSmiles(str(s)) for s in sub[sc]]
    legends = [str(i) for i in sub[id_col]]
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200), legends=legends)
    img_path = out_dir / "ligand_grid.png"
    img.save(str(img_path))
    print(f"Saved ligand grid → {img_path}")

# ---------- Section D: Model probing hooks (fill in later) ----------

def make_probe_model(model):
    """Builds a Keras model that outputs every layer's tensor (single-output layers only)."""
    import tensorflow as tf
    outputs, names = [], []
    for layer in model.layers:
        try:
            if isinstance(layer.output, (list, tuple)):
                continue
            outputs.append(layer.output)
            names.append(layer.name)
        except Exception:
            pass
    probe = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    return probe, names

def describe_tensors(tensors, names):
    for t, name in zip(tensors, names):
        try:
            dtype = getattr(t, "dtype", None)
            shape = getattr(t, "shape", None)
            dname = dtype.name if dtype is not None and hasattr(dtype, "name") else str(dtype)
            print(f"{name:30s} | dtype={dname:<8} shape={shape}")
        except Exception as e:
            print(f"{name:30s} | <unavailable> ({e})")

def batchify_samples(samples: List[Dict[str, Any]]):
    """Make a naive batch dict (np.stack when shapes match, else ragged)."""
    import tensorflow as tf
    out = {}
    if not samples:
        return out
    for k in samples[0].keys():
        vals = [s[k] for s in samples if isinstance(s[k], np.ndarray)]
        if not vals:
            continue
        try:
            out[k] = np.stack(vals, axis=0)
        except Exception:
            out[k] = tf.ragged.constant(vals)
    return out

def choose_samples_from_pkl(blob: Dict[str, Any], n: int = 10, seed: int = 0):
    if not isinstance(blob, dict):
        return [], []
    keys = list(blob.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    chosen = keys[:min(n, len(keys))]
    samples = []
    for k in chosen:
        v = blob[k]
        if isinstance(v, dict):
            # Cast float arrays to float32 to save memory
            sample = {}
            for sk, sv in v.items():
                if isinstance(sv, np.ndarray) and sv.dtype.kind == "f":
                    sample[sk] = sv.astype(np.float32)
                else:
                    sample[sk] = sv
            samples.append(sample)
    return chosen, samples

# ---------- Main ----------

def main():
    args = parse_args()
    ensure_outdir(args.out)

    if args.quiet_tf:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    csv_path = args.data_dir / args.csv
    pkl_path = args.data_dir / args.pkl

    print("=== CSV analysis ===")
    csv_info = analyze_csv(csv_path, args.out)
    df, id_col = csv_info["df"], csv_info["id_col"]

    print("\n=== SMILES grid (optional) ===")
    try_molecule_grid(df, id_col, args.out, sample=args.sample)

    print("\n=== PKL profiling ===")
    pkl_info = profile_pkl(pkl_path, args.out, max_rows=50)
    blob = pkl_info["blob"]

    # --- OPTIONAL: model probing (fill in when your Keras model is available) ---
    # Example usage:
    # from your_model_module import build_model
    # model = build_model(...)
    # probe, names = make_probe_model(model)
    # chosen_ids, samples = choose_samples_from_pkl(blob, n=args.sample, seed=0)
    # x_batch = batchify_samples(samples)
    # tensors = probe(x_batch, training=False)
    # describe_tensors(tensors, names)

    print(f"\nAll outputs saved under: {args.out.resolve()}")
    print("Done.")

if __name__ == "__main__":
    main()
