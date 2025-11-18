import os, pickle, random
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load CSV & PKL ---
DIR = Path("/home/exouser/multi-objective-pdbbind/multi-objective-pdbbind/Datasets")
CSV = DIR / "pdbbind_100.csv"
PKL = DIR / "PDBBind_100.pkl"

df = pd.read_csv(CSV)
print("Columns:", list(df.columns))
print(df.head(3))

# Identify ID + target columns
CAND_ID = [c for c in df.columns if c.lower() in {"pdb_id","pdbid","complex-name","id"}]
ID_COL = CAND_ID[0] if CAND_ID else df.columns[0]
print("Using ID column:", ID_COL)

# Sample a few rows
sample_df = df.sample(10, random_state=0)
print("\nSample rows:")
print(sample_df[[ID_COL, "ddg", "enthalpy-gb", "entropy"]])

# --- Plot pair correlations ---
plt.figure(figsize=(8,6))
sns.heatmap(df[["ddg","enthalpy-gb","entropy"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation among DDG, Enthalpy, Entropy")
plt.tight_layout()
plt.show()

# --- Scatter: enthalpy vs ddg ---
plt.figure(figsize=(7,5))
plt.scatter(df["enthalpy-gb"], df["ddg"], alpha=0.7)
plt.xlabel("Enthalpy (GB)")
plt.ylabel("ΔΔG (ddg)")
plt.title("Binding free energy vs enthalpy")
plt.grid(True)
plt.show()

# --- Load and inspect PKL ---
if PKL.exists():
    with open(PKL, "rb") as f:
        blob = pickle.load(f)

    if isinstance(blob, dict):
        print("\nPKL keys:", list(blob.keys())[:10])
        first_key = next(iter(blob))
        print(f"First entry [{first_key}] type:", type(blob[first_key]))
        try:
            sample_data = blob[first_key]
            if isinstance(sample_data, dict):
                print("Subkeys:", list(sample_data.keys()))
            elif isinstance(sample_data, np.ndarray):
                print("Array shape:", sample_data.shape)
        except Exception as e:
            print("Couldn't inspect PKL contents:", e)
