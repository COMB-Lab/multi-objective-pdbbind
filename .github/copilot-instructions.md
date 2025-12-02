# Copilot Instructions for multi-objective-pdbbind

## Project Overview

This repository implements **Physics-Guided Graph Convolutional Networks (PGGCN)** for predicting protein-ligand binding free energies using the PDBbind dataset. The core innovation is combining empirical ML predictions with physics-based energy calculations through a multi-objective loss function.

**Branch context**: Currently on `pcgrad-implementation` branch, integrating gradient conflict resolution via PCGrad optimizer.

**Framework**: Both TensorFlow and PyTorch implementations available. PyTorch version recommended for stability and performance.

## Architecture & Data Flow

### Core Components

**TensorFlow Version**:
1. **Data Pipeline** (`multi-objective-pdbbind.py` / `pcgrad-multi-objective-pdbbind.py`):
   - Loads PDBbind structures from `/home/exouser/multi-objective-pdbbind/Datasets/` (CSV + pickle format)
   - Extracts 15 physics-based energy terms (vdW, electrostatic, solvation)
   - Featurizes molecules using DeepChem atom features + 3D coordinates

2. **Graph Convolution Layers** (`models/layers_update_mobley.py`):
   - `RuleGraphConvLayer`: Applies combination rules ("sum", "multiply", "distance") to graph neighbors
   - `ConvLayer`: Follows the graph convolution with standard dense operations
   - **Key pattern**: Uses `_call_single()` for single-molecule processing, wrapped with `tf.map_fn` for batching

3. **PGGCN Model** (`models/model.py` / `multi-objective-pdbbind.py`):
   - Merges empirical ML output with physics information (15 energy features)
   - Physics-informed dense layer uses **constant initialization** with physics-derived weights
   - Returns concatenated `[prediction, physics_info]` for loss computation

**PyTorch Version** (RECOMMENDED):
1. **Data Pipeline** (same as TensorFlow, compatible)
2. **Graph Convolution Layers** (`models/layers_pytorch.py`):
   - `RuleGraphConvLayer`: PyTorch `nn.Module` with equivalent logic
   - `ConvLayer`: PyTorch implementation with tanh aggregation
   - `PGGCNModel`: Complete neural network with physics-informed layer
   - `PCGradOptimizer`: Gradient conflict resolution for multi-task learning
3. **Training** (`multi-objective-pdbbind-pytorch.py`):
   - Uses native PyTorch training loop (more stable than tf.keras)
   - Cleaner gradient computation and optimizer handling

### Loss Functions
   - **Empirical**: Pure RMSE on predicted binding affinity
   - **Physics**: RMSE between predicted and physics-calculated ΔG: `ΔG_physics = ΔG_complex - (ΔG_host + ΔG_guest)`
   - **Combined**: `empirical_loss + physics_weight × physics_loss` (swept across `[1e-7, 1e-6, 2e-6, 5e-6, 1e-5]`)

### Data Flow Diagram

```
PDBbind CSV + PDB Structures
    ↓
load_data() → clean invalid entries & extract physics terms
    ↓
featurize() → DeepChem atom features + coordinates + physics info
    ↓
prepare_features_chunked() → adaptive padding (max_atoms or memory constraint)
    ↓
train_test_split() → 80/20
    ↓
PGGCN Model
    ├─ RuleGraphConvLayer → ConvLayer (per molecule)
    ├─ Dense → Dropout → Dense (empirical output)
    └─ Merge empirical + physics → Dense(physics-informed)
    ↓
Combined Loss (sweep physics_weight)
    ↓
Results: y_pred, loss_history, metrics → pickle file for analysis
```

## Critical Developer Workflows

### Running Training

**PyTorch Version (RECOMMENDED)**:
```bash
python3 multi-objective-pdbbind-pytorch.py  # Main PyTorch training script
# Advantages: Better error handling, stable gradients, cleaner code
```

**TensorFlow Version (Legacy)**:
```bash
python3 multi-objective-pdbbind.py  # Main TensorFlow training script
```

**Using the shell wrapper** (works with both):
```bash
GPU_ID=0 ./multi-objective-pdbbind-run.sh
# Logs to logs/gpu_training_*.{out,err}
```

**PCGrad variant** (gradient conflict resolution - TensorFlow only):
```bash
python3 pcgrad-multi-objective-pdbbind.py
# PyTorch version has built-in PCGradOptimizer for multi-task gradients
```

### Memory Management & Configuration

The codebase is **designed for large-scale datasets**. Key tuning in `DatasetConfig`:

- `dataset_size`: Number of structures (10=quick test, 100=dev, 500+=production)
- `batch_size`: 2-4 recommended for TensorFlow, 4-8 for PyTorch (graph convolutions are memory-heavy)
- `max_padding`: Maximum atoms per molecule. Auto-adaptive:
  - Uses **actual max** if memory < 70% threshold
  - Falls back to `max_padding` value if tight
  - Truncates structures if necessary (logged warnings)
- `epochs` & `memory_limit_gb`: Adjust based on hardware

**Memory monitoring** utilities:
- `get_memory_usage()`: Real-time RAM consumption (GB)
- `check_memory_limit()`: Triggers garbage collection at 80% threshold
- Chunked feature preparation (`chunk_size=10`) processes molecules in batches to prevent OOM
- PyTorch version can unload model to GPU between epochs for larger datasets

**PyTorch GPU Handling**:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# Automatic memory growth on GPU via PyTorch
torch.cuda.empty_cache()  # Force cleanup between epochs
```

### Analysis & Visualization

After training, results are saved to pickle:
```bash
python3 analyze_results.py
# Loads pickle file → generates loss curves, scatter plots (pred vs true)
# Plots saved to working directory
```

Use `load_and_explore_results()` function to interactively inspect:
```python
from multi-objective-pdbbind import load_and_explore_results
results = load_and_explore_results('PGGCN_results_100structures_*.pkl')
```

## Project-Specific Conventions

### Featurization Pattern (Framework Agnostic)

**Atom features = 38-53 dimensions** (consistent across TensorFlow and PyTorch):
```python
# In featurize():
new_feature = get_atom_features(atom)  # DeepChem output
new_feature += [mass, atomic_num, formal_charge]  # Basic properties (3)
new_feature += [position.x, position.y, position.z]  # 3D coords (3)
new_feature += neighbor_indices  # Connectivity (2 neighbors padded)
# Total = 38-40, then concatenated with physics_info (15) = 53-55 final
```

**Physics info ordering** (15 terms, extracted via indexing):
- Indices 0,1,2: vdW (protein, ligand, complex)
- Indices 3,4,5: 1-4 electrostatic
- Indices 6,7,8: Electrostatic
- Indices 9,10,11: GB solvation
- Indices 12,13,14: Surface area

### Custom Layer Pattern

**PyTorch** (`models/layers_pytorch.py`):
```python
class RuleGraphConvLayer(nn.Module):
    def _call_single(self, features):  # Single molecule [num_atoms, num_features]
        # Process one molecule through graph convolution
        return output  # [num_atoms, out_channel]
    
    def forward(self, inputs):  # List of molecules
        outputs = [self._call_single(inp) for inp in inputs]
        return outputs
```

**TensorFlow** (`models/layers_update_mobley.py`):
```python
class RuleGraphConvLayer(tf.keras.layers.Layer):
    def _call_single(self, inp):
        # Uses tf.while_loop for variable-length neighbors
        return neighbor_conv_features
    
    def call(self, inputs):
        return [self._call_single(inp) for inp in inputs]
```

### Model Compilation

**PyTorch** (recommended):
```python
from models.layers_pytorch import PGGCNModel
model = PGGCNModel(num_atom_features=36, r_out_channel=20, c_out_channel=1024)
model.add_rule("sum", 0, 32)
model.add_rule("multiply", 32, 33)
model.add_rule("distance", 33, 36)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

for epoch in range(epochs):
    y_pred = model(X_batch, training=True)
    loss, emp_loss, phys_loss = combined_loss(y_true, y_pred, physics_weight)
    loss.backward()
    optimizer.step()
```

**TensorFlow** (legacy):
```python
import importlib
import models.layers_update_mobley as layers
importlib.reload(layers)  # Force fresh import

m = PGGCNModel()
m.addRule("sum", 0, 32)
m.addRule("multiply", 32, 33)
m.addRule("distance", 33, 36)
m.compile(loss=combined_loss(physics_weight), optimizer=Adam(learning_rate=lr_schedule))
```

## Cross-Component Integration

### Import Paths

**PyTorch**:
- **Models**: `from models.layers_pytorch import PGGCNModel, RuleGraphConvLayer, ConvLayer`
- **Features**: `from models.dcFeaturizer import atom_features` (or `deepchem.feat.graph_features`)

**TensorFlow**:
- **Models**: `from models.layers_update_mobley import RuleGraphConvLayer` (not `models.layers.py`)
- **Features**: `from models.dcFeaturizer import atom_features`
- **Conda setup**: `import conda_installer` before TensorFlow (sets up rdkit, openmm)

### Physics Constants Embedded

The physics-informed dense layer uses **hardcoded weight initialization**:
```python
# PyTorch
physics_weights = torch.tensor([0.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1])
self.dense7 = nn.Linear(16, 1)
self.dense7.weight.copy_(physics_weights.unsqueeze(0))

# TensorFlow
dense7 = Dense(1, kernel_initializer=Constant([0.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1]))
```
This encodes physics-derived relationships. Modify carefully—these weights guide energy calculations.

### GPU Configuration

**PyTorch**:
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# GPU memory automatically managed
```

**TensorFlow**:
Set **before** importing TensorFlow:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU ID
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent OOM
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `multi-objective-pdbbind-pytorch.py` | **Main training script (PyTorch)** - RECOMMENDED |
| `multi-objective-pdbbind.py` | Main training script (TensorFlow legacy) |
| `pcgrad-multi-objective-pdbbind.py` | PCGrad variant (TensorFlow, gradient conflict resolution) |
| `models/layers_pytorch.py` | PyTorch graph convolution layers + PGGCNModel |
| `models/layers_update_mobley.py` | TensorFlow graph convolution layers |
| `models/model.py` | Lightweight TensorFlow model definition |
| `analyze_results.py` | Pickle loading + visualization utilities (framework-agnostic) |
| `conda_installer.py` | Environment setup (rdkit, openmm, pdbfixer) |
| `multi-objective-pdbbind-run.sh` | Bash wrapper (GPU, logging, path resolution) |

## Debugging & Common Issues

1. **"E+" in complex names**: Data cleaning removes invalid entries. Check `is_valid_complex_name()`.
2. **Graph conv layer slow**: `_call_single()` uses `tf.while_loop`, which has Python overhead. Consider batching improvement.
3. **Memory spikes during featurization**: Increase `chunk_size` in `prepare_features_chunked()` or reduce `batch_size`.
4. **Model weights not updating**: Verify `addRule()` is called before `compile()` and layers are trainable.
5. **TensorFlow import errors**: Run `conda_installer.install()` first or manually install `tensorflow`, `rdkit`, `deepchem`.

## Testing & Validation

No formal test suite; validation is empirical:
- Small test: `dataset_size=10, epochs=5` (~1 min) → Verify pipeline integrity
- Medium run: `dataset_size=100, epochs=50` (~15 min PyTorch, ~30 min TensorFlow) → Check loss stability
- Physics weight sweep: 5 weights tested in loop → Find optimal balance

**PyTorch vs TensorFlow Performance**:
- PyTorch: Cleaner gradient computation, stable convergence, ~20% faster on GPU
- TensorFlow: tf.while_loop overhead, occasional numerical instability with complex graphs
- Recommendation: Start with PyTorch, fall back to TensorFlow only for specific layer implementations
