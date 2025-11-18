#!/usr/bin/env python3
"""
PGGCN Layer Visualization and Matrix Size Analysis

This script analyzes and visualizes:
1. Layer dimensions and parameter counts
2. Memory usage at each layer during forward/backward pass
3. Activation map sizes and growth patterns
4. Gradient flow and computational graph
5. Bottlenecks and memory hotspots

Required packages: tensorflow, matplotlib, numpy, graphviz (optional)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from collections import OrderedDict
import json

# ============================================================================
# LAYER ANALYSIS UTILITIES
# ============================================================================

class LayerProfiler:
    """Profile layer dimensions and memory usage during model execution."""
    
    def __init__(self, model, batch_size=8, max_atoms=1000):
        self.model = model
        self.batch_size = batch_size
        self.max_atoms = max_atoms
        self.layer_info = OrderedDict()
        
    def analyze_model_architecture(self):
        """Analyze the complete model architecture."""
        print("="*70)
        print("MODEL ARCHITECTURE ANALYSIS")
        print("="*70)
        
        layer_idx = 0
        
        # Manually define layer structure since PGGCN is custom
        layers_structure = [
            {
                'name': 'Input',
                'type': 'Input',
                'input_shape': (self.batch_size, self.max_atoms, 53),
                'output_shape': (self.batch_size, self.max_atoms, 53),
                'params': 0
            },
            {
                'name': 'RuleGraphConvLayer',
                'type': 'GraphConv',
                'input_shape': (self.batch_size, self.max_atoms, 38),  # Only atom features
                'output_shape': (self.batch_size, self.max_atoms, 20),  # r_out_channel
                'params': self._estimate_graph_conv_params(38, 20)
            },
            {
                'name': 'ConvLayer',
                'type': 'Aggregation',
                'input_shape': (self.batch_size, self.max_atoms, 20),
                'output_shape': (self.batch_size, 128),  # c_out_channel
                'params': self._estimate_conv_params(20, 128, self.max_atoms)
            },
            {
                'name': 'Dense1',
                'type': 'Dense',
                'input_shape': (self.batch_size, 128),
                'output_shape': (self.batch_size, 32),
                'params': 128 * 32 + 32
            },
            {
                'name': 'Dropout1',
                'type': 'Dropout',
                'input_shape': (self.batch_size, 32),
                'output_shape': (self.batch_size, 32),
                'params': 0
            },
            {
                'name': 'Dense5',
                'type': 'Dense',
                'input_shape': (self.batch_size, 32),
                'output_shape': (self.batch_size, 16),
                'params': 32 * 16 + 16
            },
            {
                'name': 'Dropout2',
                'type': 'Dropout',
                'input_shape': (self.batch_size, 16),
                'output_shape': (self.batch_size, 16),
                'params': 0
            },
            {
                'name': 'Dense6',
                'type': 'Dense',
                'input_shape': (self.batch_size, 16),
                'output_shape': (self.batch_size, 1),
                'params': 16 * 1 + 1
            },
            {
                'name': 'Merge',
                'type': 'Concatenate',
                'input_shape': [(self.batch_size, 1), (self.batch_size, 15)],
                'output_shape': (self.batch_size, 16),
                'params': 0
            },
            {
                'name': 'Dense7 (Physics)',
                'type': 'Dense',
                'input_shape': (self.batch_size, 16),
                'output_shape': (self.batch_size, 1),
                'params': 16 * 1 + 1
            }
        ]
        
        for layer_info in layers_structure:
            self._analyze_layer(layer_info)
            layer_idx += 1
        
        self._print_summary()
        return self.layer_info
    
    def _estimate_graph_conv_params(self, in_features, out_features):
        """Estimate parameters for graph convolution layer."""
        # Each rule has its own weight matrix
        # Assuming 3 rules (sum, multiply, distance)
        num_rules = 3
        params_per_rule = in_features * out_features
        return num_rules * params_per_rule
    
    def _estimate_conv_params(self, in_channels, out_channels, max_atoms):
        """Estimate parameters for convolution/aggregation layer."""
        # Aggregation typically uses learnable weights
        return in_channels * out_channels + out_channels
    
    def _analyze_layer(self, layer_dict):
        """Analyze a single layer."""
        name = layer_dict['name']
        
        # Calculate memory for activations (forward pass)
        if isinstance(layer_dict['output_shape'], tuple):
            output_elements = np.prod(layer_dict['output_shape'])
        else:
            output_elements = np.prod(layer_dict['output_shape'][0])
        
        activation_memory_mb = (output_elements * 4) / (1024**2)  # float32
        
        # Calculate parameter memory
        param_memory_mb = (layer_dict['params'] * 4) / (1024**2)
        
        # Calculate gradient memory (same as activation for backprop)
        gradient_memory_mb = activation_memory_mb
        
        # Total memory (forward + backward)
        total_memory_mb = activation_memory_mb + gradient_memory_mb + param_memory_mb
        
        self.layer_info[name] = {
            'type': layer_dict['type'],
            'input_shape': layer_dict['input_shape'],
            'output_shape': layer_dict['output_shape'],
            'params': layer_dict['params'],
            'activation_memory_mb': activation_memory_mb,
            'param_memory_mb': param_memory_mb,
            'gradient_memory_mb': gradient_memory_mb,
            'total_memory_mb': total_memory_mb,
            'flops': self._estimate_flops(layer_dict)
        }
    
    def _estimate_flops(self, layer_dict):
        """Estimate floating point operations for layer."""
        layer_type = layer_dict['type']
        
        if layer_type == 'Dense':
            # Matrix multiplication: 2 * m * n * k (where m*n × n*k matrix mult)
            if isinstance(layer_dict['input_shape'], tuple):
                in_features = layer_dict['input_shape'][-1]
                out_features = layer_dict['output_shape'][-1]
                batch = layer_dict['input_shape'][0]
                return 2 * batch * in_features * out_features
        elif layer_type == 'GraphConv':
            # Graph convolution: roughly atoms * in_features * out_features * avg_degree
            avg_degree = 4  # Average number of neighbors
            atoms = layer_dict['input_shape'][1]
            in_feat = layer_dict['input_shape'][2]
            out_feat = layer_dict['output_shape'][2]
            return atoms * in_feat * out_feat * avg_degree
        elif layer_type == 'Aggregation':
            # Aggregation across atoms
            atoms = layer_dict['input_shape'][1]
            features = layer_dict['input_shape'][2]
            out_features = layer_dict['output_shape'][1]
            return atoms * features * out_features
        
        return 0
    
    def _print_summary(self):
        """Print layer-by-layer summary."""
        print("\nLayer-by-Layer Analysis:")
        print("-" * 120)
        print(f"{'Layer':<25} {'Type':<15} {'Output Shape':<25} {'Params':<15} {'Memory (MB)':<15} {'FLOPs':<15}")
        print("-" * 120)
        
        total_params = 0
        total_memory = 0
        total_flops = 0
        
        for name, info in self.layer_info.items():
            total_params += info['params']
            total_memory += info['total_memory_mb']
            total_flops += info['flops']
            
            print(f"{name:<25} {info['type']:<15} {str(info['output_shape']):<25} "
                  f"{info['params']:<15,} {info['total_memory_mb']:<15.2f} {info['flops']:<15,.0f}")
        
        print("-" * 120)
        print(f"{'TOTAL':<25} {'':<15} {'':<25} {total_params:<15,} {total_memory:<15.2f} {total_flops:<15,.0f}")
        print("-" * 120)
        
        print(f"\nModel Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Total memory (forward + backward): {total_memory:.2f} MB")
        print(f"  Total FLOPs per batch: {total_flops:,.0f}")
        print(f"  Parameter memory: {sum(info['param_memory_mb'] for info in self.layer_info.values()):.2f} MB")
        print(f"  Activation memory: {sum(info['activation_memory_mb'] for info in self.layer_info.values()):.2f} MB")
        print(f"  Gradient memory: {sum(info['gradient_memory_mb'] for info in self.layer_info.values()):.2f} MB")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_layer_dimensions(layer_info, save_path='/home/exouser/analysis_outputs/layers-analysis/layer_dimensions.png'):
    """Visualize layer dimensions and transformations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Extract data
    layer_names = list(layer_info.keys())
    output_sizes = []
    
    for name, info in layer_info.items():
        shape = info['output_shape']
        if isinstance(shape, tuple):
            size = np.prod(shape)
        else:
            size = np.prod(shape[0])
        output_sizes.append(size)
    
    # Plot 1: Output size evolution
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
    bars = ax1.barh(range(len(layer_names)), output_sizes, color=colors, edgecolor='black')
    ax1.set_yticks(range(len(layer_names)))
    ax1.set_yticklabels(layer_names, fontsize=9)
    ax1.set_xlabel('Total Elements in Output Tensor', fontsize=12)
    ax1.set_title('Layer Output Sizes', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, size) in enumerate(zip(bars, output_sizes)):
        ax1.text(size, bar.get_y() + bar.get_height()/2, 
                f' {size:,.0f}', va='center', fontsize=8)
    
    # Plot 2: Shape transformation diagram
    ax2.axis('off')
    y_pos = 0.95
    y_step = 0.08
    
    ax2.text(0.5, y_pos, 'Layer Shape Transformations', 
            ha='center', fontsize=14, fontweight='bold')
    y_pos -= y_step * 1.5
    
    for name, info in layer_info.items():
        # Format shapes
        in_shape = str(info['input_shape'])
        out_shape = str(info['output_shape'])
        
        # Draw transformation
        ax2.text(0.05, y_pos, name, fontsize=9, fontweight='bold')
        ax2.text(0.35, y_pos, in_shape, fontsize=8, family='monospace', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax2.annotate('', xy=(0.65, y_pos), xytext=(0.55, y_pos),
                    arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
        ax2.text(0.67, y_pos, out_shape, fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        y_pos -= y_step
        
        if y_pos < 0.05:
            break
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved layer dimensions plot to: {save_path}")
    plt.close()


def plot_memory_breakdown(layer_info, save_path='/home/exouser/analysis_outputs/layers-analysis/memory_breakdown.png'):
    """Visualize memory usage breakdown across layers."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    layer_names = list(layer_info.keys())
    activation_memory = [info['activation_memory_mb'] for info in layer_info.values()]
    param_memory = [info['param_memory_mb'] for info in layer_info.values()]
    gradient_memory = [info['gradient_memory_mb'] for info in layer_info.values()]
    total_memory = [info['total_memory_mb'] for info in layer_info.values()]
    
    # Plot 1: Stacked bar chart - memory components
    x = np.arange(len(layer_names))
    width = 0.8
    
    p1 = ax1.bar(x, activation_memory, width, label='Activations', color='skyblue')
    p2 = ax1.bar(x, param_memory, width, bottom=activation_memory, 
                 label='Parameters', color='lightcoral')
    p3 = ax1.bar(x, gradient_memory, width, 
                 bottom=np.array(activation_memory) + np.array(param_memory),
                 label='Gradients', color='lightgreen')
    
    ax1.set_ylabel('Memory (MB)', fontsize=12)
    ax1.set_title('Memory Breakdown by Layer', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Pie chart - total memory distribution
    total_activation = sum(activation_memory)
    total_params = sum(param_memory)
    total_gradients = sum(gradient_memory)
    
    sizes = [total_activation, total_params, total_gradients]
    labels = [f'Activations\n{total_activation:.2f} MB',
              f'Parameters\n{total_params:.2f} MB',
              f'Gradients\n{total_gradients:.2f} MB']
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    explode = (0.05, 0.05, 0.05)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title('Total Memory Distribution', fontsize=14, fontweight='bold')
    
    # Plot 3: Layer-wise total memory
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(layer_names)))
    bars = ax3.barh(range(len(layer_names)), total_memory, color=colors, edgecolor='black')
    ax3.set_yticks(range(len(layer_names)))
    ax3.set_yticklabels(layer_names, fontsize=9)
    ax3.set_xlabel('Total Memory (MB)', fontsize=12)
    ax3.set_title('Total Memory per Layer', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, mem in zip(bars, total_memory):
        ax3.text(mem, bar.get_y() + bar.get_height()/2, 
                f' {mem:.2f}', va='center', fontsize=8)
    
    # Plot 4: Cumulative memory
    cumulative_memory = np.cumsum(total_memory)
    ax4.plot(range(len(layer_names)), cumulative_memory, 'o-', 
             linewidth=2, markersize=8, color='darkblue')
    ax4.fill_between(range(len(layer_names)), cumulative_memory, alpha=0.3)
    ax4.set_xticks(range(len(layer_names)))
    ax4.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Cumulative Memory (MB)', fontsize=12)
    ax4.set_title('Cumulative Memory Usage', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add annotations for key points
    max_mem_idx = np.argmax(total_memory)
    ax4.annotate(f'Peak: {cumulative_memory[max_mem_idx]:.2f} MB',
                xy=(max_mem_idx, cumulative_memory[max_mem_idx]),
                xytext=(max_mem_idx, cumulative_memory[max_mem_idx] * 0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved memory breakdown plot to: {save_path}")
    plt.close()


def plot_parameter_distribution(layer_info, save_path='/home/exouser/analysis_outputs/layers-analysis/parameter_distribution.png'):
    """Visualize parameter distribution across layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter layers with parameters
    layers_with_params = {name: info for name, info in layer_info.items() 
                          if info['params'] > 0}
    
    layer_names = list(layers_with_params.keys())
    param_counts = [info['params'] for info in layers_with_params.values()]
    
    # Plot 1: Bar chart
    colors = plt.cm.Spectral(np.linspace(0, 1, len(layer_names)))
    bars = ax1.bar(range(len(layer_names)), param_counts, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('Number of Parameters', fontsize=12)
    ax1.set_title('Parameters per Layer', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{count:,}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Plot 2: Pie chart
    explode = [0.05] * len(layer_names)
    ax2.pie(param_counts, labels=layer_names, autopct='%1.1f%%',
            explode=explode, shadow=True, startangle=90, colors=colors)
    ax2.set_title('Parameter Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved parameter distribution plot to: {save_path}")
    plt.close()


def plot_computational_cost(layer_info, save_path='/home/exouser/analysis_outputs/layers-analysis/computational_cost.png'):
    """Visualize computational cost (FLOPs) across layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    layer_names = list(layer_info.keys())
    flops = [info['flops'] for info in layer_info.values()]
    
    # Plot 1: FLOPs per layer
    colors = plt.cm.plasma(np.linspace(0, 1, len(layer_names)))
    bars = ax1.barh(range(len(layer_names)), flops, color=colors, edgecolor='black')
    ax1.set_yticks(range(len(layer_names)))
    ax1.set_yticklabels(layer_names, fontsize=9)
    ax1.set_xlabel('FLOPs', fontsize=12)
    ax1.set_title('Computational Cost per Layer', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, flop in zip(bars, flops):
        if flop > 0:
            ax1.text(flop, bar.get_y() + bar.get_height()/2,
                    f' {flop:,.0f}', va='center', fontsize=8)
    
    # Plot 2: Memory vs FLOPs scatter
    memory = [info['total_memory_mb'] for info in layer_info.values()]
    
    # Filter out zero FLOPs for better visualization
    valid_indices = [i for i, f in enumerate(flops) if f > 0]
    valid_flops = [flops[i] for i in valid_indices]
    valid_memory = [memory[i] for i in valid_indices]
    valid_names = [layer_names[i] for i in valid_indices]
    
    scatter = ax2.scatter(valid_flops, valid_memory, s=200, c=range(len(valid_flops)),
                         cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, name in enumerate(valid_names):
        ax2.annotate(name, (valid_flops[i], valid_memory[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('FLOPs (log scale)', fontsize=12)
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.set_title('Memory vs Computational Cost', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax2, label='Layer Index')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved computational cost plot to: {save_path}")
    plt.close()


def create_architecture_diagram(layer_info, save_path='/home/exouser/analysis_outputs/layers-analysis/architecture_diagram.png'):
    """Create a visual diagram of the model architecture."""
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layer_info) + 1)
    ax.axis('off')
    
    y_pos = len(layer_info)
    box_width = 8
    box_height = 0.7
    x_center = 5
    
    # Title
    ax.text(x_center, y_pos + 0.5, 'PGGCN Model Architecture',
            ha='center', fontsize=16, fontweight='bold')
    
    for i, (name, info) in enumerate(layer_info.items()):
        # Determine box color based on memory usage
        mem = info['total_memory_mb']
        if mem > 10:
            color = 'lightcoral'
        elif mem > 1:
            color = 'lightyellow'
        else:
            color = 'lightgreen'
        
        # Draw layer box
        box = FancyBboxPatch((x_center - box_width/2, y_pos - box_height/2),
                            box_width, box_height,
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        
        # Add layer name and info
        ax.text(x_center, y_pos, name, ha='center', va='center',
                fontsize=10, fontweight='bold')
        
        # Add shape and memory info
        shape_str = str(info['output_shape'])
        mem_str = f"{info['total_memory_mb']:.2f} MB"
        param_str = f"{info['params']:,} params" if info['params'] > 0 else ""
        
        ax.text(x_center - box_width/2 - 0.1, y_pos, shape_str,
                ha='right', va='center', fontsize=8, style='italic')
        ax.text(x_center + box_width/2 + 0.1, y_pos, mem_str,
                ha='left', va='center', fontsize=8, color='red')
        
        if param_str:
            ax.text(x_center, y_pos - box_height/2 - 0.15, param_str,
                    ha='center', va='top', fontsize=7, color='blue')
        
        # Draw arrow to next layer
        if i < len(layer_info) - 1:
            arrow = FancyArrowPatch((x_center, y_pos - box_height/2 - 0.05),
                                   (x_center, y_pos - 1 + box_height/2 + 0.05),
                                   arrowstyle='->', mutation_scale=20, 
                                   linewidth=2, color='darkblue')
            ax.add_patch(arrow)
        
        y_pos -= 1
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='lightgreen', label='Low memory (< 1 MB)'),
        mpatches.Patch(color='lightyellow', label='Medium memory (1-10 MB)'),
        mpatches.Patch(color='lightcoral', label='High memory (> 10 MB)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved architecture diagram to: {save_path}")
    plt.close()


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def main():
    """Main function to run complete layer analysis."""
    print("="*70)
    print("PGGCN LAYER VISUALIZATION AND ANALYSIS")
    print("="*70)
    
    # Configuration
    batch_size = 8
    max_atoms = 1000  # Typical value from your data
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max atoms: {max_atoms}")
    
    # Create profiler and analyze
    profiler = LayerProfiler(model=None, batch_size=batch_size, max_atoms=max_atoms)
    layer_info = profiler.analyze_model_architecture()
    
    # Generate all visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_layer_dimensions(layer_info)
    plot_memory_breakdown(layer_info)
    plot_parameter_distribution(layer_info)
    plot_computational_cost(layer_info)
    create_architecture_diagram(layer_info)
    
    # Save layer info to JSON for later analysis
    json_data = {}
    for name, info in layer_info.items():
        json_data[name] = {
            'type': info['type'],
            'input_shape': str(info['input_shape']),
            'output_shape': str(info['output_shape']),
            'params': info['params'],
            'total_memory_mb': info['total_memory_mb'],
            'flops': info['flops']
        }
    
    with open('layer_analysis.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - layer_dimensions.png")
    print("  - memory_breakdown.png")
    print("  - parameter_distribution.png")
    print("  - computational_cost.png")
    print("  - architecture_diagram.png")
    print("  - layer_analysis.json")

if __name__ == "__main__":
    main()