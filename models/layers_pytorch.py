"""
PyTorch implementation of Graph Convolutional Layers
Translated from TensorFlow layers_update_mobley.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RuleGraphConvLayer(nn.Module):
    """
    Rule-based Graph Convolution Layer with custom combination rules.
    Processes molecular graphs using specified aggregation functions (sum, multiply, distance).
    """
    
    def __init__(self, out_channel, num_features=36, num_bond=2, activation_fn=None, 
                 combination_rules=None):
        """
        Args:
            out_channel: Number of output channels
            num_features: Number of input features per atom (default 36 for atom properties)
            num_bond: Number of bond/neighbor features (default 2 for two neighbors)
            activation_fn: Optional activation function
            combination_rules: List of rules to apply
        """
        super(RuleGraphConvLayer, self).__init__()
        self.out_channel = out_channel
        self.num_features = num_features
        self.num_bond = num_bond
        self.activation_fn = activation_fn
        self.combination_rules = combination_rules if combination_rules is not None else []
        
        # Weight matrices for self and neighbor convolutions
        self.w_s = nn.Parameter(torch.randn(num_features, out_channel) * np.sqrt(2.0 / (num_features + out_channel)))
        self.w_n = nn.Parameter(torch.randn(num_features + num_bond, out_channel) * np.sqrt(2.0 / (num_features + num_bond + out_channel)))
        
        # Initialize with Glorot uniform
        nn.init.xavier_uniform_(self.w_s)
        nn.init.xavier_uniform_(self.w_n)
    
    def atom_distance(self, x, y):
        """Calculate Euclidean distance between two atom feature vectors."""
        return torch.sqrt(torch.sum((x - y) ** 2) + 1e-8)
    
    def add_rule(self, rule, start_index, end_index=None):
        """
        Add a combination rule.
        
        Args:
            rule: Either "sum", "multiply", "distance", "divide", "subtract" or a callable
            start_index: Starting feature index
            end_index: Ending feature index (optional)
        """
        rules_dict = {
            "sum": torch.add,
            "multiply": torch.mul,
            "distance": "distance",
            "divide": torch.div,
            "subtract": torch.sub,
        }
        
        if isinstance(rule, str):
            rule = rules_dict[rule]
        
        if end_index is None:
            self.combination_rules.append([[start_index], rule])
        else:
            self.combination_rules.append([[start_index, end_index], rule])
    
    def _call_single(self, features):
        """
        Process a single molecule's atoms through graph convolution.
        
        Args:
            features: Tensor of shape [num_atoms, num_features + num_bond]
                     Last num_bond dimensions contain neighbor indices
        
        Returns:
            Tensor of shape [num_atoms, out_channel]
        """
        device = features.device
        num_atoms = features.shape[0]
        
        # Self convolution: features @ w_s
        atom_features = features[:, :self.num_features]  # [num_atoms, num_features]
        self_conv_features = torch.matmul(atom_features, self.w_s)  # [num_atoms, out_channel]
        
        # Neighbor convolution
        neighbor_conv_features = torch.zeros(num_atoms, self.out_channel, device=device)
        
        for atom_idx in range(num_atoms):
            atom_self_features = features[atom_idx, :self.num_features]  # [num_features]
            neighbor_indices = features[atom_idx, self.num_features:]  # [num_bond]
            
            # Process each neighbor (up to 2)
            for neighbor_slot in range(self.num_bond):
                neighbor_idx = int(neighbor_indices[neighbor_slot].item())
                
                # Skip invalid neighbor indices (e.g., -1 for padding)
                if neighbor_idx < 0 or neighbor_idx >= num_atoms:
                    continue
                
                neighbor_features = features[neighbor_idx, :self.num_features]  # [num_features]
                
                # Apply combination rules and concatenate results
                rule_outputs = []
                distance_factor = 1.0
                
                for rule_indices, rule_func in self.combination_rules:
                    start_idx = rule_indices[0]
                    
                    if rule_func == "distance":
                        # Distance metric: calculate distance and scale by 1/distance^2
                        if len(rule_indices) == 2:
                            end_idx = rule_indices[1]
                            dist = self.atom_distance(
                                atom_self_features[start_idx:end_idx],
                                neighbor_features[start_idx:end_idx]
                            )
                        else:
                            dist = self.atom_distance(atom_self_features, neighbor_features)
                        
                        distance_factor = 1.0 / torch.clamp(dist ** 2, min=1e-6)
                        # Store the neighbor features for later scaling
                        if len(rule_indices) == 2:
                            rule_outputs.append(neighbor_features[start_idx:rule_indices[1]])
                        else:
                            rule_outputs.append(neighbor_features)
                    else:
                        # Apply the combination rule
                        if len(rule_indices) == 2:
                            end_idx = rule_indices[1]
                            rule_output = rule_func(
                                atom_self_features[start_idx:end_idx],
                                neighbor_features[start_idx:end_idx]
                            )
                        else:
                            rule_output = rule_func(atom_self_features, neighbor_features)
                        rule_outputs.append(rule_output)
                
                # Concatenate all rule outputs with neighbor indices to form [num_features + num_bond]
                if len(rule_outputs) > 0:
                    scaled_features = torch.cat(rule_outputs + [neighbor_indices], dim=0)  # [num_features + num_bond]
                    
                    # Apply distance scaling if distance rule was used
                    if distance_factor != 1.0:
                        # Scale only the rule output part, not the neighbor indices
                        scaled_features[:len(torch.cat(rule_outputs, dim=0))] *= distance_factor
                    
                    # Add neighbor contribution
                    neighbor_contribution = torch.matmul(scaled_features.unsqueeze(0), self.w_n).squeeze(0)
                    neighbor_conv_features[atom_idx] += neighbor_contribution
            
            # Add self contribution to neighbor output
            neighbor_conv_features[atom_idx] += self_conv_features[atom_idx]
        
        return neighbor_conv_features
    
    def forward(self, inputs):
        """
        Process a batch of molecules.
        
        Args:
            inputs: List of tensors, each of shape [num_atoms, num_features + num_bond]
        
        Returns:
            List of tensors, each of shape [num_atoms, out_channel]
        """
        outputs = []
        for inp in inputs:
            outputs.append(self._call_single(inp))
        return outputs


class ConvLayer(nn.Module):
    """
    Graph Convolution Layer that aggregates features across atoms.
    """
    
    def __init__(self, out_channel, num_features=20):
        """
        Args:
            out_channel: Number of output channels
            num_features: Number of input features
        """
        super(ConvLayer, self).__init__()
        self.out_channel = out_channel
        self.num_features = num_features
        
        # Weight matrix for feature transformation
        self.w = nn.Parameter(torch.randn(num_features, out_channel) * np.sqrt(2.0 / (num_features + out_channel)))
        nn.init.xavier_uniform_(self.w)
    
    def _call_single(self, inp):
        """
        Process a single molecule.
        
        Args:
            inp: Tensor of shape [num_atoms, num_features]
        
        Returns:
            Tensor of shape [out_channel]
        """
        out = torch.zeros(self.out_channel, device=inp.device, dtype=inp.dtype)
        
        for feature in inp:
            feature = feature.view(1, -1)  # [1, num_features]
            transformed = torch.tanh(torch.matmul(feature, self.w))  # [1, out_channel]
            out += transformed.squeeze(0)
        
        return out
    
    def forward(self, inputs):
        """
        Process a batch of molecules.
        
        Args:
            inputs: List of tensors, each of shape [num_atoms, num_features]
        
        Returns:
            Tensor of shape [batch_size, out_channel]
        """
        outputs = []
        for inp in inputs:
            outputs.append(self._call_single(inp))
        
        return torch.stack(outputs)


class PGGCNModel(nn.Module):
    """
    Physics-Guided Graph Convolutional Network for binding affinity prediction.
    Combines empirical ML with physics-based energy calculations.
    """
    
    def __init__(self, num_atom_features=36, r_out_channel=20, c_out_channel=1024,
                 l2=1e-2, dropout_rate=0.2):
        """
        Args:
            num_atom_features: Number of atom features (default 36)
            r_out_channel: Output channels for RuleGraphConv (default 20)
            c_out_channel: Output channels for ConvLayer (default 1024)
            l2: L2 regularization weight
            dropout_rate: Dropout probability
        """
        super(PGGCNModel, self).__init__()
        
        # Graph convolution layers
        self.rule_graph_conv = RuleGraphConvLayer(r_out_channel, num_atom_features, num_bond=2)
        self.conv = ConvLayer(c_out_channel, r_out_channel)
        
        # Dense layers with regularization
        self.dense1 = nn.Linear(c_out_channel, 32)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.dense5 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.dense6 = nn.Linear(16, 1)
        
        # Physics-informed dense layer with custom initialization
        # Hardcoded physics weights: empirically derived energy relationship
        physics_weights = torch.tensor([0.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1], dtype=torch.float32)
        self.dense7 = nn.Linear(16, 1)  # 1 prediction + 15 physics features = 16
        with torch.no_grad():
            self.dense7.weight.copy_(physics_weights.unsqueeze(0))
            self.dense7.bias.fill_(0)
        
        # L2 regularization
        self.l2 = l2
    
    def add_rule(self, rule, start_index, end_index=None):
        """Add a combination rule to the graph convolution layer."""
        self.rule_graph_conv.add_rule(rule, start_index, end_index)
    
    def forward(self, inputs, training=True):
        """
        Forward pass.
        
        Args:
            inputs: Tensor of shape [batch_size, max_atoms, num_features]
                   Features include atom properties (first 38 dims) + physics info (15 dims)
            training: Boolean for training mode (affects dropout)
        
        Returns:
            Tensor of shape [batch_size, 16] containing [prediction, physics_info]
        """
        # Extract physics info from first atom of each molecule
        physics_info = inputs[:, 0, 38:]  # [batch_size, 15]
        
        # Extract atom features (first 38 dimensions)
        atom_features = inputs[:, :, :38]  # [batch_size, max_atoms, 38]
        
        # Process each sample through graph convolution
        conv_outputs = []
        for i in range(atom_features.shape[0]):
            sample = atom_features[i]  # [max_atoms, 38]
            
            # Remove padding (zero rows)
            mask = torch.sum(torch.abs(sample), dim=1) > 0
            sample_actual = sample[mask]
            
            # Process through graph conv layers
            x_list = self.rule_graph_conv([sample_actual])
            x = self.conv(x_list)  # [1, c_out_channel]
            conv_outputs.append(x)
        
        # Stack all batch outputs
        x = torch.cat(conv_outputs, dim=0)  # [batch_size, c_out_channel]
        
        # Apply dense layers
        x = F.relu(self.dense1(x))
        x = self.dropout1(x) if training else x
        x = F.relu(self.dense5(x))
        x = self.dropout2(x) if training else x
        model_pred = self.dense6(x)  # [batch_size, 1]
        
        # Merge with physics information
        merged = torch.cat([model_pred, physics_info], dim=1)  # [batch_size, 16]
        out = self.dense7(merged)  # [batch_size, 1]
        
        # Return prediction concatenated with physics info
        return torch.cat([out, physics_info], dim=1)  # [batch_size, 16]
    
    def get_l2_loss(self):
        """Calculate L2 regularization loss."""
        l2_loss = torch.tensor(0.0, device=self.dense1.weight.device)
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2 * l2_loss


class PCGradOptimizer(torch.optim.Optimizer):
    """
    PCGrad optimizer for multi-task learning.
    Resolves gradient conflicts by projecting conflicting gradients.
    
    Reference: Gradient Surgery for Multi-Task Learning (https://arxiv.org/abs/2001.06782)
    """
    
    def __init__(self, params, base_optimizer=None, **kwargs):
        """
        Args:
            params: Parameters to optimize
            base_optimizer: Underlying optimizer (default: Adam)
        """
        if base_optimizer is None:
            base_optimizer = torch.optim.Adam(params, lr=1e-5)
        
        self.base_optimizer = base_optimizer
        super().__init__(params, {})
    
    def step(self, losses_list, tape=None):
        """
        Compute PCGrad projected gradients and apply optimizer step.
        
        Args:
            losses_list: List of loss tensors for different tasks (empirical + physics)
            tape: GradientTape (for TensorFlow compatibility, ignored in PyTorch)
        """
        # Compute gradients for each task
        grads_task = []
        
        for loss in losses_list:
            self.base_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            # Collect gradients
            grads = []
            for param in self.base_optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    grads.append(param.grad.clone())
                else:
                    grads.append(torch.zeros_like(param))
            grads_task.append(grads)
        
        # Flatten all gradients
        def flatten(grads):
            return torch.cat([g.view(-1) for g in grads])
        
        flat_grads_task = [flatten(g) for g in grads_task]
        flat_grads_task = torch.stack(flat_grads_task)
        
        # PCGrad projection
        def project(g, others):
            for o in others:
                dot = torch.sum(g * o)
                if dot < 0:
                    g = g - (dot / (torch.sum(o * o) + 1e-12)) * o
            return g
        
        projected = []
        for i in range(len(flat_grads_task)):
            others = torch.cat([flat_grads_task[:i], flat_grads_task[i+1:]], dim=0)
            projected.append(project(flat_grads_task[i].clone(), [others[j] for j in range(len(others))]))
        
        projected = torch.stack(projected)
        mean_grad = torch.mean(projected, dim=0)
        
        # Reshape back to parameter shapes
        self.base_optimizer.zero_grad()
        idx = 0
        for param in self.base_optimizer.param_groups[0]['params']:
            size = param.data.numel()
            param.grad = mean_grad[idx:idx + size].view(param.shape)
            idx += size
        
        # Apply optimizer step
        self.base_optimizer.step()
