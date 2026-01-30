import torch
import torch.nn as nn
import numpy as np
import time


class RuleGraphConvLayer(nn.Module):
    """
    PyTorch implementation of RuleGraphConvLayer that handles nested molecular structures.
    """

    def __init__(self,
                 out_channel,
                 num_features=81,
                 num_bond=22,
                 activation_fn=None,
                 combination_rules=None):
        """
        Args:
            out_channel: Number of output channels
            num_features: Number of atom features (default: 81)
            num_bond: Number of bond features (default: 22)
            activation_fn: Activation function to apply (default: None)
            combination_rules: List of [indices, operation] pairs for combining features
        """
        super(RuleGraphConvLayer, self).__init__()
        self.out_channel = out_channel
        self.num_features = num_features
        self.num_bond = num_bond
        self.activation_fn = activation_fn
        self.combination_rules = combination_rules if combination_rules is not None else []

        # Weight matrices
        self.w_s = nn.Parameter(torch.empty(num_features, out_channel))
        self.w_n = nn.Parameter(torch.empty(num_features, out_channel))

        # Initialize weights
        nn.init.xavier_uniform_(self.w_s)
        nn.init.xavier_uniform_(self.w_n)

    def atom_distance(self, x, y):
        """Calculate Euclidean distance between two atoms."""
        return torch.sqrt(torch.sum((x - y) ** 2))

    def add_rule(self, rule, start_index, end_index=None):
        """
        Add a combination rule for feature processing.

        Args:
            rule: Either a string ('sum', 'multiply', 'distance', 'divide', 'subtract')
                  or a function
            start_index: Start index for feature slice
            end_index: End index for feature slice (None for single index)
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
    def _call_single(self, features_with_info):
        """
        Vectorized processing of a single molecule.
        """
        # Separate features from neighbor info
        atom_features = features_with_info[:, :self.num_features]
        neighbor_indices = features_with_info[:, self.num_features:self.num_features + 2]
        num_atoms = features_with_info.shape[0]

        # Convert to long and create valid mask
        neighbor_indices = neighbor_indices.long()
        valid_mask = (neighbor_indices >= 1) & (neighbor_indices < num_atoms)

        # Self convolution (already vectorized)
        self_conv_features = torch.matmul(atom_features, self.w_s)  # [num_atoms, out_channel]
        
        # ========== VECTORIZED NEIGHBOR PROCESSING ==========
        
        # Flatten neighbor indices and mask
        # Shape: [num_atoms, 2] -> [num_atoms * 2]
        neighbor_indices_flat = neighbor_indices.flatten()
        valid_mask_flat = valid_mask.flatten()
        
        # Get valid neighbor pairs
        valid_neighbor_indices = neighbor_indices_flat[valid_mask_flat]  # [num_valid]
        
        # Get corresponding atom indices for each valid neighbor
        # atom_ids tells us which atom each neighbor belongs to
        atom_ids = torch.arange(num_atoms, device=atom_features.device).repeat_interleave(2)
        atom_ids = atom_ids[valid_mask_flat]  # [num_valid]
        
        if len(valid_neighbor_indices) == 0:
            # No valid neighbors, return self convolution only
            output = self_conv_features
            if self.activation_fn is not None:
                output = self.activation_fn(output)
            return output
        
        # Gather all self and neighbor features
        self_feats = atom_features[atom_ids]  # [num_valid, num_features]
        neighbor_feats = atom_features[valid_neighbor_indices]  # [num_valid, num_features]
        
        # Apply combination rules to ALL valid pairs in parallel
        combined_features = []
        distances = None
        for j, (indices, operation) in enumerate(self.combination_rules):
            if len(indices) == 1:
                # Single index - take features from that index onward
                start_idx = indices[0]
                result = operation(
                    self_feats[:, start_idx:], 
                    neighbor_feats[:, start_idx:]
                )  # [num_valid, remaining_features]
            else:
                # Range of indices
                start_idx, end_idx = indices[0], indices[1]
                
                if operation == "distance":
                    # Calculate distances for all valid pairs
                    diff = self_feats[:, start_idx:end_idx] - neighbor_feats[:, start_idx:end_idx]
                    distances = torch.sqrt(torch.sum(diff ** 2, dim=1))  # [num_valid]
                    result = neighbor_feats[:, start_idx:end_idx]
                else:
                    # Apply operation
                    result = operation(
                        self_feats[:, start_idx:end_idx],
                        neighbor_feats[:, start_idx:end_idx]
                    )
            
            combined_features.append(result)
        
        # Concatenate all combined features
        new_ordered_features = torch.cat(combined_features, dim=1)  # [num_valid, total_feature_size]
        
        # Apply distance scaling if distance was calculated
        if distances is not None:
            # Clamp distances to avoid division by very small numbers
            distances = torch.clamp(distances, min=1e-3)
            # Scale each feature vector by distance squared
            new_ordered_features = new_ordered_features / (distances.unsqueeze(1) ** 2)
        
        # Apply neighbor weight: [num_valid, num_features] @ [num_features, out_channel]
        neighbor_contributions = torch.matmul(new_ordered_features, self.w_n)  # [num_valid, out_channel]
        
        # Scatter contributions back to atoms
        # atom_ids tells us which atom each contribution belongs to
        neighbor_conv_features = torch.zeros(
            num_atoms, 
            self.out_channel, 
            device=atom_features.device
        )
        neighbor_conv_features.index_add_(0, atom_ids, neighbor_contributions)
        
        # Combine self and neighbor convolutions
        output = self_conv_features + neighbor_conv_features
        
        # Apply activation if specified
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        
        return output

    def forward(self, inputs):
        """
        Forward pass for a batch of molecules.

        Args:
            inputs: List of tensors, each of shape [num_atoms_i, num_features + 2 + info_size]
                   Each tensor represents a different molecule with variable number of atoms

        Returns:
            List of tensors, each of shape [num_atoms_i, out_channel]
        """
        outputs = []
        for mol_features in inputs:
            output = self._call_single(mol_features)
            outputs.append(output)
        return outputs


class ConvLayer(nn.Module):
    """
    Convolution layer that aggregates atom features to molecule-level features.
    """

    def __init__(self, out_channel, num_features=20):
        super(ConvLayer, self).__init__()
        self.out_channel = out_channel
        self.num_features = num_features
        self.w = nn.Parameter(torch.empty(num_features, out_channel))
        nn.init.xavier_uniform_(self.w)

    def _call_single(self, atom_features):
        """
        Process a single molecule's atom features.

        Args:
            atom_features: Tensor of shape [num_atoms, num_features]

        Returns:
            Tensor of shape [out_channel]
        """
        # Sum over all atoms with tanh activation
        mol_feature = torch.zeros(self.out_channel, device=atom_features.device)

        for atom_feat in atom_features:
            transformed = torch.matmul(atom_feat.unsqueeze(0), self.w)  # [1, out_channel]
            mol_feature += torch.tanh(transformed).squeeze(0)

        return mol_feature

    def forward(self, inputs):
        """
        Forward pass for a batch of molecules.

        Args:
            inputs: List of tensors, each of shape [num_atoms_i, num_features]

        Returns:
            Tensor of shape [batch_size, out_channel]
        """
        outputs = []
        for mol_features in inputs:
            output = self._call_single(mol_features)
            outputs.append(output)
        return torch.stack(outputs)


class PGGCNModel(nn.Module):
    """
    Complete PGGCN model for molecular property prediction.
    CORRECTED: Matches TensorFlow architecture exactly.
    """
    
    def __init__(self, num_atom_features=36, r_out_channel=20, c_out_channel=128, dropout_rate=0.2):
        super(PGGCNModel, self).__init__()

        self.num_atom_features = num_atom_features
        self.num_physics_features = 15  # Number of physics-based features to include
        
        self.rule_graph_conv = RuleGraphConvLayer(r_out_channel, num_atom_features, 0)
        self.conv = ConvLayer(c_out_channel, r_out_channel)
        
        # Dense layers matching TensorFlow exactly
        # TF: dense1 (128→32, relu) → dropout → dense5 (32→16, relu) → dropout → dense6 (16→1)
        self.dense1 = nn.Linear(c_out_channel, 32)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.dense2 = nn.Linear(32, 16)  # Called 'dense5' in TF
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.dense3 = nn.Linear(16, 1)  # Called 'dense6' in TF - produces model_var
        
        # Final layer (called 'dense7' in TF) combines model_var + physics_info
        # Input: 1 (model_var) + 15 (physics_info) = 16 features
        self.dense_final = nn.Linear(16, 1)
        
        # Initialize final layer weights to match dataset([.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1])
        # Also try .5 like sahar's pdbbind code
        with torch.no_grad():
            init_weights = torch.tensor([
                0.3,   # model_var weight
                -1.0, -1.0, 1.0,  # VDW: protein, ligand, complex
                -1.0, -1.0, 1.0,  # protein, ligand, complex
                -1.0, -1.0, 1.0,  # protein, ligand, complex
                -1.0, -1.0, 1.0,  # protein, ligand, complex
                -1.0, -1.0, 1.0   # protein, ligand, complex
            ]).reshape(1, 16)  # Shape: [out_features=1, in_features=16] for nn.Linear
            self.dense_final.weight.copy_(init_weights)
            self.dense_final.bias.zero_()
        
        self.relu = nn.ReLU()

    def add_rule(self, rule, start_index, end_index=None):
        """Add a combination rule to the RuleGraphConvLayer."""
        self.rule_graph_conv.add_rule(rule, start_index, end_index)

    def forward(self, batch_molecules, training=True):
        """
        Forward pass matching TensorFlow exactly.
        
        Args:
            batch_molecules: List of tensors, each of shape [num_atoms_i, num_features + 2 neighbors + physics_info]
            training: Boolean indicating if in training mode (for dropout)

        Returns:
            Tuple of (predictions, model_var, physics_info_tensor)
        """
        # Extract atom features and physics info
        atom_features_batch = []
        physics_info_batch = []

        for mol in batch_molecules:
            # Atom features: [0:num_atom_features] + neighbor indices [num_atom_features:num_atom_features+2]
            atom_feat = mol[:, :self.num_atom_features + 2]  # Include neighbor indices
            atom_features_batch.append(atom_feat)

            # Physics info: last 15 columns
            # Take from first atom (same for all atoms in molecule)
            physics_info = mol[0, -self.num_physics_features:]  # Shape: [15]
            physics_info_batch.append(physics_info)

        # Stack physics info into a batch tensor
        physics_info_tensor = torch.stack(physics_info_batch)  # [batch_size, 15]

        # Apply rule-based graph convolution
        x = self.rule_graph_conv(atom_features_batch)

        # Apply convolution to get molecule-level features
        x = self.conv(x)  # [batch_size, 128]

        # Dense layer 1 with ReLU and dropout
        x = self.dense1(x)  # [batch_size, 32]
        x = self.relu(x)
        if training:
            x = self.dropout1(x)
        
        # Dense layer 2 (called dense5 in TF) with ReLU and dropout
        x = self.dense2(x)  # [batch_size, 16]
        x = self.relu(x)
        if training:
            x = self.dropout2(x)
        
        # Dense layer 3 (called dense6 in TF) - produces model_var
        model_var = self.dense3(x)  # [batch_size, 1]

        # Concatenate model_var with physics_info
        merged = torch.cat([model_var, physics_info_tensor], dim=1)  # [batch_size, 16]

        # Final dense layer (called dense7 in TF)
        out = self.dense_final(merged)  # [batch_size, 1]

        # Return final prediction, model_var (for loss), and physics_info (for loss)
        return out, model_var, physics_info_tensor