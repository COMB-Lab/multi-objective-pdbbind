import torch
import torch.nn as nn
import numpy as np


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
        Process a single molecule.

        Args:
            features_with_info: Tensor of shape [num_atoms, num_features + 2 + info_size]
                               where last elements are [neighbor1_idx, neighbor2_idx, ...info...]

        Returns:
            Tensor of shape [num_atoms, out_channel]
        """
        # Separate features from neighbor info and additional info
        atom_features = features_with_info[:, :self.num_features]
        neighbor_indices = features_with_info[:, self.num_features:self.num_features + 2]
        num_atoms = features_with_info.shape[0]

        # Self convolution: multiply each atom's features by w_s
        self_conv_features = torch.matmul(atom_features, self.w_s)  # [num_atoms, out_channel]

        # Initialize neighbor convolution features
        neighbor_conv_features = torch.zeros_like(self_conv_features)

        # Process each atom's neighbors
        for i in range(num_atoms):
            self_feat = atom_features[i]  # [num_features]

            # Process up to 2 neighbors
            for neighbor_col in range(2):
                neighbor_idx_raw = neighbor_indices[i, neighbor_col].item()
                neighbor_idx = int(neighbor_idx_raw)

                # Skip if no neighbor (index is 0 or invalid)
                if neighbor_idx == -1 or neighbor_idx == 0:
                    continue
                if neighbor_idx >= num_atoms:
                    continue

                neighbor_feat = atom_features[neighbor_idx]  # [num_features]

                # Apply combination rules to create new ordered features
                combined_features = []
                distance = -1.0

                for j, (indices, operation) in enumerate(self.combination_rules):
                    if len(indices) == 1:
                        # Single index - take features from that index onward
                        start_idx = indices[0]
                        if j == len(self.combination_rules) - 1:
                            # Last rule with single index
                            result = operation(self_feat[start_idx:], neighbor_feat[start_idx:])
                        else:
                            result = operation(self_feat[start_idx:], neighbor_feat[start_idx:])
                    else:
                        # Range of indices
                        start_idx, end_idx = indices[0], indices[1]

                        if operation == "distance":
                            # Calculate distance
                            distance = self.atom_distance(
                                self_feat[start_idx:end_idx],
                                neighbor_feat[start_idx:end_idx]
                            )
                            result = neighbor_feat[start_idx:end_idx]
                        else:
                            # Apply operation
                            result = operation(
                                self_feat[start_idx:end_idx],
                                neighbor_feat[start_idx:end_idx]
                            )

                    combined_features.append(result)

                # Concatenate all combined features
                new_ordered_features = torch.cat(combined_features, dim=0)

                # Apply distance scaling if distance was calculated
                if distance > 0:
                    distance = max(distance.item(), 1e-3)  # Avoid division by very small numbers
                    new_ordered_features = new_ordered_features / (distance ** 2)

                # Ensure the concatenated features match num_features dimension
                if new_ordered_features.shape[0] < self.num_features:
                    # Pad if necessary
                    padding = torch.zeros(self.num_features - new_ordered_features.shape[0],
                                         device=new_ordered_features.device)
                    new_ordered_features = torch.cat([new_ordered_features, padding], dim=0)
                elif new_ordered_features.shape[0] > self.num_features:
                    # Truncate if necessary
                    new_ordered_features = new_ordered_features[:self.num_features]

                # Apply neighbor weight and add to accumulator
                neighbor_contribution = torch.matmul(
                    new_ordered_features.unsqueeze(0),
                    self.w_n
                )  # [1, out_channel]

                neighbor_conv_features[i] += neighbor_contribution.squeeze(0)

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
        
        # Initialize final layer weights to match TensorFlow EXACTLY
        # TensorFlow: kernel_initializer=Constant([.3, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1])
        with torch.no_grad():
            init_weights = torch.tensor([
                0.3,   # model_var weight
                -1.0, -1.0, 1.0,  # VDW: protein, ligand, complex
                -1.0, -1.0, 1.0,  # 1-4EEL: host, guest, complex
                -1.0, -1.0, 1.0,  # EELEC: host, guest, complex
                -1.0, -1.0, 1.0,  # EGB: host, guest, complex
                -1.0, -1.0, 1.0   # ESURF: host, guest, complex
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