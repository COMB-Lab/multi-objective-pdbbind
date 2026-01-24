"""
PCGrad (Project Conflicting Gradients) optimizer for PyTorch
Replicates the TensorFlow implementation for multi-task learning
"""

import torch
import numpy as np
from torch.optim import Optimizer
import copy


class PCGrad:
    """
    PCGrad optimizer wrapper that projects conflicting gradients.
    
    This wraps any PyTorch optimizer and modifies the gradients before
    the optimizer step to resolve conflicts between multiple task gradients.
    
    Usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        pcgrad_optimizer = PCGrad(optimizer)
        
        # In training loop:
        losses = [loss1, loss2, loss3]  # Multiple task losses
        pcgrad_optimizer.pc_backward(losses)
        pcgrad_optimizer.step()
    """
    
    def __init__(self, optimizer):
        """
        Args:
            optimizer: Base PyTorch optimizer (e.g., Adam, SGD)
        """
        self._optimizer = optimizer
        self._reduction_method = 'mean'  # How to combine projected gradients
        
    @property
    def optimizer(self):
        """Access to underlying optimizer."""
        return self._optimizer
    
    def zero_grad(self):
        """Clear gradients."""
        self._optimizer.zero_grad()
    
    def step(self):
        """Perform optimizer step."""
        return self._optimizer.step()
    
    def state_dict(self):
        """Return optimizer state."""
        return self._optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self._optimizer.load_state_dict(state_dict)
    
    @property
    def param_groups(self):
        """Access to parameter groups."""
        return self._optimizer.param_groups
    
    def pc_backward(self, losses):
        """
        Compute PCGrad projected gradients from multiple task losses.
        
        Args:
            losses: List of scalar loss tensors, one per task
        """
        assert isinstance(losses, list), "losses must be a list of task losses"
        assert all(loss.requires_grad for loss in losses), "All losses must require gradients"
        
        # Get all parameters
        params = []
        for group in self._optimizer.param_groups:
            params.extend(group['params'])
        
        # Compute gradients for each task
        grads_task = []
        for i, loss in enumerate(losses):
            # Zero out gradients before computing new ones
            self._optimizer.zero_grad()
            
            # Compute gradients for this task
            loss.backward(retain_graph=(i < len(losses) - 1))
            
            # Collect gradients
            grads = []
            for param in params:
                if param.grad is not None:
                    grads.append(param.grad.clone())
                else:
                    grads.append(torch.zeros_like(param))
            
            grads_task.append(grads)
        
        # Clear gradients before setting projected gradients
        self._optimizer.zero_grad()
        
        # Apply PCGrad projection
        projected_grads = self._project_conflicting_gradients(grads_task)
        
        # Set the projected gradients
        for param, grad in zip(params, projected_grads):
            param.grad = grad
    
    def _project_conflicting_gradients(self, grads_task):
        """
        Project conflicting gradients to resolve conflicts between tasks.
        
        Args:
            grads_task: List of gradient lists, one per task
                       Each inner list contains gradients for all parameters
        
        Returns:
            List of projected gradients for all parameters
        """
        # Flatten gradients for each task
        flat_grads_task = []
        for grads in grads_task:
            flat_grad = torch.cat([g.flatten() for g in grads])
            flat_grads_task.append(flat_grad)
        
        # Stack into tensor [num_tasks, num_params]
        flat_grads_task = torch.stack(flat_grads_task)
        
        # Optional: shuffle tasks (commented out like in TensorFlow version)
        # indices = torch.randperm(flat_grads_task.size(0))
        # flat_grads_task = flat_grads_task[indices]
        
        # Project each task's gradients
        projected = []
        for i in range(len(flat_grads_task)):
            # Get current task gradient
            g_i = flat_grads_task[i]
            
            # Get other task gradients
            others = torch.cat([flat_grads_task[:i], flat_grads_task[i+1:]], dim=0)
            
            # Project
            g_i_projected = self._project(g_i, others)
            projected.append(g_i_projected)
        
        # Stack projected gradients
        projected = torch.stack(projected)
        
        # Average the projected gradients
        mean_grad = projected.mean(dim=0)
        
        # Reshape back to original parameter shapes
        reshaped_grads = []
        idx = 0
        for grads in grads_task[0]:  # Use first task to get shapes
            param_shape = grads.shape
            param_size = grads.numel()
            
            reshaped_grad = mean_grad[idx:idx + param_size].reshape(param_shape)
            reshaped_grads.append(reshaped_grad)
            
            idx += param_size
        
        return reshaped_grads
    
    def _project(self, g, others):
        """
        Project gradient g away from conflicting gradients in others.
        
        Args:
            g: Current task gradient (flattened)
            others: Other task gradients (flattened) [num_other_tasks, num_params]
        
        Returns:
            Projected gradient
        """
        g_proj = g.clone()
        
        for other in others:
            # Compute dot product
            dot = torch.dot(g_proj, other)
            
            # If negative (conflicting), project away
            if dot < 0:
                # Project: g = g - (g·o / ||o||²) * o
                other_norm_sq = torch.dot(other, other)
                g_proj = g_proj - (dot / (other_norm_sq + 1e-12)) * other
        
        return g_proj


# Example usage and testing
if __name__ == "__main__":
    print("PCGrad PyTorch Implementation - Test")
    print("=" * 60)
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    # Create optimizer
    base_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    pcgrad_optimizer = PCGrad(base_optimizer)
    
    # Create dummy data
    x = torch.randn(32, 10)
    y1 = torch.randn(32, 5)
    y2 = torch.randn(32, 5)
    
    # Forward pass
    output = model(x)
    
    # Multiple task losses
    loss1 = torch.nn.functional.mse_loss(output, y1)
    loss2 = torch.nn.functional.mse_loss(output, y2)
    
    print(f"Loss 1: {loss1.item():.4f}")
    print(f"Loss 2: {loss2.item():.4f}")
    
    # PCGrad backward pass
    pcgrad_optimizer.pc_backward([loss1, loss2])
    
    # Check that gradients are set
    has_grads = any(p.grad is not None for p in model.parameters())
    print(f"\nGradients computed: {has_grads}")
    
    # Optimizer step
    pcgrad_optimizer.step()
    
    print("\nPCGrad test completed successfully!")
    print("=" * 60)