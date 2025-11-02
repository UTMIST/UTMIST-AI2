import pytest
import ttnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

"""
my_agent_tt demonstrates how to take a PyTorch model / agent and allows you to reimplement your agent using TTNN to leverage
Tenstorrent hardware.

Usual workflow is first define your PyTorch modules using torch operations as seen in class MLPPolicy
The next step is to convert your weights to ttnn tensors and run forward pass on them.

Check out these ttnn tutorials here for how to get started with using ttnn APIs: 
- https://github.com/tenstorrent/tt-metal/tree/main/ttnn    
- https://github.com/tenstorrent/tt-metal/blob/main/ttnn/tutorials/001.ipynb
- https://github.com/tenstorrent/tt-metal/blob/main/ttnn/tutorials/002.ipynb
- ...
"""

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.bfloat16)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.bfloat16)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, action_dim, dtype=torch.bfloat16)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TTMLPPolicy(nn.Module):
    def __init__(self, state_dict, mesh_device):
        super(TTMLPPolicy, self).__init__()
        self.mesh_device = mesh_device
        self.fc1 = ttnn.from_torch(
            state_dict["fc1.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc2 = ttnn.from_torch(
            state_dict["fc2.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc3 = ttnn.from_torch(
            state_dict["fc3.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        self.fc1_b = ttnn.from_torch(state_dict["fc1.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.fc2_b = ttnn.from_torch(state_dict["fc2.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.fc3_b = ttnn.from_torch(state_dict["fc3.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        print("Running TT forward pass!")
        obs = obs.to(torch.bfloat16)
        tt_obs = ttnn.from_torch(obs, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)

        x1 = ttnn.linear(tt_obs, self.fc1, bias=self.fc1_b, activation="relu")
        tt_obs.deallocate()

        x2 = ttnn.linear(x1, self.fc2, bias=self.fc2_b, activation="relu")
        x1.deallocate()

        x3 = ttnn.linear(x2, self.fc3, bias=self.fc3_b)
        x2.deallocate()

        tt_out = ttnn.to_torch(x3).flatten().to(torch.float32)

        return tt_out


def check_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor, threshold: float = 0.99) -> bool:
    """
    Check if the Pearson correlation coefficient (PCC) between two tensors exceeds a given threshold.
    
    Args:
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.
        threshold (float): Minimum acceptable correlation (default: 0.99).
    
    Returns:
        bool: True if PCC >= threshold, else False.
    """
    # Flatten tensors to 1D
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()

    # Ensure same shape
    if t1.shape != t2.shape:
        raise ValueError("Input tensors must have the same number of elements")

    # Compute Pearson correlation coefficient
    t1_mean = t1.mean()
    t2_mean = t2.mean()
    numerator = torch.sum((t1 - t1_mean) * (t2 - t2_mean))
    denominator = torch.sqrt(torch.sum((t1 - t1_mean) ** 2) * torch.sum((t2 - t2_mean) ** 2))
    pcc = numerator / denominator

    # Check if it exceeds threshold
    return pcc.item() >= threshold


def test_mlp_policy():
    
    # Open the device (since we are only using single devices N150 cards, your mesh shape will be 1x1)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))
    
    # Dimensions based on our custom RL environment
    batch_size = 1
    action_dim = 10 
    hidden_dim = 64
    obs_dim = 64

    # Create torch input
    x = torch.randn(1, obs_dim, dtype=torch.bfloat16)
    # Create torch model
    policy = MLPPolicy(obs_dim, action_dim, hidden_dim)

    # Create TTNN model, we pass the torch model state dict to use its weights
    tt_policy = TTMLPPolicy(policy.state_dict(), mesh_device)

    # Run forward pass
    y = policy(x)
    tt_y = tt_policy(x)

    # Check that the Pearson Correlation Coefficient is above 0.99 (meaning that these 2 tensors are very very close to eachother) to check for correctness
    if check_pcc(y, tt_y):
        print("✅ PCC check passed!")
    else:
        print("❌ PCC below threshold.")

if __name__ == "__main__":
    test_mlp_policy()