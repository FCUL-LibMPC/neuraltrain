import torch.nn as nn


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FeedForwardNN, self).__init__()

        # Check if passed hidden_dims list or tuple is valid
        if not isinstance(hidden_dims, (list, tuple)) or not all(hs > 0 for hs in hidden_dims):
            raise ValueError(f"'hidden_dims' must be a list or tuple of positive integers, got {hidden_dims}.")
        
        if len(hidden_dims) != 2:
            raise ValueError(f"'hidden_dims' must contain exactly two positive integers, got {len(hidden_dims)}.")

        # Unpack hidden layers dimensions
        hidden_dim_1, hidden_dim_2 = hidden_dims

        # Initialize the feedforward neural network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim),
        )

    def forward(self, x):
        return self.net(x)
