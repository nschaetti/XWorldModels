
import torch.nn as nn
import torch


class MLP(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )
    # end def __init__

    def forward(self, x):
        return self.layers(x)
    # end def forward

# end class MLP
