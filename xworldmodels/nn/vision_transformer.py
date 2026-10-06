
import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    """
    A Vision Transformer
    """

    def __init__(
            self,
            hidden_dim=768,
            patch_kernel_size=14,
            num_heads=12,
            num_layers=12,
    ):
        super().__init__()

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_kernel_size,
            stride=14
        )

        # CLS token (1, 1, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 16*16+1, hidden_dim))
    # end def __init__

    def __call__(self, x):
        """
        Forward pass
        """
        # B, C, H, W => B, D, H/14, W/14
        x = self.patch_embed(x)

        # B, D, H/14, W/14 => B, H/14, W/14, D
        x = x.permute(0, 2, 3, 1)

        # B, H/14, W/14, D => B, H/14*W/14, D
        x = torch.flatten(x, start_dim=1, end_dim=2)

        # Concatenate CLS token
        # B, H/14*W/14, D => B, H/14*W/14+1, D
        x = torch.cat((self.cls_token, x), dim=1)

        # Add position embeddings
        # B, H/14*W/14+1, D => B, H/14*W/14+1, D (no change)
        x = x + self.pos_embed

        return x
    # end def __call__

# end class VisionTransformer
