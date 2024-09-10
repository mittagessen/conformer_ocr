import torch
from torch import nn


class PromptEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        """
        Encoder for quadratic Bézier curve prompts for decoder.

        Args:
            embed_dim: The prompts' embedding dimension. Needs to be divisible
            by 8.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, embed_dim // 8)))

    def forward(self, curves: torch.Tensor) -> torch.Tensor:
        """
        Embeds a quadratic Bézier curve.

        Args:
          curves: point coordinates of shape (B, 4, 2)

        Returns:
          Embeddings for the points with shape (B, E)
        """
        bs = curves.shape[0]
        curve_embeddings = torch.empty((bs, 0, self.embed_dim), device=curves.device)

        coords = curves.clone()
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1).view(bs, -1)
