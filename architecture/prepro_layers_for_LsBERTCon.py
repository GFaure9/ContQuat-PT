"""
Definition of the SBERTLossPreproLayers class.
Apply temporal pooling and matrix multiplication to go from
tensors of shape (N_batch, T, hidden_size) to tensors of shape (N_batch, output_dim).
In our case, output_dim will typically be of the size of SBERT embeddings dimension.
"""

import torch
import torch.nn as nn


class SBERTLossPreproLayers(nn.Module):
    """
    Class to apply temporal pooling and a linear layer with no bias to get the
    embeddings of the decoder's self-attention layers to the dimension of the SBERT embeddings
    before computing the SBERT similarity-based supervised contrastive loss.
    See g(.) function in https://arxiv.org/pdf/2508.14574
    """
    def __init__(self, input_dim: int, output_dim: int, temporal_pooling: str = "average"):
        super(SBERTLossPreproLayers, self).__init__()
        self.temporal_pooling = temporal_pooling
        self.linear = nn.Linear(input_dim, output_dim, bias=False)  # FFN with no bias: FFN(x) = W.x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # since `x` is supposed to be self-attention PT decoder output, its shape will be (N_batch, T, hidden_size)
        # typically it will be (64, T, 512)...

        if self.temporal_pooling == "average":
            x_pooled = x.mean(dim=1)  # averaging over temporal dimension
        elif self.temporal_pooling == "active_times_weighting":
            weights = torch.softmax(x.sum(dim=-1), dim=1)  # (N_batch, T) | w_t = exp(sum(feats_t)) / sum(exp(sum(feats_k))
            x_pooled = (weights.unsqueeze(-1) * x).sum(dim=1)  # larger weights given to "active" times
        else:
            raise ValueError(f"'{self.temporal_pooling}' is not a valid pooling option")

        return self.linear(x_pooled)