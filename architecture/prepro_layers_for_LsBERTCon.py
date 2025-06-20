import torch
import torch.nn as nn


class SBERTLossPreproLayers(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, temporal_pooling: str = "average"):
        super(SBERTLossPreproLayers, self).__init__()
        self.temporal_pooling = temporal_pooling
        self.linear = nn.Linear(input_dim, output_dim, bias=False)  # FFN with no bias: FFN(x) = W.x

    def forward(self, x: torch.Tensor):
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