from dataclasses import dataclass
from typing import Optional

import torch as t
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Config:
    n_features: int
    n_hidden: int

    # We optimize n_instances models in a single training loop
    # to let us sweep over sparsity or importance curves
    # efficiently.

    # We could potentially use t.vmap instead.
    n_instances: int


class Model(nn.Module):
    def __init__(
        self,
        config: Config,
        feature_probability: Optional[t.Tensor] = None,
        importance: Optional[t.Tensor] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.config = config
        self.W = nn.Parameter(
            t.empty((config.n_instances, config.n_features, config.n_hidden), device=device)
        )
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(
            t.zeros((config.n_instances, config.n_features), device=device)
        )

        if feature_probability is None:
            feature_probability = t.ones(())
        self.feature_probability = feature_probability.to(device)
        if importance is None:
            importance = t.ones(())
        self.importance = importance.to(device)

    def forward(self, features: t.Tensor) -> t.Tensor:
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        hidden = t.einsum("...if,ifh->...ih", features, self.W)
        out = t.einsum("...ih,ifh->...if", hidden, self.W)
        out = out + self.b_final
        out = F.relu(out)
        return out

    def generate_batch(self, n_batch: int) -> t.Tensor:
        feat = t.rand(
            (n_batch, self.config.n_instances, self.config.n_features), device=self.W.device
        )
        batch = t.where(
            t.rand(
                (n_batch, self.config.n_instances, self.config.n_features),
                device=self.W.device,
            )
            <= self.feature_probability,
            feat,
            t.zeros((), device=self.W.device),
        )
        return batch
