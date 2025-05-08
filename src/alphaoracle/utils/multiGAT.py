#@title Multiview Model
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor

from typing import Dict, List, Tuple, Optional, Union

from .weightedGAT import WeightedGATConv, Interp


class Adj:
    """Compatibility class for the Adj tuple from older PyG versions."""

    def __init__(self, edge_index: torch.Tensor, e_id: torch.Tensor, weights: torch.Tensor, size: Tuple[int, int]):
        self.edge_index = edge_index
        self.e_id = e_id
        self.weights = weights
        self.size = size

    def to(self, device):
        return Adj(
            self.edge_index.to(device),
            self.e_id.to(device),
            self.weights.to(device),
            self.size,
        )

    def __iter__(self):
        return iter((self.edge_index, self.e_id, self.weights, self.size))


class GATEncoder(nn.Module):
    def __init__(self, in_size: int, gat_shapes: Dict[str, int], alpha: float = 0.1):
        """Network encoder module.

        Args:
            in_size (int): Number of nodes in input networks.
            gat_shapes (Dict[str, int]): Graph attention layer hyperparameters.
            alpha (float, optional): LeakyReLU negative slope. Defaults to 0.1.

        Returns:
            Tensor: 2D tensor of node features. Each row is a node, each column is a feature.
        """
        super(GATEncoder, self).__init__()
        self.in_size = in_size
        self.dimension: int = gat_shapes["dimension"]
        self.n_heads: int = gat_shapes["n_heads"]
        self.n_layers: int = gat_shapes["n_layers"]
        self.alpha = alpha

        self.pre_gat = nn.Linear(self.in_size, self.dimension * self.n_heads)
        self.gat = WeightedGATConv(
            (self.dimension * self.n_heads,) * 2,
            self.dimension,
            heads=self.n_heads,
            dropout=0,
            negative_slope=self.alpha,
            add_self_loops=True,
        )

    def forward(self, data_flow, device=None):
        _, n_id, adjs = data_flow

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Handle both list and single adjacency formats
        if isinstance(adjs, list):
            adjs = [adj.to(device) for adj in adjs]
        else:
            adjs = [adjs.to(device)]

        x_store_layer = []

        # Iterate over flow (pass data through GAT)
        for j, adj in enumerate(adjs):
            # Unpack the Adj object - compatible with both NamedTuple and custom class
            if hasattr(adj, '_fields'):  # NamedTuple
                edge_index, e_id, weights, size = adj
            else:  # Custom Adj class or iterable
                edge_index, e_id, weights, size = adj

            # Initial `x` is feature matrix
            if j == 0:
                x = torch.t(self.pre_gat.weight)[n_id] + self.pre_gat.bias

            if j != 0:
                x_store_layer = [x_s[: size[1]] for x_s in x_store_layer]
                x_pre = x[: size[1]]
                x_store_layer.append(x_pre)

            # Updated to match WeightedGATConv's API in PyG 2.3
            x = self.gat(
                (x, x[: size[1]]),
                edge_index,
                edge_attr=weights,  # Use edge_attr instead of edge_weights
                size=size
            )

        x = sum(x_store_layer) + x  # Compute tensor with residuals

        return x


class MultiViewGAT(nn.Module):
    def __init__(
            self,
            in_size: int,
            gat_shapes: Dict[str, int],
            emb_size: int,
            n_modalities: int,
            alpha: float = 0.1
    ):
        """The MultiViewGAT model.

        Args:
            in_size (int): Number of nodes in input networks.
            gat_shapes (Dict[str, int]): Graph attention layer hyperparameters.
            emb_size (int): Dimension of learned node features.
            n_modalities (int): Number of input networks.
            alpha (float, optional): LeakyReLU negative slope. Defaults to 0.1.
        """

        super(MultiViewGAT, self).__init__()

        self.in_size = in_size
        self.emb_size = emb_size
        self.alpha = alpha
        self.n_modalities = n_modalities
        self.gat_shapes = gat_shapes

        self.dimension: int = self.gat_shapes["dimension"]
        self.n_heads: int = self.gat_shapes["n_heads"]

        self.encoders = nn.ModuleList()

        # GAT
        # Create an encoder for each modality
        for i in range(self.n_modalities):
            self.encoders.append(GATEncoder(self.in_size, self.gat_shapes, self.alpha))

        self.integration_size = self.dimension * self.n_heads
        self.interp = Interp(self.n_modalities)

        # Embedding
        self.emb = nn.Linear(self.integration_size, self.emb_size)

    def forward(
            self,
            data_flows: List[Tuple[int, Tensor, Union[List[Adj], Adj]]],
            masks: Tensor,
            evaluate: bool = False,
            device=None
    ):
        """Forward pass logic.

        Args:
            data_flows (List[Tuple[int, Tensor, Union[List[Adj], Adj]]]): Sampled bi-partite data flows.
                See PyTorch Geometric documentation for more details.
            masks (Tensor): 2D masks indicating which nodes (rows) are in which networks (columns)
            evaluate (bool, optional): Used to turn off random sampling in forward pass.
                Defaults to False.
            device (torch.device, optional): Device to run the model on. Defaults to None.

        Returns:
            Tensor: 2D tensor of final reconstruction to be used in loss function.
            Tensor: 2D tensor of integrated node features. Each row is a node, each column is a feature.
            Tensor: Learned network scaling coefficients.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        idxs = list(range(self.n_modalities))

        net_scales, interp_masks = self.interp(masks, idxs, evaluate, device=device)

        # Define encoder logic.
        batch_size = data_flows[0][0]
        x_store_modality = torch.zeros(
            (batch_size, self.integration_size), device=device
        )  # Tensor to store results from each modality.

        # Iterate over input networks
        for i, data_flow in enumerate(data_flows):
            net_idx = idxs[i]

            x = self.encoders[net_idx](data_flow, device=device)
            x = net_scales[:, i] * interp_masks[:, i].reshape((-1, 1)) * x
            x_store_modality += x

        # Embedding
        emb = self.emb(x_store_modality)

        # Dot product (network reconstruction)
        dot = torch.mm(emb, torch.t(emb))

        return dot, emb, net_scales