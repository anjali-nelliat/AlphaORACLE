#@title Preprocess Data
import json
import typer
import torch
import numpy as np
import pandas as pd

from typing import List, Optional, Union, Tuple
from pathlib import Path
from functools import reduce

from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataPreprocess:
    def __init__(
            self,
            input_names: List[Path],
            delimiter: str = " ",
    ):
        """Preprocesses input networks.

        Args:
            input_names (List[Path]): Paths to input networks.
            delimiter (str, optional): Delimiter used in input network files. Defaults to " ".
        """

        self.input_names = input_names
        self.graphs = self.data_load(delimiter)
        self.node_sets, self.union = self.get_union()

    def data_load(self, delimiter):

        # Import networks
        graphs = [pd.read_csv(name, delimiter=delimiter, header=None) for name in self.input_names]

        # Add weights of 1.0 if weights are missing
        for G in graphs:
            if G.shape[1] < 3:
                G[2] = pd.Series([1.0] * len(G))
        return graphs

    def get_union(self):

        node_sets = [np.union1d(G[0].values, G[1].values) for G in self.graphs]
        union = reduce(np.union1d, node_sets)
        return node_sets, union

    def create_masks(self):

        masks = torch.FloatTensor([np.isin(self.union, nodes) for nodes in self.node_sets])
        masks = torch.t(masks)
        masks = masks.to(device)
        return masks

    def create_weights(self):

        weights = torch.FloatTensor([1.0 for G in self.graphs])
        weights = weights.to(device)
        return weights

    def create_graphs(self):
        """Create graph representations using SparseTensor from torch_sparse."""

        typer.echo("Preprocessing input networks...")

        # Import required libraries
        from torch_sparse import SparseTensor

        # Uniquely map node names to integers
        mapper = {name: idx for idx, name in enumerate(self.union)}

        # Transform networks to PyG graphs
        pyg_graphs = []
        for G in self.graphs:
            # Map node names to integers given by `mapper`
            G[[0, 1]] = G[[0, 1]].applymap(lambda node: mapper[node])

            # Extract weights and edges from `G` and convert to tensors
            weights = torch.FloatTensor(G[2].values)
            edge_index = torch.LongTensor(G[[0, 1]].values.T)

            # Remove existing self loops and add self loops from `union` nodes
            edge_index, weights = remove_self_loops(edge_index, edge_attr=weights)
            edge_index, weights = to_undirected(edge_index, edge_attr=weights)

            # Add self-loops explicitly
            union_idxs = list(range(len(self.union)))
            self_loops = torch.LongTensor([union_idxs, union_idxs])
            edge_index = torch.cat([edge_index, self_loops], dim=1)

            # Make sure self-loop weights are properly created
            self_loop_weights = torch.ones(len(self.union), dtype=torch.float)
            weights = torch.cat([weights, self_loop_weights])

            # Create SparseTensor directly instead of using ToSparseTensor transform
            row, col = edge_index
            N = len(self.union)  # Number of nodes

            # Create SparseTensor explicitly
            adj_t = SparseTensor(
                row=col,           # Note the swap of row/col for transposed adj matrix
                col=row,           # This creates adj_t directly (transposed adjacency)
                value=weights,
                sparse_sizes=(N, N)
            )

            # Create PyG Data object with the SparseTensor
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pyg_graph = Data()
            pyg_graph.adj_t = adj_t
            pyg_graph.num_nodes = N
            pyg_graph = pyg_graph.to(self.device)

            pyg_graphs.append(pyg_graph)

        typer.echo(f"Preprocessing finished")

        return pyg_graphs

    def process(self):
        """Calls preprocessing functions.

        Returns:
            np.ndarray: Array of all nodes present across input networks (union of nodes).
            Tensor: 2D binary mask tensor indicating nodes (rows) present in each network (columns).
            Tensor: 1D network weight tensor.
            List[SparseTensor]: Processed networks in Pytorch Geometric `SparseTensor` format.
        """

        masks: Tensor = self.create_masks()
        weights: Tensor = self.create_weights()
        pyg_graphs: List[Union[SparseTensor, Data]] = self.create_graphs()

        typer.echo(f"Preprocessing finished")

        return self.union, masks, weights, pyg_graphs