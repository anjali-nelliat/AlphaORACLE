import torch
from torch.utils.data import Sampler
from torch_geometric.loader import NeighborLoader
from torch_sparse import SparseTensor
from torch_geometric.sampler import NeighborSampler as PYGNeighborSampler

from typing import List, Tuple, NamedTuple, Optional, Union, Dict
from torch import Tensor


class SampleStates(Sampler):
    """A random sampler that ensures instances share the same permutation.

    Instances are passed to PyTorch Geometric `NeighborLoader`. Each instance
    returns an iterable of the class variable `perm`, ensuring each instance
    has the same random ordering. Calling `step` will create a new current
    random permutation. `step` should be called each epoch.
    """

    perm = None  # replaced with a new random permutation on `step` call

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(SampleStates.perm.tolist())

    def __len__(self):
        return len(self.data_source)

    @classmethod
    def step(cls, n_samples=None, random=True):
        if n_samples is None and cls.perm is None:
            raise Exception("`n_samples` must be passed on first call to `step`.")
        elif n_samples is None:
            cls.perm = torch.randperm(len(cls.perm))
        else:
            cls.perm = torch.randperm(n_samples)

        if not random:
            cls.perm = torch.arange(len(cls.perm))
        return cls.perm


class Adj(NamedTuple):
    """Custom adjacency structure that includes weights.

    This is a compatibility layer for code designed to work with the deprecated
    NeighborSampler that now needs to work with newer PyG implementations.
    """
    edge_index: torch.Tensor
    e_id: torch.Tensor
    weights: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(
            self.edge_index.to(*args, **kwargs),
            self.e_id.to(*args, **kwargs),
            self.weights.to(*args, **kwargs),
            self.size,
        )


class NeighborSamplerWithWeights:
    """A compatibility layer for the deprecated NeighborSampler that handles weighted networks.

    This class provides a similar interface to the old NeighborSampler class from PyG
    but uses the newer NeighborLoader internally. It's intended for transitioning code
    that used the old API.
    """

    def __init__(self, data, sizes, **kwargs):
        """Initialize the weighted neighbor sampler.

        Args:
            data: The graph data to sample from
            sizes: The number of neighbors to sample per node per layer
            **kwargs: Additional arguments for the sampler
        """
        # Move data to CPU for sampling
        data = data.to("cpu")

        # Extract edge information - handle both SparseTensor and regular tensor formats
        if hasattr(data, 'adj_t'):
            # Get indices and values from SparseTensor
            row, col, self.weights = data.adj_t.coo()
            edge_index = torch.stack([col, row], dim=0)  # Note: swap col/row because adj_t is transposed
            data.edge_index = edge_index
        elif hasattr(data, 'edge_index'):
            # Handle regular tensor format
            edge_index = data.edge_index
            self.weights = data.edge_weight if hasattr(data, 'edge_weight') else None
        else:
            raise ValueError("Input data must have either adj_t or edge_index attribute")

        # Convert sizes to the format expected by NeighborLoader
        num_neighbors = sizes if isinstance(sizes, list) else [sizes]

        # Remove parameters not supported in older NeighborSampler
        kwargs.pop('subgraph_type', None)  # Remove subgraph_type if present

        # Create the underlying sampler
        self.sampler = PYGNeighborSampler(
            data,
            num_neighbors=num_neighbors,
            replace=kwargs.get('replace', False),
            directed=kwargs.get('directed', True),
        )

        # Create the loader for sampling
        self.loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=kwargs.get('batch_size', 1),
            directed=kwargs.get('directed', True),
            neighbor_sampler=self.sampler,
            shuffle=kwargs.get('shuffle', False),
        )

    def sample(self, batch):
        """Sample a neighborhood around the given batch of nodes.

        Args:
            batch: The batch of nodes to sample around

        Returns:
            batch_size, n_id, adjs: Similar to the return format of the old NeighborSampler
        """
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size = len(batch)

        # Use the loader's collate function to get a subgraph
        subgraph = self.loader.collate_fn(batch)

         # In PyG 2.3, the subgraph is a SamplerOutput object
          # Access node_id through the proper attribute
        if hasattr(subgraph, 'node_id'):
            n_id = subgraph.node_id
        elif hasattr(subgraph, 'node_ids'):
            n_id = subgraph.node_ids
        elif hasattr(subgraph, 'n_id'):
            n_id = subgraph.n_id
        else:
            # Try to find the node IDs in the subgraph object
            for attr_name in dir(subgraph):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(subgraph, attr_name)
                if isinstance(attr, torch.Tensor) and attr.dtype == torch.long and len(attr.shape) == 1:
                    if len(attr) >= batch_size:
                        n_id = attr
                        break
            else:
                raise ValueError("Could not find node IDs in the sampler output")

        # Convert to the old format
        adjs = []
        # Convert to the old format
        adjs = []
        # Try to access edge information
        if hasattr(subgraph, 'edge_index'):
            edge_index = subgraph.edge_index
            e_id = getattr(subgraph, 'e_id', None)
            if e_id is None and hasattr(subgraph, 'edge_id'):
                e_id = subgraph.edge_id

            # Get weights for these edges
            if e_id is not None and self.weights is not None:
                weights = self.weights[e_id]
            else:
                weights = torch.ones(edge_index.size(1), device=edge_index.device)

            # Create the size tuple
            size = (n_id.size(0), batch.size(0))

            # Create the Adj object
            adjs.append(Adj(edge_index, e_id, weights, size))
        else:
            # Fallback approach - create a single adjacency structure
            # This is a simplification and might need to be adjusted based on your model
            dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)
            dummy_e_id = torch.zeros(0, dtype=torch.long)
            dummy_weights = torch.zeros(0)
            size = (n_id.size(0), batch.size(0))
            adjs.append(Adj(dummy_edge_index, dummy_e_id, dummy_weights, size))

        # Return in the format expected by the old API
        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]