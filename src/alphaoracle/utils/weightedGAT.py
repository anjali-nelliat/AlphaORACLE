import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_scatter import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import GATConv

from typing import Optional, Tuple, Union
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_sparse import SparseTensor, set_diag


def weighted_softmax(
        src: Tensor,
        index: Tensor,
        edge_weights: Tensor,
        ptr: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
) -> Tensor:
    """Extends the PyTorch Geometric `softmax` functionality to incorporate edge weights.

    See the PyTorch Geomtric `softmax` documentation for details on arguments.
    """

    if ptr is None:
        N = maybe_num_nodes(index, num_nodes)
        out = src - scatter(src, index, dim=0, dim_size=N, reduce="max")[index]
        out = edge_weights.unsqueeze(-1) * out.exp()  # multiply softmax by `edge_weights`
        out_sum = scatter(out, index, dim=0, dim_size=N, reduce="sum")[index]
        return out / out_sum.clamp(min=1e-16)  # Updated for numerical stability
    else:
        raise NotImplementedError("Using `ptr` with `weighted_softmax` has not been implemented.")


class WeightedGATConv(GATConv):
    """Weighted version of the Graph Attention Network (`GATConv`)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store references to the linear layers with both possible naming conventions
        if hasattr(self, 'lin_l'):
            self._lin_l = self.lin_l
            self._lin_r = self.lin_r
        elif hasattr(self, 'lin_src'):
            self._lin_l = self.lin_src
            self._lin_r = self.lin_dst
        else:
            raise AttributeError("Could not find linear layers in GATConv")

        # Similarly for attention parameters
        if hasattr(self, 'att_l'):
            self._att_l = self.att_l
            self._att_r = self.att_r
        elif hasattr(self, 'att_src'):
            self._att_l = self.att_src
            self._att_r = self.att_dst
        else:
            raise AttributeError("Could not find attention parameters in GATConv")

    def forward(
            self,
            x,
            edge_index,
            edge_attr: Optional[Tensor] = None,
            size=None,
            return_attention_weights=None,
    ):
        """Adapted forward method with compatibility for PyG 2.3."""
        # Convert edge_attr to edge_weights for backward compatibility
        edge_weights = edge_attr

        H, C = self.heads, self.out_channels
        x_l = None
        x_r = None
        alpha_l = None
        alpha_r = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in `GATConv`."
            x_l = x_r = self._lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self._att_l).sum(dim=-1)
            alpha_r = (x_r * self._att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, "Static graphs not supported in `GATConv`."
            x_l = self._lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self._att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self._lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self._att_r).sum(dim=-1)
        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, edge_weights = remove_self_loops(edge_index, edge_weights)
                edge_index, edge_weights = add_self_loops(
                    edge_index, edge_weights, fill_value='mean', num_nodes=num_nodes
                )
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # Store edge_weights for use in message
        self.edge_weights = edge_weights

        # Use propagate for message passing
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)

        # Process output
        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        # Handle attention weights if requested
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(
            self,
            x_j: Tensor,
            alpha_j: Tensor,
            alpha_i: OptTensor,
            index: Tensor,
            ptr: OptTensor,
            size_i: Optional[int],
    ) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = weighted_softmax(alpha, index, self.edge_weights, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class Interp(nn.Module):
    """Stochastic summation layer.

    Performs random node feature dropout and feature scaling.
    """

    def __init__(self, n_modalities: int):
        super(Interp, self).__init__()

        self.temperature = (
            1.0  # can modify this to change the relative magnitudes of network scales
        )

        self.net_scales = nn.Parameter(
            (torch.FloatTensor([1.0 for _ in range(n_modalities)]) / n_modalities).reshape((1, -1))
        )

    def forward(
            self, mask: Tensor, idxs: Tensor, evaluate: bool = False, device=None
    ) -> Tuple[Tensor, Tensor]:

        net_scales = F.softmax(self.net_scales / self.temperature, dim=-1)
        net_scales = net_scales[:, idxs]

        # Updated random mask generation
        if evaluate:
            random_mask = torch.randint(1, 2, mask.shape, dtype=torch.int32).float()
        else:
            random_mask = torch.randint(0, 2, mask.shape, dtype=torch.int32).float()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        random_mask = random_mask.to(device)

        mask_sum = 1 / (1 + torch.sum(random_mask, dim=-1)).pow(20)
        random_mask += mask_sum.reshape((-1, 1))
        random_mask += 1 / (torch.sum(mask, dim=-1).pow(20)).reshape((-1, 1))
        random_mask = random_mask.int().float()
        random_mask = random_mask / random_mask.clamp(min=1e-10)  # Using clamp for stability

        mask = mask * random_mask
        mask = F.softmax(mask + ((1 - mask) * -1e10), dim=-1)

        return net_scales, mask


def masked_scaled_mse(
        output: Tensor,
        target: Tensor,
        weight: Tensor,
        node_ids: Tensor,
        mask: Tensor,
        device=None,
):
    """Masked and scaled MSE loss.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Subset `target` to current batch and make dense
    target = target.to(device)
    target = target.adj_t[node_ids, node_ids].to_dense()

    loss = weight * torch.mean(mask.reshape((-1, 1)) * (output - target) ** 2 * mask)

    return loss
