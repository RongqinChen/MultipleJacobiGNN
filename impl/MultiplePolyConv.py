import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
import numpy as np


def buildAdj(edge_index: Tensor, edge_weight: Tensor, n_node: int, aggr: str):
    """
    convert edge_index and edge_weight to the sparse adjacency matrix.
    Args:
        edge_index (Tensor): shape (2, number of edges).
        edge_attr (Tensor): shape (number of edges).
        n_node (int): number of nodes in the graph.
        aggr (str): how adjacency matrix is normalized. choice: ["mean", "sum", "gcn"]
    """
    deg = degree(edge_index[0], n_node)
    deg[deg < 0.5] += 1.0
    ret = None
    if aggr == "mean":
        val = (1.0 / deg)[edge_index[0]] * edge_weight
    elif aggr == "sum":
        val = edge_weight
    elif aggr == "gcn":
        deg = torch.pow(deg, -0.5)
        val = deg[edge_index[0]] * edge_weight * deg[edge_index[1]]
    else:
        raise NotImplementedError
    ret = SparseTensor(
        row=edge_index[0], col=edge_index[1], value=val, sparse_sizes=(n_node, n_node)
    ).coalesce()
    ret = ret.cuda() if edge_index.is_cuda else ret
    return ret


def MultipleJacobiConv(k, adj, xs_list, alphas_list, a_list, b_list):

    if k == 0:
        return xs_list[0]

    xs = xs_list[k - 1]
    alphas = alphas_list[k - 1]

    if k == 1:
        new_xs = [
            alpha * (a - b) / 2 * x + alpha * (a + b + 2) / 2 * (adj @ x)
            for x, alpha, a, b in zip(xs, alphas, a_list, b_list)
        ]
        return new_xs

    pre_xs = xs_list[k - 2]
    pre_alphas = alphas_list[k - 2]
    new_xs = []
    for x, pre_x, alpha, pre_alpha, a, b in zip(
        xs, pre_xs, alphas, pre_alphas, a_list, b_list
    ):
        coef_nu1 = (2 * k + a + b + 1) * (2 * k + a + b + 2)
        coef_de1 = 2 * (k + 1) * (k + a + b + 1)
        coef_nu2 = (a**2 - b**2) * (2 * k + a + b + 1)
        coef_de2 = 2 * (k + 1) * (k + a + b + 1) * (2 * k + a + b)
        coef_nu3 = (k + a) * (k + b) * (2 * k + a + b + 2)
        coef_de3 = (k + 1) * (k + a + b + 1) * (2 * k + a + b)
        coef1 = alpha * coef_nu1 / coef_de1
        coef2 = alpha * coef_nu2 / coef_de2
        coef3 = alpha * pre_alpha * coef_nu3 / coef_de3
        new_x = coef1 * (adj @ x) + coef2 * x - coef3 * pre_x
        new_xs.append(new_x)

    return new_xs


class MultiplePolyConvFrame(nn.Module):
    """
    A framework for polynomial graph signal filter.
    Args:
        conv_fn: the filter function, like PowerConv, LegendreConv,...
        depth (int): the order of polynomial.
        cached (bool): whether or not to cache the adjacency matrix.
        alpha (float):  the parameter to initialize polynomial coefficients.
        fixed_w (bool): whether or not to fix to weight function coefficients.
    """

    def __init__(
        self,
        conv_fn,
        depth: int = 3,
        ab_tuple_list=None,
        aggr: int = "gcn",
        cached: bool = True,
        alpha: float = 1.0,
        fixed_w: float = True,
    ):
        super().__init__()
        self.depth = depth

        if ab_tuple_list is None:
            step = 0.5
            ab_tuple_list = [
                (0.0, b) for b in np.arange(0, 3, step)
            ] + [
                (a, 0.0) for a in np.arange(0 + step, 3, step)
            ]

        self.ab_tuple_list = ab_tuple_list
        self.a_list = nn.ParameterList([
            nn.Parameter(torch.tensor(ab_tuple[0]), requires_grad=not fixed_w)
            for ab_tuple in ab_tuple_list
        ])
        self.b_list = nn.ParameterList([
            nn.Parameter(torch.tensor(ab_tuple[1]), requires_grad=not fixed_w)
            for ab_tuple in ab_tuple_list
        ])
        self.num_ab_tuples = len(ab_tuple_list)

        self.base_alpha = float(min(1 / alpha, 1))
        self.alphas_dict = nn.ParameterDict(
            {
                f"{k}_{m}": nn.Parameter(torch.tensor(self.base_alpha))
                for k in range(depth)
                for m in range(self.num_ab_tuples)
            }
        )

        self.cached = cached
        self.aggr = aggr
        self.adj = None
        self.conv_fn = conv_fn
        self.w = nn.Parameter(torch.ones((1, len(self.ab_tuple_list), 1, 1)))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        """
        Args:
            x: node embeddings. of shape (number of nodes, node feature dimension)
            edge_index and edge_attr: If the adjacency is cached, they will be ignored.
        """
        if self.adj is None or not self.cached:
            n_node = x.shape[0]
            self.adj = buildAdj(edge_index, edge_attr, n_node, self.aggr)

        alphas_list = [
            [self.alphas_dict[f"{k}_{m}"] for m in range(self.num_ab_tuples)]
            for k in range(self.depth)
        ]
        xs = [x] * self.num_ab_tuples
        xs_list = [xs]

        for k in range(1, self.depth + 1):
            xs = self.conv_fn(k, self.adj, xs_list, alphas_list, self.a_list, self.b_list)
            xs_list.append(xs)

        for idx in range(len(xs_list)):
            xs_list[idx] = torch.stack(xs_list[idx], 1)

        h = torch.stack(xs_list, dim=2)
        wh = self.w * h
        sh = wh.sum(1)
        return sh
