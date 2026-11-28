"""Pure-PyTorch stand-in for the torch_scatter functions DPVO and DROID-SLAM use.

Installed only when no prebuilt torch-scatter wheel matches the runtime's
torch/CUDA (e.g. Colab torch 2.14 + CUDA 13). Semantics follow torch_scatter:
a 1-D index is broadcast along `dim`; empty groups yield 0.
"""

import torch

__version__ = "0.0.0+shim"


def _broadcast(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim % src.dim()
    if index.dim() == 1:
        index = index.view([1] * dim + [-1] + [1] * (src.dim() - dim - 1))
    else:
        while index.dim() < src.dim():
            index = index.unsqueeze(-1)
    return index.expand_as(src)


def _out_shape(src, index, dim, dim_size):
    size = list(src.shape)
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    size[dim % src.dim()] = dim_size
    return size


def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    index = _broadcast(index, src, dim)
    if out is None:
        out = src.new_zeros(_out_shape(src, index, dim, dim_size))
    return out.scatter_add_(dim, index, src)


scatter_add = scatter_sum


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    """Mean per group; empty groups yield 0 (count clamped to 1, as torch_scatter does)."""
    total = scatter_sum(src, index, dim, out, dim_size)
    count = scatter_sum(torch.ones_like(src), index, dim, dim_size=total.shape[dim % src.dim()])
    return total / count.clamp(min=1)


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    """Returns (max, None). DPVO only reads the values; argmax is not provided."""
    index = _broadcast(index, src, dim)
    if out is None:
        out = src.new_zeros(_out_shape(src, index, dim, dim_size))
    return out.scatter_reduce_(dim, index, src, reduce="amax", include_self=False), None


def scatter_softmax(src, index, dim=-1, eps=1e-12, dim_size=None):
    index = _broadcast(index, src, dim)
    group_max = scatter_max(src, index, dim, dim_size=dim_size)[0]
    exp = (src - group_max.gather(dim, index)).exp()
    group_sum = scatter_sum(exp, index, dim, dim_size=dim_size)
    return exp / (group_sum + eps).gather(dim, index)
