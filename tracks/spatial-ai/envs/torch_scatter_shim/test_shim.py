"""Check the shim against explicit per-group loops, with DPVO's and DROID-SLAM's call shapes.

    python tracks/spatial-ai/envs/torch_scatter_shim/test_shim.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from torch_scatter import scatter_max, scatter_mean, scatter_softmax, scatter_sum


def loop_reduce(src, index, n, fn):
    """Reference over dim 1, index 1-D, empty groups -> 0."""
    out = src.new_zeros(src.shape[0], n, *src.shape[2:])
    for g in range(n):
        sel = index == g
        if sel.any():
            out[:, g] = fn(src[:, sel])
    return out


def main():
    torch.manual_seed(0)
    E, n = 50, 13  # some groups stay empty
    index = torch.randint(0, n - 2, (E,))

    # ba.py: scatter_sum(A[:, v], ii*m + jj, dim=1, dim_size=n*m) on (1, E, 6, 6)
    A = torch.randn(1, E, 6, 6, dtype=torch.float64)
    got = scatter_sum(A, index, dim=1, dim_size=n)
    assert torch.allclose(got, loop_reduce(A, index, n, lambda x: x.sum(1))), "sum"

    # blocks.py: softmax + sum over dim 1 on (1, E, D), index (E,)
    x = torch.randn(1, E, 8, dtype=torch.float64) * 5
    w = scatter_softmax(x, index, dim=1)
    ref = torch.empty_like(x)
    for g in index.unique():
        sel = index == g
        ref[:, sel] = torch.softmax(x[:, sel], dim=1)
    assert torch.allclose(w, ref), "softmax"
    assert torch.allclose(scatter_sum(w, index, dim=1)[:, index.unique()], torch.ones(1, len(index.unique()), 8, dtype=torch.float64))

    # long_term.py: scatter_max(residual, kk)[0], 1-D, default dim=-1
    r = torch.randn(E, dtype=torch.float64)
    got = scatter_max(r, index)[0]
    ref = torch.zeros(int(index.max()) + 1, dtype=torch.float64)
    for g in range(len(ref)):
        if (index == g).any():
            ref[g] = r[index == g].max()
    assert torch.allclose(got, ref), "max"

    # DROID droid_net.py: scatter_mean(net, ix, dim=1) on (1, E, C, H, W), index (E,)
    net = torch.randn(1, E, 4, 3, 5, dtype=torch.float64)
    got = scatter_mean(net, index, dim=1)
    assert got.shape[1] == int(index.max()) + 1, "mean shape"
    assert torch.allclose(got, loop_reduce(net, index, got.shape[1], lambda x: x.mean(1))), "mean"

    # half precision under autocast-like inputs
    xh = x.half()
    assert torch.isfinite(scatter_softmax(xh, index, dim=1)).all(), "half softmax"
    print("torch_scatter shim OK")


if __name__ == "__main__":
    main()
