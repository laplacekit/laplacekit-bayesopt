import pytest
import torch
from torch import nn

from laplacekit_bayesopt import PosthocMarglik

torch.set_default_dtype(torch.float64)

FAST = PosthocMarglik(epochs=30, lr=1e-2, batch_size=8)
"""A short training run, so each fit takes well under a second."""


class Nets:
    """A factory of fresh networks that keeps every network it made."""

    def __init__(self, d: int, k: int) -> None:
        self.d, self.k = d, k
        self.made: list[nn.Module] = []

    def __call__(self) -> nn.Module:
        torch.manual_seed(len(self.made))
        net = nn.Sequential(nn.Linear(self.d, 8), nn.Tanh(), nn.Linear(8, self.k))
        self.made.append(net)

        return net


@pytest.fixture
def data1():
    g = torch.Generator().manual_seed(0)
    X = torch.rand(12, 3, generator=g)
    Y = X.sin().sum(-1, keepdim=True) + 0.1 * torch.randn(12, 1, generator=g)

    return X, Y


@pytest.fixture
def data3():
    g = torch.Generator().manual_seed(1)
    X = torch.rand(12, 3, generator=g)
    Y = torch.stack([X.sum(-1), X.prod(-1), X[:, 0]], -1) + 0.1 * torch.randn(12, 3, generator=g)

    return X, Y


def flat_params(post):
    return {k: v.detach() for k, v in post.mean.items()}


def dense_cov(post, net, X_train):
    """Σ = (Σ_n J_nᵀ J_n / σ² + λ I)⁻¹ for an isotropic prior, from torch.func."""

    params = flat_params(post)
    names = list(params)
    sizes = [params[n].numel() for n in names]
    flat = torch.cat([params[n].reshape(-1) for n in names])

    def f(theta, x):
        chunks = torch.split(theta, sizes)
        p = {n: c.reshape(params[n].shape) for n, c in zip(names, chunks, strict=True)}

        return torch.func.functional_call(net, p, (x,))

    J = torch.func.jacrev(f)(flat, X_train).reshape(-1, flat.numel())  # [N K, P]
    sigma = post.likelihood.sigma.reshape(())
    lam = post.prior.tunable()
    precision = J.T @ J / sigma**2 + lam * torch.eye(flat.numel())

    return f, flat, torch.linalg.inv(precision)


def joint_reference(post, net, X_train, Xq):
    """Mean [q K] and joint covariance [q K, q K] of the linearized posterior at Xq."""

    f, flat, cov = dense_cov(post, net, X_train)
    J = torch.func.jacrev(f)(flat, Xq).reshape(-1, flat.numel())

    return f(flat, Xq).reshape(-1), J @ cov @ J.T
