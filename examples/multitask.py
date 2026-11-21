"""Maximize a composite of three outcomes with Monte Carlo log expected improvement.

The objective is a function of all three outcomes of a contextual Hartmann problem, as in
BoTorch's tutorial on composite multi-task Bayesian optimization. The surrogate models the
three outcomes jointly, and the acquisition function draws joint samples of them.
"""

import time

import torch
from botorch.acquisition import GenericMCObjective, qLogExpectedImprovement
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions import Hartmann
from torch import nn

from laplacekit_bayesopt import LaplaceBNNSurrogate

torch.set_default_dtype(torch.float64)
torch.manual_seed(10)

ROUNDS = 15
TASKS = 3
hartmann = Hartmann(dim=6, negate=True)
contexts = torch.linspace(0, 1, TASKS)
bounds = torch.stack([torch.zeros(5), torch.ones(5)])
weights = torch.randn(TASKS)


def problem(X):
    """Hartmann at the 5 inputs plus each context, one outcome per context."""

    full = torch.cat(
        [
            X.unsqueeze(-2).expand(*X.shape[:-1], TASKS, 5),
            contexts[:, None].expand(*X.shape[:-1], TASKS, 1),
        ],
        -1,
    )

    return hartmann(full) + 0.001 * torch.randn(*X.shape[:-1], TASKS)


def composite(samples, X=None):
    return -torch.cos(samples**2 + samples * weights).sum(-1)


objective = GenericMCObjective(composite)
train_X = torch.rand(5, 5)
train_Y = problem(train_X)


def make_net():
    return nn.Sequential(
        nn.Linear(5, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, TASKS)
    )


start = time.perf_counter()

for i in range(ROUNDS):
    model = LaplaceBNNSurrogate(
        make_net(), train_X, train_Y, outcome_transform=Standardize(m=TASKS)
    )
    acq = qLogExpectedImprovement(
        model,
        best_f=objective(train_Y).max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])),
        objective=objective,
    )
    x, _ = optimize_acqf(acq, bounds=bounds, q=4, num_restarts=10, raw_samples=64)
    train_X, train_Y = torch.cat([train_X, x]), torch.cat([train_Y, problem(x)])

    print(
        f"round {i + 1:2d}: best objective = {objective(train_Y).max().item():6.3f}, "
        f"{time.perf_counter() - start:5.1f} s"
    )
