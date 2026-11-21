"""Minimize Branin with log expected improvement."""

import time

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin
from torch import nn

from laplacekit_bayesopt import LaplaceBNNSurrogate

torch.set_default_dtype(torch.float64)
torch.manual_seed(1)

ROUNDS = 30
problem = Branin(negate=True)  # BoTorch maximizes, so maximize -Branin
bounds = problem.bounds

train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(20, 2)
train_Y = problem(train_X).unsqueeze(-1)


def make_net():
    return nn.Sequential(
        nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 1)
    )


start = time.perf_counter()

for i in range(ROUNDS):
    model = LaplaceBNNSurrogate(
        make_net(),
        train_X,
        train_Y,
        input_transform=Normalize(d=2),
        outcome_transform=Standardize(m=1),
    )
    acq = LogExpectedImprovement(model, best_f=train_Y.max())
    x, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=20)
    y = problem(x).unsqueeze(-1)
    train_X, train_Y = torch.cat([train_X, x]), torch.cat([train_Y, y])

    print(
        f"round {i + 1:2d}: f(x) = {-y.item():8.3f}, best = {-train_Y.max().item():8.3f} "
        f"(global min 0.398), {time.perf_counter() - start:5.1f} s"
    )
