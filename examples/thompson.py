"""Minimize Branin over a fixed set of Sobol candidates with Thompson sampling.

Each round draws one function from the posterior with BoTorch's
`PathwiseThompsonSampling` and evaluates the candidate where that function is largest.
"""

import time

import torch
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf_discrete
from botorch.test_functions import Branin
from botorch.utils.sampling import draw_sobol_samples
from torch import nn

from laplacekit_bayesopt import LaplaceBNNSurrogate

torch.set_default_dtype(torch.float64)
torch.manual_seed(2)

ROUNDS = 30
problem = Branin(negate=True)
bounds = problem.bounds

# A scrambled Sobol set covers the box more evenly than uniform draws, as in BoTorch's
# Thompson sampling tutorials. A power of two keeps its balance properties.
candidates = draw_sobol_samples(bounds=bounds, n=2**13, q=1, seed=0).squeeze(1)
train_X = candidates[:20]
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
    acq = PathwiseThompsonSampling(model)
    x, _ = optimize_acqf_discrete(acq, q=1, choices=candidates, max_batch_size=2048)
    y = problem(x).unsqueeze(-1)
    train_X, train_Y = torch.cat([train_X, x]), torch.cat([train_Y, y])

    print(
        f"round {i + 1:2d}: f(x) = {-y.item():8.3f}, best = {-train_Y.max().item():8.3f} "
        f"(global min 0.398), {time.perf_counter() - start:5.1f} s"
    )
