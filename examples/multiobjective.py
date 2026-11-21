"""Find the Pareto front of Branin-Currin with log expected hypervolume improvement.

As in BoTorch's multi-objective tutorial: two noisy objectives, batches of four points per
round, and the hypervolume of the true values of the points found so far.
"""

import time

import torch
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
from torch import nn

from laplacekit_bayesopt import LaplaceBNNSurrogate

torch.set_default_dtype(torch.float64)
torch.manual_seed(10)

ROUNDS = 15
NOISE_SE = torch.tensor([15.19, 0.63])
problem = BraninCurrin(negate=True)
bounds = problem.bounds


def observe(X):
    clean = problem(X)

    return clean, clean + NOISE_SE * torch.randn_like(clean)


def log_hv_gap(Y_clean):
    hv = DominatedPartitioning(ref_point=problem.ref_point, Y=Y_clean).compute_hypervolume()

    return torch.log10(problem.max_hv - hv).item()


def make_net():
    return nn.Sequential(
        nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 2)
    )


train_X = draw_sobol_samples(bounds=bounds, n=6, q=1).squeeze(1)
clean_Y, train_Y = observe(train_X)

start = time.perf_counter()

for i in range(ROUNDS):
    model = LaplaceBNNSurrogate(
        make_net(),
        train_X,
        train_Y,
        input_transform=Normalize(d=2),
        outcome_transform=Standardize(m=2),
    )

    with torch.no_grad():
        predicted = model.posterior(train_X.unsqueeze(-2)).mean.squeeze(-2)

    acq = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point,
        partitioning=FastNondominatedPartitioning(ref_point=problem.ref_point, Y=predicted),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
    )
    x, _ = optimize_acqf(acq, bounds=bounds, q=4, num_restarts=10, raw_samples=20, sequential=True)
    clean, noisy = observe(x)
    train_X, train_Y = torch.cat([train_X, x]), torch.cat([train_Y, noisy])
    clean_Y = torch.cat([clean_Y, clean])

    print(
        f"round {i + 1:2d}: log10 hypervolume gap = {log_hv_gap(clean_Y):6.3f}, "
        f"{time.perf_counter() - start:5.1f} s"
    )
