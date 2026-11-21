"""BoTorch's own acquisition functions and optimizers on the surrogate."""

import torch
from botorch.acquisition import (
    LogExpectedImprovement,
    qKnowledgeGradient,
    qLogNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.utils.sampling import draw_sobol_samples

from laplacekit_bayesopt import LaplaceBNNSurrogate

from .conftest import FAST, Nets

BOUNDS = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def surrogate(data):
    X, Y = data

    # A new transform per model, as in BoTorch's tutorials.

    return LaplaceBNNSurrogate(
        Nets(3, 1)(),
        X,
        Y,
        trainer=FAST,
        input_transform=Normalize(d=3),
        outcome_transform=Standardize(m=1),
    )


def inside(x):
    return bool(((x >= 0) & (x <= 1)).all())


def test_log_ei(data1):
    model = surrogate(data1)
    acq = LogExpectedImprovement(model, best_f=data1[1].max())
    x, val = optimize_acqf(acq, BOUNDS, q=1, num_restarts=2, raw_samples=16)

    assert x.shape == (1, 3) and inside(x) and torch.isfinite(val)


def test_q_log_nei_and_ucb(data1):
    model = surrogate(data1)

    for acq in (
        qLogNoisyExpectedImprovement(model, X_baseline=data1[0], prune_baseline=False),
        qUpperConfidenceBound(model, beta=2.0),
    ):
        x, val = optimize_acqf(acq, BOUNDS, q=2, num_restarts=2, raw_samples=16)
        assert x.shape == (2, 3) and inside(x) and torch.isfinite(val).all()


def test_knowledge_gradient_fantasizes(data1):
    model = surrogate(data1)
    acq = qKnowledgeGradient(model, num_fantasies=4)
    x, val = optimize_acqf(acq, BOUNDS, q=1, num_restarts=2, raw_samples=16)

    assert x.shape == (1, 3) and inside(x) and torch.isfinite(val)


def test_pathwise_thompson_on_discrete_candidates(data1):
    model = surrogate(data1)
    candidates = draw_sobol_samples(bounds=BOUNDS, n=256, q=1, seed=0).squeeze(1)
    x, val = optimize_acqf_discrete(PathwiseThompsonSampling(model), q=1, choices=candidates)

    assert x.shape == (1, 3)
    assert any(torch.equal(x[0], c) for c in candidates)


def test_a_few_rounds_rebuilding_the_model(data1):
    X, Y = data1

    def f(x):
        return x.sin().sum(-1, keepdim=True)

    for _ in range(2):
        model = surrogate((X, Y))
        acq = LogExpectedImprovement(model, best_f=Y.max())
        x, _ = optimize_acqf(acq, BOUNDS, q=1, num_restarts=2, raw_samples=16)
        X, Y = torch.cat([X, x]), torch.cat([Y, f(x)])

    assert surrogate((X, Y)).train_X.shape == (14, 3)


def test_pathwise_thompson_with_optimize_acqf(data1):
    model = surrogate(data1)
    x, val = optimize_acqf(
        PathwiseThompsonSampling(model), BOUNDS, q=2, num_restarts=2, raw_samples=16
    )

    assert x.shape == (2, 3) and inside(x) and torch.isfinite(val)
