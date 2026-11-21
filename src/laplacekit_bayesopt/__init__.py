r"""Bayesian optimization with Laplace-approximated neural networks, as BoTorch models.

`LaplaceBNNSurrogate` is a BoTorch `Model`. Build it from the data, as `SingleTaskGP`, and use
it with BoTorch's acquisition functions and optimizers:

```python
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from laplacekit_bayesopt import LaplaceBNNSurrogate

for _ in range(rounds):
    model = LaplaceBNNSurrogate(make_net(), train_X, train_Y)
    acq = LogExpectedImprovement(model, best_f=train_Y.max())
    x, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=512)
    train_X, train_Y = torch.cat([train_X, x]), torch.cat([train_Y, objective(x)])
```

The posterior is the linearized Laplace approximation (Kristiadi et al. 2023, "Promises
and Pitfalls of the Linearized Laplace in Bayesian Optimization"). It is a Gaussian
process, so `LaplaceBNNSurrogate.posterior` is joint over the points of a batch, and
`LaplaceBNNSurrogate.condition_on_observations` conditions it on new points without training,
which BoTorch's fantasies (`qKnowledgeGradient`, say) rely on.

Importing the package registers the model with BoTorch's `draw_matheron_paths`, so
`PathwiseThompsonSampling` draws functions from the linearized posterior.

A `Trainer` trains the network and fits the approximation: `PosthocMarglik` (the default)
tunes the prior precision and the noise on the marginal likelihood after training,
`OnlineMarglik` during training. A reused network starts from its trained weights (a warm
start); pass a fresh one for a fresh start.

This is the successor of `laplace-bayesopt`, built on **laplacekit** instead of
`laplace-torch`.
"""

from importlib import metadata

import laplacekit as lk
from laplacekit_bayesopt import paths as _paths  # noqa: F401  # registers the paths
from laplacekit_bayesopt.surrogate import LaplaceBNNSurrogate
from laplacekit_bayesopt.training import OnlineMarglik, PosthocMarglik, Trainer

__all__ = ["LaplaceBNNSurrogate", "OnlineMarglik", "PosthocMarglik", "Trainer"]


def health() -> tuple[lk.HealthCheck, ...]:
    """Check for `lk.health()` that BoTorch and GPyTorch are installed, with their versions.

    Returns:
        One check per library.
    """

    return tuple(_installed(dist) for dist in ("botorch", "gpytorch"))


def _installed(dist: str) -> lk.HealthCheck:
    try:
        return lk.HealthCheck(dist, "ok", metadata.version(dist))
    except metadata.PackageNotFoundError:
        return lk.HealthCheck(dist, "error", "not installed (required)")


plugin = lk.Plugin(supports=">=0.1a0,<0.2", health=health)
"""The record behind the `laplacekit.plugins` entry point."""
