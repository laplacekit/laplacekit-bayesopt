# laplacekit-bayesopt

A <code><b>laplacekit</b></code> plugin that turns a network into a
[BoTorch](https://botorch.org) surrogate, Bayesian optimization with
Laplace-approximated neural networks.

## Install

```bash
pip install laplacekit-bayesopt
```

## Usage

`LaplaceBNNSurrogate` is a BoTorch `Model`. Use it in a BoTorch loop in place of
`SingleTaskGP`:

```python
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from torch import nn

from laplacekit_bayesopt import LaplaceBNNSurrogate


def make_net():
    return nn.Sequential(
        nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 1)
    )


for _ in range(rounds):
    model = LaplaceBNNSurrogate(
        make_net(),
        train_X,
        train_Y,
        input_transform=Normalize(d=2),
        outcome_transform=Standardize(m=1),
    )
    acq = LogExpectedImprovement(model, best_f=train_Y.max())
    x, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=20)
    train_X, train_Y = torch.cat([train_X, x]), torch.cat([train_Y, objective(x)])
```

As with `SingleTaskGP`, each round builds a new model from all the data, and the data
can go in by keyword (`train_X=`, `train_Y=`) or through
`LaplaceBNNSurrogate.construct_inputs(dataset, net=make_net())`, as Ax builds models. The
constructor trains the network it is given in place. A fresh network starts over; the
network of the last round starts from its trained weights, a warm start.

The posterior is the linearized Laplace approximation, which is a Gaussian process:

- `model.posterior(X)` is joint over the `q` points of each batch, so Monte Carlo
  acquisition functions (`qLogNoisyExpectedImprovement`, `qLogExpectedHypervolumeImprovement`, ...)
  see the correlations between points and outcomes.
- `model.condition_on_observations(X, Y)` conditions the GP on new points without
  training. BoTorch's fantasies need this, so `qKnowledgeGradient` works.
- BoTorch's `PathwiseThompsonSampling` works on it: importing the package registers the
  model with `draw_matheron_paths`. Each path is one function f(x; μ) + J(x) ε drawn from
  the posterior, smooth in x, so it works with `optimize_acqf` and `optimize_acqf_discrete`
  alike.

### Training

A `Trainer` trains the network and fits the Laplace approximation:

| Trainer                      | What it does                                                                                                                                 |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `PosthocMarglik()` (default) | Adam on the mean squared error, then tunes the prior precision and the noise on the marginal likelihood.                                     |
| `OnlineMarglik()`            | Tunes the prior precision and the noise during training ([Immer et al. 2021](https://arxiv.org/abs/2104.04975)), through `lk.marglik.train`. |

They are laplace-bayesopt's `marglik_mode="posthoc"` and `"online"`. Both use KFAC over
all weights by default. `PosthocMarglik(subset=lk.subsets.last_layer)` gives a last-layer
Laplace BNN, and `hessian_approx=` takes another curvature:

```python
import laplacekit as lk

from laplacekit_bayesopt import PosthocMarglik

trainer = PosthocMarglik(epochs=500, subset=lk.subsets.last_layer, hessian_approx=lk.DenseGGN())
model = LaplaceBNNSurrogate(make_net(), train_X, train_Y, trainer=trainer)
```

Training and path draws use torch's global RNG, so `torch.manual_seed`
makes a run repeatable.

`train_Yvar=` fixes the observation noise variance instead of learning it, as in
`SingleTaskGP`. It is in the space of `train_Y` and goes through the outcome transform; one
value for all points and outcomes is supported. `model.noise_variance` is the noise, learned
or given, and `model.laplace_posterior` the Laplace posterior. Any callable with the
signature of `Trainer.__call__` works as a trainer.

## Examples

`examples/` holds runnable scripts:

- `singletask.py`: Branin with log expected improvement.
- `thompson.py`: Branin over 8,192 Sobol candidates with Thompson sampling.
- `multitask.py`: a composite objective over three outcomes, with Monte Carlo log expected
  improvement.
- `multiobjective.py`: the Pareto front of Branin-Currin with log expected hypervolume
  improvement.

## Development

```bash
make sync    # uv sync
make check   # ruff, ty, pytest
make examples
```

## Citation

```bibtex
@inproceedings{kristiadi2023promises,
  title={Promises and Pitfalls of the Linearized {L}aplace in {B}ayesian Optimization},
  author={Kristiadi, Agustinus and Immer, Alexander and Eschenhagen, Runa and Fortuin, Vincent},
  booktitle={AABI},
  year={2023}
}
```
