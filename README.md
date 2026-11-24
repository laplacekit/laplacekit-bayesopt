# laplacekit-bayesopt

A <code><b>laplacekit</b></code> plugin for Bayesian optimization (BO) with [BoTorch](https://botorch.org).
Essentially, a drop-in alternative for Gaussian process (GP) surrogates.
Here, the surrogate is a [Laplace-approximated Bayesian neural network (BNN)](https://openreview.net/forum?id=gDcaUj4Myhn).

## Install

```bash
pip install laplacekit-bayesopt
```

## Usage

`LaplaceBNNSurrogate` is a BoTorch `Model`.
Use it in a BoTorch loop in place of standard BoTorch models such as `SingleTaskGP`, etc.

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
        nn.Linear(2, 50), 
        nn.ReLU(), 
        nn.Linear(50, 50), 
        nn.ReLU(), 
        nn.Linear(50, 1)
    )


for _ in range(rounds):
    # The only difference!
    model = LaplaceBNNSurrogate(
        make_net(),
        train_X,
        train_Y,
        input_transform=Normalize(d=2),
        outcome_transform=Standardize(m=1),
    )

    # Everything below stays the same!
    acq = LogExpectedImprovement(model, best_f=train_Y.max())
    x, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=20)
    train_X, train_Y = torch.cat([train_X, x]), torch.cat([train_Y, objective(x)])
```

As with standard BoTorch models, we build a new model in each iteration.
The posterior by default is the linearized Laplace approximation (which is theoretically a Gaussian process!) with KFAC.

### Training

Since the surrogate is a neural network, it needs to be trained.
A `Trainer` object does this for us, and there are two options (see the [paper](https://openreview.net/forum?id=gDcaUj4Myhn))

| Trainer                      | What it does                                                                                                                                               |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PosthocMarglik()` (default) | Adam on the mean squared error, then tunes the prior precision and the noise on the marginal likelihood. Cheaper, but theoretically less correct.          |
| `OnlineMarglik()`            | Tunes the prior precision and the noise during training ([Immer et al. 2021](https://arxiv.org/abs/2104.04975)). More expensive, but theoretically better. |

Various options are available---see docstrings.
For the Laplace approximation itself, refer to <https://laplacekit.org>.

Example:

```python
import laplacekit as lk

from laplacekit_bayesopt import PosthocMarglik

trainer = PosthocMarglik(
    epochs=500, 
    subset=lk.subsets.last_layer, 
    hessian_approx=lk.DenseGGN()
)

model = LaplaceBNNSurrogate(make_net(), train_X, train_Y, trainer=trainer)
```

## Example Usages

See the [`examples/` directory](https://github.com/laplacekit/laplacekit-bayesopt/tree/main/examples).

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
