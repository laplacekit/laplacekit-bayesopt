"""Trainers: they train a fresh network on the data and return its Laplace posterior.

Trainers draw from torch's global RNG, so `torch.manual_seed` makes a run repeatable.

`LaplaceBNNSurrogate` hands its training run to a `Trainer`. `PosthocMarglik` trains the
network on the mean squared error, then tunes the prior and the noise on the marginal
likelihood after training (laplace-bayesopt's `marglik_mode="posthoc"`). `OnlineMarglik`
tunes them during training (Immer et al. 2021; `marglik_mode="online"`). Any callable with
the signature of `Trainer.__call__` works too.
"""

import dataclasses
import math
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

import laplacekit as lk
from laplacekit.core import HessianApprox, LaplacePosterior, Prior, Subset

__all__ = ["OnlineMarglik", "PosthocMarglik", "Trainer"]


_FIT_BATCH = 256
"""Examples per batch in the Laplace fit after training."""

type _BuiltinPrior = lk.priors.Isotropic | lk.priors.Layerwise | lk.priors.Diagonal


@runtime_checkable
class Trainer(Protocol):
    """Trains a network on the data and returns its Laplace posterior.

    `LaplaceBNNSurrogate` calls its trainer with the network and the training data, after the
    input and outcome transforms. The posterior must have an `lk.Gaussian` likelihood
    with a one-element `sigma`, the noise scale of the outcomes.
    """

    def __call__(
        self, net: nn.Module, train_X: Tensor, train_Y: Tensor, *, train_Yvar: float | None
    ) -> LaplacePosterior[Any, Any, lk.Gaussian]:
        """Train `net` on the data and fit the Laplace approximation.

        Args:
            net: The network given to `LaplaceBNNSurrogate`.
            train_X: The inputs, `[n, d]`.
            train_Y: The outcomes, `[n, m]`.
            train_Yvar: The noise variance to keep fixed, or `None` to learn it.

        Returns:
            The posterior. Its likelihood is an `lk.Gaussian` with a one-element `sigma`,
            `sqrt(train_Yvar)` when that is given.
        """
        ...


def _check_positive(owner: str, **values: float) -> None:
    for name, value in values.items():
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{owner} needs a positive {name}, got {value}")


def _batches(X: Tensor, Y: Tensor, size: int) -> list[tuple[Tensor, Tensor]]:
    return [(X[i : i + size], Y[i : i + size]) for i in range(0, X.shape[0], size)]


def _likelihood(train_Yvar: float | None, like: Tensor) -> lk.Gaussian:
    """A Gaussian with the noise left to the tuner, or fixed at `sqrt(train_Yvar)`."""

    if train_Yvar is None:
        return lk.Gaussian()

    sigma = torch.tensor(math.sqrt(train_Yvar), dtype=like.dtype, device=like.device)

    return lk.Gaussian(sigma=sigma)


@dataclasses.dataclass(frozen=True)
class PosthocMarglik:
    """Train on the mean squared error, then tune the prior and noise on the marginal likelihood.

    The network trains with Adam and a cosine learning-rate schedule. `lk.laplace` then
    fits the parameters that `subset` picks with `hessian_approx`, and its default hyper
    tuner, `lk.MarginalLikelihood()`, tunes the prior precision and the noise scale
    together. With a fixed `train_Yvar`, only the prior precision is tuned.

    ```python
    trainer = PosthocMarglik(epochs=500, subset=lk.subsets.last_layer)
    model = LaplaceBNNSurrogate(make_net(), train_X, train_Y, trainer=trainer)
    ```

    Args:
        epochs: Passes over the data.
        lr: Adam's learning rate at the start of the schedule.
        weight_decay: Adam's weight decay.
        batch_size: Examples per training step.
        subset: The parameters the Laplace approximation covers. `lk.subsets.all` covers
            every weight; `lk.subsets.last_layer` gives a neural-linear model.
        hessian_approx: The curvature method, KFAC by default. `None` lets `lk.laplace`
            pick one per block.
        prior: The prior, whose precision the tune starts from.

    Raises:
        ValueError: An option is not positive (`weight_decay` may be 0).
    """

    epochs: int = 1000
    lr: float = 0.1
    weight_decay: float = 1e-3
    batch_size: int = 10
    subset: Subset = lk.subsets.all
    hessian_approx: HessianApprox[Any] | None = lk.KFAC()
    prior: Prior = lk.priors.Isotropic()

    def __post_init__(self) -> None:
        _check_positive(
            "PosthocMarglik", epochs=self.epochs, lr=self.lr, batch_size=self.batch_size
        )

        if not (math.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise ValueError(f"PosthocMarglik needs weight_decay >= 0, got {self.weight_decay}")

    def __call__(
        self, net: nn.Module, train_X: Tensor, train_Y: Tensor, *, train_Yvar: float | None
    ) -> LaplacePosterior[Any, Any, lk.Gaussian]:
        """Train `net` on the data, then fit and tune the Laplace approximation.

        Args:
            net: The network. It is trained in place and left in eval mode.
            train_X: The inputs, `[n, d]`.
            train_Y: The outcomes, `[n, m]`.
            train_Yvar: The noise variance to keep fixed, or `None` to learn it.

        Returns:
            The tuned posterior.
        """

        X, Y = train_X, train_Y
        steps = math.ceil(X.shape[0] / self.batch_size)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs * steps)

        net.train()

        for _ in range(self.epochs):
            order = torch.randperm(X.shape[0], device=X.device)

            for i in range(0, X.shape[0], self.batch_size):
                idx = order[i : i + self.batch_size]
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(net(X[idx]), Y[idx])
                loss.backward()
                optimizer.step()
                scheduler.step()

        net.eval()

        return lk.laplace(
            net,
            _batches(X, Y, _FIT_BATCH),
            _likelihood(train_Yvar, X),
            subset=self.subset,
            hessian_approx=self.hessian_approx,
            prior=self.prior,
        )


@dataclasses.dataclass(frozen=True)
class OnlineMarglik:
    """Train with marginal-likelihood steps on the prior and noise (Immer et al. 2021).

    This runs `lk.marglik.train` with Adam and a cosine learning-rate schedule. Every
    `marglik_frequency` epochs, a Laplace fit of all trainable parameters gives the
    marginal likelihood, and the prior precision and the noise scale take steps on it. A
    fixed `train_Yvar` stays as given.

    ```python
    trainer = OnlineMarglik(epochs=500, marglik_frequency=25)
    model = LaplaceBNNSurrogate(make_net(), train_X, train_Y, trainer=trainer)
    ```

    Args:
        epochs: Passes over the data.
        lr: Adam's learning rate at the start of the schedule.
        marglik_frequency: Epochs between marginal-likelihood steps.
        batch_size: Examples per training step.
        prior: How the weights share the prior precision, a class or an instance with one
            shared starting precision, as in `lk.marglik.train`.
        hessian_approx: The curvature method of each Laplace fit, KFAC by default. `None`
            lets `lk.laplace` pick one per block.

    Raises:
        ValueError: An option is not positive.
    """

    epochs: int = 1000
    lr: float = 0.1
    marglik_frequency: int = 50
    batch_size: int = 10
    prior: type[_BuiltinPrior] | _BuiltinPrior = lk.priors.Layerwise
    hessian_approx: HessianApprox[Any] | None = lk.KFAC()

    def __post_init__(self) -> None:
        _check_positive(
            "OnlineMarglik",
            epochs=self.epochs,
            lr=self.lr,
            marglik_frequency=self.marglik_frequency,
            batch_size=self.batch_size,
        )

    def __call__(
        self, net: nn.Module, train_X: Tensor, train_Y: Tensor, *, train_Yvar: float | None
    ) -> LaplacePosterior[Any, Any, lk.Gaussian]:
        """Train `net` and its hyperparameters on the data.

        Args:
            net: The network. It is trained in place and left in eval mode.
            train_X: The inputs, `[n, d]`.
            train_Y: The outcomes, `[n, m]`.
            train_Yvar: The noise variance to keep fixed, or `None` to learn it.

        Returns:
            The posterior of the epoch with the best marginal likelihood.
        """

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_X, train_Y),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs * len(loader))

        net.train()
        trained = lk.marglik.train(
            net,
            loader,
            _likelihood(train_Yvar, train_X),
            n_data=train_X.shape[0],
            optimizer=optimizer,
            epochs=self.epochs,
            marglik_frequency=self.marglik_frequency,
            hessian_approx=self.hessian_approx,
            prior=self.prior,
            scheduler=scheduler,
        )
        net.eval()

        return trained.posterior
