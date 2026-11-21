r"""The BoTorch surrogate: a linearized Laplace posterior as a BoTorch `Model`.

The linearized Laplace posterior is a Gaussian process. With $f(x; \mu)$ the network at
the posterior mean, $J(x)$ its Jacobian with respect to the weights, and $\Sigma$ the
weight covariance, the outputs at any inputs are jointly Gaussian with mean $f(x; \mu)$
and covariance $J(x) \Sigma J(x')^\top$. `LaplaceBNNSurrogate.posterior` returns that joint
Gaussian, and `LaplaceBNNSurrogate.condition_on_observations` conditions it on new points as a
GP would, without training.
"""

import copy
import dataclasses
from collections.abc import Callable
from typing import Any, Self, cast

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import FantasizeMixin, Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.datasets import SupervisedDataset
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from linear_operator.operators import DenseLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor, nn

import laplacekit as lk
from laplacekit.core import (
    Inputs,
    LaplacePosterior,
    ModelFn,
    Outputs,
    Params,
    StructureError,
)
from laplacekit_bayesopt.training import PosthocMarglik, Trainer

__all__ = ["LaplaceBNNSurrogate"]


@dataclasses.dataclass(frozen=True)
class _Conditioning:
    r"""Points the surrogate is conditioned on, in the transformed spaces.

    `X` is `[*c, n, d]`, and `Y` and `noise` (variances) are `[*c, n, m]`. They keep their
    autograd graph, so gradients reach the points of a fantasy.
    """

    X: Tensor
    Y: Tensor
    noise: Tensor


def _check_data(X: Tensor, Y: Tensor, Yvar: float | Tensor | None) -> None:
    if X.ndim != 2:
        raise ValueError(f"LaplaceBNNSurrogate needs train_X of shape [n, d], got {tuple(X.shape)}")

    if Y.ndim != 2:
        raise ValueError(f"LaplaceBNNSurrogate needs train_Y of shape [n, m], got {tuple(Y.shape)}")

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"LaplaceBNNSurrogate needs as many rows in train_Y as in train_X, got "
            f"{Y.shape[0]} and {X.shape[0]}"
        )

    if isinstance(Yvar, Tensor) and Yvar.ndim != 0 and Yvar.shape != Y.shape:
        raise ValueError(
            f"LaplaceBNNSurrogate needs a float, a 0-dim tensor or train_Yvar of shape "
            f"{list(Y.shape)} (that of train_Y), got {list(Yvar.shape)}"
        )


def _constant(var: Tensor) -> bool:
    return torch.allclose(var, var.reshape(-1)[0].expand_as(var))


def _noise_variance(train_Yvar: Tensor, given: Tensor) -> float:
    """The one noise variance in `train_Yvar` (transformed), refusing anything else.

    `given` is the untransformed `train_Yvar`, to say when the transform made a constant
    variance vary.
    """

    var = train_Yvar.detach()

    if not bool(torch.isfinite(var).all()) or not bool((var > 0).all()):
        raise ValueError(
            "LaplaceBNNSurrogate needs a positive finite train_Yvar, got "
            f"{var.flatten()[:4].tolist()}"
        )

    if not _constant(var) and _constant(given.detach()):
        raise ValueError(
            "LaplaceBNNSurrogate supports one noise variance for every point and outcome, but "
            "the outcome transform (per-outcome Standardize) divides train_Yvar by each "
            "outcome's variance, so the transformed variance differs between outcomes; pass "
            "train_Yvar=None to learn the noise, or standardize the outcomes yourself; got "
            f"transformed values from {float(var.min()):.4g} to {float(var.max()):.4g}"
        )

    if not _constant(var):
        raise ValueError(
            "LaplaceBNNSurrogate supports one noise variance for every point and outcome (after "
            "the outcome transform); pass a constant train_Yvar, or None to learn the noise; "
            f"got values from {float(var.min()):.4g} to {float(var.max()):.4g}"
        )

    return float(var.reshape(-1)[0])


def _expand(t: Tensor, batch: torch.Size) -> Tensor:
    return t.expand(*batch, *t.shape[-2:])


def _on_sets(model: ModelFn) -> Callable[[Params, Inputs], Outputs]:
    """The model on sets of points: `[*s, q, d]` inputs to `[*s, q, m]` outputs."""

    def forward(params: Params, inputs: Inputs) -> Outputs:
        x = cast(Tensor, inputs)  # `_joint` passes a tensor
        out = torch.as_tensor(model(params, x.reshape(-1, x.shape[-1])))

        return out.reshape(*x.shape[:-1], *out.shape[1:])

    return forward


def _check_posterior(post: object) -> LaplacePosterior[Any, Any, lk.Gaussian]:
    if not isinstance(post, LaplacePosterior):
        raise StructureError(
            f"the trainer must return a LaplacePosterior, got {type(post).__name__}"
        )

    lik = post.likelihood

    if not isinstance(lik, lk.Gaussian) or lik.sigma.numel() != 1:
        raise StructureError(
            "the trainer must return a posterior with an lk.Gaussian likelihood and a "
            f"one-element sigma, got {lik!r}"
        )

    return post


class LaplaceBNNSurrogate(Model, FantasizeMixin):
    r"""A Laplace-approximated neural network as a BoTorch model.

    Build it from the data, as any BoTorch model, and use it wherever BoTorch takes a
    `Model`: analytic and Monte Carlo acquisition functions, `optimize_acqf`, and
    fantasy-based acquisition functions such as `qKnowledgeGradient`. The constructor
    trains the network and fits the Laplace approximation, so each round of Bayesian
    optimization builds a new model from all the data:

    ```python
    for _ in range(rounds):
        model = LaplaceBNNSurrogate(
            make_net(), train_X, train_Y, outcome_transform=Standardize(m=1)
        )
        acq = LogExpectedImprovement(model, best_f=train_Y.max())
        x, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=512)
        train_X, train_Y = torch.cat([train_X, x]), torch.cat([train_Y, objective(x)])
    ```

    The posterior is the linearized Laplace GP. For a batch of $q$ points $X$ and $m$
    outcomes, it is Gaussian with mean $f(X; \mu)$ and covariance $J \Sigma J^\top$, where
    $J$ is the Jacobian of the $q m$ outputs with respect to the weights. Batches are
    independent, so for marginals at many points pass them as `X.unsqueeze(-2)`, a batch
    of one point each; a 2-dim `X` of $N$ points gives one joint $N m \times N m$
    covariance.

    As in BoTorch's own models, the transforms learn their coefficients from the training
    data in the constructor and are then kept in eval mode. The network and the posterior
    work in the transformed spaces, and `posterior` returns outcomes in the original space.
    Training draws from torch's global RNG, so `torch.manual_seed` makes it repeatable.

    Args:
        net: A network with `d` inputs and `m` outputs. It is trained in place, so a
            network reused from the last round starts from its trained weights (a warm
            start); pass a freshly initialized one to start over.
        train_X: The inputs, `[n, d]`.
        train_Y: The outcomes, `[n, m]`.
        trainer: Trains the network and fits the Laplace approximation.
        train_Yvar: The observation noise variance, as in `SingleTaskGP`: a float or a
            tensor `[n, m]` in the space of `train_Y`, kept fixed. It goes through the
            outcome transform with `train_Y`, and must then be one value. `None` learns the
            noise on the marginal likelihood.
        input_transform: A BoTorch input transform, such as `Normalize(d=d)`.
        outcome_transform: A BoTorch outcome transform, such as `Standardize(m=m)`.

    Raises:
        ValueError: `train_X` is not `[n, d]`, `train_Y` is not `[n, m]` with the same `n`,
            or `train_Yvar` is not positive and finite, or not one value after the outcome
            transform.
        StructureError: The trainer returned something other than a `LaplacePosterior` with
            an `lk.Gaussian` likelihood and a one-element `sigma`.
    """

    laplace_posterior: LaplacePosterior[Any, Any, lk.Gaussian]
    train_X: Tensor
    train_Y: Tensor

    def __init__(
        self,
        net: nn.Module,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: float | Tensor | None = None,
        *,
        trainer: Trainer = PosthocMarglik(),
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
    ) -> None:
        super().__init__()

        _check_data(train_X, train_Y, train_Yvar)
        X_tf, Y_tf = train_X, train_Y
        Yvar = (
            None
            if train_Yvar is None
            else torch.as_tensor(train_Yvar, dtype=train_Y.dtype, device=train_Y.device)
        )
        Yvar = given = None if Yvar is None else Yvar.expand_as(train_Y)

        # Train mode learns the transforms' coefficients from the data; eval mode then
        # freezes them for every later call.
        with torch.no_grad():
            if input_transform is not None:
                input_transform.train()
                X_tf = input_transform(train_X)
                input_transform.eval()

            if outcome_transform is not None:
                outcome_transform.train()
                Y_tf, Yvar = outcome_transform(train_Y, Yvar, X=X_tf)
                outcome_transform.eval()

        noise = None if Yvar is None or given is None else _noise_variance(Yvar, given)
        post = _check_posterior(trainer(net, X_tf, Y_tf, train_Yvar=noise))
        self._setup(post, train_X, train_Y, input_transform, outcome_transform, None)

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        net: nn.Module | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """The constructor's arguments for a BoTorch dataset, as `Model.construct_inputs`.

        ```python
        model = LaplaceBNNSurrogate(**LaplaceBNNSurrogate.construct_inputs(dataset, net=net))
        ```

        Args:
            training_data: A `SupervisedDataset` with `X`, `Y` and, optionally, `Yvar`.
            net: The network to train.
            **kwargs: Other constructor arguments, such as `trainer` or the transforms,
                passed on as they are.

        Returns:
            `train_X`, `train_Y`, `train_Yvar` when the dataset has it, `net` and `kwargs`.

        Raises:
            ValueError: `net` is missing.
        """

        if net is None:
            raise ValueError("LaplaceBNNSurrogate.construct_inputs needs the network: net=...")

        return {**super().construct_inputs(training_data), "net": net, **kwargs}

    def _setup(
        self,
        post: LaplacePosterior[Any, Any, lk.Gaussian],
        train_X: Tensor,
        train_Y: Tensor,
        input_transform: InputTransform | None,
        outcome_transform: OutcomeTransform | None,
        conditioning: _Conditioning | None,
    ) -> None:
        self.laplace_posterior = post
        # The same posterior, with a model that takes a set of points [q, d] as one input:
        # its per-input covariance is then the joint covariance over the q points.
        self._sets = copy.replace(post, model=ModelFn(_on_sets(post.model), post.model.module))
        self.train_X = train_X
        self.train_Y = train_Y
        self._conditioning = conditioning
        # BoTorch's `fantasize` reads `likelihood` to look for fixed noise.
        self.likelihood = post.likelihood

        # BoTorch checks for these attributes by name, so set them only when given.
        if input_transform is not None:
            self.input_transform = input_transform

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

    @classmethod
    def _conditioned(cls, base: "LaplaceBNNSurrogate", conditioning: _Conditioning) -> Self:
        """A copy of `base` conditioned on points, without training."""

        new = cls.__new__(cls)
        Model.__init__(new)
        new._setup(
            base.laplace_posterior,
            base.train_X,
            base.train_Y,
            getattr(base, "input_transform", None),
            getattr(base, "outcome_transform", None),
            conditioning,
        )

        return new

    @property
    def num_outputs(self) -> int:
        """The number of outcomes $m$."""

        return self.train_Y.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        """The batch shape of the conditioning points, and empty without conditioning."""

        if self._conditioning is None:
            return torch.Size([])

        return torch.Size(self._conditioning.X.shape[:-2])

    @property
    def noise_variance(self) -> Tensor:
        """The noise variance of the transformed outcomes, learned or given, a 0-dim tensor."""

        # The constructor checked that the likelihood has a one-element sigma.
        return self.laplace_posterior.likelihood.sigma.reshape(()) ** 2

    def _set_transformed_inputs(self) -> None:
        # The training inputs are transformed once, in `__init__`; BoTorch's version would warn
        # that there are no `train_inputs`.
        pass

    def _revert_to_original_inputs(self) -> None:
        pass

    def _joint(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """The mean `[*b, q, m]` and covariance `[*b, q m, q m]` of the latent outputs.

        `X` is `[*b, q, d]`, in the transformed space. The covariance is in point-major order:
        point 0's outputs, then point 1's.
        """

        batch, q, m = X.shape[:-2], X.shape[-2], self.num_outputs
        sets = X.reshape(-1, q, X.shape[-1])
        out = lk.predict_outputs(self._sets, sets, n_samples="exact")
        mean = torch.as_tensor(out.mean)

        if mean.shape[1:] != (q, m):
            raise StructureError(
                f"the network must map inputs [N, d] to outputs [N, {m}], got outputs of shape "
                f"{tuple(mean.shape[1:])} for {q} inputs"
            )

        cov = out.cov.per_input()

        return mean.reshape(*batch, q, m), cov.reshape(*batch, q * m, q * m)

    def _latent(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """The mean `[*b, q, m]` and covariance `[*b, q m, q m]` of the latent outputs at `X`.

        `X` is in the transformed space. With conditioning points, this is the GP posterior
        given them.
        """

        cond = self._conditioning

        if cond is None:
            return self._joint(X)

        batch = torch.broadcast_shapes(X.shape[:-2], cond.X.shape[:-2])
        n, q, m = cond.X.shape[-2], X.shape[-2], self.num_outputs
        mean, cov = self._joint(torch.cat([_expand(cond.X, batch), _expand(X, batch)], -2))

        # The Schur complement of the conditioning block: with K the noisy covariance of the
        # conditioning points and r their residuals, mean + Kqc K⁻¹ r and cov - Kqc K⁻¹ Kcq.
        nm = n * m
        noise = _expand(cond.noise, batch).reshape(*batch, nm)
        residual = (_expand(cond.Y, batch) - mean[..., :n, :]).reshape(*batch, nm, 1)
        chol = psd_safe_cholesky(cov[..., :nm, :nm] + torch.diag_embed(noise))
        W = torch.linalg.solve_triangular(chol, cov[..., :nm, nm:], upper=False)
        alpha = torch.linalg.solve_triangular(chol, residual, upper=False)

        mean = mean[..., n:, :] + (W.mT @ alpha).reshape(*batch, q, m)
        cov = cov[..., nm:, nm:] - W.mT @ W

        return mean, 0.5 * (cov + cov.mT)

    # BoTorch's FantasizeMixin declares a narrower stub; Model.posterior is the contract, as
    # for BoTorch's own GPyTorchModel.
    def posterior(  # ty: ignore[incompatible-method-override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> Posterior:
        r"""The joint Gaussian over the outcomes at `X`.

        Args:
            X: The inputs, `[*b, q, d]`, in the original space. Each batch is joint over its
                `q` points. A 2-dim `X` is one batch of `q` points.
            output_indices: The outcomes to keep. `None` keeps all.
            observation_noise: `True` adds the noise variance (`noise_variance`) to the diagonal. A
                tensor `[*, 1, m]` or `[*, q, m]` adds those variances instead, given in
                the transformed space. `False` gives the latent outputs.
            posterior_transform: Applied last, after the outcome transform.
            **kwargs: Ignored.

        Returns:
            A `GPyTorchPosterior`. Its distribution is a `MultivariateNormal` for one
            outcome and a `MultitaskMultivariateNormal` (point-major order) for several.

        Raises:
            ValueError: `X` has fewer than 2 dims.
        """

        if X.ndim < 2:
            raise ValueError(f"posterior needs X of shape [*b, q, d], got {tuple(X.shape)}")

        self.eval()
        X_tf = self.transform_inputs(X)
        mean, cov = self._latent(X_tf)
        q, m = mean.shape[-2], mean.shape[-1]

        if observation_noise is True:
            cov = cov + self.noise_variance * torch.eye(q * m, dtype=cov.dtype, device=cov.device)
        elif isinstance(observation_noise, Tensor):
            noise = observation_noise.expand(*observation_noise.shape[:-2], q, m)
            cov = cov + torch.diag_embed(noise.reshape(*noise.shape[:-2], q * m))

        idx = list(range(m)) if output_indices is None else list(output_indices)

        if len(idx) != m:
            rows = torch.tensor([i * m + k for i in range(q) for k in idx], device=cov.device)
            mean = mean[..., idx]
            cov = cov[..., rows[:, None], rows]

        # A lazy covariance: gpytorch then factors it with jitter when it samples, instead of
        # a plain Cholesky at construction, which fails on a singular covariance.
        lazy = DenseLinearOperator(cov)

        if len(idx) == 1:
            dist = MultivariateNormal(mean.squeeze(-1), covariance_matrix=lazy)
        else:
            dist = MultitaskMultivariateNormal(mean, covariance_matrix=lazy, interleaved=True)

        result = GPyTorchPosterior(dist)

        if hasattr(self, "outcome_transform"):
            transform = self.outcome_transform

            if len(idx) != m:
                transform = transform.subset_output(idx)

            result = transform.untransform_posterior(result, X=X_tf)

        if posterior_transform is not None:
            return posterior_transform(result)

        return result

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, noise: Tensor | None = None, **kwargs: Any
    ) -> Self:
        r"""Condition the linearized GP on new observations, without training.

        The result is a GP posterior given the new points: for a query $x$, the mean moves
        by $k(x, X) (K + N)^{-1} (Y - f(X; \mu))$ and the covariance shrinks by
        $k(x, X) (K + N)^{-1} k(X, x)$, where $k$ is the linearized kernel
        $J(x) \Sigma J(x')^\top$, $K = k(X, X)$ and $N$ the noise. The network and the
        weight posterior stay as they are. This is what BoTorch's fantasies use; to train
        on new data, build a new model.

        Batch dims of `X` and `Y` broadcast, so a batch of fantasies gives a batched
        surrogate. Conditioning again adds the new points to the earlier ones.

        Args:
            X: The inputs, `[*b, n, d]`, in the original space.
            Y: The outcomes, `[*b', n, m]`, in the original space, with `*b'` broadcasting
                to `*b`.
            noise: The noise variances of `Y`, shaped like `Y` and in the transformed space.
                `None` uses `noise_variance`.
            **kwargs: Ignored.

        Returns:
            A new model that shares this one's posterior and training data.
        """

        self.eval()
        X_tf = self.transform_inputs(X)

        if hasattr(self, "outcome_transform"):
            Y, _ = self.outcome_transform(Y, X=X_tf)

        var = self.noise_variance.to(Y) if noise is None else noise
        old = self._conditioning
        parts = [(X_tf, Y, torch.as_tensor(var).expand_as(Y))]

        if old is not None:
            parts.insert(0, (old.X, old.Y, old.noise))

        batch = torch.broadcast_shapes(*(t.shape[:-2] for part in parts for t in part))
        X_all, Y_all, var_all = (
            torch.cat([_expand(part[i], batch) for part in parts], -2) for i in range(3)
        )

        return type(self)._conditioned(self, _Conditioning(X_all, Y_all, var_all))
