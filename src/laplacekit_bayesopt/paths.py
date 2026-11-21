r"""Sample paths of `LaplaceBNNSurrogate` for BoTorch's pathwise sampling.

A path is one function drawn from the linearized Laplace posterior,
$f_s(x) = f(x; \mu) + J(x) \varepsilon_s$, with one weight draw
$\varepsilon_s \sim \mathcal{N}(0, \Sigma)$ for the whole input space. Importing the
package registers `LaplaceBNNSurrogate` with BoTorch's `draw_matheron_paths`, so BoTorch's
pathwise tools work on it unchanged:

```python
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling

acq = PathwiseThompsonSampling(model)
x, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=512)
```

The draws come from torch's global RNG, so `torch.manual_seed` makes them repeatable.
"""

import math
from typing import Any

import torch
from botorch.sampling.pathwise import SamplePath
from botorch.sampling.pathwise.posterior_samplers import DrawMatheronPaths
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths
from botorch.sampling.pathwise.update_strategies import gaussian_update
from torch import Size, Tensor

from laplacekit.core import CapabilityError
from laplacekit_bayesopt.surrogate import LaplaceBNNSurrogate

__all__: list[str] = []


class _LinearizedPaths(SamplePath):
    """`sample_shape` functions drawn from the linearized posterior of `model`.

    Called on inputs `[*batch, n, d]` in the original space, the paths return
    `[*broadcast(batch, sample_shape), n]` for one outcome. For several outcomes, the
    inputs carry an extra dim of size 1 before `n`, as `MatheronPathModel` passes them,
    and the result is `[..., m, n]`, the layout of BoTorch's multi-output GP paths. The
    outcomes are in the original space.
    """

    def __init__(self, model: LaplaceBNNSurrogate, sample_shape: Size) -> None:
        super().__init__()

        noise = model.laplace_posterior.sample_centered(math.prod(sample_shape))
        self.surrogate = model
        self.sample_shape = sample_shape
        self.eps = {k: v.reshape(*sample_shape, *v.shape[1:]) for k, v in noise.items()}

    def set_ensemble_as_batch(self, ensemble_as_batch: bool) -> None:
        """Do nothing: the model is not an ensemble.

        Args:
            ensemble_as_batch: Ignored.
        """

    def forward(self, X: Tensor, **kwargs: Any) -> Tensor:
        """The paths' values at `X`.

        Args:
            X: The inputs, `[*batch, n, d]`, or `[*batch, 1, n, d]` for several outcomes.
            **kwargs: Ignored.

        Returns:
            `[*batch', n]` for one outcome, `[*batch', m, n]` for several, where `batch'`
            broadcasts `batch` with the sample shape.
        """

        model = self.surrogate
        m = model.num_outputs

        if m > 1:
            X = X.squeeze(-3)

        X_tf = model.transform_inputs(X)
        n, d = X_tf.shape[-2:]
        batch = torch.broadcast_shapes(X_tf.shape[:-2], self.sample_shape)
        lead = len(batch) - len(self.sample_shape)

        inputs = X_tf.expand(*batch, n, d).reshape(-1, n, d)
        mean = {k: v.detach() for k, v in model.laplace_posterior.mean.items()}
        eps = {
            k: e.reshape(*[1] * lead, *e.shape)
            .expand(*batch, *e.shape[len(self.sample_shape) :])
            .reshape(inputs.shape[0], *e.shape[len(self.sample_shape) :])
            .to(mean[k])
            for k, e in self.eps.items()
        }
        fn = model.laplace_posterior.model

        # f(x; μ) + J(x) ε for each input set with its own ε, one JVP per set.
        def one(x: Tensor, e: dict[str, Tensor]) -> Tensor:
            f, lin = torch.func.jvp(lambda p: torch.as_tensor(fn(p, x)), (mean,), (e,))[:2]

            return f + lin

        out = torch.vmap(one)(inputs, eps).reshape(*batch, n, m)

        if hasattr(model, "outcome_transform"):
            out, _ = model.outcome_transform.untransform(out, X=X_tf)

        return out.squeeze(-1) if m == 1 else out.transpose(-1, -2)


@DrawMatheronPaths.register(LaplaceBNNSurrogate)
def _draw_linearized_paths(
    model: LaplaceBNNSurrogate,
    *,
    sample_shape: Size,
    prior_sampler: object = draw_kernel_feature_paths,
    update_strategy: object = gaussian_update,
) -> _LinearizedPaths:
    """Draw `sample_shape` paths of the linearized Laplace posterior.

    `draw_matheron_paths` passes a GP prior sampler and update strategy. The Laplace paths
    have neither, so only BoTorch's defaults are accepted.
    """

    if prior_sampler is not draw_kernel_feature_paths or update_strategy is not gaussian_update:
        raise CapabilityError(
            "LaplaceBNNSurrogate draws its paths from the weight posterior, so it takes no "
            "prior_sampler or update_strategy; leave them at BoTorch's defaults"
        )

    if model._conditioning is not None:
        raise CapabilityError(
            "paths of LaplaceBNNSurrogate come from the fitted posterior, so they can't "
            "follow condition_on_observations; build a new model with the data instead"
        )

    return _LinearizedPaths(model, sample_shape)
