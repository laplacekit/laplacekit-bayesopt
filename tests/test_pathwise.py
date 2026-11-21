"""BoTorch's pathwise Thompson sampling on the Laplace model, through its path dispatcher."""

import pytest
import torch
from botorch.acquisition.objective import LinearMCObjective
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch.models.deterministic import MatheronPathModel
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.pathwise import SamplePath, draw_matheron_paths

import laplacekit as lk
from laplacekit_bayesopt import LaplaceBNNSurrogate

from .conftest import FAST, Nets, flat_params


def mean_and_jacobian(model, net, Xq):
    """f(x; μ) `[N]` and the Jacobian J(x) `[N, P]` at the posterior mean, from torch.func."""

    params = flat_params(model.laplace_posterior)
    names = list(params)
    sizes = [params[n].numel() for n in names]
    flat = torch.cat([params[n].reshape(-1) for n in names])

    def f(theta):
        chunks = torch.split(theta, sizes)
        p = {n: c.reshape(params[n].shape) for n, c in zip(names, chunks, strict=True)}

        return torch.func.functional_call(net, p, (Xq,)).reshape(-1)

    return f(flat), torch.func.jacrev(f)(flat)


def test_draw_matheron_paths_dispatches_to_the_laplace_path(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    paths = draw_matheron_paths(model, sample_shape=torch.Size([4]))

    assert isinstance(paths, SamplePath)
    assert paths(torch.rand(5, 3)).shape == (4, 5)
    assert paths(torch.rand(4, 5, 3)).shape == (4, 5)  # sample dims broadcast as batch


def test_each_path_is_one_linearized_function(data1):
    X, Y = data1
    nets = Nets(3, 1)
    model = LaplaceBNNSurrogate(nets(), X, Y, trainer=FAST)
    paths = draw_matheron_paths(model, sample_shape=torch.Size([2]))

    # f_s(x) - f(x; μ) = J(x) ε_s with one ε_s per path: solve for ε_s on many points,
    # then check it predicts the path at new points.
    g = torch.Generator().manual_seed(1)
    fit_x, new_x = torch.rand(200, 3, generator=g), torch.rand(20, 3, generator=g)
    mean, J = mean_and_jacobian(model, nets.made[-1], fit_x)
    mean_new, J_new = mean_and_jacobian(model, nets.made[-1], new_x)
    eps = torch.linalg.lstsq(J, (paths(fit_x) - mean).T).solution  # [P, 2]

    torch.testing.assert_close(paths(new_x), (mean_new[:, None] + J_new @ eps).T)
    assert not torch.allclose(eps[:, 0], eps[:, 1])


def test_paths_follow_the_posterior(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(
        Nets(3, 1)(), X, 5 * Y + 2, trainer=FAST, outcome_transform=Standardize(m=1)
    )
    Xq = torch.rand(3, 3)
    torch.manual_seed(0)
    values = draw_matheron_paths(model, sample_shape=torch.Size([20_000]))(Xq)  # [S, 3]

    post = model.posterior(Xq)
    torch.testing.assert_close(values.mean(0), post.mean.squeeze(-1), atol=0.02, rtol=0.0)
    torch.testing.assert_close(
        values.T.cov(), post.distribution.covariance_matrix, atol=0.02, rtol=0.05
    )


def test_manual_seed_makes_paths_repeatable(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    Xq = torch.rand(5, 3)

    torch.manual_seed(3)
    a = draw_matheron_paths(model, sample_shape=torch.Size([2]))
    torch.manual_seed(3)
    b = draw_matheron_paths(model, sample_shape=torch.Size([2]))

    torch.testing.assert_close(a(Xq), b(Xq))
    torch.testing.assert_close(a(Xq), a(Xq))


def test_matheron_path_model_multi_output(data3):
    X, Y = data3
    model = LaplaceBNNSurrogate(Nets(3, 3)(), X, Y, trainer=FAST)
    path_model = MatheronPathModel(model, sample_shape=torch.Size([4]))

    assert path_model(torch.rand(4, 5, 3)).shape == (4, 5, 3)


def test_pathwise_thompson_sampling_is_the_sum_over_q(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    acq = PathwiseThompsonSampling(model)
    Xq = torch.rand(6, 2, 3)

    value = acq(Xq)
    path = acq.samples  # one path per q point, drawn on the first call

    assert value.shape == (6,)
    torch.testing.assert_close(value, path(Xq.unsqueeze(-2)).squeeze(-1).squeeze(-1).sum(-1))
    torch.testing.assert_close(acq(Xq), value)


def test_pathwise_thompson_sampling_with_a_linear_objective(data3):
    X, Y = data3
    model = LaplaceBNNSurrogate(Nets(3, 3)(), X, Y, trainer=FAST)
    w = torch.tensor([1.0, -1.0, 0.5])
    acq = PathwiseThompsonSampling(model, objective=LinearMCObjective(w))
    Xq = torch.rand(6, 1, 3)

    value = acq(Xq)

    assert value.shape == (6,)
    torch.testing.assert_close(value, (acq.samples(Xq.unsqueeze(-2)) @ w).reshape(6))


def test_paths_are_differentiable(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    Xq = torch.rand(6, 1, 3, requires_grad=True)

    PathwiseThompsonSampling(model)(Xq).sum().backward()

    assert Xq.grad is not None and Xq.grad.abs().sum() > 0


def test_a_conditioned_model_is_refused(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    cond = model.condition_on_observations(torch.rand(1, 3), torch.randn(1, 1))

    with pytest.raises(lk.CapabilityError, match="condition"):
        draw_matheron_paths(cond, sample_shape=torch.Size([1]))
