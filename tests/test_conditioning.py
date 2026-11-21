import pytest
import torch
from botorch.models.transforms.outcome import Standardize
from botorch.sampling import SobolQMCNormalSampler

from laplacekit_bayesopt import LaplaceBNNSurrogate

from .conftest import FAST, Nets
from .test_posterior import EXACT


def gp_condition(mean, cov, n, y, noise):
    """Condition a joint Gaussian over [new; query] on the first `n` entries, by the book."""

    Kcc = cov[:n, :n] + torch.diag(noise)
    Kqc = cov[n:, :n]
    gain = Kqc @ torch.linalg.inv(Kcc)

    return mean[n:] + gain @ (y - mean[:n]), cov[n:, n:] - gain @ Kqc.T


@pytest.mark.parametrize("data, k", [("data1", 1), ("data3", 3)])
def test_condition_is_gp_conditioning_in_the_linearized_model(request, data, k):
    X, Y = request.getfixturevalue(data)
    model = LaplaceBNNSurrogate(Nets(3, k)(), X, Y, trainer=EXACT)
    g = torch.Generator().manual_seed(3)
    Xn, Yn, Xq = torch.rand(2, 3, generator=g), torch.randn(2, k, generator=g), torch.rand(4, 3)

    joint = model.posterior(torch.cat([Xn, Xq]))
    sigma2 = model.noise_variance
    mean, cov = gp_condition(
        joint.mean.reshape(-1),
        joint.distribution.covariance_matrix,
        2 * k,
        Yn.reshape(-1),
        sigma2.expand(2 * k),
    )

    post = model.condition_on_observations(Xn, Yn).posterior(Xq)

    torch.testing.assert_close(post.mean.reshape(-1), mean)
    torch.testing.assert_close(post.distribution.covariance_matrix, cov)


def test_conditioning_twice_is_conditioning_once(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    Xn, Yn, Xq = torch.rand(3, 3), torch.randn(3, 1), torch.rand(4, 3)

    once = model.condition_on_observations(Xn, Yn).posterior(Xq)
    twice = (
        model.condition_on_observations(Xn[:1], Yn[:1])
        .condition_on_observations(Xn[1:], Yn[1:])
        .posterior(Xq)
    )

    torch.testing.assert_close(twice.mean, once.mean)
    torch.testing.assert_close(
        twice.distribution.covariance_matrix, once.distribution.covariance_matrix
    )


def test_given_noise_replaces_sigma(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    Xn, Yn, Xq = torch.rand(2, 3), torch.randn(2, 1), torch.rand(4, 3)
    noise = torch.tensor([[0.01], [0.5]])

    joint = model.posterior(torch.cat([Xn, Xq]))
    mean, cov = gp_condition(
        joint.mean.reshape(-1),
        joint.distribution.covariance_matrix,
        2,
        Yn.reshape(-1),
        noise.reshape(-1),
    )
    post = model.condition_on_observations(Xn, Yn, noise=noise).posterior(Xq)

    torch.testing.assert_close(post.mean.reshape(-1), mean)
    torch.testing.assert_close(post.distribution.covariance_matrix, cov)


def test_condition_keeps_the_fit_and_the_original(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    Xq = torch.rand(4, 3)
    before = model.posterior(Xq).mean

    new = model.condition_on_observations(torch.rand(2, 3), torch.randn(2, 1))

    assert new.laplace_posterior is model.laplace_posterior
    torch.testing.assert_close(model.posterior(Xq).mean, before)
    torch.testing.assert_close(new.train_X, X)  # fantasies are not training data


def test_outcome_transform_applies_to_conditioning_data(data1):
    X, Y = data1
    Y = 5 * Y + 3
    model = LaplaceBNNSurrogate(
        Nets(3, 1)(), X, Y, trainer=FAST, outcome_transform=Standardize(m=1)
    )
    Xn, Xq = torch.rand(2, 3), torch.rand(4, 3)
    Yn = model.posterior(Xn).mean.detach() + 2.0

    joint = model.posterior(torch.cat([Xn, Xq]))
    sigma2 = model.noise_variance
    sd = model.outcome_transform.stdvs.reshape(())
    mean, cov = gp_condition(
        joint.mean.reshape(-1),
        joint.distribution.covariance_matrix,
        2,
        Yn.reshape(-1),
        (sd**2 * sigma2).expand(2),
    )

    post = model.condition_on_observations(Xn, Yn).posterior(Xq)

    torch.testing.assert_close(post.mean.reshape(-1), mean)
    torch.testing.assert_close(post.distribution.covariance_matrix, cov)


def test_fantasize_batches(data3):
    X, Y = data3
    model = LaplaceBNNSurrogate(Nets(3, 3)(), X, Y, trainer=FAST)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([5]))

    fantasy = model.fantasize(torch.rand(2, 1, 3), sampler)

    assert fantasy.batch_shape == torch.Size([5, 2])
    post = fantasy.posterior(torch.rand(2, 4, 3))
    assert post.mean.shape == (5, 2, 4, 3)
    assert post.distribution.covariance_matrix.shape == (5, 2, 12, 12)


def test_each_fantasy_is_its_own_conditioning(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    Xn = torch.rand(2, 3)
    Yn = torch.randn(4, 2, 1)  # four fantasies of the same two points
    Xq = torch.rand(3, 3)

    batch = model.condition_on_observations(Xn, Yn).posterior(Xq)

    assert batch.mean.shape == (4, 3, 1)

    for i in range(4):
        one = model.condition_on_observations(Xn, Yn[i]).posterior(Xq)
        torch.testing.assert_close(batch.mean[i], one.mean)
        torch.testing.assert_close(
            batch.distribution.covariance_matrix[i], one.distribution.covariance_matrix
        )


def test_gradient_flows_through_conditioning(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    Xn = torch.rand(2, 3, requires_grad=True)

    post = model.condition_on_observations(Xn, torch.randn(2, 1)).posterior(torch.rand(3, 3))
    post.variance.sum().backward()

    assert Xn.grad is not None and Xn.grad.abs().sum() > 0


def test_empty_conditioning(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    Xq = torch.rand(4, 3)

    same = model.condition_on_observations(torch.rand(0, 3), torch.rand(0, 1)).posterior(Xq)

    torch.testing.assert_close(same.mean, model.posterior(Xq).mean)
