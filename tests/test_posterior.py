import pytest
import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal

import laplacekit as lk
from laplacekit_bayesopt import LaplaceBNNSurrogate, PosthocMarglik

from .conftest import FAST, Nets, joint_reference

EXACT = PosthocMarglik(epochs=30, lr=1e-2, batch_size=8, hessian_approx=lk.DenseGGN(joint=True))
"""Full GGN, so the reference covariance is the exact inverse of one dense matrix."""


def test_single_output_shapes(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    post = model.posterior(torch.rand(5, 4, 3))

    assert isinstance(post, GPyTorchPosterior)
    assert type(post.distribution) is MultivariateNormal
    assert post.mean.shape == (5, 4, 1)
    assert post.distribution.covariance_matrix.shape == (5, 4, 4)
    assert model.num_outputs == 1
    assert model.batch_shape == torch.Size([])


def test_multi_output_shapes(data3):
    X, Y = data3
    model = LaplaceBNNSurrogate(Nets(3, 3)(), X, Y, trainer=FAST)
    post = model.posterior(torch.rand(5, 4, 3))

    assert isinstance(post.distribution, MultitaskMultivariateNormal)
    assert post.mean.shape == (5, 4, 3)
    assert post.distribution.covariance_matrix.shape == (5, 12, 12)
    assert model.num_outputs == 3


def test_two_dim_input_is_one_joint_batch(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    post = model.posterior(torch.rand(6, 3))

    assert post.mean.shape == (6, 1)
    assert post.distribution.covariance_matrix.shape == (6, 6)


@pytest.mark.parametrize("data, k", [("data1", 1), ("data3", 3)])
def test_matches_dense_linearized_reference(request, data, k):
    X, Y = request.getfixturevalue(data)
    nets = Nets(3, k)
    model = LaplaceBNNSurrogate(nets(), X, Y, trainer=EXACT)
    Xq = torch.rand(4, 3, generator=torch.Generator().manual_seed(7))

    mean, cov = joint_reference(model.laplace_posterior, nets.made[-1], X, Xq)
    post = model.posterior(Xq)

    # Interleaved (point-major) order, as MultitaskMultivariateNormal's default.
    torch.testing.assert_close(post.mean.reshape(-1), mean)
    torch.testing.assert_close(post.distribution.covariance_matrix, cov)


def test_observation_noise_adds_sigma_squared(data3):
    X, Y = data3
    model = LaplaceBNNSurrogate(Nets(3, 3)(), X, Y, trainer=FAST)
    Xq = torch.rand(2, 4, 3)
    sigma2 = model.noise_variance

    latent = model.posterior(Xq).distribution.covariance_matrix
    noisy = model.posterior(Xq, observation_noise=True).distribution.covariance_matrix

    torch.testing.assert_close(noisy - latent, sigma2 * torch.eye(12).expand(2, 12, 12))


def test_observation_noise_tensor(data3):
    X, Y = data3
    model = LaplaceBNNSurrogate(Nets(3, 3)(), X, Y, trainer=FAST)
    Xq = torch.rand(4, 3)
    noise = torch.tensor([[0.1, 0.2, 0.3]])  # 1 x m: the same for every point

    latent = model.posterior(Xq).distribution.covariance_matrix
    noisy = model.posterior(Xq, observation_noise=noise).distribution.covariance_matrix

    torch.testing.assert_close(noisy - latent, torch.diag(noise.expand(4, 3).reshape(-1)))


def test_output_indices(data3):
    X, Y = data3
    model = LaplaceBNNSurrogate(Nets(3, 3)(), X, Y, trainer=FAST)
    Xq = torch.rand(5, 2, 3)

    full = model.posterior(Xq)
    sub = model.posterior(Xq, output_indices=[0, 2])
    one = model.posterior(Xq, output_indices=[1])
    cov = full.distribution.covariance_matrix.reshape(5, 2, 3, 2, 3)

    torch.testing.assert_close(sub.mean, full.mean[..., [0, 2]])
    torch.testing.assert_close(
        sub.distribution.covariance_matrix,
        cov[:, :, [0, 2]][:, :, :, :, [0, 2]].reshape(5, 4, 4),
    )
    assert type(one.distribution) is MultivariateNormal
    torch.testing.assert_close(one.mean, full.mean[..., [1]])


def test_posterior_transform(data3):
    X, Y = data3
    model = LaplaceBNNSurrogate(Nets(3, 3)(), X, Y, trainer=FAST)
    w = torch.tensor([1.0, -2.0, 0.5])
    Xq = torch.rand(5, 2, 3)

    full = model.posterior(Xq)
    scal = model.posterior(Xq, posterior_transform=ScalarizedPosteriorTransform(weights=w))

    torch.testing.assert_close(scal.mean.squeeze(-1), full.mean @ w)


def test_outcome_transform_matches_standardizing_by_hand(data1):
    X, Y = data1
    mu, sd = Y.mean(0), Y.std(0)

    by_hand = LaplaceBNNSurrogate(Nets(3, 1)(), X, (Y - mu) / sd, trainer=FAST)
    model = LaplaceBNNSurrogate(
        Nets(3, 1)(), X, Y, trainer=FAST, outcome_transform=Standardize(m=1)
    )
    Xq = torch.rand(3, 2, 3)

    ref = by_hand.posterior(Xq)
    post = model.posterior(Xq)

    torch.testing.assert_close(post.mean, ref.mean * sd + mu)
    torch.testing.assert_close(
        post.distribution.covariance_matrix, ref.distribution.covariance_matrix * sd**2
    )
    torch.testing.assert_close(model.train_Y, Y)


def test_input_transform_matches_normalizing_by_hand(data1):
    X, Y = data1
    X = 10 * X - 3
    lo, hi = X.min(0).values, X.max(0).values

    by_hand = LaplaceBNNSurrogate(Nets(3, 1)(), (X - lo) / (hi - lo), Y, trainer=FAST)
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST, input_transform=Normalize(d=3))
    Xq = 10 * torch.rand(3, 2, 3) - 3

    ref = by_hand.posterior((Xq - lo) / (hi - lo))
    post = model.posterior(Xq)

    torch.testing.assert_close(post.mean, ref.mean)
    torch.testing.assert_close(
        post.distribution.covariance_matrix, ref.distribution.covariance_matrix
    )
    torch.testing.assert_close(model.train_X, X)


def test_transforms_are_attached_and_fitted_as_in_botorch(data1):
    X, Y = data1
    ot, it = Standardize(m=1), Normalize(d=3)
    model = LaplaceBNNSurrogate(
        Nets(3, 1)(), X, Y, trainer=FAST, outcome_transform=ot, input_transform=it
    )

    assert model.outcome_transform is ot and model.input_transform is it
    assert not ot.training and not it.training
    torch.testing.assert_close(ot.means, Y.mean(0, keepdim=True))
    torch.testing.assert_close(it.mins, X.min(0, keepdim=True).values)


def test_eval_and_train_do_not_warn(data1, recwarn):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST, input_transform=Normalize(d=3))
    model.eval()
    model.train()
    model.posterior(torch.rand(2, 3))

    assert not [w for w in recwarn if "train_inputs" in str(w.message)]


@pytest.mark.parametrize("trainer", ["posthoc", "marglik"])
def test_backprop_through_the_posterior_twice(data1, trainer):
    from laplacekit_bayesopt import LaplaceBNNSurrogate, OnlineMarglik

    X, Y = data1
    t = FAST if trainer == "posthoc" else OnlineMarglik(epochs=10, marglik_frequency=5, lr=1e-2)
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=t)

    for _ in range(2):
        Xq = torch.rand(5, 4, 3, requires_grad=True)
        post = model.posterior(Xq)
        (post.mean.sum() + post.variance.sum()).backward()

        assert Xq.grad is not None and torch.isfinite(Xq.grad).all()
        assert Xq.grad.abs().sum() > 0


def test_posterior_needs_a_feature_dim(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)

    with pytest.raises(ValueError, match="X"):
        model.posterior(torch.rand(3))


@pytest.mark.filterwarnings("ignore:A not p.d., added jitter")
@pytest.mark.parametrize("data, k", [("data1", 1), ("data3", 3)])
def test_singular_covariance_still_samples(request, data, k):
    X, Y = request.getfixturevalue(data)
    # The last layer has 9 k parameters, fewer than the 12 k outputs at 12 points.
    model = LaplaceBNNSurrogate(
        Nets(3, k)(), X, Y, trainer=PosthocMarglik(epochs=5, subset=lk.subsets.last_layer)
    )
    post = model.posterior(torch.rand(2, 12, 3))

    assert torch.linalg.matrix_rank(post.distribution.covariance_matrix[0]) < 12 * k
    draws = post.rsample(torch.Size([4]))

    assert draws.shape == (4, 2, 12, k) and torch.isfinite(draws).all()
