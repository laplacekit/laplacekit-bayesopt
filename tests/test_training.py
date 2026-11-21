import math

import pytest
import torch

import laplacekit as lk
from laplacekit_bayesopt import LaplaceBNNSurrogate, OnlineMarglik, PosthocMarglik, Trainer

from .conftest import FAST, Nets


def test_trainers_are_trainers():
    assert isinstance(PosthocMarglik(), Trainer)
    assert isinstance(OnlineMarglik(), Trainer)


def test_defaults_follow_laplace_bayesopt():
    assert PosthocMarglik() == PosthocMarglik(epochs=1000, lr=0.1, weight_decay=1e-3, batch_size=10)
    assert PosthocMarglik().subset is lk.subsets.all
    assert (
        PosthocMarglik().hessian_approx == lk.KFAC() and OnlineMarglik().hessian_approx == lk.KFAC()
    )
    assert (
        OnlineMarglik().epochs == 1000
        and OnlineMarglik().marglik_frequency == 50
        and OnlineMarglik().batch_size == 10
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: PosthocMarglik(epochs=0),
        lambda: PosthocMarglik(lr=0.0),
        lambda: PosthocMarglik(weight_decay=-1.0),
        lambda: PosthocMarglik(batch_size=0),
        lambda: OnlineMarglik(epochs=0),
        lambda: OnlineMarglik(marglik_frequency=0),
        lambda: OnlineMarglik(lr=-1.0),
        lambda: OnlineMarglik(batch_size=0),
    ],
)
def test_bad_options_raise(make):
    with pytest.raises(ValueError):
        make()


def test_posthoc_learns_sigma_and_prior(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=FAST)
    post = model.laplace_posterior

    assert isinstance(post.likelihood, lk.Gaussian)
    assert post.likelihood.sigma.numel() == 1
    assert post.likelihood.sigma.item() != 1.0
    assert post.prior.tunable().item() != 1.0


def test_fixed_noise_is_kept_and_the_prior_is_tuned(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, train_Yvar=0.09, trainer=FAST)
    post = model.laplace_posterior

    assert post.likelihood.sigma.item() == pytest.approx(0.3)

    # The tuned precision maximizes the marginal likelihood at the fixed sigma.
    lam = post.prior.tunable().detach()
    best = -post.log_marglik().item()

    for factor in (0.5, 2.0):
        other = _with_prec(post, lam * factor)
        assert -other.log_marglik().item() >= best - 1e-8


def _with_prec(post, prec):
    import copy

    return copy.replace(post, prior=copy.replace(post.prior, prior_precision=prec))


@pytest.mark.parametrize("var", [0.0, -1.0, math.inf, math.nan])
def test_bad_noise_raises(data1, var):
    X, Y = data1

    with pytest.raises(ValueError, match="train_Yvar"):
        LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, train_Yvar=var, trainer=FAST)


def test_float_noise_keeps_its_value_in_float64(data1):
    X, Y = data1
    X, Y = X.double(), Y.double()
    seen = []

    def trainer(net, train_X, train_Y, *, train_Yvar):
        seen.append(train_Yvar)
        return FAST(net.double(), train_X, train_Y, train_Yvar=train_Yvar)

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)  # a float32 detour would round 0.05

    try:
        LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, train_Yvar=0.05, trainer=trainer)
    finally:
        torch.set_default_dtype(old)

    assert seen == [0.05]


def test_noise_varying_over_points_raises(data1):
    X, Y = data1
    var = torch.linspace(0.1, 0.2, Y.shape[0]).unsqueeze(-1)

    with pytest.raises(ValueError, match=r"one noise variance.*got values from 0\.1 to 0\.2"):
        LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, train_Yvar=var, trainer=FAST)


@pytest.mark.parametrize("shape", [(12,), (12, 2), (1, 1), (3,)])
def test_noise_of_another_shape_raises_before_training(data1, shape):
    """`train_Yvar` is a float, a 0-dim tensor or one entry per entry of `train_Y`."""

    X, Y = data1
    var = torch.full(shape, 0.1)

    with pytest.raises(ValueError, match=r"train_Yvar of shape \[12, 1\]"):
        LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, train_Yvar=var, trainer=FAST)


def test_constant_noise_made_per_outcome_by_standardize_says_so(data3):
    """Per-outcome `Standardize` divides one noise variance by each outcome's variance."""

    from botorch.models.transforms.outcome import Standardize

    X, Y = data3
    Y = Y * torch.tensor([1.0, 3.0, 5.0])

    with pytest.raises(ValueError, match="Standardize"):
        LaplaceBNNSurrogate(
            Nets(3, 3)(),
            X,
            Y,
            train_Yvar=0.09,
            trainer=FAST,
            outcome_transform=Standardize(m=3),
        )


def test_data_by_keyword_and_noise_through_the_outcome_transform(data1):
    """As `SingleTaskGP`: `train_Yvar` is in the space of `train_Y`, and `Standardize`
    divides it by the outcome variance."""

    from botorch.models.transforms.outcome import Standardize

    X, Y = data1
    Y = 3.0 * Y
    model = LaplaceBNNSurrogate(
        net=Nets(3, 1)(),
        train_X=X,
        train_Y=Y,
        train_Yvar=torch.full_like(Y, 0.09),
        trainer=FAST,
        outcome_transform=Standardize(m=1),
    )
    want = 0.09 / Y.var().item()

    assert model.noise_variance.item() == pytest.approx(want, rel=1e-5)
    assert model.laplace_posterior.likelihood.sigma.item() == pytest.approx(want**0.5, rel=1e-5)


def test_construct_inputs_builds_the_model(data1):
    from botorch.utils.datasets import SupervisedDataset

    X, Y = data1
    dataset = SupervisedDataset(
        X, Y, Yvar=torch.full_like(Y, 0.04), feature_names=["a", "b", "c"], outcome_names=["y"]
    )
    inputs = LaplaceBNNSurrogate.construct_inputs(dataset, net=Nets(3, 1)(), trainer=FAST)
    model = LaplaceBNNSurrogate(**inputs)

    assert set(inputs) == {"train_X", "train_Y", "train_Yvar", "net", "trainer"}
    assert model.noise_variance.item() == pytest.approx(0.04)

    with pytest.raises(ValueError, match="net="):
        LaplaceBNNSurrogate.construct_inputs(dataset)


def test_last_layer(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(
        Nets(3, 1)(), X, Y, trainer=PosthocMarglik(epochs=5, subset=lk.subsets.last_layer)
    )

    assert set(model.laplace_posterior.mean) == {"2.weight", "2.bias"}


def test_manual_seed_makes_a_fit_repeatable(data1):
    X, Y = data1

    def make():
        return torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))

    torch.manual_seed(4)
    a = LaplaceBNNSurrogate(make(), X, Y, trainer=FAST)
    torch.manual_seed(4)
    b = LaplaceBNNSurrogate(make(), X, Y, trainer=FAST)
    torch.manual_seed(5)
    c = LaplaceBNNSurrogate(make(), X, Y, trainer=FAST)

    assert all(
        torch.equal(a.laplace_posterior.mean[k], b.laplace_posterior.mean[k])
        for k in a.laplace_posterior.mean
    )
    assert not all(
        torch.equal(a.laplace_posterior.mean[k], c.laplace_posterior.mean[k])
        for k in a.laplace_posterior.mean
    )


def test_marglik_trainer(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(
        Nets(3, 1)(), X, Y, trainer=OnlineMarglik(epochs=20, marglik_frequency=5, lr=1e-2)
    )
    post = model.laplace_posterior

    assert isinstance(post.likelihood, lk.Gaussian)
    assert post.likelihood.sigma.item() != 1.0
    assert torch.isfinite(model.posterior(X).mean).all()


def test_marglik_trainer_fixed_sigma(data1):
    X, Y = data1
    model = LaplaceBNNSurrogate(
        Nets(3, 1)(),
        X,
        Y,
        train_Yvar=0.04,
        trainer=OnlineMarglik(epochs=10, marglik_frequency=5, lr=1e-2),
    )

    assert model.laplace_posterior.likelihood.sigma.item() == pytest.approx(0.2)


def test_custom_trainer_gets_the_given_net(data1):
    X, Y = data1
    nets = Nets(3, 1)
    calls = []

    def trainer(net, train_X, train_Y, *, train_Yvar):
        calls.append((net, train_X, train_Y, train_Yvar))
        lik = lk.Gaussian() if train_Yvar is None else lk.Gaussian(sigma=train_Yvar**0.5)

        return lk.laplace(net, [(train_X, train_Y)], lik, subset=lk.subsets.all)

    model = LaplaceBNNSurrogate(nets(), X, Y, trainer=trainer)
    ((net, X_, Y_, var),) = calls

    assert net is nets.made[-1] and var is None
    assert torch.equal(X_, X) and torch.equal(Y_, Y)
    assert model.laplace_posterior.likelihood.sigma.numel() == 1


def test_trainer_must_return_a_gaussian_posterior(data1):
    X, Y = data1

    def trainer(net, train_X, train_Y, *, train_Yvar):
        return lk.laplace(
            net,
            [(train_X, train_Y.squeeze(-1).round().long().clamp(0, 0))],
            lk.Categorical(),
            subset=lk.subsets.all,
        )

    with pytest.raises(lk.StructureError, match="Gaussian"):
        LaplaceBNNSurrogate(Nets(3, 1)(), X, Y, trainer=trainer)


@pytest.mark.parametrize("shapes", [((12, 3), (12,)), ((2, 12, 3), (2, 12, 1)), ((12, 3), (11, 1))])
def test_bad_data_raises(shapes):
    xs, ys = shapes

    with pytest.raises(ValueError, match="train_X|train_Y"):
        LaplaceBNNSurrogate(Nets(3, 1)(), torch.rand(xs), torch.rand(ys), trainer=FAST)


@pytest.mark.parametrize(
    "approx", [lk.KFAC(), lk.DiagGGN(), lk.DenseGGN(), lk.DenseGGN(joint=True)], ids=repr
)
@pytest.mark.parametrize("trainer", [PosthocMarglik, OnlineMarglik])
def test_curvature_methods(data3, approx, trainer):
    X, Y = data3
    t = trainer(epochs=10, lr=1e-2, hessian_approx=approx)
    model = LaplaceBNNSurrogate(Nets(3, 3)(), X, Y, trainer=t)
    post = model.posterior(torch.rand(4, 2, 3))

    assert torch.isfinite(post.mean).all()
    assert (post.variance > 0).all()
