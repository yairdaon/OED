from itertools import product

import numpy as np
import pytest
from joblib import delayed, Parallel
from matplotlib import pyplot as plt

from forward import Heat
from numpy.testing import assert_allclose
from observations import PointObservation
from probability import Prior, Posterior
from scipy.interpolate import interp1d


@pytest.fixture
def prior(transform):
    np.random.seed(253234236)
    N = 500
    L = 3
    gamma = -.6
    return Prior(gamma=gamma, N=N, L=L, transform=transform)

@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_prior_sample_from_coefficients(prior):

    sample, coefficients = prior.sample(return_coeffs=True)
    sample = sample.squeeze()
    sample_from_cefficients = prior.coeff2u(coefficients.squeeze())
    assert_allclose(sample, sample_from_cefficients)



@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_prior_sample_mean(prior):
    """Test that sampling the prior we get zero mean."""
    n_sample = 500000

    sample, coeff = prior.sample(return_coeffs=True, n_sample=n_sample)
    assert sample.shape == (n_sample, prior.N)
    assert coeff.shape == (n_sample, prior.N)

    mean = np.mean(sample, axis=0)
    assert mean.shape == (prior.N,)
    assert_allclose(mean, np.zeros_like(mean), rtol=0, atol=1e-2)


@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_prior_sample_covariance(prior):
    """Test the empirical covariance is the same as the matrix representation of the prior.
    TODOs - check eigenvectors and eigenvalues agree???"""
    n_sample = 50000
    k = 40
    function = lambda: np.cov(prior.sample(n_sample=n_sample), rowvar=False)
    empiric_covariance = sum(Parallel(n_jobs=6)(delayed(function)() for _ in range(k))) / k

    assert empiric_covariance.shape == (prior.N, prior.N)
    assert_allclose(empiric_covariance, empiric_covariance.conjugate().T)

    operator_covariance = prior.matrix
    empiric_ratio = np.mean(operator_covariance / empiric_covariance)
    assert abs(empiric_ratio - prior.h) < 1e-4
    assert_allclose(operator_covariance, empiric_covariance * prior.h, rtol=0, atol=1e-4)
    # operator_covariance ~ empiric_covariance * h

@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_prior_covariance_multiplier(prior):
    operator_covariance = prior.matrix
    covariance = prior.mult2time(np.diag(prior.multiplier))
    assert covariance.shape == (prior.N, prior.N)
    assert_allclose(covariance, operator_covariance, rtol=0, atol=1e-9)

@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_posterior(prior):

    sig = 0.0000005
    time = 5e-2
    alpha = 0.6
    L, N, transform = prior.L, prior.N, prior.transform

    meas = np.linspace(0.05, L - 0.05, endpoint=False, num=7)
    meas += np.random.normal(scale=0.01, size=meas.size)
    meas = np.random.uniform(low=0.05, high=L-0.05, size=79)

    np.random.seed(134567)
    obs = PointObservation(meas=meas, L=L, N=N, transform=transform)
    fwd = Heat(N=N, L=L, alpha=alpha, time=time, transform=transform)
    prior.multiplier[4:] = 0
    post = Posterior(fwd=fwd,
                     prior=prior,
                     sigSqr=sig**2,
                     L=L,
                     N=N,
                     transform=transform)
    u0 = prior.sample(return_coeffs=False).squeeze()

    uT = fwd(u0)
    data = obs(uT) # + np.random.normal(scale=sig, size=obs.shape[0])
    post.update(obs, data)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    ax[0].plot(fwd.x, u0.real, label='IC')
    ax[0].plot(fwd.x, uT.real, label='FC')
    ax[0].scatter(obs.meas, np.dot(post.A, post.to_freq_domain(u0)).real, label='Matrix FC')
    ax[0].scatter(obs.meas, data.real, label='Measurements', marker='*', s=10, color='r', zorder=10)
    line, = ax[0].plot(post.x, post.m, label='Posterior mean')
    # ax[0].plot(post.x, post.m + 2*post.ptwise, color=line.get_color(), label='Posterior std', linestyle=':')
    # ax[0].plot(post.x, post.m - 2*post.ptwise, color=line.get_color(), linestyle=':')
    ax[0].legend()

    ax[1].plot(post.x, post.ptwise.real, label='posterior STD')
    ax[1].scatter(obs.meas, np.zeros(obs.shape[0]), label='measurement locations on x-axis')
    ax[1].legend()
    fig.suptitle(f"Transform = {transform}")

    # print(np.diag(post.Sigma)[:9])
    # tra = post.to_freq_domain(post.m)
    # plt.close()
    # plt.plot(tra)
    plt.show()
