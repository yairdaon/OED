from itertools import product

import numpy as np
import pytest
from joblib import delayed, Parallel
from matplotlib import pyplot as plt

from forward import Heat
from numpy import allclose
from numpy.testing import assert_allclose
from observations import PointObservation
from probability import Prior, Posterior
from scipy.interpolate import interp1d
from scipy.linalg import solve


@pytest.fixture
def prior(transform):
    N = 2000
    L = 3
    gamma = -2
    return Prior(gamma=gamma, N=N, L=L, transform=transform)

@pytest.fixture
def short_prior(transform):
    N = 200
    L = 3
    gamma = -1
    return Prior(gamma=gamma, N=N, L=L, transform=transform)


@pytest.mark.parametrize("transform", ['dct', 'dst', 'fft'])
def test_prior_matrix(short_prior):
    matrix = short_prior.matrix
    plt.imshow(matrix.real)
    plt.show()


@pytest.mark.parametrize("transform", ['dct', 'fft', 'dst'])
def test_prior_sample_from_coefficients(prior):

    sample, coefficients = prior.sample(return_coeffs=True)
    sample = sample.squeeze()
    sample_from_cefficients = prior.to_time_domain(coefficients.squeeze())
    assert_allclose(sample, sample_from_cefficients)


@pytest.mark.parametrize("transform", ['dct', 'fft', 'dst'])
def test_prior_sample_mean(short_prior):
    """Test that sampling the prior we get zero mean."""
    n_sample = 25000
    k = 12
    get_mean = lambda: short_prior.sample(return_coeffs=False, n_sample=n_sample).mean(axis=0)
    sample = sum(Parallel(n_jobs=7)(delayed(get_mean)() for _ in range(k))) / k
    assert sample.shape == (short_prior.N,)

    assert_allclose(sample, np.zeros_like(sample), rtol=0, atol=1e-2)


@pytest.mark.parametrize("transform", ['dct', 'fft', 'dst'])
def test_prior_sample_covariance(short_prior):
    """Test the empirical covariance is the same as the matrix representation of the prior.
    TODOs - check eigenvectors and eigenvalues agree???

    We compare the empiric covariance with the operator. The covariance operator C
    is defined for a Hilbert space H and a corresponding measure P as (see Wikipedia)

    C: H -> H , <Cx, y> = \int_H <x,z><y,z> dP(z).

    We think of H as as discretization of L2. H contains functions that
    are constant on every interval [nh, nh+h) and the inner product is

    <x,y> = \int_0^L x(u)y*(u) du = h sum_i=1^N x_iy_i*.

    Thus (\dot is the linear algebraic dot product):

    h sum (Cx)_iy_i* = <Cx, y> = h^2 \int_H x\dotz y \dot z dP(z).

    Taking x = e_i and y = e_j, we get that

    C_ij = h E[z_i z_j].

    On the left - the covariance operator. On the right - the empiric covariance. Note the h factor
    premultiplying the expectation."""
    n_sample = 200000 if short_prior.transform == 'dst' else 50000
    k = 60 if short_prior.transform == 'dst' else 40
    function = lambda: np.cov(short_prior.sample(n_sample=n_sample), rowvar=False)
    empiric_covariance = sum(Parallel(n_jobs=7)(delayed(function)() for _ in range(k))) / k

    assert empiric_covariance.shape == (short_prior.N, short_prior.N)
    assert_allclose(empiric_covariance, empiric_covariance.conjugate().T)

    # See details in the docs above concerning the factor h.
    assert_allclose(short_prior.matrix, empiric_covariance * short_prior.h, rtol=0, atol=1e-4)


@pytest.mark.parametrize("transform", ['dct', 'dst', 'fft'])
def test_prior_pointwise_std(prior):
    Sigma = np.einsum('i, ij->ij', prior.multiplier, prior.to_freq_domain(np.eye(prior.N), axis=0))
    Sigma = prior.to_time_domain(Sigma, axis=0)
    std = np.sqrt(np.diag(Sigma))
    plt.plot(std)
    plt.title(f'transform = {prior.transform}')
    plt.xlabel('x')
    plt.ylabel('prior pointwise standard deviation')
    plt.show()


@pytest.mark.parametrize("transform", ['dct', 'fft', 'dst'])
def test_posterior_utility(prior):
    sig = 1e-2
    time = 2e-2
    alpha = 0.6
    L, N, transform = prior.L, prior.N, prior.transform

    meas = np.random.uniform(low=0, high=L, size=50)

    obs = PointObservation(measurements=meas, L=L, N=N, transform=transform)
    fwd = Heat(N=N, L=L, alpha=alpha, time=time, transform=transform)
    post = Posterior(fwd=fwd,
                     prior=prior,
                     sigSqr=sig**2,
                     L=L,
                     N=N,
                     transform=transform)
    info_gain = post.utility(obs)
    assert info_gain > 0


@pytest.mark.parametrize("transform", ['dct', 'fft', 'dst'])
def test_posterior(prior):

    sig = 1e-2
    time = 2e-2
    alpha = 0.6
    L, N, transform = prior.L, prior.N, prior.transform

    # meas = np.random.uniform(low=0, high=L, size=33)
    # meas = np.array([prior.L/2, prior.L/2.1])
    meas = np.random.choice(prior.x, size=33, replace=True)
    obs = PointObservation(measurements=meas, L=L, N=N, transform=transform)
    fwd = Heat(N=N, L=L, alpha=alpha, time=time, transform=transform)
    post = Posterior(fwd=fwd,
                     prior=prior,
                     sigSqr=sig**2,
                     L=L,
                     N=N,
                     transform=transform)
    u0 = prior.sample(return_coeffs=False).squeeze()

    uT = fwd(u0)
    data = obs(uT) + np.random.normal(scale=sig, size=obs.shape[0])
    mean, pointwise_std = post.mean_std(obs, data)

    err = np.abs(mean - u0).max()
    if err < 1e-2:
        assert err < 1e-2
    else:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 16), sharex=True)
        ax[0].plot(fwd.x, u0.real, label='IC')
        ax[0].plot(fwd.x, uT.real, label='FC')
        ax[0].scatter(obs.measurements, np.dot(post.A, post.to_freq_domain(u0)).real, label='Matrix FC')
        ax[0].scatter(obs.measurements, data.real, label='Measurements', marker='*', s=10, color='r', zorder=10)
        line, = ax[0].plot(post.x, mean, label='Posterior mean')
        ax[0].plot(post.x, mean + 2*pointwise_std, color=line.get_color(), label='95% Posterior Interval', linestyle=':')
        ax[0].plot(post.x, mean - 2*pointwise_std, color=line.get_color(), linestyle=':')
        ax[0].legend()

        ax[1].plot(post.x, pointwise_std, label='posterior STD')
        ax[1].scatter(obs.measurements, np.ones(obs.shape[0]) * pointwise_std.mean(), label='measurement locations on x-axis')
        ax[1].legend()
        fig.suptitle(f"Transform = {transform}, error = {err:.4f}")

        # print(np.diag(post.Sigma)[:9])
        # tra = post.to_freq_domain(post.m)
        # plt.close()
        # plt.plot(tra)
        plt.tight_layout()
        plt.show()
