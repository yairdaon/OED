import numpy as np
import pytest
from matplotlib import pyplot as plt

from forward import Heat
from numpy.testing import assert_allclose
from observations import PointObservation
from probability import Prior, Posterior


@pytest.mark.parametrize("transform", ['dct'])#, 'fft'])
def test_prior_sample(transform):
    np.random.seed(253234236)
    N = 30
    L = 3
    gamma = -.6
    n_sample = 500000

    prior = Prior(gamma=gamma, N=N, L=L, transform=transform)
    sample, coeff = prior.sample(return_coeffs=True, n_sample=n_sample)
    assert sample.shape == (n_sample, N)
    assert coeff.shape == (n_sample, N)

    mean = np.mean(sample, axis=0)
    assert mean.shape == (N,)
    assert_allclose(mean, np.zeros_like(mean), rtol=0, atol=1e-2)

    empiric_covariance = np.einsum('ki, kj-> ij', sample.conjugate(), sample) / n_sample
    assert empiric_covariance.shape == (N, N)
    assert_allclose(empiric_covariance, empiric_covariance.conjugate().T)

    empiric_eigenvalues, empiric_basis = np.linalg.eigh(empiric_covariance)
    multiplier = np.sort(prior.multiplier)
    empiric_eigenvalues.sort()
    assert_allclose(multiplier, empiric_eigenvalues, rtol=0, atol=1e-3)

    # covariance = prior.mult2time(np.diag(prior.multiplier))
    # assert covariance.shape == (N, N)
    # assert_allclose(covariance, empiric_covariance, rtol=0, atol=1e-2)

    # fwd = Heat(N=N, L=L, alpha=alpha, time=time)
    # prior = Prior(gamma=gamma, N=N, L=L)
    # meas = [0.2356323, 0.9822345, 1.451242, 1.886632215, 2.43244,
    #         2.89235633, 1, 1.2]
    # obs = PointObservation(meas=meas, L=L, N=N)
    #
    # # IC
    # u0, coeffs0 = prior.sample(return_coeffs=True)
    #
    # # Analytic
    # coeffsT = coeffs0 * fwd.multiplier
    # uT = prior.coeff2u(coeffsT)
    #
    # # Numeric solution
    # uT_numeric = fwd(u0)
    #
    # numeric_success = np.allclose(uT, uT_numeric, rtol=0, atol=1e-3)
    # inversion_success = np.allclose(prior(prior.inverse(uT)), prior.inverse(prior(uT)), rtol=0, atol=1e-3)
    #
    # interpolant = interp1d(fwd.x, uT)
    # measure_success = np.allclose(obs(uT), interpolant(obs.meas), atol=1e-2, rtol=0)


def test_posterior():
    np.random.seed(134567)
    sig = 0.0005
    N = 400
    time = 5e-2
    L = 2
    alpha = 0.6
    gamma = -0.6

    meas = np.linspace(0.05, L - 0.05, endpoint=False, num=700)
    meas += np.random.normal(scale=0.01, size=meas.size)
    obs = PointObservation(meas=meas, L=L, N=N)
    fwd = Heat(N=N, L=L, alpha=alpha, time=time)
    prior = Prior(gamma=gamma, N=N, L=L)
    prior.multiplier[4:] = 0
    post = Posterior(fwd=fwd, prior=prior, sigSqr=sig ** 2, L=L, N=N)
    u0 = prior.sample(return_coeffs=False)
    uT = fwd(u0)
    data = obs(uT)  # + np.random.normal(scale=sig, size=obs.size)
    post.update(obs, data)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax[0].plot(fwd.x, u0.real, label='IC')
    ax[0].plot(fwd.x, uT.real, label='FC')
    ax[0].scatter(obs.meas, np.dot(post.A, post.to_freq_domain(u0)).real, label='Matrix FC')
    ax[0].scatter(obs.meas, data.real, label='Measurements', marker='*', s=10, color='w', zorder=10)
    line, = ax[0].plot(post.x, post.m, label='Posterior mean')
    # ax[0].plot(post.x, post.m + 2*post.ptwise, color=line.get_color(), label='Posterior std', linestyle=':')
    # ax[0].plot(post.x, post.m - 2*post.ptwise, color=line.get_color(), linestyle=':')
    ax[0].legend()
    ax[0].set_title("Error bars seem too small, no?")

    ax[1].plot(post.x, post.ptwise.real, label='posterior STD')
    ax[1].scatter(obs.meas, np.zeros(obs.size), label='measurements')
    # print(np.diag(post.Sigma)[:9])
    # tra = post.to_freq_domain(post.m)
    # plt.close()
    # plt.plot(tra)
    plt.show()