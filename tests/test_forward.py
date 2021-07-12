import numpy as np
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d

from forward import Heat
from observations import PointObservation
from probability import Prior


def test_heat(plot=False):
    np.random.seed(5842)
    N = 500
    L = 3
    time = 3e-3
    alpha = 0.6
    gamma = -.6

    fwd = Heat(N=N, L=L, alpha=alpha, time=time)
    prior = Prior(gamma=gamma, N=N, L=L)
    obs = PointObservation(meas=[0.2356323, 0.9822345, 1.451242, 1.886632215, 2.43244, 2.89235633, 1, 1.2], L=L, N=N)

    # IC
    u0, coeffs0 = prior.sample(return_coeffs=True)

    # Analytic
    coeffsT = coeffs0 * fwd.multiplier
    uT = prior.coeff2u(coeffsT)

    # Numeric solution
    uT_numeric = fwd(u0)

    assert_allclose(uT, uT_numeric, rtol=1e-5, atol=1e-9)
    assert_allclose(prior(prior.inverse(uT)), prior.inverse(prior(uT)), rtol=0, atol=1e-9)

    interpolant = interp1d(fwd.x, uT)
    assert_allclose(obs(uT), interpolant(obs.meas))

    if plot:
        plt.plot(fwd.x, u0.real, label='IC')
        plt.plot(fwd.x, uT.real, label='Analytic FC')
        plt.plot(fwd.x, uT_numeric.real + 0.025, label='Numeric FC')
        plt.plot(prior.x, prior(prior.inverse(uT)).real + 0.05, label=str(prior) + prior.invstr + 'FC')
        plt.plot(prior.x, prior.inverse(prior(uT)).real + 0.075, label=prior.invstr + str(prior) + 'FC')
        plt.plot(prior.x, prior(uT).real, label=str(prior) + 'FC')
        # plt.plot(prior.x, prior.inverse(uT), label= prior.inv_str + 'FC')
        plt.scatter(obs.meas, obs(uT).real, label='Measurements of FC', marker='*', s=10, color='w', zorder=10)
        plt.legend()
        plt.show()

