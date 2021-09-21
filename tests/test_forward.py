import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from src.forward import Heat
from src.observations import PointObservation
from src.probability import Prior


@pytest.mark.parametrize("transform", ['dst', 'dct', 'fft'])
def test_heat(transform):
    """Check that analytic and numerical solution are identical.
    Check that we can measure correctly also."""
    # np.random.seed(532842)
    N = 300
    L = 3
    time = 3e-3
    alpha = 0.6
    gamma = -.6

    fwd = Heat(N=N, L=L, alpha=alpha, time=time, transform=transform)
    prior = Prior(gamma=gamma, N=N, L=L, transform=transform)
    meas = [0.23563, 0.9822345, 1.451242, 1.886632215,
            2.43244, 2.8923563, 1.0, 1.2]
    obs = PointObservation(measurements=meas, L=L, N=N, transform=transform)

    # IC
    u0, coeffs0 = prior.sample(return_coeffs=True)
    u0 = np.squeeze(u0)

    # Analytic
    coeffsT = coeffs0 * fwd.multiplier
    uT = prior.to_time_domain(coeffsT).squeeze()
    # uT = np.squeeze(uT)
    # Numeric solution
    uT_numeric = fwd(u0)

    numeric_success = np.allclose(uT, uT_numeric, rtol=0, atol=1e-3)
    inversion_success = np.allclose(prior(prior.inverse(uT)), prior.inverse(prior(uT)), rtol=0, atol=1e-3)

    interpolant = interp1d(fwd.x, uT)
    measure_success = np.allclose(obs(uT), interpolant(obs.measurements), atol=1e-2, rtol=0)

    success = numeric_success and inversion_success and measure_success
    if success:
        assert success
    else:
        plt.plot(fwd.x, u0.real, label='IC')
        plt.plot(fwd.x, uT.real, label='Analytic FC')
        plt.plot(fwd.x, uT_numeric.real + 0.025, label='Numeric FC')
        plt.plot(prior.x, prior(prior.inverse(uT)).real + 0.05, label=str(prior) + prior.invstr + 'FC')
        plt.plot(prior.x, prior.inverse(prior(uT)).real + 0.075, label=prior.invstr + str(prior) + 'FC')
        plt.plot(prior.x, prior(uT).real, label=str(prior) + 'FC')
        # plt.plot(prior.x, prior.inverse(uT), label= prior.inv_str + 'FC')
        plt.scatter(obs.measurements, obs(uT).real, label='Measurements of FC', marker='*', s=10, color='w', zorder=10)
        plt.legend()
        title = f'{transform} successes: numeric {numeric_success} inversion {inversion_success} measure {measure_success}'
        plt.title(title)
        plt.show()
        assert success

