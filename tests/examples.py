import numpy as np
import pytest

from src.forward import Heat
from src.multiplier import FourierMultiplier
from src.observations import DiagObservation, PointObservation
from src.probability import Prior, Posterior

N = 600
L = 3
time = 3e-3
alpha = 0.6
gamma = -.6


@pytest.fixture
def multiplier(transform):
    """Create a multiplier object for testing with eigenvalues
    (10, 9, ..., 1, 0, 0, 0, ...)"""
    multiplier = FourierMultiplier(L=L, N=N, transform=transform)
    multiplier.multiplier = np.zeros(multiplier.N)
    multiplier.multiplier[:10] = np.arange(10, 0, -1)
    return multiplier


@pytest.fixture
def diag_obs(transform):
    obs = DiagObservation(multiplier=np.abs(np.random.randn(6)),
                          N=N,
                          L=L,
                          transform=transform)
    return obs


@pytest.fixture
def prior(transform):
    return Prior(gamma=gamma, N=N, L=L, transform=transform)


@pytest.fixture
def short_prior(transform):
    return Prior(gamma=gamma, N=300, L=L, transform=transform)


@pytest.fixture
def fwd(transform):
    return Heat(N=N, L=L, alpha=alpha, time=time, transform=transform)


@pytest.fixture
def point_observation(transform):
    meas = [0.23563, 0.9822345, 1.451242, 1.886632215,
            2.43244, 2.8923563, 1.0, 1.2]
    return PointObservation(measurements=meas, L=L, N=N, transform=transform)


@pytest.fixture
def many_point_observation(transform):
    return PointObservation(measurements=np.random.uniform(0, L, 11),
                            L=L,
                            N=10000,
                            transform=transform)


@pytest.fixture
def posterior(transform):
    sig = 1e-2
    forward = Heat(N=N, L=L, transform=transform, alpha=alpha, time=time)
    pr = Prior(N=N, L=L, transform=transform, gamma=gamma)
    return Posterior(fwd=forward,
                     prior=pr,
                     sigSqr=sig**2,
                     L=L,
                     N=N,
                     transform=transform)
