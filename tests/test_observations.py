import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose
from observations import PointObservation, DiagObservation
from scipy.interpolate import interp1d

from tests.helpers import align_eigenvectors

from src.probability import Prior
COLORS = ['r', 'g', 'b', 'k', 'c', 'm', 'y']


@pytest.mark.parametrize("transform", ['fft', 'dct'])
def test_point_observation(transform):
    """Test we can measure an observable correctly."""
    N, L = 2000, 3
    prior = Prior(N=N, L=L, transform=transform, gamma=-1.6)
    u = prior.sample(return_coeffs=False).squeeze()
    measurements = np.linspace(0, L, 33, endpoint=False)
    obs = PointObservation(N=N, L=L, meas=measurements)
    measured = interp1d(prior.x, u)(obs.meas)
    err = np.abs(measured - obs(u)).max()
    if err < 1e-3:
        assert err < 1e-3
    else:
        plt.figure(figsize=(6, 3))
        plt.plot(obs.x, u)
        plt.scatter(obs.meas, obs(u).real, color='r')
        plt.title(f'Max abs err {err}')
        plt.show()
        assert err < 1e-3


@pytest.mark.parametrize("transform", ['fft', 'dct'])
def test_diagonal_observation_eigenvectors(transform):
    """Test that a measurement operator with diagonal multiplier indeed has
    the correct eigenvectors for OstarO (note that this is up to multiplication
    by a complex unit)."""
    np.random.seed(342424)
    n = 3  # len(COLORS)
    singular_values = np.random.randn(n) ** 2
    # singular_values[1] = 0
    obs = DiagObservation(singular_values=singular_values,
                          N=200,
                          random_U=True,
                          transform=transform)

    D, P = np.linalg.eig(obs.OstarO)
    xtra = P[:, n]
    P = P[:, :n].T.conjugate()
    P = align_eigenvectors(P)

    eigs = np.vstack([obs.eigenvector(i) for i in range(n)])
    eigs = align_eigenvectors(eigs)

    assert_allclose(np.linalg.norm(eigs, axis=1), 1)
    assert_allclose(np.linalg.norm(P, axis=1), 1)
    assert P.shape == eigs.shape

    errors = np.abs(P - eigs)
    err = np.max(errors)
    if err < 1e-5:
        assert True
    else:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
        for i in range(n):
            color = COLORS[i]
            p = P[i, :]
            eig = eigs[i, :]
            real_diff = np.max(np.abs(p.real - eig.real))
            imag_diff = np.max(np.abs(p.imag - eig.imag))
            real_diff = f'{i} Error = {real_diff:.4f}'
            imag_diff = f'{i} Error = {imag_diff:.4f}'
            ax[0, 0].plot(obs.x, p.real, color='r', label=i)
            ax[0, 1].plot(obs.x, p.imag, color='r', label=i)
            ax[1, 0].plot(obs.x, eig.real, color=color, linestyle='-', label=real_diff)
            ax[1, 0].plot(obs.x, p.real, color=color, linestyle=':')
            ax[1, 1].plot(obs.x, eig.imag, color=color, linestyle='-', label=imag_diff)
            ax[1, 1].plot(obs.x, p.imag, color=color, linestyle=':')
        ax[0, 0].plot(obs.x, xtra, label='$n+1$', color='k')
        ax[0, 1].plot(obs.x, xtra, label='$n+1$', color='k')
        ax[1, 0].legend()
        ax[1, 1].legend()
        OstarO = "OstarO"  # r"$\mathcal{O}^{*}\mathcal{O}$"
        # ax[0].set_title(f"First $n={n}$ eigs of a diagonalizable " + OstarO)
        ax[0, 0].set_title(f"First $n+1={n + 1}$ eigs of " + OstarO + '  (real)')
        ax[0, 1].set_title(f"First $n+1={n + 1}$ eigs of " + OstarO + '  (imaginary)')
        ax[1, 0].set_title(f"First $n={n}$ modes of {obs.transform} (real)")
        ax[1, 1].set_title(f"First $n={n}$ modes of {obs.transform} (imaginary)")

        # ind = np.where(np.abs(D) > 1e-9)[0]
        # ax[3].plot(np.arange(n), D[:n], label="e_i") # "'$\mathbf{e}_i$')
        # ax[3].plot(np.arange(n), np.zeros(n), label='y=0')
        # ax[3].set_title(f"Nonzero eigenvalues: {ind}")

        plt.tight_layout()
        plt.show()
        assert False

