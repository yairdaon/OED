import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

from observations import PointObservation, DiagObservation

COLORS = ['r', 'g', 'b', 'k', 'c', 'm', 'y']

def test_point_observation():
    k = 7
    obs = PointObservation(N=1400, L=2, meas=np.linspace(0.1, 1.9, 50, endpoint=False))
    u = obs.eigenvector(k)
    measured = interp1d(obs.x, u)(obs.meas)

    if np.allclose(measured, obs(u), rtol=0, atol=1e-3):
        assert True
    else:
        plt.figure(figsize=(6, 3))
        plt.plot(obs.x, u)
        plt.scatter(obs.meas, obs(u).real)
        plt.show()


@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_diagonal_observation_eigenvectors(transform):

    n = 3  # len(COLORS)
    singular_values = np.random.randn(n) ** 2
    # singular_values[1] = 0
    obs = DiagObservation(singular_values=singular_values,
                          N=200,
                          random_U=True,
                          transform=transform)
    k = obs.N // 5

    D, P = np.linalg.eig(obs.OstarO)
    xtra = P[:, n]
    P = P[:, :n].T.conjugate()
    P = np.einsum('ij, i -> ij', P, np.sign(P[:, k].real))
    assert np.all(P[:, k].real > 0)
    P = P[np.argsort(P[:, k].real)]

    eigs = np.vstack([obs.eigenvector(i) for i in range(n)])
    eigs = np.einsum('ij, i -> ij', eigs, np.sign(eigs[:, k].real))
    assert np.all(eigs[:, k].real > 0)
    eigs = eigs[np.argsort(eigs[:, k].real)]

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
            real_diff = f'{i} {real_diff:.4f}'
            imag_diff = f'{i} {imag_diff:.4f}'
            ax[0, 0].plot(obs.x, p.real, color='r', label=i)
            ax[0, 1].plot(obs.x, p.imag, color='r', label=i)
            ax[1, 0].plot(obs.x, eig.real, color=color, linestyle='-', label=real_diff)
            ax[1, 0].plot(obs.x, 0.005 + p.real, color=color, linestyle=':')
            ax[1, 1].plot(obs.x, eig.imag, color=color, linestyle='-', label=imag_diff)
            ax[1, 1].plot(obs.x, 0.005 + p.imag, color=color, linestyle=':')
        ax[0, 0].plot(obs.x, xtra, label='$n+1$', color='k')
        ax[0, 1].plot(obs.x, xtra, label='$n+1$', color='k')
        ax[1, 0].legend()
        ax[1, 1].legend()
        OstarO = "OstarO"  # r"$\mathcal{O}^{*}\mathcal{O}$"
        # ax[0].set_title(f"First $n={n}$ eigs of a diagonalizable " + OstarO)
        ax[0, 0].set_title(f"First $n+1={n + 1}$ eigs of " + OstarO + '  (real)')
        ax[0, 1].set_title(f"First $n+1={n + 1}$ eigs of " + OstarO + '  (imaginary)')
        ax[1, 0].set_title(f"First $n={n}$ modes of {obs.transform}")
        ax[1, 1].set_title(f"First $n={n}$ modes of {obs.transform}")

        # ind = np.where(np.abs(D) > 1e-9)[0]
        # ax[3].plot(np.arange(n), D[:n], label="e_i") # "'$\mathbf{e}_i$')
        # ax[3].plot(np.arange(n), np.zeros(n), label='y=0')
        # ax[3].set_title(f"Nonzero eigenvalues: {ind}")

        plt.tight_layout()
        plt.show()
        assert False

