import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose

from src.forward import Heat
from src.multiplier import FourierMultiplier
from src.observations import DiagObservation

COLORS = ['r', 'g', 'b', 'k', 'c', 'm', 'y']


@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_orthogonality(transform):
    fwd = Heat(L=3, N=30, transform=transform, time=1e-2, alpha=0.5)

    identity = np.eye(fwd.N)
    forward = fwd.to_freq_domain(identity)
    assert_allclose(np.dot(forward, forward.conjugate().T), identity, atol=1e-9, rtol=0)

    inverse = fwd.to_time_domain(identity)
    assert_allclose(np.dot(inverse, inverse.conjugate().T), identity, atol=1e-9, rtol=0)

    assert_allclose(np.dot(inverse, forward), identity, atol=1e-9, rtol=0)
    assert_allclose(np.dot(forward, inverse), identity, atol=1e-9, rtol=0)


@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_eigen_norm(transform):
    fwd = Heat(L=3, N=30, transform=transform, time=1e-2, alpha=0.5)
    for i in range(fwd.N):
        eigenfunction = fwd.eigenvector(i)
        assert not np.any(np.isnan(eigenfunction))


@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_eigenfunction(transform):
    n = len(COLORS)
    singular_values = np.random.randn(n) ** 2
    # singular_values[1] = 0
    obs = DiagObservation(singular_values=singular_values,
                          N=200,
                          random_U=True,
                          transform=transform)
    k = obs.N // 7

    D, P = np.linalg.eig(obs.OstarO)
    xtra = P[:, n]
    P = P[:, :n].T
    P = np.einsum('ij, i -> ij', P, np.exp(-1j * np.angle(P[:, k])))
    assert np.all(P[:, k] > 0)
    P = P[P[:, k].argsort()]

    eigs = np.vstack([obs.eigenvector(i) for i in range(n)])
    eigs = np.einsum('ij, i -> ij', eigs, np.exp(-1j * np.angle(eigs[:, k])))
    assert np.all(eigs[:, k] > 0)
    eigs = eigs[eigs[:, k].argsort()]

    assert_allclose(np.linalg.norm(eigs, axis=1), 1)
    assert_allclose(np.linalg.norm(P, axis=1), 1)
    assert P.shape == eigs.shape

    err = np.max(np.abs(P - eigs))
    if err > 1e-5:
        assert True
    else:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
        for i in range(n):
            color = COLORS[i]
            ax[0].plot(obs.x, P[i, :], label=i, color='r')
            ax[1].plot(obs.x, eigs[i, :], color=color, linestyle='-')
            ax[1].plot(obs.x, 0.001 + P[i, :], color, linestyle=':')
        ax[0].plot(obs.x, xtra, label='$n+1$', color='k')

        OstarO = "OstarO" # r"$\mathcal{O}^{*}\mathcal{O}$"
        #ax[0].set_title(f"First $n={n}$ eigs of a diagonalizable " + OstarO)
        ax[0].set_title(f"First $n+1={n + 1}$ eigs of " + OstarO)
        ax[1].set_title(f"First $n={n}$ modes of {obs.transform}")

        # ind = np.where(np.abs(D) > 1e-9)[0]
        # ax[3].plot(np.arange(n), D[:n], label="e_i") # "'$\mathbf{e}_i$')
        # ax[3].plot(np.arange(n), np.zeros(n), label='y=0')
        # ax[3].set_title(f"Nonzero eigenvalues: {ind}")

        plt.tight_layout()
        plt.show()
        assert False


@pytest.mark.parametrize("transform", ['dct', 'fft', 'dst'])
def test_eigenvector(transform):
    multiplier = FourierMultiplier(N=100, L=1.53, transform=transform)
    for i in range(multiplier.N):
        assert_allclose(multiplier.eigenvector(i), multiplier.eigenfunction(i)(multiplier.x), rtol=0, atol=1e-12)


def test_coeff2u():
    pass


def test_mult2time():
    pass

