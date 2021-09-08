import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy import allclose
from numpy.testing import assert_allclose
from probability import Prior
from scipy.sparse.linalg import eigsh, gmres, cg

from src.multiplier import FourierMultiplier

from tests.helpers import align_eigenvectors

COLORS = ['r', 'g', 'b', 'k', 'c', 'm', 'y']


@pytest.fixture
def multiplier(transform):
    """Create a multiplier object for testing with eigenvalues
    (11, 10, ..., 1, 0, 0, 0, ...)"""
    multiplier = FourierMultiplier(L=3, N=500, transform=transform)
    multiplier.multiplier = np.zeros(multiplier.N)
    multiplier.multiplier[:10] = np.arange(10, 0, -1)
    return multiplier


@pytest.mark.parametrize('transform', ['dct', 'fft'])
def test_matvec(multiplier):
    v = np.ones(multiplier.N)
    matvec = multiplier(v)
    assert matvec.max() - matvec.min() < 1e-9


@pytest.mark.parametrize('transform', ['dct', 'fft'])
def test_matmat(multiplier):
    """Test matmat is the same as matvec on every column"""
    M = multiplier.normal(n_sample=10).T
    assert M.shape == (multiplier.N, 10)
    matmat = multiplier(M)
    matvec = np.vstack([multiplier(col) for col in M.T]).T
    assert_allclose(matmat, matvec)


@pytest.mark.parametrize('transform', ['dct', 'fft'])
def test_matrix_hermitian(multiplier):
    """ Test matrix representation is hermitian"""
    matrix = multiplier.matrix
    assert_allclose(matrix, matrix.conjugate().T)


@pytest.mark.parametrize('transform', ['dct', 'fft'])
def test_linearoperator(multiplier):
    b = sum([np.random.randn() * multiplier.eigenvector(i) for i in np.where(multiplier.multiplier)[0]])
    x, _ = cg(multiplier, b)
    b_hat = multiplier(x)
    assert_allclose(b, b_hat)

    x, _ = gmres(multiplier, b)
    b_hat = multiplier(x)
    assert_allclose(b, b_hat)


@pytest.mark.parametrize('transform', ['dct', 'fft'])
def test_orthogonality(multiplier):
    identity = np.eye(multiplier.N)
    forward = multiplier.to_freq_domain(identity, axis=0)
    assert_allclose(np.dot(forward, forward.conjugate().T), identity, atol=1e-9, rtol=0)

    inverse = multiplier.to_time_domain(identity, axis=0)
    assert_allclose(np.dot(inverse, inverse.conjugate().T), identity, atol=1e-9, rtol=0)

    assert_allclose(np.dot(inverse, forward), identity, atol=1e-9, rtol=0)
    assert_allclose(np.dot(inverse, forward), identity, atol=1e-9, rtol=0)

    vector = multiplier.normal().squeeze()
    assert_allclose(np.dot(forward, vector), multiplier.to_freq_domain(vector), rtol=0, atol=1e-9)
    assert_allclose(np.dot(inverse, vector), multiplier.to_time_domain(vector), rtol=0, atol=1e-9)


@pytest.mark.parametrize('transform', ['dct', 'fft'])
def test_eigenvectors_agree(multiplier):
    for i in range(multiplier.N//2, multiplier.N):
        eigenvector = np.zeros(multiplier.N)
        eigenvector[i] = 1
        eigenvector = multiplier.to_time_domain(eigenvector)
        # eigenvector = eigenvector / np.linalg.norm(eigenvector)
        class_eigenvector = multiplier.eigenvector(i)
        diff = np.abs(eigenvector - class_eigenvector)
        assert allclose(diff, 0, atol=1e-3, rtol=0)


@pytest.mark.parametrize('transform', ['fft','dct'])
def test_basis_matrix_agree(multiplier):
    identity = np.eye(multiplier.N)
    forward = multiplier.to_time_domain(identity, axis=0)
    U = np.vstack(multiplier.eigenvector(i) for i in range(multiplier.N)).T
    assert_allclose(U, forward, atol=1e-9, rtol=0)
    assert_allclose(np.dot(U.conjugate().T, forward), identity, atol=1e-9, rtol=0)


@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_eigen_norm(multiplier):
    """ Verify norm of eigenfunctions is 1 in the L2 sense and norm
    of eigenvector is 1 in the standaed linear algebraic sense"""
    for i in range(multiplier.N):
        eigenvector = multiplier.eigenvector(i)
        assert not np.any(np.isnan(eigenvector))
        assert abs(np.linalg.norm(eigenvector) - 1) < 1e-9

        eigenfunction = multiplier.eigenfunction(i)(multiplier.x)
        assert abs(multiplier.norm(eigenfunction) - 1) < 1e-9


@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_eigenfunction(multiplier):

    number_eigenvectors = 4
    D, P = eigsh(multiplier, which='LM', k=number_eigenvectors)
    assert P.shape == (multiplier.N, number_eigenvectors)
    P = P[:, :number_eigenvectors].T
    P = align_eigenvectors(P)

    eigs = np.vstack([multiplier.eigenvector(i) for i in range(multiplier.N)])
    eigs = eigs[:number_eigenvectors,:]
    eigs = align_eigenvectors(eigs)

    assert P.shape == eigs.shape
    assert_allclose(np.linalg.norm(eigs, axis=1), 1)
    assert_allclose(np.linalg.norm(P, axis=1), 1)

    errors = np.abs(P - eigs)
    err = np.max(errors)
    if err < 1e-5:
        assert True
    else:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
        for i in range(number_eigenvectors):
            color = COLORS[i]
            p = P[i, :]
            eig = eigs[i, :]
            real_diff = np.max(np.abs(p.real-eig.real))
            imag_diff = np.max(np.abs(p.imag-eig.imag))
            real_diff = f'{i} {real_diff:.4f}'
            imag_diff = f'{i} {imag_diff:.4f}'
            ax[0].plot(multiplier.x, eig.real, color=color, linestyle='-', label=real_diff)
            ax[0].plot(multiplier.x, p.real, color=color, linestyle=':')
            ax[1].plot(multiplier.x, eig.imag, color=color, linestyle='-', label=imag_diff)
            ax[1].plot(multiplier.x, p.imag, color=color, linestyle=':')
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title(f"First $n={number_eigenvectors}$ modes of {multiplier.transform}")
        ax[1].set_title(f"First $n={number_eigenvectors}$ modes of {multiplier.transform}")

        # ind = np.where(np.abs(D) > 1e-9)[0]
        # ax[3].plot(np.arange(n), D[:n], label="e_i") # "'$\mathbf{e}_i$')
        # ax[3].plot(np.arange(n), np.zeros(n), label='y=0')
        # ax[3].set_title(f"Nonzero eigenvalues: {ind}")

        plt.tight_layout()
        plt.show()
        assert False


@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_eigenvalues(multiplier):

    fig, axes = plt.subplots(nrows=1, ncols=2)
    success = True
    for k in range(5):
        eigenvector = multiplier.eigenvector(k)
        transformed = multiplier.to_freq_domain(eigenvector)
        if abs(transformed[k] - 1) > 1e-3:
            success = False
            axes[0].plot(multiplier.x, eigenvector, label=k)
            axes[1].plot(transformed, label=k)
    if success:
        assert True
    else:
        plt.legend()
        plt.show()
        assert False


@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_eigenvector(transform):
    multiplier = FourierMultiplier(N=100, L=1.53, transform=transform)
    for i in range(multiplier.N):
        ef = multiplier.eigenfunction(i)(multiplier.x)
        ev = multiplier.eigenvector(i)
        assert_allclose(ev, ef/np.linalg.norm(ef), rtol=0, atol=1e-9)


@pytest.mark.parametrize("transform", ['dct', 'fft'])
def test_coeff2u(transform):
    prior = Prior(L=3, N=30, transform=transform, gamma=-0.6)
    samples, coefficients = prior.sample(return_coeffs=True, n_sample=3)
    for sample, coefficient in zip(samples, coefficients):
        calculated_sample = sum(c * prior.eigenfunction(i)(prior.x) for i, c in enumerate(coefficient))
        assert_allclose(sample, calculated_sample, rtol=0, atol=1e-12)