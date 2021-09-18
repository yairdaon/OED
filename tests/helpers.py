import numpy as np


def align_eigenvectors(P):
    """P contains vectors in its rows. We normalize thesm to norm 1 and real positive first entry"""
    P = np.einsum('ij, i -> ij', P, np.exp(-1j * np.angle(P[:, 0])) / np.linalg.norm(P, axis=1))
    assert np.all(P[:, 0].real > 0)
    assert np.all(np.abs(P[:, 0].imag) < 1e-12)
    ind = np.argsort(P[:, 1])
    P = P[ind, :]
    # P = np.sort(P, axis=0)
    return P


if __name__ == '__main__':
    N = 10
    P = np.random.randn(N, N)
    P = align_eigenvectors(P)
    for row in P:
        imag = abs(row[0].imag)
        if imag > 1e-13:
            print(f'abs(Imaginary part) = {imag} > 0')
        real = row[0].real
        if real <= 0:
            print(f'Real part = {real} <= 0')
        norm = np.linalg.norm(row)
        if abs(norm-1) > 1e-12:
            print(f'1-norm == {1-norm} > 1e-12')
