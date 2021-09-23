import numpy as np


def align_eigenvectors(P, k=0):
    """P contains vectors in its rows. We normalize thesm to norm 1 and real positive first entry"""
    P = np.einsum('ij, i -> ij', P, np.exp(-1j * np.angle(P[:, k])) / np.linalg.norm(P, axis=1))
    assert np.all(P[:, k].real >= 0)
    assert np.all(np.abs(P[:, k].imag) < 1e-12)
    ind = np.argsort(P[:, 1])
    P = P[ind, :]
    # P = np.sort(P, axis=0)
    return P


if __name__ == '__main__':
    N = 100
    k = 9
    P = np.random.randn(N, N)
    P = align_eigenvectors(P, k)
    for row in P:
        imag = abs(row[k].imag)
        if imag > 1e-13:
            print(f'abs(Imaginary part) = {imag} > 0')
        real = row[k].real
        if real <= 0:
            print(f'Real part = {real} <= 0')
        norm = np.linalg.norm(row)
        if abs(norm-1) > 1e-12:
            print(f'1-norm == {1-norm} > 1e-12')
