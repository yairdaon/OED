import numpy as np

from src.multiplier import FourierMultiplier

class Observation(FourierMultiplier):
    @property
    def OstarO(self):
        O = self.matrix
        return np.einsum('ki, kj->ij', O.conjugate(), O)

class PointObservation(Observation):
    def __init__(self, measurements=[], **kwargs):
        super().__init__(**kwargs, size=len(measurements))
        self.measurements = np.array(measurements)
        self.multiplier = np.zeros(self.shape, dtype=self.dtype)
        for k in range(self.N):
                self.multiplier[:, k] = self.eigenfunction(k)(self.measurements)

    def __str__(self):
        return 'Point observations at ' + ', '.join([f'{me:.4f}' for me in self.measurements])

    def _matvec(self, v):
        v_hat = self.to_freq_domain(v)
        return np.einsum('ij, j -> i', self.multiplier, v_hat)

    def _matmat(self, M):
        M_hat = self.to_freq_domain(M, axis=0)
        return np.einsum('ij, jk -> ik', self.multiplier, M_hat)

    @property
    def matrix(self):
        return self.to_freq_domain_from_right(self.multiplier)

    def eigenvalues(self):
        O = self.multiplier
        return np.linalg.eigh(O @ O.conjugate().T)[0] * self.h


class DiagObservation(Observation):
    def __init__(self, multiplier, random_U=False, **kwargs):
        super().__init__(**kwargs, size=len(multiplier))
        self.multiplier = np.zeros(self.N)
        self.multiplier[:self.shape[0]] = multiplier
        assert np.all(self.multiplier.imag < 1e-14)
        assert np.all(self.multiplier.real >= 0)

        # if random_U:
        #     # From https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy
        #     H = np.random.randn(self.singular_values.shape[0], self.singular_values.shape[0])
        #     Q, R = np.linalg.qr(H)
        #     # Q = np.vdot(Q, np.diag(np.sign(np.diag(R))))
        #     self.multiplier = np.dot(Q, self.multiplier)
        #     self.U = Q
        # else:
        #     self.U = np.eye(len(singular_values))

    def _matvec(self, v):
        return self.to_freq_domain(v) * self.multiplier

    def _matmat(self, M):
        M_hat = self.to_freq_domain(M, axis=0)
        return np.einsum('i, ij -> ij', self.multiplier, M_hat)

    @property
    def matrix(self):
        return self(np.eye(self.N))

    def singular_values(self):
        return self.multiplier * self.sqrt_h

    def __str__(self):
        return 'Diagonal observation with singular values ' + ', '.join([f'{s:.4f}' for s in self.multiplier])


