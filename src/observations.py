import numpy as np

from multiplier import FourierMultiplier

class Observation(FourierMultiplier):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs, size=size)
        self.multiplier = np.zeros(self.shape, dtype=np.complex128)

    def _matvec(self, v):
        v_hat = self.to_freq_domain(v)
        return np.einsum('ij, j -> i', self.multiplier, v_hat)

    def _matmat(self, M):
        M_hat = self.to_freq_domain(M)
        return np.einsum('ij, jk -> ik', self.multiplier, M_hat)


class PointObservation(Observation):
    def __init__(self, meas=[], **kwargs):
        super().__init__(**kwargs, size=len(meas))
        self.meas = np.array(meas)

        if self.transform == 'fft':
            for k in range(self.N):
                self.multiplier[:, k] = self.eigenfunction(k)(self.meas) / np.sqrt(self.N)
        elif self.transform in ('dct', 'dst'):
            for k in range(self.N):
                self.multiplier[:, k] = self.eigenfunction(k)(self.meas) / np.sqrt(0.5 * self.N)

    def __str__(self):
        return 'Point observations ' + ', '.join([f'{me:.4f}' for me in self.meas])


class DiagObservation(Observation):
    def __init__(self, singular_values, random_U=False, **kwargs):
        super().__init__(**kwargs, size=len(singular_values))
        self.singular_values = np.array(singular_values)
        assert len(self.singular_values.shape) == 1
        np.fill_diagonal(self.multiplier, self.singular_values)
        if random_U:
            H = np.random.randn(self.singular_values.shape[0], self.singular_values.shape[0])
            Q, R = np.linalg.qr(
                H)  # From https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy
            Q = np.vdot(Q, np.diag(np.sign(np.diag(R))))
            self.multiplier = np.dot(Q, self.multiplier)
        O = self.to_time_domain(self.multiplier)
        Ostar = O.T.conj()
        self.OstarO = np.dot(Ostar, O)

    def __str__(self):
        return 'Observation with singular values ' + ', '.join([f'{s:.4f}' for s in self.singular_values])


