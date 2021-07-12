import numpy as np

from multiplier import FourierMultiplier


class PointObservation(FourierMultiplier):
    def __init__(self, meas=[], **kwargs):
        super().__init__(**kwargs)
        self.meas = np.array(meas)
        self.size = len(meas)
        self.multiplier = np.zeros((self.size, self.N), dtype=np.complex128)
        if self.transform == 'fft':
            for k in range(self.N):
                self.multiplier[:, k] = self.eigenfunction(k)(self.meas) / np.sqrt(self.N)
        elif self.transform in ('dct', 'dst'):
            for k in range(self.N):
                self.multiplier[:, k] = self.eigenfunction(k)(self.meas) / np.sqrt(0.5 * self.N)

    def __call__(self, v):
        v_hat = self.to_freq_domain(v)
        return np.einsum('ij, j -> i', self.multiplier, v_hat)


class DiagObservation(FourierMultiplier):
    def __init__(self, singular_values, random_U=False, **kwargs):
        super().__init__(**kwargs)
        singular_values = np.array(singular_values)
        assert len(singular_values.shape) == 1
        self.multiplier = np.zeros((singular_values.shape[0], self.N))
        np.fill_diagonal(self.multiplier, singular_values)
        if random_U:
            H = np.random.randn(singular_values.shape[0], singular_values.shape[0])
            Q, R = np.linalg.qr(
                H)  # From https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy
            Q = np.vdot(Q, np.diag(np.sign(np.diag(R))))
            self.multiplier = np.dot(Q, self.multiplier)
        O = self.to_time_domain(self.multiplier)
        Ostar = O.T.conj()
        self.OstarO = np.dot(Ostar, O)


