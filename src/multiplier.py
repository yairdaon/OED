import numpy as np
from numpy.random import randn
from scipy import fft as fft
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve
from functools import partial


class Operator(LinearOperator):
    def __init__(self, N=500, L=1, size=None):
        shape = (N, N) if size is None else (size, N)
        super().__init__(dtype=np.float, shape=shape)
        self.L = L
        self.N = N
        self.shape = shape
        self.x = np.linspace(0, self.L, num=self.N, endpoint=False)
        self.dtype = np.float


    def norm(self, v):
        return np.sqrt(np.dot(v.conjugate(), v) * self.L / self.N)


class FourierMultiplier(Operator):

    def __init__(self, transform='dct', **kwargs):
        super().__init__(**kwargs)
        self.freqs = fft.fftfreq(self.N, d=self.L / self.N)
        self.multiplier = None # Need to implement this on particular case
        self.transform = transform
        if self.transform == 'dct':
            self.to_freq_domain = partial(fft.dct, norm='ortho')
            self.to_time_domain = partial(fft.idct, norm='ortho')
            self.func = np.cos
        # elif self.transform == 'dst':
        #     self.to_freq_domain = partial(fft.dst, norm='ortho')
        #     self.to_time_domain = partial(fft.idst, norm='ortho')
        #     self.func = np.sin
        elif self.transform == 'fft':
            self.to_freq_domain = partial(fft.fft, norm='ortho')
            self.to_time_domain = partial(fft.ifft, norm='ortho')
            self.func = lambda x: np.exp(2j * x)

    def eigenfunction(self, i):
        if self.transform == 'fft':
            eigen = lambda x: np.exp(2j * np.pi * self.freqs[i] * x)
        elif self.transform == 'dct':
            eigen = lambda x: np.cos(np.pi*i/(2*self.N) + np.pi * self.freqs[i] * x)
        norm = np.linalg.norm(eigen(self.x)) * np.sqrt(self.L / self.N)
        return lambda x: eigen(x) / norm

    def eigenvector(self, i):
        v = self.eigenfunction(i)(self.x)
        return v / np.linalg.norm(v)

    def coeff2u(self, coeff):
        eigenfunctions = np.vstack([self.eigenfunction(i)(self.x) for i in range(self.N)])
        res = np.einsum('ik, kj->ij', coeff, eigenfunctions)
        return res

    def normal(self, n_sample=1):
        if self.transform == 'fft':
            # Divide by sqrt(2) to get real and imaginary parts with sqrt(2) standard deviation,
            # which results in a unit variance complex random variable.
            Z = randn(n_sample, self.N, 2).view(np.complex128).reshape(n_sample, self.N) / np.sqrt(2)
        else:
            Z = randn(n_sample, self.N)
        return Z

    def mult2time(self, mult):
        """mult is assumed 2D!!!"""
        assert len(mult.shape) == 2
        # M4 = ifft(mult, axis=0)
        # M4 = ifft(M4.H, axis=0)
        M = self.to_time_domain(mult, axis=0)
        M = self.to_time_domain(M.conjugate().T, axis=0)
        return M

    def _matvec(self, v):
        shp = v.shape
        v_hat = self.to_freq_domain(np.squeeze(v))
        Av_hat = v_hat * self.multiplier
        Av = self.to_time_domain(Av_hat)
        return Av.reshape(*shp)

    def _matmat(self, M):
        """M = PDP*, where P* == to_freq_domain, D == (diagonal) multiplier
        P == to_time_domain """

        M_hat = self.to_freq_domain(M, axis=0)
        AM_hat = np.einsum('i, ij-> ij', self.multiplier, M_hat)
        AM = self.to_time_domain(AM_hat, axis=0)
        return AM

    @property
    def matrix(self):
        return self(np.eye(self.N))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Generic Fourier multiplier with ' + self.transform
