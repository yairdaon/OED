import numpy as np
from scipy import fft as fft
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve
from functools import partial


class Operator(LinearOperator):
    def __init__(self, N=500, L=1):
        self.L = L
        self.N = N
        self.shape = (N, N)
        self.x = np.linspace(0, self.L, num=self.N, endpoint=False)

    def norm(self, v):
        return np.sqrt(np.dot(v.conjugate(), v) * self.L / self.N)


class FourierMultiplier(Operator):

    def __init__(self, transform='dct', **kwargs):
        super().__init__(**kwargs)
        self.dtype = None
        self.freqs = fft.fftfreq(self.N, d=self.L / self.N)
        self.multiplier = None  # Need to implement this on particular case
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
        eigen = lambda x: self.func(np.pi * self.freqs[i] * x)
        norm = np.linalg.norm(eigen(self.x)) * np.sqrt(self.L / self.N)
        return lambda x: eigen(x) / norm

    def eigenvector(self, i):
        v = self.eigenfunction(i)(self.x)
        return v / np.linalg.norm(v)

    def coeff2u(self, coeff):
        res = sum(coeff[i] * self.eigenfunction(i)(self.x) for i in np.where(np.abs(coeff) > 1e-5)[0])
        return res

    def mult2time(self, mult):
        """mult is assumed 2D!!!"""
        assert len(mult.shape) == 2
        # M4 = ifft(mult, axis=0)
        # M4 = ifft(M4.H, axis=0)
        M = self.to_time_domain(mult, axis=0)
        M = self.to_time_domain(M.conjugate().T, axis=0)
        return M

    def _matvec(self, v):
        return self(np.squeeze(v))

    def __call__(self, v, mult=None):
        if mult is None:
            mult = self.multiplier
        v_hat = self.to_freq_domain(v)
        Av_hat = v_hat * mult
        Av = self.to_time_domain(Av_hat)
        return Av + np.mean(v)

    #     @property
    #     def matrix(self):
    #         M = np.diag(self.multiplier) if len(self.multiplier.shape) == 1 else self.multiplier
    #         M = self.inv(M)
    #         M = M.conjugate().T
    #         M = self.inv(M).conjugate().T
    #         return M

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Generic Fourier multiplier'
