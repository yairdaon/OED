import numpy as np
from numpy.random import randn
from scipy import fft as fft
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve
from functools import partial


class Operator(LinearOperator):
    def __init__(self, N=500, L=1, size=None, dtype=None, **kwargs):
        shape = (N, N) if size is None else (size, N)
        super().__init__(dtype=np.dtype(dtype), shape=shape)
        self.L = L
        self.N = N
        self.x = np.linspace(0, self.L, num=self.N, endpoint=False)
        self.h = self.L / self.N
        self.sqrt_h = np.sqrt(self.h)

    def norm(self, v):
        return np.linalg.norm(v) * self.sqrt_h


class FourierMultiplier(Operator):

    def __init__(self, transform='dct', **kwargs):
        dtype = 'complex' if transform == 'fft' else 'float'
        super().__init__(**kwargs, dtype=dtype)
        self.multiplier = None # Need to implement this on particular case
        self.transform = transform
        if self.transform == 'dct':
            self.freqs = np.arange(self.N) / self.L
        elif self.transform == 'dst':
            self.freqs = np.arange(1,self.N+1) / self.L
        elif self.transform == 'fft':
            self.freqs = fft.fftfreq(self.N, d=self.h)

            
    @property
    def specs(self):
        return {"N": self.N, "L": self.L, "transform":self.transform}

    def to_freq_domain(self, x, axis=-1):
        if self.transform == 'dct':
            return fft.dct(x, norm='ortho', type=2, axis=axis) * self.sqrt_h
        elif self.transform == 'fft':
            return fft.fft(x, norm='ortho', axis=axis) * self.sqrt_h
        elif self.transform == 'dst':
             return fft.dst(x, norm='ortho', type=2, axis=axis) * self.sqrt_h

         
    def to_freq_domain_from_right(self, x):
        return self.to_time_domain(x.conjugate(), axis=1).conjugate() * self.h

    
    def to_time_domain(self, x, axis=-1):
        if self.transform == 'dct':
            return fft.dct(x, norm='ortho', type=3, axis=axis) / self.sqrt_h
        elif self.transform == 'fft':
            return fft.ifft(x, norm='ortho', axis=axis) / self.sqrt_h
        elif self.transform == 'dst':
            return fft.dst(x, norm='ortho', type=3, axis=axis) / self.sqrt_h

        
    def eigenfunction(self, i, normalize=True):
        if self.transform == 'fft':
            eigen = lambda x: np.exp(2j * np.pi * self.freqs[i] * x)
        elif self.transform == 'dct':
            eigen = lambda x: np.cos(np.pi*i/2/self.N + np.pi * self.freqs[i] * x)
        elif self.transform == 'dst':
            eigen = lambda x: np.sin(np.pi*(i+1)/2/self.N + np.pi * self.freqs[i] * x)

        norm = self.norm(eigen(self.x)) if normalize else 1
        return lambda x: eigen(x) / norm

    def block(self, x):
        tmp = np.einsum('i, j -> ij', self.freqs, x).T
        i = np.arange(self.N)
        if self.transform == 'fft':
            return np.exp(2j * np.pi * tmp).T
        elif self.transform == 'dct':
            return np.cos(np.pi*i/2/self.N + np.pi * tmp).T
        elif self.transform == 'dst':
            return np.sin(np.pi*(i+1)/2/self.N + np.pi * tmp).T

    def normalized_block(self, x):
        block = self.block(x)
        norms = self.norms()
        return np.einsum("ij, i-> ij", block, 1/norms)
        
    def norms(self):
        return np.linalg.norm(self.block(self.x), axis=1) * self.sqrt_h


    def eigenvector(self, i):
        return self.eigenfunction(i)(self.x)

    # def coeff2u(self, coeff):
    #     return self.to_time_domain(coeff) / self.sqrt_h

    def normal(self, n_sample=1):
        if self.transform == 'fft':
            # Divide by sqrt(2) to get real and imaginary parts with sqrt(2) standard deviation,
            # which results in a unit variance complex random variable.
            Z = randn(n_sample, self.N, 2).view(np.complex128).reshape(n_sample, self.N) / np.sqrt(2)
        else:
            Z = randn(n_sample, self.N)
        return Z

    # def mult2time(self, mult):
    #     """mult is assumed 2D!!!"""
    #     assert len(mult.shape) == 2
    #     # M4 = ifft(mult, axis=0)
    #     # M4 = ifft(M4.H, axis=0)
    #     M = self.to_time_domain(mult, axis=0)
    #     M = self.to_time_domain(M.conjugate().T, axis=0)
    #     return M

    def _matvec(self, v):
        shp = v.shape
        v_hat = self.to_freq_domain(np.squeeze(v))
        Av_hat = v_hat * self.multiplier
        Av = self.to_time_domain(Av_hat)
        return Av.reshape(*shp)

    def _matmat(self, M):
        """A = PDP*, where P* == to_freq_domain, D == (diagonal) multiplier
        P == to_time_domain. We find AM """

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
