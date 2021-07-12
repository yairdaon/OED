import numpy as np

from multiplier import FourierMultiplier


class Prior(FourierMultiplier):  # Actually, negative Laplacian
    def __init__(self, gamma, **kwargs):
        assert gamma < 0
        super().__init__(**kwargs)
        ind = self.freqs != 0
        multiplier = np.ones(self.freqs.shape)
        multiplier[ind] = np.power(np.pi ** 2 * self.freqs[ind] ** 2, gamma)
        inv_mult = np.ones(self.freqs.shape)
        # (multiplier < 1e-4) #| (inv_mult < 1e-4) #| np.isnan(multiplier) | np.isnan(inv_mult) | (multiplier > 1e4) | (inv_mult > 1e4)
        inv_mult[ind] = np.power(np.pi ** 2 * self.freqs[ind] ** 2, -gamma)
        # ind[0] = True
        # ind[-1] = True
        # multiplier[ind] = 0
        self.multiplier = multiplier
        # inv_mult[ind] = 0
        self.inv_mult = inv_mult
        self.gamma = gamma
        # self.kwargs = kwargs
        # self.ind = ind
        # self.ind = ind

    def sample(self, return_coeffs=False):
        if self.transform == 'fft':
            coeffs = np.random.randn(self.N, 2).view(np.complex128)
            coeffs = np.squeeze(coeffs)
        else:
            coeffs = np.random.randn(self.freqs.size)
        coeffs *= self.multiplier
        coeffs[0] = 0

        u0 = self.coeff2u(coeffs)
        return u0, coeffs if return_coeffs else u0

    def inverse(self, v):
        return self(v, mult=self.inv_mult)

    #         return Laplacian(gamma=-self.gamma, N=self.N, L=self.L)

    def __str__(self):
        return '$(-\Delta)^{' + str(self.gamma) + '}$'

    @property
    def invstr(self):
        return '$(-\Delta)^{' + str(-self.gamma) + '}$'


class Posterior(FourierMultiplier):
    ''' Based on Stuart 2.16a and 2.16b'''

    def __init__(self, fwd, prior, sigSqr, **kwargs):
        super().__init__(**kwargs)
        self.fwd = fwd
        self.prior = prior
        self.sigSqr = sigSqr

    def update(self, obs, data):
        self.A = np.einsum('ij,j->ij', obs.multiplier, self.fwd.multiplier)
        self.A_H = self.A.T if self.transform == 'dct' else self.A.conjugate().T
        self.AstarA = np.einsum('ij, jk-> ik', self.A_H, self.A)
        self.precision = self.AstarA / self.sigSqr + np.diag(self.prior.inv_mult)
        Sigma = np.linalg.inv(self.precision)

        #         self.A = A
        #         self.Sigma = Sigma
        #         self.precision = precision
        #         self.obs = obs
        #         self.AstarA = AstarA
        self.Astar_data = np.einsum('ij, j->i', self.A_H, data)
        # Astar_data = self.to_freq_domain(Astar_data)
        # print(np.linalg.cond(Sigma))
        mean = solve(self.precision, self.Astar_data, assume_a='her') / self.sigSqr
        # mean[80:] = 0

        self.m = self.to_time_domain(mean)
        self.ptwise = np.sqrt(np.abs(np.diag(self.mult2time(Sigma)))).real

    def utility(self, meas):
        obs = PointObservation(meas=meas, N=self.N, L=self.L)
        A = np.einsum('ij,j->ij', obs.multiplier, self.fwd.multiplier)
        AstarA = np.einsum('ji, jk->ik', A.conjugate(), A)
        C_sqrt = np.sqrt(self.prior.multiplier)
        tmp = np.einsum('i,ij,j->ij', C_sqrt, AstarA, C_sqrt.conjugate()) / self.sigSqr + np.eye(self.N)
        utility = -np.linalg.slogdet(tmp)[1]
        assert utility < 0
        return {'meas': meas, 'utility': utility}

    def __le__(self, other):
        return self.utility <= other.utility

