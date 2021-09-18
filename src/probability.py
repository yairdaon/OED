import numpy as np
from scipy.linalg import solve

from multiplier import FourierMultiplier


class Prior(FourierMultiplier):
    """ Implement a negative Laplacian prior Delta^{gamma}, where gamma < 0"""
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

    def sample(self, return_coeffs=False, n_sample=1):
        """ Generate a sample and return its coefficients if needed. This effectively uses the
        Karhunen-Loeve expansion"""
        assert n_sample > 0
        coeffs = self.normal(n_sample)

        coeffs = np.einsum('ij, j->ij', coeffs, np.power(self.multiplier, 0.5))
        # coeffs[:, 0] = 0

        u0 = self.to_time_domain(coeffs)
        if return_coeffs:
            return u0, coeffs
        return u0

    def inverse(self, v):
        ''' A bit of a hack - change multiplier to inverse multiplier and back'''
        tmp = self.multiplier
        self.multiplier = self.inv_mult
        ret = self(v)
        self.multiplier = tmp
        return ret


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
        self.AstarA = np.einsum('ji, jk-> ik', self.A.conjugate(), self.A)
        self.precision_sigSqr = self.AstarA + np.diag(self.prior.inv_mult) * self.sigSqr
        self.Astar_data = np.einsum('ji, j->i', self.A.conjugate(), data)

        mean = solve(self.precision_sigSqr, self.Astar_data, assume_a='her')
        self.m = self.to_time_domain(mean)

        Sigma_over_sigSqr = np.linalg.inv(self.precision_sigSqr)
        Sigma_over_sigSqr = self.to_time_domain(self.to_time_domain(Sigma_over_sigSqr).conjugate().T).conjugate().T
        self.ptwise = np.sqrt(np.abs(np.diag(Sigma_over_sigSqr * self.sigSqr))).real

    def utility(self):
        C_sqrt = np.sqrt(self.prior.multiplier)
        tmp = np.einsum('i,ij,j->ij', C_sqrt, self.AstarA, C_sqrt.conjugate()) / self.sigSqr + np.eye(self.N)
        utility = np.linalg.slogdet(tmp)
        assert utility[0].real > 0
        assert abs(utility[0].imag) < 1e-14
        return utility[1]

    def __le__(self, other):
        return self.utility <= other.utility

