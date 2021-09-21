import numpy as np
from scipy.linalg import solve

from multiplier import FourierMultiplier
from observations import PointObservation, Observation


class Prior(FourierMultiplier):
    """ Implement a negative Laplacian prior Delta^{gamma}, where gamma < 0"""
    def __init__(self, gamma, **kwargs):
        assert gamma < 0
        super().__init__(**kwargs)
        ind = self.freqs != 0
        multiplier = np.ones(self.freqs.shape)
        multiplier[ind] = np.power(np.pi**2 * self.freqs[ind]**2, gamma)
        inv_mult = np.ones(self.freqs.shape)
        inv_mult[ind] = np.power(np.pi**2 * self.freqs[ind]**2, -gamma)
        self.multiplier = multiplier
        assert not np.any(np.isnan(multiplier)), 'NaN in prior multiplier'
        self.inv_mult = inv_mult
        assert not np.any(np.isnan(inv_mult)), 'NaN in prior inverse multiplier'
        self.gamma = gamma
        self.ind = ind

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

    def operators(self, obs):
        if type(obs) in (list, tuple, np.ndarray):
            obs = self.observation(obs)
        self.A = np.einsum('ij,j->ij', obs.multiplier, self.fwd.multiplier)
        assert not np.any(np.isnan(self.A)), 'NaN in A'
        assert not np.any(np.isinf(self.A)), 'inf in A'

        self.AstarA = np.einsum('ji, jk-> ik', self.A.conjugate(), self.A)
        assert not np.any(np.isnan(self.AstarA)), 'NaN in AstarA'
        assert not np.any(np.isinf(self.AstarA)), 'inf in AstarA'

        self.precision = self.AstarA / self.sigSqr + np.diag(self.prior.inv_mult)
        assert not np.any(np.isnan(self.precision)), 'NaN in precision'
        assert not np.any(np.isinf(self.precision)), 'inf in precision'

        return self

    def Astar_data(self, data):
        return np.einsum('ji, j->i', self.A.conjugate(), data)

    def mean_std(self, obs, data=None):
        self.operators(obs)
        Sigma = self.to_time_domain(solve(self.precision, self.to_freq_domain(np.eye(self.N), axis=0)), axis=0)
        pointwise_std = np.sqrt(np.diag(Sigma).real) / self.sqrt_h
        if data is None:
            return pointwise_std
        mean = solve(self.precision, self.Astar_data(data), assume_a='pos') / self.sigSqr
        mean = self.to_time_domain(mean)
        return mean, pointwise_std

    def utility(self, obs):
        self.operators(obs)
        C_sqrt = np.sqrt(self.prior.multiplier)
        tmp = np.einsum('i,ij,j->ij', C_sqrt, self.AstarA, C_sqrt.conjugate()) / self.sigSqr + np.eye(self.N)
        utility = np.linalg.slogdet(tmp)
        assert utility[0].real > 0
        assert abs(utility[0].imag) < 1e-12, f'utility via slogdet {utility}'
        return utility[1]

    def observation(self, measurements):
        return PointObservation(N=self.N,
                                L=self.L,
                                transform=self.transform,
                                measurements=measurements)

    def minimization_target(self, measurements):
        obs = self.observation(measurements)
        return -self.utility(obs)

