import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize, LinearConstraint
from functools import partial
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import pdb

from src.multiplier import FourierMultiplier
from src.observations import PointObservation#, DiagObservation
from src.forward import Heat


class Prior(FourierMultiplier):
    """ Implement a prior with the following *precision* \delta I +  \Delta^{-gamma}, where gamma < 0"""
    def __init__(self,
                 gamma=-0.6,
                 delta=0.5,
                 **kwargs):
        """
        Parameters:

        gamma: float
            the power of the Laplacian is the prior precision.
        delta: float
            a regularizing constant used for the case of Homogeneous Neumann boundary condition, so that
            \delta I + \Delta^{-gamma} is invertible.

        """

        assert gamma < 0
        super().__init__(**kwargs)
        inv_mult = np.power(np.pi**2 * self.freqs**2, -gamma) + delta
        multiplier = 1 / inv_mult 
        self.multiplier = multiplier
        assert not np.any(np.isnan(multiplier)), 'NaN in prior multiplier'
        self.inv_mult = inv_mult
        assert not np.any(np.isnan(inv_mult)), 'NaN in prior inverse multiplier'
        self.gamma = gamma
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
        '''Implement the inverse of  A bit of a hack - change multiplier to inverse multiplier and back'''
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
    '''Posterior Gaussian. Formula for posterior covariance/ precision is based on
    equations 2.16a and 2.16b from Andrew Stuart's Acta Numerica (2010) paper.'''

    def __init__(self,
                 fwd=None,
                 prior=None,
                 sigSqr=0.1,
                 model_error=0,
                 **kwargs):
        """
        Parameters:
        fwd: the forward operator. We implement the heat equation in 1D
        prior: prior operator. We implement a Laplacian-like prior.

        """

        super().__init__(**kwargs)
        self.fwd = Heat(**self.specs) if fwd is None else fwd
        self.prior = Prior(**self.specs) if prior is None else prior
        self.sigSqr = sigSqr
        self.model_error = True
        self.C_sqrt_fwd = np.sqrt(self.prior.multiplier) * self.fwd.multiplier
        assert np.all(np.abs(self.C_sqrt_fwd.imag) < 1e-12)
        assert np.all(self.C_sqrt_fwd.real >= 0)
        

    # def make_diagonal(self, m, k):
    #     eigs = self.sigSqr / self.C_sqrt_fwd[:k]**2
    #     uniform = np.mean(eigs) + m / k
    #     return uniform - eigs
    #
    # def make_optimal_diagonal(self, m):
    #     k = 1
    #     while True:
    #         eta = self.make_diagonal(m, k)
    #         if np.any(eta <= 0):
    #             break
    #         self.optimal_diagonal_O = np.sqrt(eta)
    #         k += 1
    #
    #     self.optimal_diagonal_O_matrix = np.zeros((m, self.N))
    #     np.fill_diagonal(self.optimal_diagonal_O_matrix, self.optimal_diagonal_O)
    #
    #     power = np.sum(self.optimal_diagonal_O**2)
    #     assert abs(power - m) < 1e-12, (power, m)
    #     return self.optimal_diagonal_O
    #
    # def operators(self, obs):
    #     self.A = np.einsum('ij,j->ij', obs.multiplier, self.fwd.multiplier)
    #     assert not np.any(np.isnan(self.A)), 'NaN in A'
    #     assert not np.any(np.isinf(self.A)), 'inf in A'
    #
    #     self.AstarA = np.einsum('ik, kj-> ij', self.A.conjugate().T, self.A)
    #     assert not np.any(np.isnan(self.AstarA)), 'NaN in AstarA'
    #     assert not np.any(np.isinf(self.AstarA)), 'inf in AstarA'
    #     assert np.allclose(self.AstarA.conjugate().T, self.AstarA, atol=1e-12, rtol=1e-3)
    #
    #     self.precision = self.AstarA / self.sigSqr + np.diag(self.prior.inv_mult)
    #     assert not np.any(np.isnan(self.precision)), 'NaN in precision'
    #     assert not np.any(np.isinf(self.precision)), 'inf in precision'
    #
    #     return self
    #
    # def Astar_data(self, data):
    #     return np.einsum('ji, j->i', self.A.conjugate(), data)
    #
    # def mean_std(self, obs, data=None):
    #     self.operators(obs)
    #     Sigma = self.to_time_domain(solve(self.precision, self.to_freq_domain(np.eye(self.N), axis=0)), axis=0)
    #     pointwise_std = np.sqrt(np.diag(Sigma).real) / self.sqrt_h
    #     if data is None:
    #         return pointwise_std
    #     mean = solve(self.precision, self.Astar_data(data), assume_a='pos') / self.sigSqr
    #     mean = self.to_time_domain(mean)
    #     return mean, pointwise_std


    def point_utility(self, measurements):

        m = len(measurements)
        obs = PointObservation(**self.specs, measurements=measurements)
        Sigma = np.eye(m) * self.sigSqr
        Sigma = Sigma + self.model_error * np.einsum('ik, k, kj-> ij',obs.multiplier, self.prior.multiplier**2, obs.multiplier.conjugate().T)
        OstarO = obs.multiplier.conjugate().T @ np.linalg.solve(Sigma, obs.multiplier)
        tmp = np.einsum('i,ij,j->ij', self.C_sqrt_fwd.conjugate(), OstarO, self.C_sqrt_fwd)
        tmp = tmp + np.eye(self.N)
        utility = np.linalg.slogdet(tmp)
        assert abs(utility[0] - 1) < 1e-7, abs(utility[0])
        return utility[1]
    

    # def diag_utility(self, diag):
    #     tmp = self.C_sqrt_fwd.copy()[:diag.size]
    #     tmp = tmp*diag
    #     tmp = tmp ** 2 / self.sigSqr + 1
    #     return np.sum(np.log(tmp))

                
    # def close2diagonal(self, measurements):
    #     obs = PointObservation(**self.specs,
    #                            measurements=measurements)
    #     return np.linalg.norm(obs.multiplier-self.optimal_diagonal_O_matrix)

    def optimize(self,
                 m,
                 target='utility',
                 n_iterations=1,
                 n_jobs=6,
                 eps=0,
                 full=False):

        # self.make_optimal_diagonal(m)
        f = self.minimization_point #if target == 'utility' else self.close2diagonal
        bounds = [(0+eps, self.L-eps)] * m
        parallelized = partial(minimize, bounds=bounds)
        x0s = np.random.uniform(low=0, high=self.L, size=(n_iterations, m))
        results = Parallel(n_jobs=n_jobs)(delayed(parallelized)(f, x0=x0) for x0 in x0s)
        agg = []
        for res in results:
            x = np.sort(res['x'])
            obs = PointObservation(**self.specs, measurements=x)
            tmp = {}
            tmp['x'] = x
            tmp['sum_eigenvalues'] = np.sum(obs.eigenvalues())
            #tmp['utility'] = -res['fun'] if target == 'utility' else self.point_utility(res['x'])
            #tmp['diag_dist'] = res['fun'] if 'diag' in target else self.close2diagonal(res['x'])
            tmp['target'] = target
            tmp['m'] = m
            tmp['transform'] = self.transform
            tmp['N'] = self.N
            tmp['L'] = self.L
            tmp['success'] = res['success']
            tmp['fun'] = res['fun']
            tmp['object'] = obs
            agg.append(tmp)
        return agg if full else min(agg, key=lambda x: x['fun'])


    # def minimization_diag(self, diag):
    #     return -self.diag_utility(np.sqrt(diag))

    
    # def optimize_diag(self,
    #                   m,
    #                   n_iterations=1,
    #                   n_jobs=6,
    #                   full=False):
    #
    #     f = self.minimization_diag
    #     constraints = {'type': 'eq',
    #                    'fun': lambda x: np.sum(x) - m,
    #                    'jac': lambda x: np.ones_like(x)}
    #
    #     #constraints = LinearConstraint(A=np.ones((1,self.N)), lb=[0], ub=[m])
    #     #import pdb; pdb.set_trace()
    #     parallelized = partial(minimize, constraints=constraints)
    #     x0s = np.random.uniform(low=0, high=m, size=(n_iterations, m))
    #     results = Parallel(n_jobs=n_jobs)(delayed(parallelized)(f, x0=x0) for x0 in x0s)
    #     agg = []
    #     for res in results:
    #         x = res['x']
    #         obs = DiagObservation(**self.specs, multiplier=x)
    #         tmp = {}
    #         tmp['x'] = x
    #         tmp['sum_eigenvalues'] = np.sum(obs.multiplier**2)
    #         tmp['utility'] = -res['fun']
    #         tmp['m'] = m
    #         tmp['transform'] = self.transform
    #         tmp['N'] = self.N
    #         tmp['L'] = self.L
    #         tmp['success'] = res['success']
    #         tmp['object'] = obs
    #         tmp['fun'] = res['fun']
    #         agg.append(tmp)
    #     return agg if full else min(agg, key=lambda x: x['fun'])


    def minimization_point(self, measurements):
        return -self.point_utility(measurements)

    def __str__(self):
        return f'Posterior with prior {self.prior} and {self.fwd}'

    def __repr__(self):
        return str(self)
