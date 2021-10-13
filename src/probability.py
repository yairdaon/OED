import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from functools import partial
from joblib import Parallel, delayed

from src.multiplier import FourierMultiplier
from src.observations import PointObservation, DiagObservation
from src.forward import Heat


class Prior(FourierMultiplier):
    """ Implement a negative Laplacian prior Delta^{gamma}, where gamma < 0"""
    def __init__(self, gamma=-0.6, **kwargs):
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

    def __init__(self,
                 fwd=None,
                 prior=None,
                 sigSqr=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.fwd = Heat(**self.specs) if fwd is None else fwd
        self.prior = Prior(**self.specs) if prior is None else prior
        self.sigSqr = sigSqr
        self.C_sqrt_fwd = np.sqrt(self.prior.multiplier) * self.fwd.multiplier
        self.optimal_diagonal = None
        assert np.all(np.abs(self.C_sqrt_fwd.imag) < 1e-12)
        assert np.all(self.C_sqrt_fwd.real >= 0)


    def make_optimal_diagonal(self, m, plot=False):
        """Plots and returns the optimal design multiplier"""
        eigenvalues = self.prior.inv_mult * self.sigSqr
        data = []
        for k in range(1, m + 8):
            eigs = eigenvalues[:k]
            uniform = (np.sum(eigs) + m / self.sqrt_h) / k
            extra = uniform - eigs
            # assert abs(np.sum(extra) - num_observations) < 1e-9
            if np.any(extra < 0):
                break
            data.append({'eigs': eigs, 'extra': extra})

        # if plot:
        #     N = k
        #
        #     fig, axes = plt.subplots(ncols=N - 1, figsize=(10, 5), sharex=True, sharey=True)
        #     axes = axes if hasattr(axes, '__getitem__') else [axes]
        #     width = 0.35
        #     for ax, dd in zip(axes, data):
        #         ax.bar(range(N), eigenvalues[:N], width, label=r'$\sigma^2 \lambda_i^{-1}$', color='b')
        #
        #         k = dd['extra'].size
        #         ax.bar(range(k), dd['eigs'], width, color='b')
        #         ax.bar(range(k), dd['extra'], width, bottom=dd['eigs'], label=r'$\eta_i$', color='r')
        #         ax.set_ylim((0, m))
        #
        #         if k > 0:
        #             ax.set_yticklabels([])
        #             ax.set_yticks([])
        #             ax.axes.get_yaxis().set_visible(False)
        #
        #     axes[0].legend()
        #     fig.suptitle(f"{m} observations")
        #     plt.tight_layout()
        #     plt.show()
        self.optimal_diagonal = data[-1]['extra']
        self.optimal_diagonal_matrix = np.zeros((m, self.N))
        np.fill_diagonal(self.optimal_diagonal_matrix, self.optimal_diagonal)

    def operators(self, obs):
        self.A = np.einsum('ij,j->ij', obs.multiplier, self.fwd.multiplier)
        assert not np.any(np.isnan(self.A)), 'NaN in A'
        assert not np.any(np.isinf(self.A)), 'inf in A'

        self.AstarA = np.einsum('ik, kj-> ij', self.A.conjugate().T, self.A)
        assert not np.any(np.isnan(self.AstarA)), 'NaN in AstarA'
        assert not np.any(np.isinf(self.AstarA)), 'inf in AstarA'
        assert np.allclose(self.AstarA.conjugate().T, self.AstarA, atol=1e-12, rtol=1e-3)

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


    def point_utility(self, measurements):
        obs = PointObservation(**self.specs, measurements=measurements)
        OstarO = obs.multiplier.conjugate().T @ obs.multiplier
        tmp = np.einsum('i,ij,j->ij', self.C_sqrt_fwd, OstarO, self.C_sqrt_fwd)
        tmp = tmp / self.sigSqr + np.eye(self.N)
        utility = np.linalg.slogdet(tmp)
        # assert tmp.shape == (self.N, self.N)
        # assert OstarO.shape == (self.N, self.N)
        # assert utility[0].real > 0
        # assert abs(utility[0].imag) < 1e-12, f'utility via slogdet {utility}'
        return utility[1]

    def diagonal_utility(self, diag):
        tmp = np.zeros(self.N)
        tmp[:diag.size] = diag
        tmp = self.C_sqrt_fwd ** 2 * tmp ** 2 / self.sigSqr + 1
        return np.sum(np.log(tmp))

    def close2diagonal(self, measurements):
        obs = PointObservation(**self.specs,
                               measurements=measurements)
        return np.linalg.norm(obs.multiplier-self.optimal_diagonal_matrix)

    def optimize(self,
                m,
                target=None,
                n_iterations=1,
                n_jobs=6,
                eps=0):
        target = self.minimization_point if target is None else target
        bounds = [(0+eps, self.L-eps)] * m
        parallelized = partial(minimize, bounds=bounds)
        x0s = np.random.uniform(low=0, high=self.L, size=(n_iterations, m))
        res = Parallel(n_jobs=n_jobs)(delayed(parallelized)(target, x0=x0) for x0 in x0s)
        res = min(res, key=lambda x: x['fun'])
        res['obs'] = PointObservation(**self.specs, measurements=np.sort(res['x']))
        res['utility'] = self.point_utility(res['x'])
        return res

    def minimization_point(self, measurements):
        return -self.point_utility(measurements)