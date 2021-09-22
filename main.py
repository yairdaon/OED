from forward import Heat
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from observations import PointObservation, DiagObservation
from probability import Posterior, Prior
from scipy.stats import gaussian_kde

# plt.rcParams['figure.figsize'] = [8, 4]
# plt.rcParams['figure.dpi'] = 200  # 200 e.g. is really fine, but slower
# plt.rcParams['axes.facecolor'] = 'black'
# plt.rcParams['figure.facecolor'] = 'black'
# plt.rcParams['text.color'] = 'white'


def optimal_observation(num_observations, posterior, plot=False):
    """Plots and returns the optimal design multiplier"""
    eigenvalues = posterior.prior.inv_mult * posterior.sigSqr
    data = []
    for k in range(1, num_observations+8):
        eigs = eigenvalues[:k]
        uniform = (np.sum(eigs) + num_observations) / k
        extra = uniform - eigs
        assert abs(np.sum(extra) - num_observations) < 1e-9
        if np.any(extra < 0):
            break
        data.append({'eigs': eigs, 'extra': extra})

    if plot:
        N = k

        fig, axes = plt.subplots(ncols=N-1, figsize=(10, 5), sharex=True, sharey=True)
        axes = axes if hasattr(axes, '__getitem__') else [axes]
        width = 0.35
        for ax, dd in zip(axes, data):
            ax.bar(range(N), eigenvalues[:N], width, label=r'$\sigma^2 \lambda_i^{-1}$', color='b')

            k = dd['extra'].size
            ax.bar(range(k), dd['eigs'], width, color='b')
            ax.bar(range(k), dd['extra'], width, bottom=dd['eigs'], label=r'$\eta_i$', color='r')
            ax.set_ylim((0,num_observations))

            if k > 0:
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.axes.get_yaxis().set_visible(False)

        axes[0].legend()
        fig.suptitle(f"{num_observations} observations")
        plt.tight_layout()
        plt.show()
    extra = data[-1]['extra']
    return DiagObservation(**posterior.specs, singular_values=extra, random_U=False)


def main():
    specs = {'N': 500, 'L': 2}
    ms = list(range(2, 8))
    for transform in ['dct', 'fft']:
        fig, axes = plt.subplots(nrows=1, ncols=len(ms), figsize=(20, 10))
        for num_observations, ax in zip(ms, axes):

            specs['transform'] = transform
            prior = Prior(gamma=-2, **specs)
            forward = Heat(time=3e-1, alpha=0.6, **specs)
            post = Posterior(fwd=forward, prior=prior, sigSqr=1e-2, **specs)

            # Optimal diagonal
            diagonal_observation = optimal_observation(num_observations, post)
            diagonal_utility = post.utility(diagonal_observation)
            singular_values = diagonal_observation.singular_values

            # Optimal point
            res = minimize(post.minimization_target,
                           bounds=[(0, post.L)] * num_observations,
                           x0=np.random.uniform(low=0, high=post.L, size=num_observations))
            point_utility = post.utility(res['x'])

            f = lambda: post.utility(DiagObservation(singular_values=singular_values,
                                                     **post.specs,
                                                     random_U=True))
            utilities = Parallel(n_jobs=7)(delayed(f)() for _ in range(2000))

            kde = gaussian_kde(utilities)
            x = np.linspace(min(utilities), max(utilities), 100)
            ax.plot(x, kde(x), color='b', label='Diagonals with random U')
            ax.hist(utilities, density=True)
            ax.axvline(x=point_utility, color='k', ymin=0, ymax=1, label='Optimal point')
            ax.axvline(x=diagonal_utility, color='r', ymin=0, ymax=1, label='Diagonal U = Id')
            ax.set_title(f'm={num_observations} transform = {transform}')
        plt.legend()
        plt.show()
        # print(res)
        # # measurements = res['x']
        # obs = post.observation(measurements)
        #
        # pointwise = post.mean_std(obs=measurements)
        #
        # plt.figure(figsize=(6, 3))
        # plt.plot(post.x, pointwise, label='post STD')
        # plt.scatter(measurements, obs(pointwise))
        # plt.title(f'transform {transform}')
        # plt.show()


if __name__ == '__main__':
    main()