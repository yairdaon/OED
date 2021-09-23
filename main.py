import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from src.observations import PointObservation, DiagObservation
from src.probability import Posterior, Prior
from src.forward import Heat


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
        uniform = (np.sum(eigs) + num_observations / posterior.sqrt_h) / k
        extra = uniform - eigs
        # assert abs(np.sum(extra) - num_observations) < 1e-9
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
    return DiagObservation(multiplier=extra, random_U=False, **posterior.specs)


def main():
    specs = {'N': 2000, 'L': 2}

    for m in [4, 8, 12, 16]:
        print('m =', m)

        for transform in ['dct', 'fft']:
            specs['transform'] = transform
            prior = Prior(gamma=-2, **specs)
            forward = Heat(time=3e-1, alpha=0.6, **specs)
            post = Posterior(fwd=forward, prior=prior, sigSqr=1e-2, **specs)

            # Optimal diagonal
            diagonal_observation = optimal_observation(m, post)
            diagonal_utility = post.utility(diagonal_observation)

            # Optimal point
            res = minimize(post.minimization_target,
                           bounds=[(0, post.L)]*m,
                           x0=np.random.uniform(low=0, high=post.L, size=m))
            assert res['success']
            assert res['x'].size == m
            assert np.all(res['x'] >= 0) and np.all(res['x'] <= post.L)
            print(f"Utility: Point={-res['fun']:.2f} diag={diagonal_utility:.2f} {transform}")

            point_observation = PointObservation(**post.specs,
                                                 measurements=res['x'])

            pt_sum = np.sum(point_observation.singular_values())
            diag_sum = np.sum(diagonal_observation.singular_values())
            print(f'Sum    : Point={pt_sum:.5f} diag={diag_sum:.5f} {transform}')
            print()


if __name__ == '__main__':
    main()