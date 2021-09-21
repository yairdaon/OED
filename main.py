from forward import Heat
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from observations import PointObservation, DiagObservation
from probability import Posterior, Prior

# plt.rcParams['figure.figsize'] = [8, 4]
# plt.rcParams['figure.dpi'] = 200  # 200 e.g. is really fine, but slower
# plt.rcParams['axes.facecolor'] = 'black'
# plt.rcParams['figure.facecolor'] = 'black'
# plt.rcParams['text.color'] = 'white'


def optimal_observation(num_observations, posterior, plot=False):
    """Plots and returns the optimal design multiplier"""
    eigenvalues = posterior.prior.inv_mult * posterior.sigSqr
    data = []
    for k in range(1, num_observations+1):
        eigs = eigenvalues[:k]
        uniform = (np.sum(eigs) + num_observations) / k
        extra = uniform - eigs
        if np.any(extra < 0):
            break
        data.append({'eigs': eigs, 'extra': extra})

    if plot:
        N = k

        fig, axes = plt.subplots(ncols=N-1, figsize=(10, 5), sharex=True, sharey=True)
        width = 0.35

        for ax, dd in zip(axes, data):
            ax.bar(range(N), eigenvalues[:N], width, label=r'$\sigma^2 \lambda_i^{-1}$', color='b')

            k = dd['extra'].size
            ax.bar(range(k), dd['eigs'], width, color='b')
            ax.bar(range(k), dd['extra'], width, bottom=dd['eigs'], label=r'$\eta_i$', color='r')
            ax.set_ylim((0,num_observations/2))

            if k > 0:
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.axes.get_yaxis().set_visible(False)

        axes[0].legend()
        fig.suptitle(f"{num_observations} observations")
        plt.tight_layout()
        plt.show()
    extra = data[-1]['extra']
    return DiagObservation(**posterior.specs, singular_values=extra)


def main():
    specs = {'N': 500, 'L': np.pi}
    for num_observations in range(2, 10):
        for transform in ['dct', 'fft']:
            specs['transform'] = transform
            prior = Prior(gamma=-2, **specs)
            forward = Heat(time=3e-1, alpha=0.6, **specs)
            posterior = Posterior(fwd=forward, prior=prior, sigSqr=1e-2, **specs)

            obs = optimal_observation(num_observations, posterior)
            diagonal = posterior.utility(obs)

            # Optimal design
            res = minimize(posterior.minimization_target,
                           bounds=[(0, posterior.L)] * num_observations,
                           x0=np.random.uniform(low=0, high=posterior.L, size=num_observations))
            point = posterior.utility(res['x'])
            if diagonal < point:
                sign = '<'
                message = 'BAD!!!'
            else:
                sign = '>'
                message = ''
            print(f"Diag={diagonal:.4f} {sign} {point:.4f}=point. {transform}, m={num_observations} {message}")

            # print(res)
            # # measurements = res['x']
            # obs = posterior.observation(measurements)
            #
            # pointwise = posterior.mean_std(obs=measurements)
            #
            # plt.figure(figsize=(6, 3))
            # plt.plot(posterior.x, pointwise, label='posterior STD')
            # plt.scatter(measurements, obs(pointwise))
            # plt.title(f'transform {transform}')
            # plt.show()


if __name__ == '__main__':
    main()