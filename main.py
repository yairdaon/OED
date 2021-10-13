import os

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from src.forward import Heat
from src.probability import Posterior, Prior


# plt.rcParams['figure.figsize'] = [8, 4]
# plt.rcParams['figure.dpi'] = 200  # 200 e.g. is really fine, but slower
# plt.rcParams['axes.facecolor'] = 'black'
# plt.rcParams['figure.facecolor'] = 'black'
# plt.rcParams['text.color'] = 'white'

def main(n_jobs=6):
    L, N = np.pi, 1200
    specs = {'N': N, 'L': L}
    ms = [3 , 5, 7, 37, 61]
    transforms = ['dct', 'fft']
    fig1, axes1 = plt.subplots(figsize=(30,16), nrows=len(transforms), ncols=len(ms))
    # fig2, axes2 = plt.subplots(figsize=(18, 18), nrows=len(transforms), ncols=len(ms))

    for col, m in enumerate(ms):

        for row, transform in enumerate(transforms):
            print('m =', m, transform)

            specs['transform'] = transform
            prior = Prior(gamma=-2, **specs)
            forward = Heat(time=3e-1, alpha=0.6, **specs)
            post = Posterior(fwd=forward, prior=prior, sigSqr=1e-2, **specs)

            # Optimal diagonal
            post.make_optimal_diagonal(m, plot=False)
            diagonal_utility = post.diagonal_utility(post.optimal_diagonal)
            print(f"Utility: Diag={diagonal_utility:3.3f}", end=' ')

            # Optimal approximate point
            approx = post.optimize(m=m,
                                   target=post.close2diagonal,
                                   n_iterations=3*n_jobs,
                                   n_jobs=n_jobs)
            print(f"Approx={approx['utility']:3.3f}", end=' ')

            # Optimal point
            point = post.optimize(m=m, target=post.minimization_point)
            print(f"Point={point['utility']:3.3f}")

            pt_sum = np.sum(point['obs'].eigenvalues())
            diag_sum = np.sum(post.optimal_diagonal)
            approx_sum = np.sum(approx['obs'].eigenvalues())

            print(f"Sum    : Diag={diag_sum:3.3f} approx={approx_sum:3.3f} Point={pt_sum:3.3f} ")

            random_measurements = np.random.uniform(low=0, high=L, size=(10000, m))
            random_utilities = Parallel(n_jobs=n_jobs)(delayed(post.minimization_point)(x) for x in random_measurements)
            random_utilities = -np.array(random_utilities)
            kde = gaussian_kde(random_utilities)
            xx = np.linspace(min(random_utilities), max(random_utilities), num=1000)
            ax = axes1[row, col]
            ax.hist(random_utilities, density=True, alpha=0.3, label='hist')
            ax.plot(xx, kde(xx), label='random')
            # ax.scatter(diagonal_utility, kde(diagonal_utility), label='optimal diagonal')
            ax.scatter(approx['utility'], kde(approx['utility']), label='point approximating diagonal')
            ax.scatter(point['utility'], kde(point['utility']), label='optimal point', color='r')
            ax.legend()
            ax.set_title(f'{m} measurements with {transform}')
            plt.tight_layout()
            plt.savefig(f'pix/utilities.png')
            print(f'finish {transform} {m}')


if __name__ == '__main__':
    if not os.path.exists('pix'):
        os.mkdir('pix/')
    main(os.cpu_count()-1)
