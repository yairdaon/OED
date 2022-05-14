import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from src.observations import PointObservation
from scipy.stats import gaussian_kde
from src.forward import Heat
from src.probability import Posterior, Prior


# plt.rcParams['figure.figsize'] = [8, 4]
# plt.rcParams['figure.dpi'] = 200  # 200 e.g. is really fine, but slower
# plt.rcParams['axes.facecolor'] = 'black'
# plt.rcParams['figure.facecolor'] = 'black'
# plt.rcParams['text.color'] = 'white'

def main(n_jobs=6):
    L, N = np.pi*10, 800
    ms = [1, 2, 3, 5, 7, 10, 12]
    transforms = ['fft', 'dct']
    # fig1, axes1 = plt.subplots(figsize=(30, 16), nrows=len(transforms), ncols=len(ms))
    # fig2, axes2 = plt.subplots(figsize=(30, 16), nrows=len(transforms), ncols=len(ms))
    # figs = [fig1, fig2]
    # axess = [axes1, axes2]

    for col, m in enumerate(ms):

        for transform in ['fft']:#transforms:#, fig, axes in zip(transforms, figs, axess):
            print('\nm =', m, transform)

            prior = Prior(gamma=-2, N=N, L=L, transform=transform)
            forward = Heat(time=3e-1, alpha=0.6, N=N, L=L, transform=transform)
            post = Posterior(fwd=forward, prior=prior, sigSqr=1e-2, N=N, L=L, transform=transform)

            sample = prior.sample().ravel()
        
            # Optimal diagonal
            post.make_optimal_diagonal(m)
            diagonal_utility = post.diag_utility(post.optimal_diagonal_O)
            # ax.scatter(diagonal_utility, kde(diagonal_utility), label='optimal diagonal')
            print(f"utility: Diag={diagonal_utility:3.3f}", end=' ')

            # # Random point measurements
            # measurements = np.random.uniform(low=0, high=L, size=(100, m))
            # utilities = Parallel(n_jobs=n_jobs)(delayed(post.minimization_point)(x) for x in measurements)
            # utilities = -np.array(utilities)
            # distances = Parallel(n_jobs=n_jobs)(delayed(post.close2diagonal)(x) for x in measurements)
            # f = lambda x: np.sum(PointObservation(measurements=x, **post.specs).eigenvalues())
            # sum_eigenvalues = Parallel(n_jobs=n_jobs)(delayed(f)(x) for x in measurements)
            # random = pd.DataFrame({'utility': utilities,
            #                        'diag_dist': distances,
            #                        'target': 'random',
            #                        'm': m,
            #                        'transform': transform,
            #                        'L': L,
            #                        'N': N,
            #                        'sum_eigenvalues': sum_eigenvalues})
            # random['x'] = [x for x in measurements]

            # # Optimal approximate point
            # approx = post.optimize(m=m,
            #                        target='diag',
            #                        n_iterations=1*n_jobs,
            #                        n_jobs=n_jobs,
            #                        full=True)
            # approx = pd.DataFrame(approx)
            # print(f"Approx={approx.utility.min():3.3f}", end=' ')

            # Optimal point
            point = post.optimize(m=m, target='utility')
            print(f"Point={point['utility']:3.3f}")

            #import pdb; pdb.set_trace()
            point_obs = PointObservation(measurements=point['x'], N=N, L=L, transform=transform)
            # plt.plot(prior.x, sample)
            # plt.scatter(point_obs.measurements, point_obs(sample))
            # plt.show()
            pt_sum = point['sum_eigenvalues']
            diag_sum = np.sum(post.optimal_diagonal_O**2)
            # approx_sum = np.sum(approx.loc[0, 'sum_eigenvalues'])
            print(f"Sum    : Diag={diag_sum:3.3f} Point={pt_sum:3.3f}")  #Approx={approx_sum:3.3f}

            # for row, target in enumerate(['utility', 'diag_dist']):
            #     kde = gaussian_kde(random[target])
            #     xx = np.linspace(random[target].min()*0.9, random[target].max()*1.1, num=1000, endpoint=True)
            #     ax = axes[row, col]

            #     ax.hist(random.utility, density=True, alpha=0.3, label='hist')
            #     ax.plot(xx, kde(xx), label='random')

            #     ax.scatter(approx.utility, kde(approx.utility), label='point approximating diagonal')

            #     ax.scatter(point['utility'], kde(point['utility']), label='optimal point', color='r')

            #     ax.legend()
            #     ax.set_title(f'{m} measurements with {transform}')
            #     fig.tight_layout()
            #     fig.savefig(f'pix/{transform}.png')
        # print(f'finish {transform} {m}')


if __name__ == '__main__':
    try:
        if not os.path.exists('pix'):
            os.mkdir('pix/')
        main(os.cpu_count()-1)
    except:
        import pdb, traceback, sys
        traceback.print_exc()
        _, _ , tb = sys.exc_info()        
        pdb.post_mortem(tb)
