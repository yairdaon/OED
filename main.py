import numpy as np
from src.forward import Heat
from src.probability import Posterior, Prior


# plt.rcParams['figure.figsize'] = [8, 4]
# plt.rcParams['figure.dpi'] = 200  # 200 e.g. is really fine, but slower
# plt.rcParams['axes.facecolor'] = 'black'
# plt.rcParams['figure.facecolor'] = 'black'
# plt.rcParams['text.color'] = 'white'




def main():
    specs = {'N': 400, 'L': np.pi}

    for m in [3, 7, 11, 17, 23, 29]:
        for transform in ['dct']: #, 'fft']:
            print('m =', m, transform)

            specs['transform'] = transform
            prior = Prior(gamma=-2, **specs)
            forward = Heat(time=3e-1, alpha=0.6, **specs)
            post = Posterior(fwd=forward, prior=prior, sigSqr=1e-2, **specs)

            # Optimal diagonal
            post.set_optimal_diagonal(m, plot=False)
            diagonal_utility = post.utility(post.optimal_diagonal)
            print(f"Utility: Diag={diagonal_utility:3.3f}", end=' ')

            # Optimal approximate point
            approx = post.optimal(m=m, target=post.close2diagonal, n_iterations=7)
            print(f"Approx={approx['utility']:3.3f}", end=' ')

            # Optimal point
            point = post.optimal(m=m, target=post.minimization_target)
            print(f"Point={point['utility']:3.3f}")

            pt_sum = np.sum(point['obs'].eigenvalues())
            diag_sum = np.sum(post.optimal_diagonal.singular_values())
            approx_sum = np.sum(approx['obs'].eigenvalues())
            print(f"Sum    : Diag={diag_sum:3.3f} approx={approx_sum:3.3f} Point={pt_sum:3.3f} ")
            print()


if __name__ == '__main__':
    main()