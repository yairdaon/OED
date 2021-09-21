from forward import Heat
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from observations import PointObservation
from probability import Posterior, Prior

# plt.rcParams['figure.figsize'] = [8, 4]
# plt.rcParams['figure.dpi'] = 200  # 200 e.g. is really fine, but slower
# plt.rcParams['axes.facecolor'] = 'black'
# plt.rcParams['figure.facecolor'] = 'black'
# plt.rcParams['text.color'] = 'white'


def main():
    num_observations = 10
    specs = {'N': 500, 'L': 3, 'transform': 'dct'}
    prior = Prior(gamma=-2, **specs)
    forward = Heat(time=3e-1, alpha=0.6, **specs)

    posterior = Posterior(fwd=forward, prior=prior, sigSqr=1e-3, **specs)
    initial_condition = np.random.uniform(low=0, high=posterior.L, size=num_observations)
    bounds = [(0, posterior.L)] * num_observations
    res = minimize(posterior.minimization_target, bounds=bounds, x0=initial_condition)
    print(res)
    measurements = res['x']
    obs = posterior.observation(measurements)
    ptwise = posterior.mean_std(obs=measurements)

    plt.figure(figsize=(6, 3))
    plt.plot(posterior.x, ptwise, label='posterior STD')
    plt.scatter(measurements, obs(ptwise))
    plt.show()


if __name__ == '__main__':
    main()