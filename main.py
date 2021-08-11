from matplotlib import pyplot as plt
import numpy as np

from observations import PointObservation

plt.rcParams['figure.figsize'] = [8, 4]
plt.rcParams['figure.dpi'] = 200  # 200 e.g. is really fine, but slower
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['text.color'] = 'white'


def main():
    k = 7
    obs = PointObservation(N=1400, L=2, meas=np.linspace(0, 2, 50))
    u = obs.eigenvector(k)

    fig = plt.figure(figsize=(6, 3))
    plt.plot(obs.x, u)
    plt.scatter(obs.meas, obs(u).real)
    plt.show()


if __name__ == '__main__':
    main()