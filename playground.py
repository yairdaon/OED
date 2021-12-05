import numpy as np

from src.multiplier import FourierMultiplier as FM
from src.observations import PointObservation as Point
from src.probability import Posterior

def main():
    m = 7
    post = Posterior(N=576, L=np.pi, transform='dct')

    res = post.optimal(m=m, eps=1e-3)
    obs = res['obs']

    #meas = np.arange(1,m-0.05) / m
    #meas = np.random.uniform(size=m)
    #meas = [0.005] * m
    #obs = Point(meas, **post.specs)


    print(obs.shape[0], np.sum(obs.eigenvalues()))

    res['x'].sort()
    print(res['x'])

    obs = Point(**obs.specs, measurements=obs.measurements[:-1])
    print(np.sum(obs.eigenvalues()))


if __name__ == '__main__':
    main()
