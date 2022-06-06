import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from pdb import set_trace as stahp

# from importlib import reload
# import src.probability
# reload(src.probability)
# reload(src.multiplier)
# reload(src.observations)
from src.probability import Prior, Posterior
from src.multiplier import FourierMultiplier
from src.observations import PointObservation, DiagObservation
from src.forward import Heat

class TestPrior(Prior):
    def __init__(self,
                 eigenvalues,
                 **kwargs):
        super().__init__(**kwargs)
        eigenvalues = np.array(eigenvalues)
        self.gamma = None
        self.delta = None
        self.multiplier = np.zeros(self.N)
        self.multiplier[:eigenvalues.size] = eigenvalues
        self.inv_mult = np.full(self.N, np.inf)
        self.inv_mult[:eigenvalues.size] = 1/eigenvalues
        

class TestForward(FourierMultiplier):
    def __init__(self,
                 eigenvalues,
                 **kwargs):
            
        super().__init__(**kwargs)
        eigenvalues = np.array(eigenvalues)
        self.multiplier = np.zeros(self.N)
        self.multiplier[:eigenvalues.size] = eigenvalues

    def __str__(self):
        return f'Test Forward operator'

def main():
    n = 3
    N = 1000
    L = 1
    sigSqr = 1e-1
    prior_eigs = np.ones(n)#np.power(2., -np.arange(n))
    fwd_eigs = np.ones(n)
    ms = [1, 2, 3, 5, 7, 10, 12]
    transforms = ['dct', 'fft']
    for transform in transforms:
        print('\n\n')
        print(transform)
        for m in ms:
            prior = TestPrior(N=N, L=L, transform=transform, eigenvalues=prior_eigs)
            fwd = TestForward(N=N, L=L, transform=transform, eigenvalues=fwd_eigs)
            post = Posterior(prior=prior, fwd=fwd, N=N, L=L, transform=transform, sigSqr=sigSqr)
            diagonal_O = post.make_optimal_diagonal(m)
            # print(diagonal_O)
            eigs = prior.multiplier ** 2 * fwd.multiplier
            psi = sum(np.log(1 + eig * dia**2 /sigSqr) for eig, dia in zip(eigs, diagonal_O))
            
            meas = np.random.uniform(0, L, size=m)
            pt = PointObservation(measurements=meas, N=N, L=L, transform=transform)
            # [plt.plot(pt.x, pt.to_time_domain(y)) for y in pt.multiplier]
            # plt.scatter(meas, np.ones(m))
            # plt.show()
            sample = fwd(prior.sample().squeeze())
            post.optimize_diag(m=m)
            post.make_optimal_diagonal(m)
            diagonal_utility = post.diag_utility(diagonal_O)
            point_utility = post.point_utility(meas)
            print(f"{m}: Point {point_utility:2.7f}, diagonal {diagonal_utility:2.7f}, analytic {psi:2.7f}")
        
            # diag_obs = DiagObservation(multiplier=diagonal_O,  N=N, L=L, transform=transform)
          
            # do = diag_obs(sample)
            # po = point_obs(sample)
            # import pdb;pdb.set_trace()
if __name__ == '__main__':
    try:
        main()
    except:
        import pdb, traceback, sys
        traceback.print_exc()
        _, _ , tb = sys.exc_info()        
        pdb.post_mortem(tb)

