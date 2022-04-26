import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

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
                 eigenvalues=[1],
                 **kwargs):
        super().__init__(**kwargs)
        eigenvalues = np.array(eigenvalues)
        self.gamma = None
        self.multiplier = np.zeros(self.N)
        self.multiplier[:eigenvalues.size] = eigenvalues
        self.inv_mult = np.full(self.N, np.inf)
        self.inv_mult[:eigenvalues.size] = 1/eigenvalues
        self.ind = None

class TestForward(FourierMultiplier):
    def __init__(self,
                 eigenvalues=[1],
                 **kwargs):
            
        super().__init__(**kwargs)
        eigenvalues = np.array(eigenvalues)
        self.multiplier = np.zeros(self.N)
        self.multiplier[:eigenvalues.size] = eigenvalues

    def __str__(self):
        return f'Test Forward operator'

def main():
    sigSqr = 1e-1
    prior_eigs = np.array([0.05, 0.01, 1, 0.2, 0.9, 1, 2, 0.0003])
    fwd_eigs = 1 / np.arange(1, prior_eigs.size+1)
    f_gamma_f_eigs = fwd_eigs**2 * prior_eigs
    diagonal_O = np.array([1])
    psi = sum(np.log(1 + eig * dia**2 /sigSqr) for eig, dia in zip(f_gamma_f_eigs, diagonal_O))
   
    L, N = np.sqrt(np.pi), 10000
    # m = 1
    ms = [1, 2, 3, 5, 7, 10, 12]
    transforms = ['dct', 'fft']
    for transform in transforms:
        for m in ms:
            prior = TestPrior(N=N, L=L, transform=transform, eigenvalues=prior_eigs)
            fwd = TestForward(L=L, N=N, transform=transform)
            meas = np.random.uniform(0, L, size=m)
            pt = PointObservation(measurements=meas, N=N, L=L, transform=transform)
            post = Posterior(prior=prior, fwd=fwd, N=N, L=L, transform=transform, sigSqr=sigSqr)

            # sample = fwd(prior.sample().squeeze())
            
            post.make_optimal_diagonal(m)
            diagonal_utility = post.diagonal_utility(diagonal_O)
            point_utility = post.point_utility(meas)
            print(f"Point {point_utility:2.7f}, diagonal {diagonal_utility:2.7f}, analytic {psi:2.7f}")
        
            diag_obs = DiagObservation(multiplier=diagonal_O,  N=N, L=L, transform=transform)
          
            diag_obs(sample)
            point_obs(sample)
            import pdb;pdb.set_trace()
if __name__ == '__main__':
    try:
        main()
    except:
        import pdb, traceback, sys
        traceback.print_exc()
        _, _ , tb = sys.exc_info()        
        pdb.post_mortem(tb)

