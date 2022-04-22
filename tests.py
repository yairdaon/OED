import numpy as np
from matplotlib import pyplot as plt

# from importlib import reload
# import src.probability
# reload(src.probability)
# reload(src.multiplier)
# reload(src.observations)
from src.probability import Prior, Posterior
from src.multiplier import FourierMultiplier
from src.observations import PointObservation, DiagObservation

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
    fwd_eigs = 1 / np.array([2])
    prior_eigs = np.array([0.05])
    f_gamma_f_eigs = fwd_eigs**2 * prior_eigs
    diagonal_O = np.array([1])
    psi = sum(np.log(1 + eig * dia**2 /sigSqr) for eig, dia in zip(f_gamma_f_eigs, diagonal_O))
    
    L, N = np.pi, 100
    m = 1
    ms = [1, 2, 3, 5, 7, 10, 12]
    transforms = ['fft']
    for transform in transforms:
        prior = TestPrior(N=N, L=L, transform=transform, eigenvalues=prior_eigs)
        fwd = TestForward(N=N, L=L, transform=transform, eigenvalues=fwd_eigs)
        post = Posterior(fwd=fwd, prior=prior, sigSqr=sigSqr, N=N, L=L, transform=transform)
        
        #post.make_optimal_diagonal(m)
        diagonal_utility = post.diagonal_utility(diagonal_O)
        point_utility = post.point_utility([4])
        print(f"Point {point_utility:2.7f}, diagonal {diagonal_utility:2.7f}, analytic {psi:2.7f}")
        
        point_obs = PointObservation(measurements=[2], N=N, L=L, transform=transform)
        diag_obs = DiagObservation(multiplier=diagonal_O,  N=N, L=L, transform=transform)
        sample = prior.sample(n_sample=1).ravel()
        diag_obs(sample)
        point_obs(sample)
        import pdb; pdb.set_trace()
        
if __name__ == '__main__':
    try:
        main()
    except:
        import pdb, traceback, sys
        traceback.print_exc()
        _, _ , tb = sys.exc_info()        
        pdb.post_mortem(tb)

