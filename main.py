import numpy as np
from matplotlib import pyplot as plt

from src.probability import Prior, Posterior
from src.forward import Heat
from src.observations import PointObservation, DiagObservation
from src.multiplier import FourierMultiplier



def main():

    ## DST / DCT - discrete sine / cosine transform. 
    ## Corresponds to a homogeneous Dirichlet/ Neumann boundary condition.
    ## Clusterization also happens with Neumann boundary, I just chose to present 
    ## Dirichlet boundary
    transform = 'dst'
    delta = 0. if transform == 'dst' else 0.5
    boundary = 'Dirichlet' if transform == 'dst' else 'Neumann'
    
    N = 200
    L = 1
    time = 3e-2
    alpha = 1.
    gamma = -1.
    model_error = True
    sig = 5e-2
    
    fwd = Heat(N=N, L=L, transform=transform, alpha=alpha, time=time)
    pr = Prior(N=N, L=L, transform=transform, gamma=gamma, delta=delta)
    post = Posterior(fwd=fwd,
                     prior=pr,
                     sigSqr=sig**2,
                     L=L,
                     N=N,
                     transform=transform,
                     model_error=True)

    dic = {}
    print("model error", model_error, 'transform', transform)
    for m in range(3, 8):
        print(m, end=' ')
        res = post.optimize(m=m, n_iterations=20, n_jobs=-1)
        design = res['x']
        dic[m] = design
        print(design)


    fs = 24
    fig, ax = plt.subplots(figsize=(10,5))

    for m, array in dic.items():
        colors = iter(list("brgkmcy"))

        vals = np.repeat(m, len(array))
        ax.scatter(array, vals, s=0)
        for i, val in enumerate(array):
            ax.annotate(str(i+1),
                        xy=(val, m),
                        ha='center', 
                        va='center',
                        fontsize=fs,
                        color=next(colors))
            

    ax.set_xlabel('Measurement Location', fontsize=fs)
    ax.set_ylabel('No. of Measurements', fontsize=fs)
    ax.set_yticks(list(dic.keys()))
    ax.set_xlim(0,1)
    plt.title(f"D-Optimal Measurements for Heat Equation", fontsize=fs)
    plt.tight_layout()


    plt.show()
    if model_error:
        fname = f"latex/{boundary}_model_error_sig{sig}.pdf"
    else:
        fname = f"latex/{boundary}_no_model_error_sig{sig}.pdf"
    plt.savefig(fname)
    #plt.close()

    
if __name__ == '__main__':
    main()
