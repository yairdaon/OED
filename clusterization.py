import numpy as np
from matplotlib import pyplot as plt

from src.probability import Prior, Posterior
from src.forward import Heat
from src.observations import PointObservation


def main():

    sig = 5e-2 ## Observation error amplitude
    N = 200 ## Number of spatial discretizatio points
    L = 1 ## Length of computational domain
    time = 3e-2 ## Time for heat dissipation in arbitrary units.
    alpha = 1. ## Coefficient of Laplacian in heat equation
    gamma = -1. ## Exponent of Laplacian in prior. 

    ## Choose one, comment out the other
    transform = 'dct' ## dct - Discrete Cosine Transform. Corresponds
                      ## to homogenoeus Neumann boundary condition.
    transform = 'dst' ## dst - Discrete Sine Transform. Corresponds
                      ## homogenoeus Dirichlet boundary condition.

    ## We need to regularize the Laplacian so it is invertible and can
    ## function as a useful prior. Hence we add the constant delta
    ## when employing homogeneous Neumann boundary condition.
    delta = 0. if transform == 'dst' else 0.5
    
    ## Avoid clusterization by increasing this to e.g. 4
    model_error = 0

    ## Forward model - the heat equation with BC chosen according to transform
    fwd = Heat(N=N,
               L=L,
               transform=transform,
               alpha=alpha,
               time=time)

    ## Prior - with precision \delta
    pr = Prior(N=N,
               L=L,
               transform=transform,
               gamma=gamma,
               delta=delta)

    ## Posterior. We don't care about data, so no data is involved,
    ## only observation operators. This effectively implements
    ## calculating the D-optimality criterion design and searching for
    ## a D-optimal design.
    post = Posterior(fwd=fwd,
                     prior=pr,
                     sigSqr=sig**2,
                     L=L,
                     N=N,
                     transform=transform,
                     model_error=model_error)


    dic = {}

    ## Form  number of observations in this range
    for m in range(2, 6):

        ## Calculate an optimal design
        design = post.optimize(m=m, n_iterations=25)['x']

        ## And store it in dic
        dic[m] = design

        ## Print it so we see
        print(transform, m, design)
        

    
    fs = 24 ## Font size
    fig, ax = plt.subplots(figsize=(10,5)) ## Create figure object

    ## For every pair in dic, plot a the locations of the measurements
    ## at ordinate m, with different colors.
    for m, array in dic.items():

        colors = iter(list("brgkm"))
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
    plt.tight_layout()

    plt.savefig(f"latex/{transform}_modelError{model_error}.pdf")
    plt.show()

    
main()
