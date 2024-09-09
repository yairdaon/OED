## OED

Code to generate figures from my paper "Measurement Clusterization in Bayesian
D-optimal Designs in Infinite Dimensions".

#The Directory src/

src/ contains an implementation of the parts of the inverse
problem I needed. Particularly, it contains a class called
FourierMultiplier that implements transporting the problem from the
spatial domain to the freqiuency domain. This is done with one of two
transformations: the discrete cosine transform and the discrete sine
transform. The former is used when we take a homogeneous Neumann
boundary condition, while the latter is used when we take a
homogeneous Dirichlet boundary condition.

The directory src/ contains several other modules:

fwd.py implements the time evolotion of the heat equation

probability.py implements a Prior class, which is used for its
covariance and a Posterior class, which calculates D-optimal measurement
locations

observation.py implements an observation operator by transforming
point evaluations to the frequency domain.

# The directory tests/

Please ignore this one, it is filled with old tests I wrote and I am
keeping in case I ever need them.


# Scripts

clusterization.py generates a clustered design for 1D heat equation
inverse problem with homogeneous Dirichlet boundary condition (see
manuscript for details). It can generate a D-optimal (clustered)
design for the homogeneous Neumann boundary condition as well. If one
modifies the value of variable model_error inside this script, it can
add model error to the calculation, thus avoiding clusterization.

movies.py generates a forward and a backward time evolution of the
heat equation.

zeros.py implements the constructive existence proof of Lemma 15 from
the paper. It runs said construction for many numbers of measurements
(m) and possible ranks of O^*O (i.e. k). For each such pair it
generates 2000 random matrices PSD M with trace M = m. It then
constructs A such that AA^t = M and A has unit norm columns.	
