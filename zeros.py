## This here code implements (and tests) the algorithm outlined in
## Lemma B.5 in the paper.
##
## Given M symmetric positive definite in R^{k X k} such that
##
## tr M = m > k, (m an integer),
## 
## it finds A such that:
##
## A's columns have unit norm, and
## AA^t = M. 

import numpy as np
from pandas import isnull
from numpy import sin, cos
from scipy.stats import ortho_group
import pdb

EPS = 1e-10
ortho = ortho_group.rvs
cot = lambda x: cos(x)/ sin(x)


def conj(D, U):
    ''' Returns UDU^t'''
    assert U.shape == D.shape
    return np.einsum('ij, jk, lk -> il', U, D, U)


def givens(theta, dims, lower, upper):    
    assert theta >= 0 and theta <= np.pi
    assert lower < upper
    assert upper <= dims
    R = np.zeros((dims,dims))
    for a in range(dims):
        for b in range(dims):
            if a == b: 
                if a == b == lower:
                    R[a,b] = cos(theta)
                elif a == b == upper:
                    R[a,b] = cos(theta)
                else:
                    R[a,b] = 1
            elif a == lower and b == upper:
                R[a,b] = -sin(theta)
            elif a == upper and b == lower:
                R[a,b] = sin(theta)            
    return R


def MDUS(m, k):
    assert m >= k
    
    # D = np.abs(np.random.uniform(size=k))
    D = np.random.lognormal(mean=5, sigma=1, size=k)
    # D = np.array([np.random.lognormal(mean=i, sigma=1) for i in range(k)]) 
    D = D * m / np.sum(D)
    assert abs(np.sum(D) - m) < EPS 
    assert np.all(D > 0)
    D = np.diag(D)
    assert D.shape == (k,k)
    assert np.all(~isnull(D))
    

    U = ortho(k)
    assert np.all(np.abs(np.dot(U,U.T) - np.identity(k)) < EPS)

    M = conj(D, U)
    M = (M + M.T)/2 ## Ensure it is symmetric
    
    S = np.zeros((k,m))
    for i in range(k):
        S[i,i] = np.sqrt(D[i,i])

    assert M.shape == D.shape == U.shape == (k,k)
    assert S.shape == (k,m)
    
    return M, D, U, S

def get_theta(C, upper):
    assert upper < C.shape[0]
    assert C.shape[0] == C.shape[1]
    if upper == 0:
        raise ValueError(f'Why upper == 0? C.shape == {C.shape}.')
    if abs(C[upper,upper]) < EPS:
        return None, None

    success = False
    for lower in reversed(range(upper)):
        ckk = C[upper,upper]
        cpp = C[lower,lower]
        if ckk * cpp < 0:
            success = True
            break
    assert success, f'ckk = {ckk:2.3f}, cpp = {cpp:2.3f}, sum = {ckk+cpp:2.3f}' 
    
    ## Comment refers to common names in quadratic equation lingo
    c2 = C[upper,upper] 
    c1 = 2*C[upper,lower]
    c0 = C[lower,lower] 
    assert c2 * c0 < 0
    
    ## Solving the quadratic to get cot(theta), which will soon
    ## be inverted to find theta.
    disc = c1**2 - 4*c2*c0 # Discriminant
    assert disc >= 0
    cot1 = (-c1 + np.sqrt(disc))/ 2 / c2
    cot2 = (-c1 - np.sqrt(disc))/ 2 / c2 ## Second solution
    
    
    theta1 = np.arctan(1/cot1)
    theta2 = np.arctan(1/cot2)
    if 0 <= theta1 <= np.pi:
        return theta1, lower
    elif 0 <= theta2 <= np.pi:
        return theta2, lower
    else:
        raise ValueError(f'Thetas == {theta1:2.2f}, {theta2:2.2f} not in [0,pi]')
        

    
def main(m, k):
    assert m >= k
    M, _, _, _ = MDUS(m, k)

    A = get_A(M, m)
    assert A.shape == (k,m)
    
    ## Check that AAt = M, as stated.
    AAt = np.dot(A,A.T)
    assert np.all(np.abs(AAt - M) < EPS)
    
    ## Check that A does have unit norm columns, as stated.
    norms = np.linalg.norm(A, axis=0)
    assert len(norms) == m
    assert np.all(np.abs(norms - 1) < EPS) 


    for vec in A.T:
        pos_ind = np.all(np.abs(A.T - vec) < EPS, axis=1)
        neg_ind = np.all(np.abs(A.T + vec) < EPS, axis=1)
        ind = np.logical_or(pos_ind, neg_ind).astype(int)
        assert ind.shape == (A.shape[1],)
        if np.sum(ind) > 1:
            print(ind)
            #pdb.set_trace()
    
    
def get_A(M, m):
    k = M.shape[0]
    assert abs(m - np.trace(M)) < EPS
    
    D, U = np.linalg.eigh(M)
    assert np.all(D > 0)
    
    S = np.zeros((k,m))
    for i in range(k):
        S[i,i] = np.sqrt(D[i])

    
    C = np.dot(S.T, S) - np.identity(m)
    V = np.identity(m)
    for upper in reversed(range(1,m)):
        assert abs(np.trace(C)) < EPS
        theta, lower = get_theta(C, upper)
        if theta is None and lower is None:
            continue
        R = givens(theta=theta, dims=m, lower=lower, upper=upper)
        C = conj(C, R)
        V = np.dot(R, V)

    A = np.einsum('ij, jk, lk ->il', U, S, V)

    ## Verify C has zero trace
    assert abs(np.trace(C)) < EPS

    ## Verify V is orthogonal
    assert np.all(np.abs(np.diag(np.dot(V,V.T)) - 1) < EPS)
    assert np.all(np.abs(np.dot(V,V.T) - np.identity(m)) < EPS)

    return A

    
def test_theta(big_dim=20, dim=9, upper=4):
    assert big_dim > dim > upper
    M, D, U, S = MDUS(big_dim, dim)
    
    C = M - np.identity(dim) * np.trace(M) / dim
    assert abs(np.trace(C)) < EPS
    
    theta, lower = get_theta(C, upper)
    R = givens(theta=theta, dims=dim, lower=lower, upper=upper)
    T = conj(C, R)

    cot = cos(theta) / sin(theta)
    eqn = cot**2 * C[upper, upper] + 2*cot*C[upper,lower] + C[lower,lower]
    assert abs(eqn) < EPS
    assert abs(T[upper,upper]) < EPS
  


def test_givens(k=8, lower=4, upper=6):
    M, D, U, S = MDUS(k+2, k)
    
    ## Testing Givens rotations R
    R = givens(2.3, k, lower, upper) 
    C = conj(M, R)

    res = np.abs(M - C)
    res[lower,:] = 0
    res[upper,:] = 0
    res[:,lower] = 0
    res[:,upper] = 0

    assert np.all(np.abs(res) < EPS)

    
if __name__ == '__main__':
    try:
        test_givens()
        test_theta()
        for m in range(0,10,3):
            for k in range(2, m+1, 1):
                print(m, k) 
                main(k=k, m=m)
                       
    except:
        import sys, traceback, pdb
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

        
