import numpy as np
from numpy import sin, cos
from scipy.stats import ortho_group
import pdb

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
    assert m > k
    
    D = np.abs(np.random.uniform(size=k))
    D = D * m / np.sum(D)
    assert abs(np.sum(D) - m) < 1e-12 
    D = np.diag(D)
    assert D.shape == (k,k)

    U = ortho(k)
    assert np.all(np.abs(np.dot(U,U.T) - np.identity(k)) < 1e-12)

    M = conj(D, U)

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
        raise ValueError(f'Why upper == 0 and C.shape == {C.shape}?')
    if C[upper,upper] == 0:
        return None, None

    success = False
    for lower in reversed(range(upper)):
        ckk = C[upper,upper]
        cpp = C[lower,lower]
        if ckk * cpp < 0:
            success = True
            break
    assert success
    
    ## Comment refers to common names in quadratic equation lingo
    a = C[upper,upper] 
    b = 2*C[upper,lower]
    c = C[lower,lower] 
    assert a * c < 0
    
    ## Solving the quadratic to get cot(theta), which will soon
    ## be inverted to find theta.
    cot1 = (-b + np.sqrt(b**2 - 4*a*c))/ 2 / a
    cot2 = (-b - np.sqrt(b**2 - 4*a*c))/ 2 / a ## Second solution

    theta1 = np.arctan(1/cot1)
    theta2 = np.arctan(1/cot2)
    if 0 <= theta1 <= np.pi:
        return theta1, lower
    elif 0 <= theta2 <= np.pi:
        return theta2, lower
    else:
        raise ValueError(f'Thetas == {theta1:2.2f}, {theta2:2.2f} not in [0,pi]')
        

    
def main(m, k):
    assert m > k
    M, D, U, S = MDUS(m, k)
    
    C = np.dot(S.T, S) - np.identity(m)
    assert abs(np.trace(C)) < 1e-14

    V = np.identity(m)
    for upper in reversed(range(1,k+1)):
        theta, lower = get_theta(C, upper)
        R = givens(theta=theta, dims=m, lower=lower, upper=upper)
        C = conj(C, R)
        V = np.dot(R, V)
  
    ## Verify C has zero trace
    assert abs(np.trace(C)) < 1e-14

    ## Verify V is orthogonal
    assert np.all(np.abs(np.diag(np.dot(V,V.T)) - 1) < 1e-14)
    assert np.all(np.abs(np.dot(V,V.T) - np.identity(m)) < 1e-14)

    ## Check that AAt = M, as stated.
    A = np.einsum('ij, jk, lk ->il', U, S, V)
    AAt = np.dot(A,A.T)
    assert np.all(np.abs(AAt - M) < 1e-12)
    
    ## Check that A does have unit norm columns, as stated.
    norms = np.linalg.norm(A, axis=0)
    assert np.all(np.abs(norms - 1) < 1e-14) 


    
def test_theta(big_dim=20, dim=9, upper=4):
    assert big_dim > dim > upper
    M, D, U, S = MDUS(big_dim, dim)
    
    C = M - np.identity(dim) * np.trace(M) / dim
    assert abs(np.trace(C)) < 1e-14
    
    theta, lower = get_theta(C, upper)
    R = givens(theta=theta, dims=dim, lower=lower, upper=upper)
    T = conj(C, R)

    cot = cos(theta) / sin(theta)
    eqn = cot**2 * C[upper, upper] + 2*cot*C[upper,lower] + C[lower,lower]
    assert abs(eqn) < 1e-13
    assert abs(T[upper,upper]) < 1e-14
  


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

    assert np.all(np.abs(res) < 1e-13)

    
if __name__ == '__main__':
    try:
        test_givens()
        test_theta()
        main(k=4, m=5)

    except:
        import sys, traceback, pdb
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

        
