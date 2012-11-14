'''
Created on Nov 2, 2012

@author: neeraj
'''
import numpy as np
from scipy.optimize import fmin_l_bfgs_b as bfgs
import scipy.sparse as sp
from scipy.optimize.minpack import check_gradient

def matdot(U, V, rows, cols):
    return (U[rows]*V[cols]).sum(axis=1)

def matvec(V, E, rows, cols, M, N):
    Matrix   = sp.coo_matrix((E, np.vstack((rows, cols))), shape=(M, N), dtype=float).tocsr()
    return np.array(Matrix.dot(V), copy=False)
    
def f_and_g(w, reg, rows, cols, Y, M, N, rank):
    '''Matrix factorization with sq loss'''
    #######################################
    # re-set sizes
    #######################################
    W = w.reshape((M+N, rank))
    U = W[:M]
    V = W[M:]
    
    # the following avoids unnecessary memory copy for MF.
    Yp = matdot(U, V, rows, cols) # U, V -> Yp
    E = Yp - Y
    cost  = 0.5*( np.dot(E,E) + reg*( np.dot(U.ravel(), U.ravel()) + np.dot(V.ravel(), V.ravel()) ) ) 
    
    # grad of cost = KEV, GE'U
    EV = matvec(V, E, rows, cols, M, N)
    EU = matvec(U, E, cols, rows, N, M)
    
    grad     = np.empty(W.shape)
    grad[:M] = EV + reg*U
    grad[M:] = EU + reg*V
    #print cost/len(Y)
    return cost, grad.ravel()


def mf_solver(rows, cols, vals, rank, reg, M, N):
    w0 = np.random.randn((M+N)*rank)
    w = bfgs(func=f_and_g, x0=w0, args=(reg, rows, cols, vals, M, N, rank),maxfun=500)[0]
    W = w.reshape((M+N, rank))
    U = W[:M]
    V = W[M:]
    
    return U, V