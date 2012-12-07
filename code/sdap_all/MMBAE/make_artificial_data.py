'''
Created on Jul 3, 2012

@author: neeraj
'''

import scipy.sparse as sp
from numpy.random import random
from numpy.random import randn
from numpy.random import randint 
from numpy.random.mtrand import dirichlet
from misc import sample_discrete
import numpy as np
from misc import get_loglikelihood_art_data

def make_artificial_data(M,N,D1,D2,K,L,no_obs):
    X1 = random((M,D1))
    X2 = random((N,D2))
    Xs = [X1,X2]
    
    beta = randint(low = 0, high = 10, size = (K,L,1 + X1.shape[1] + X2.shape[1])) + random((K,L,1 + X1.shape[1] + X2.shape[1]))
    sigmaY = randint(low = 0, high = 10, size = (K,L)) + random((K,L))
    
    alphas = [randint(low = 0, high = 10000, size = (K,)) + random(K,), randint(low = 0, high = 1000, size = (L,)) + random(L,)]
    alpha1 = alphas[0]
    alpha2 = alphas[1]
    pi1 = dirichlet(alpha1,(M,1))
    pi2 = dirichlet(alpha2,(N,1))
    z1 = sample_discrete(pi1,(M,1))
    z2 = sample_discrete(pi2,(N,1))
   
    made_ij = False
    I = []
    J = []
    prev_len = 0
    while made_ij == False:  
        I.extend(randint(low = 0,high = M, size = (no_obs - prev_len,)))
        J.extend(randint(low = 0,high = N, size = (no_obs - prev_len,)))
        W = sp.csr_matrix((np.ones(no_obs),(I,J)), shape=(M,N))
        I,J = sp.find(W)[:2]
        I = list(I)
        J = list(J)
        if len(I) == no_obs:
            made_ij = True
        else:
            prev_len = len(I)
    
    Xbias = np.ones((no_obs,1)) # |Yobs| x 1
    Xusers = X1[I,:].reshape((no_obs,D1)) # |Yobs| x D1
    Xitems = X2[J,:].reshape((no_obs,D2)) # |Yobs| x D2
    X = np.hstack((Xbias, Xusers, Xitems)) # |Yobs| x (1 + D1 + D2)    
    Y = np.zeros((no_obs,)) # |Yobs| x 1
    for o in range(no_obs):
        Y[o] = sigmaY[int(z1[I[o]][0]),int(z2[J[o]][0])] * randn() + np.dot(beta[int(z1[I[o]][0]),int(z2[J[o]][0]),:],X[o,:])
    
    pis = [pi1,pi2]
    zs = [z1,z2]
    betas = [beta,sigmaY]
 
    params = {'alphas':alphas, 
              'pis':pis, 
              'zs':zs, 
              'betas':betas
              }
        
    return Xs, Y, I, J, params

def get_likelihood_art_data(alphas, pis, zs, betas, I, J, Y, X1, X2, model_name):
    log_likelihood_art_data = get_loglikelihood_art_data(alphas, pis, zs, betas, I, J, Y, X1, X2, model_name)
    return log_likelihood_art_data