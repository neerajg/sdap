
'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

import numpy as np
from numpy.random import random
from numpy.random import randint
from numpy.random.mtrand import dirichlet
from scipy.special import gammaln as gammaln
from scipy.special import psi as digamma
import scipy.sparse as sp

def initialize(X1, X2, train_I, train_J, train_Y, K, L):
    M = X1.shape[0]
    N = X2.shape[0]
    no_obs = len(train_Y)
    # Form the composite attributes vector   
    Xbias = np.ones((no_obs,1)) # |Yobs| x 1
    Xusers = X1[train_I,:] # |Yobs| x D1
    Xitems = X2[train_J,:] # |Yobs| x D2
    X_composite = np.hstack((Xbias, Xusers, Xitems)) # |Yobs| x (1 + D1 + D2)
    
    # Initialize alphas, betas, r and gammas
    alphas, gammas, betas, thetas, r, s = init_params(K, L, M, N, X1, X2, no_obs, train_I, train_J)
    return alphas, gammas, betas, thetas, r, s, X_composite, M, N

def init_params(K, L, M, N, X1, X2, no_obs, train_I, train_J):
    # TO DO : need a way to make initialization of sigma in such a way that in the beginning not too much of r gets even out because of this or gets neglected
    # (update r log exp problem)
    alphas = [random(K,),random(L,)]
    alphas[0] = alphas[0]/np.sum(alphas[0])
    alphas[1] = alphas[1]/np.sum(alphas[1])
    gammas = [randint(low = 50, high = 500, size = (M,K)) + random((M,K)), randint(low = 1.46, high = 3, size = (N,L)) + random((N,L))]
    beta_shape = (K,L,1 + X1.shape[1] + X2.shape[1])
    sigmaY_shape = (K,L)
    #randint(low = -1, high = 1, size = beta_shape) + 
    betas = [random(beta_shape), randint(low = 10, high = 50, size = sigmaY_shape) + random(sigmaY_shape)]
      
    m1 = np.zeros((K,X1.shape[1])) + random((K,X1.shape[1]))
    m2 = np.zeros((L,X2.shape[1])) + random((L,X2.shape[1]))
    sigma1 = np.zeros((K,X1.shape[1])) + random((K,X1.shape[1]))
    sigma2 = np.zeros((L,X2.shape[1])) + random((L,X2.shape[1]))
    theta1 = [m1,sigma1]
    theta2 = [m2,sigma2]
    thetas = [theta1,theta2]
        
    r1 = dirichlet(alphas[0], M)
    r1[r1<1e-4] = 1e-4
    #r1[r1>0.99] = 0.9
    r2 = dirichlet(alphas[1], N)
    r2[r2<1e-6] = 1e-6
    #r2[r2>0.9] = 0.9    
    r = [r1,r2]
    ones = np.ones((len(train_I),))
    mu = sp.csr_matrix((ones, (train_I,train_J)), shape=(M,N)).sum(1)
    mv = sp.csr_matrix((ones, (train_I,train_J)), shape=(M,N)).sum(0).transpose()
    mu[mu<1] = 1
    mv[mv<1] = 1
    
    s1 = dirichlet(alphas[0], M)
    s1[s1<1e-4] = 1e-4
    #r1[r1>0.99] = 0.9
    s2 = dirichlet(alphas[1], N)
    s2[s2<1e-6] = 1e-6
    #r2[r2>0.9] = 0.9    
    s = [s1,s2]     
    
    gammas[0] = np.tile(alphas[0].reshape(1,K), (M,1)) + s[0] + np.multiply(r[0],mu) # M x K
    gammas[1] = np.tile(alphas[1].reshape(1,L), (N,1)) + s[1] + np.multiply(r[1],mv) # N x L
                  
    return alphas, gammas, betas, thetas, r, s

""" Helper Function to build the objective function to be maximized for updating the
alphas. Code built using the notations for alpha1, works for alpha2 too since
the operations on alpha2 are the same with different notation 
Inputs -
    Alpha - K/L x 1
    Gamma - M/N x K/L """
def objective_fn(alpha, gamma, reg_alpha):
    # TO DO: vectorize this over k
    M,K = gamma.shape
    fn = M*(gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) - reg_alpha*np.sum(alpha**2))
    grad = M*digamma(np.sum(alpha))*np.ones((K,))  
    
    for k in range(K):
        # Denominator of B(alpha)
        #fn -= M*gammaln(alpha[k])
        grad[k] += - M*(digamma(alpha[k]) - 2*reg_alpha*alpha[k]) # TO DO : This gives overflow errors sometime (check it out)
        fn += (alpha[k] - 1)*np.sum(digamma(gamma[:,k]) - digamma(np.sum(gamma,1)))
        grad[k] += np.sum(digamma(gamma[:,k]) - digamma(np.sum(gamma,1)))
    
    return -fn, -grad

def sample_discrete(pi,shape):
    Z = (np.zeros(shape))
    K = pi.shape[2]
    rand_numbers = random(shape)
    pi_cumsum = np.cumsum(pi, 2)
    for m in range(shape[0]):
        for n in range(shape[1]):
            if rand_numbers[m,n] == 0 or rand_numbers[m,n] <=pi_cumsum[m,0,0]:
                continue
            if rand_numbers[m,n] == 1 or rand_numbers[m,n] >=pi_cumsum[m,0,K-1]:
                Z[m,n] = K-1
                continue
            #Z[m,n] = filter(lambda x: x>=rand_numbers[m,n], pi_cumsum[m,0,:])[0]
            Z[m,n] = int((pi_cumsum[m,0,:] >=rand_numbers[m,n]).tolist().index(True))
    return Z

# TO DO : Vectorize this
def get_loglikelihood_art_data(alphas, pis, zs, betas, I, J, Y, X1, X2, model_name):
    log_likelihood_data = 0
    M,temp,K = pis[0].shape
    N,temp,L = pis[1].shape
    beta = betas[0]
    sigmaY = betas[1]
    users = list(set(I))
    movies = list(set(J))
    z1 = zs[0][users]
    z2 = zs[1][movies]
   
    no_obs = len(I)
    Xbias = np.ones((no_obs,1)) # |Yobs| x 1
    Xusers = X1[I,:] # |Yobs| x D1
    Xitems = X2[J,:] # |Yobs| x D2
    X = np.hstack((Xbias, Xusers, Xitems))    
    
    if model_name == 'bae_linear':
        
        log_likelihood_data += len(users)*(gammaln(np.sum(alphas[0])) - np.sum(gammaln(alphas[0])))
        for m in range (M):
            if m in users:
                for k in range(K):
                    log_likelihood_data +=  (alphas[0][k]-1)*np.log(pis[0][m,0,k])
                log_likelihood_data +=  np.log(pis[0][m,0,int(zs[0][m])])
            else:
                continue

        log_likelihood_data += len(movies)*(gammaln(np.sum(alphas[1])) - np.sum(gammaln(alphas[1])))
        for n in range (N):
            if n in movies:
                for l in range(L):
                    log_likelihood_data +=  (alphas[1][l]-1)*np.log(pis[1][n,0,l])
                log_likelihood_data +=  np.log(pis[1][n,0,int(zs[1][n])])
            else:
                continue
        
        for o in range(no_obs):
            beta_times_x = np.dot(X[o],beta[int(z1[I[o]][0]),int(z2[J[o]][0]),:])
            sigma_Y = sigmaY[int(z1[I[o]][0]),int(z2[J[o]][0])]
            log_likelihood_data += (-.5*np.log(2*np.pi*(sigma_Y**2))-(.5/(sigma_Y**2))*(Y[o]**2 - 2*Y[o]*beta_times_x + beta_times_x**2))
        
    return log_likelihood_data