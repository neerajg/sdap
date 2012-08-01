"""

Author  : Neeraj Gaur
Date    : 02/21/2012
Title   : Learn parameter of the Mixed Membership Bayesian Affinity Estimation

"""

import sys
import pdb

import numpy as np
import scipy.sparse as sp

from scipy.special import psi as digamma
from scipy.special import gammaln as gammaln

from scikits.learn.linear_model import Ridge
from scikits.learn.utils.extmath import safe_sparse_dot as dot
#from sklearn.utils.extmath import safe_sparse_dot as dot
#from sklearn.linear_model import Ridge
import scipy.optimize as opt

import copy

from numpy.random.mtrand import dirichlet as sample_dirichlet
from numpy.random import random as np_random

import gc

sigma_tolerance = 1e-3

# TO DO: Modify all this to be able to accept tensors and to be able to accept
# the interaction term attributes

# TO DO: RIGHT NOW I AM USING LINEAR REGRESSION AS THE MODEL FOR PREDICTION FOR
# Y i.e Y = (BETA).(X) BECAUSE CLINT USED IT, THIS WILL PROBABLY CHANGE

""" Main function to learn the parameters of the MMBAE model
Inputs -
    Y - M x N ratings matrix
    X - V x N(v) x D(v) Covariates matrix.
        V - Number of modes (2 for our case right now)
        N(v) - Number of entities in mode v
        D(v) - Number of covariates for entities in mode v
    K - Number of clusters in mode 0 (right now hard-coded)
    L - Number of clusters in mode 1 (right now hard-coded)
    num_iter - maximum number of iterations
    W - M x N matrix with Wij = 1 is Yij is observed, else Wij = 0
    reg_lambda - Regularization constant lambda
    num_iter, precision - these control the convergence

Outputs - Parameters for the mmbae model
    Alphas - The priors for the 1 out of K/L categoricals
    Thetas - The parameters for the covariates
    Betas - The parameters for the ratings(interactions) """
def learn_bae(Y, W, X, K, L, num_runs, num_iter, precision, reg_lambda, initial_T=50, eta=0.5, init_point=None):
    
    P,Q, temp = sp.find(W) # Rows and col corresponding to Yobs
    Yobs = np.array(Y[(P,Q)]).reshape((len(P),1))
    # Form the composite attributes vectors
    X1 = X[0] # M x D1
    X2 = X[1] # N x D2    
    Xbias = np.ones((len(P),1)) # |Yobs| x 1
    Xusers = X1[P,:] # |Yobs| x D1
    Xitems = X2[Q,:] # |Yobs| x D2
    X_composite = np.hstack((Xbias, Xusers, Xitems)) # |Yobs| x (1 + D1 + D2)

    parameters = []
    log_likelihood = np.zeros(num_iter)
    M = Y.shape[0]
    N = Y.shape[1]

    alphas,thetas,betas,r,s,gammas = initialize(Y,X,K,L,M,N,P,Q,W,sigma_tolerance)    

    best_log_likelihood = -1e99
    for run in range(num_runs):

        print "Run %d" % (run + 1)

        # EM Algorithm
        # Parameters for the EM
        delta_likelihood = 1e99
        #precision = 1e-5
        t = 0
    
        crnt_log_likelihood = get_lower_bound(Y, X, alphas, thetas, betas, r[0], r[1], s[0], s[1],gammas, P,Q, reg_lambda)
        print('LOG LIKELIHOOD BEFORE EM = %f' %crnt_log_likelihood)
        em_convergence = False
        number_times_convg = 0
        while em_convergence==False and t < num_iter:
            print 'iteration :' + str(t + 1) +' out of ' + str(num_iter)
            # M Step
            thetas = update_thetas(K,L,M,N,X,s,thetas)
            #print 'updated thetas'
            #crnt_log_likelihood = get_lower_bound(Y, X, alphas, thetas, betas, r[0], r[1], s[0], s[1],gammas, P,Q, reg_lambda)
            #print('LOG LIKELIHOOD AFTER thetas UPDATE = %f' %crnt_log_likelihood)
        
            betas = update_betas(Yobs,X_composite,r,betas,reg_lambda,K,L,W,P,Q,)
            #print 'updated betas'
            #crnt_log_likelihood = get_lower_bound(Y, X, alphas, thetas, betas, r[0], r[1], s[0], s[1],gammas, P,Q, reg_lambda)
            #print('LOG LIKELIHOOD AFTER betas UPDATE = %f' %crnt_log_likelihood)
        
            if t>0:
                alphas = update_alphas(M,N,K,L,gammas,alphas)
                #print 'updated alphas'
                #crnt_log_likelihood = get_lower_bound(Y, X, alphas, thetas, betas, r[0], r[1], s[0], s[1],gammas, P,Q, reg_lambda)
                #print('LOG LIKELIHOOD AFTER M UPDATE = %f' %crnt_log_likelihood)

            # E Step
            iters_run = 0
            mean_field_convergence = 'Not converged'
            delta = 1e99
            while mean_field_convergence == 'Not converged':
                gc.collect()

                gammas = update_gammas(alphas,r,s,M,N,K,L,P,Q)
                #print 'updated gammas'
                #crnt_log_likelihood = get_lower_bound(Y, X, alphas, thetas, betas, r[0], r[1], s[0], s[1],gammas, P,Q, reg_lambda)
                #print('LOG LIKELIHOOD AFTER gammas UPDATE = %f' %crnt_log_likelihood)
            
                # TO DO: ADD THE SIMULATED ANNEALING PART OR LOOK FOR SOME OTHER ALTERNATIVE
                # to both r and s update
                r_new = update_r(gammas,r,Yobs,betas,K,L,M,N,P,Q,W,X_composite)
                #print 'updated r'
                #crnt_log_likelihood = get_lower_bound(Y, X, alphas, thetas, betas, r_new[0], r_new[1], s[0], s[1],gammas, P,Q, reg_lambda)
                #print('LOG LIKELIHOOD AFTER r UPDATE = %f' %crnt_log_likelihood)
            
                s_new = update_s(gammas,X,thetas,s,M,N,K,L)
                #print 'updated s'
                #crnt_log_likelihood = get_lower_bound(Y, X, alphas, thetas, betas, r_new[0], r_new[1], s_new[0], s_new[1],gammas, P,Q, reg_lambda)
                #print('LOG LIKELIHOOD AFTER s UPDATE = %f' %crnt_log_likelihood)
            
                delta = np.sum((r_new[0]-r[0])**2) + np.sum((r_new[1]-r[1])**2) +  np.sum((s_new[0]-s[0])**2) + np.sum((s_new[1]-s[1])**2)
                r = r_new
                s = s_new
                #print delta,'\t', iters_run
                if delta < precision/100 or iters_run>num_iter:
                    mean_field_convergence = 'converged'

                iters_run += 1
                #if iters_run % 50 == 0:
                    #print delta,'\t', iters_run
                    #print "%d iterations of mean field." % iters_run
                
            log_likelihood[t] = get_lower_bound(Y, X, alphas, thetas, betas, r[0], r[1], s[0], s[1],gammas, P,Q, reg_lambda)
            print('LOG LIKELIHOOD AFTER EM UPDATE = %f' %log_likelihood[t])

            # Update the parameter which determines convergence of the EM Algorithm
            if t>0:
                delta_likelihood = log_likelihood[t]-log_likelihood[t-1]
                if delta_likelihood >= precision:
                    number_times_convg = 0
                    em_convergence = False
                if delta_likelihood < precision:
                    number_times_convg += 1
                    if number_times_convg >=3:
                        em_convergence = True
            t +=1

        parameters.append({})
        parameters[run]['alphas'] = alphas
        parameters[run]['theta'] = thetas
        parameters[run]['betas'] = betas[0]
        parameters[run]['sigmas_Y'] = betas[1]
        parameters[run]['log_likelihood'] = log_likelihood[t-1]
        parameters[run]['r'] = r
        parameters[run]['s'] = s
        parameters[run]['r_mnk'] = r[0]
        parameters[run]['r_nml'] = r[1]
        parameters[run]['r_m0k'] = s[0]
        parameters[run]['r_n0l'] = s[1]
        parameters[run]['gamma_mk'] = gammas[0]
        parameters[run]['gamma_nl'] = gammas[1]

        alphas,thetas,betas,r,s,gammas = initialize(Y,X,K,L,M,N,P,Q,W,sigma_tolerance)

        if log_likelihood[t-1] > best_log_likelihood:
            best_log_likelihood = log_likelihood[t-1]
            best_run = run

        #r = parameters[best_run]['r']
        #s = parameters[best_run]['s']        

    return parameters, best_run

""" Function to Initialize the parameters
Inputs - Y,K,L
Outputs - Initialized alphas, thetas, betas, r, and s """
def initialize(Y,X,K,L,M,N,P,Q,W,sigma_tolerance):

    alphas = [np_random(K,),np_random(L,)]    
    
    r1 = np.zeros((len(P),K)) # Sparse r1 vector
    try:
        r1[:,:] = sample_dirichlet(alphas[0])
    except:
        r1[:,:] = np_random()
    r2 = np.zeros((len(Q),L))
    try:
        r2[:,:] = sample_dirichlet(alphas[1])
    except:
        r2[:,:] = np_random()
    r = [r1,r2]
    s1 = np.zeros((M,K))
    try:
        s1[:] = sample_dirichlet(alphas[0])
    except:
        s1[:] = np_random()
    s2 = np.zeros((N,L))
    try:
        s2[:] = sample_dirichlet(alphas[1])
    except:
        s2[:] = np_random()
    s = [s1,s2]
    
    # Thetas = [[means-KxNumber of attributes for entity1, Sigmas - KxNumber of
    # attributes for e1],[means,sigmas for entity 2]]
    m1 = np.zeros((K,X[0].shape[1])) + np_random()
    m2 = np.zeros((L,X[1].shape[1])) + np_random()
    sigma1 = np.zeros((K,X[0].shape[1])) + sigma_tolerance + np_random()
    sigma2 = np.zeros((L,X[1].shape[1])) + sigma_tolerance + np_random()
    theta1 = [m1,sigma1]
    theta2 = [m2,sigma2]
    thetas = [theta1,theta2]
    # Betas = [KxLx(1 + sum of No. of attributes of the two entities)]
    betas = [np_random() + np.zeros((K,L,1 + X[0].shape[1] + X[1].shape[1])),np.zeros((K,L)) + sigma_tolerance + np_random()]

    # concentration parameter
    gammas = [np_random((M,K)),np_random((N,L))]
    
    return  alphas,thetas,betas,r,s,gammas

""" Function to Update the Thetas 
Inputs - X,s
Outputs - Updated thetas """
def update_thetas(K,L,M,N,X,s,thetas):
    # No. of attributes
    D1 = X[0].shape[1]
    D2 = X[1].shape[1]

    #Initializations
    theta1_old = thetas[0]
    theta2_old = thetas[1]
    m1_old = theta1_old[0]
    sigma1_old = theta1_old[1]
    m2_old = theta2_old[0]
    sigma2_old = theta2_old[1]    
    m1 = np.zeros((K,D1)) # K x D1
    m2 = np.zeros((L,D2)) # L x D2
    sigma1 = np.zeros((K,D1)) # K x D1
    sigma2 = np.zeros((L,D2)) # L x D2
    X1 = X[0] # M x D1
    X2 = X[1] # N x D2
    s1 = s[0] # M x K
    s2 = s[1] # N x L

    assert((s1 >= 0).all)
    assert((s2 >= 0).all)

    # Update for Entity 1
    for k in range(K):
        if np.sum(s1[:,k])>1e-10:
            m1[k,:] = np.sum(s1[:,k].reshape(M,1)*X1,0)/np.sum(s1[:,k])
            x_minus_mean_sq = (X1 - np.tile(m1[k,:],(M,1)))**2
            sigma1[k,:] = np.sum(s1[:,k].reshape(M,1)*x_minus_mean_sq.reshape(M,D1),0)/np.sum(s1[:,k])
        else:
            m1[k,:] = m1_old[k,:]
            sigma1 = sigma1_old[k,:]
    sigma1[sigma1<sigma_tolerance] = sigma_tolerance
    sigma1[np.isnan(sigma1)] = sigma_tolerance            
    theta1 = [m1,sigma1]

    # Update for Entity 2
    for l in range(L):
        if np.sum(s2[:,l])>1e-10:
            m2[l,:] = np.sum(s2[:,l].reshape(N,1)*X2.reshape(N,D2),0)/np.sum(s2[:,l])
            x_minus_mean_sq = (X2.reshape(N,D2) - np.tile(m2[l,:],(N,1)))**2
            sigma2[l,:] = np.sum(s2[:,l].reshape(N,1)*x_minus_mean_sq.reshape(N,D2),0)/np.sum(s2[:,l])          
        else:
            m2[l,:] = m2_old[l,:]
            sigma2[l,:] = sigma2_old[l,:] 
    sigma2[sigma2<sigma_tolerance] = sigma_tolerance
    sigma2[np.isnan(sigma2)] = sigma_tolerance  
    theta2 = [m2,sigma2]

    thetas = [theta1,theta2]

    return thetas

""" Function to Update the alphas 
Inputs - M,N,K,L,gammas
Outputs - Updated Alphas """
def update_alphas(M,N,K,L,gammas,alphas):

    # Initializations
    gamma1 = gammas[0] # M x K
    gamma2 = gammas[1] # N x L
    alpha1 = alphas[0] # K x 1
    alpha2 = alphas[1] # L x 1

    alphas_new = [np.zeros(K,),np.zeros(L,)]

    # objective_fn is a function handler which returns the function value and gradient
    alphas_new[0], f1, d1= opt.fmin_l_bfgs_b(func=objective_fn, x0=alpha1,args=(gamma1,), bounds=[(0,None) for i in range(K)], maxfun=50000)
    if d1['warnflag']>0 :
        print "WARNING: Update for alpha[0] did not converge"
        '''ofile=open('debug0.txt','a')
        ofile.write('update in alpha[0] failed:\n')
        ofile.write('alpha_old \n')
        ofile.write(str(alphas))
        ofile.write('\ngammas\n')
        ofile.write(str(gammas))
        ofile.write('\nalphas_new\n')
        ofile.write(str(alphas_new))
        ofile.close()'''
        alphas_new = alphas
        
    alphas_new[1], f2, d2= opt.fmin_l_bfgs_b(func=objective_fn, x0=alpha2,args=(gamma2,), bounds=[(0,None) for i in range(L)], maxfun=50000)
    if d2['warnflag']>0 :
        print "WARNING: Update for alpha[1] did not converge"
        '''ofile=open('debug1.txt','a')
        ofile.write('update in alpha[1] failed:\n')
        ofile.write('alpha_old \n')
        ofile.write(str(alphas))
        ofile.write('\ngammas\n')
        ofile.write(str(gammas))
        ofile.write('\nalphas_new\n')
        ofile.write(str(alphas_new))
        ofile.close()'''
        alphas_new = alphas  

    alphas = alphas_new
    return alphas_new

""" Function to update the betas 
Inputs - Y, X, r, W
Outputs - Updated Betas """
# TO DO: RIGHT NOW I USE A LINEAR REGRESSION FOR Y. SHOULD CHANGE THIS TO
# WHICHEVER MODEL WE USE

# TO DO: also check out regularization of sigmaY
def update_betas(Y_obs,X,r,betas,reg_lambda,K,L,W,I,J):
    # initializations
    betas_new = np.zeros((betas[0].shape)) # K X L x (1 + D1 + D2)
    sigmas_Y_new = np.zeros((betas[1].shape)) # K x L
    r1 = r[0] # |Yobs| x K
    r2 = r[1] # |Yobs| x L
    D = betas_new.shape[2]  
    num_nonzero = len(I)
    Y_obs = Y_obs.reshape((num_nonzero,))
    Ysq = Y_obs**2
    
    # Update eqns
    for k in range(K):
        for l in range(L):
            weights = (r1[:,k]*r2[:,l])
            assert ((r1>=0).all)
            assert ((r2>=0).all)
            linear_weights = weights.reshape((num_nonzero,))
            if ((linear_weights>0.01).all):
                sqrt_weights = np.sqrt(linear_weights).reshape((num_nonzero,))
                nzweight=len(sqrt_weights[linear_weights>1e-10])
                weighted_X = np.tile(sqrt_weights[linear_weights>1e-10].reshape(nzweight,1),(1,D))*X[linear_weights>1e-10]
                weighted_Y_values = Y_obs[linear_weights>1e-10]*sqrt_weights[linear_weights>1e-10]
                
                # Calculate new beta[k][l]
                '''outfile = '../debug.txt'
                f = open(outfile,'w')
                print >>f,'r1'
                print >>f,r1
                print >>f,'\n'
                print >>f,'r2'
                print >>f,r2
                print >>f,'\n'
                print >>f,'nzweight'
                print >>f,nzweight
                print >>f,'\n'
                print >>f,'weighted X'
                print >>f,weighted_X
                print >>f,'\n'
                print >> f,'weighted Y'
                print >>f,weighted_Y_values
                f.close()'''
                try:
                    regressor = Ridge(alpha=reg_lambda,fit_intercept=False) 
                    regressor.fit(weighted_X,weighted_Y_values)
                    betas_new[k,l,:] = regressor.coef_[:]
                
                    # Calculate new sigmas_Y[k][l]
                    new_beta_times_x = np.dot(X,betas_new[k,l,:].reshape(D,1)).reshape(num_nonzero,)
                    numerator = np.sum(weights*(Ysq + new_beta_times_x**2 - 2*Y_obs*new_beta_times_x))#+reg_lambda*np.sum(betas_new[k,l,:]**2)
                    denominator = np.sum(weights)
                    if denominator > 1e-10:
                        sigmas_Y_new[k,l] = numerator/denominator
                        if sigmas_Y_new[k,l] < sigma_tolerance:
                            sigmas_Y_new[k,l] = sigma_tolerance
                except:
                    betas_new[k,l,:]=betas[0][k,l,:]
                    sigmas_Y_new[k,l] = betas[1][k,l]
            else:
                betas_new[k,l,:]=betas[0][k,l,:]
                sigmas_Y_new[k,l] = betas[1][k,l]                           
    sigmas_Y_new[sigmas_Y_new<sigma_tolerance] = sigma_tolerance
    sigmas_Y_new[np.isnan(sigmas_Y_new)] = sigma_tolerance
    
    # Update the calculated betas
    betas = [betas_new,sigmas_Y_new]
    
    return betas

""" Function to update the gammas 
Inputs - Alphas, r, s, W
Outputs - Updated gammas """
def update_gammas(alphas,r,s,M,N,K,L,I,J):
    # Initializations
    alpha1 = alphas[0] # K x 1
    alpha2 = alphas[1] # L x 1

    s1 = s[0] # M x K
    s2 = s[1] # N x L
    
    r1 = r[0] # |Yobs| x K
    r2 = r[1] # |Yobs| x L    

    gamma1 = np.zeros((M,K)) # M x K
    gamma2 = np.zeros((N,L)) # N x L
    
    gamma1 = s1 + alpha1.T
    gamma2 = s2 + alpha2.T

    col_of_ones = (np.ones((N,1)))
    row_of_ones = (np.ones((1,M)))
    for k in range(K): 
        gamma1[:,k] += dot(sp.csr_matrix((r1[:,k],(I,J)),shape=(M,N)),col_of_ones).reshape(M,)
    for l in range(L): 
        gamma2[:,l] += dot(row_of_ones,sp.csr_matrix((r2[:,l],(I,J)),shape=(M,N))).reshape(N,)
        
    gammas = [gamma1,gamma2]    

    return gammas         
            
'''    for k in range(K): 
        r1_sp = sp.coo_matrix((r1[:,k],(I,J)),shape=(M,N))
        gamma1[:,k] += np.array(r1*col_of_ones.todense().transpose())[0]
    for l in range(L):
        r2_sp = sp.coo_matrix((r2[:,l],(I,J)),shape=(M,N))
        gamma2[:,l] += np.array(row_of_ones.dot(r2_sp).todense().transpose()).reshape(N,)
        
    a = np.zeros((M,K)) # M x K
    b = np.zeros((N,L)) # N x L
    
    # TO DO: SEE IF THIS ADDITION CAN BE VECTORIZED
    a = s1 + alpha1.T
    b = s2 + alpha2.T        
    for m in range(len(I)):
        a[I[m],:] += r1[m,:]
        b[J[m],:] += r2[m,:]
        
    gammas = [a,b]'''

""" Function to update r 
Inputs - gammas, r, Y, X
Outputs - Updated r """
def update_r(gammas,r,Yobs,betas,K,L,M,N,P,Q,W,X):
    # Initializations
    r1 = r[0] # |Yobs| x K
    r2 = r[1] # |Yobs| x L
    gamma1 = gammas[0] # M x K
    gamma2 = gammas[1] # N x L
    beta = betas[0] # K x L x (1 + D1 + D2)
    sigmaY = betas[1] # K x L

    log_r1 = np.zeros((r1.shape))
    log_r2 = np.zeros((r2.shape))
    r1_new = np.zeros((r1.shape))
    r2_new = np.zeros((r2.shape))

    Yobs = Yobs.reshape(len(P),)
    
    # r1,r2 update
    for k in range(K):
        # p(y|beta*x)
        for l in range(L):
            beta_times_x = np.dot(X,beta[k,l,:]) # (|Yobs| x 1)
            log_r1[:,k] += digamma(gamma1[P,k])# - digamma(np.sum(gamma1[P,:]))            
            log_r2[:,l] += digamma(gamma2[Q,l])# - digamma(np.sum(gamma2[Q,:]))                        
            log_r1[:,k] += (-.5*np.log(2*np.pi*sigmaY[k,l])-(.5/sigmaY[k,l])*(Yobs**2 - 2*Yobs*beta_times_x + beta_times_x**2))*r2[:,l]
            log_r2[:,l] += (-.5*np.log(2*np.pi*sigmaY[k,l]) - (.5/sigmaY[k,l])*(Yobs**2 - 2*Yobs*beta_times_x + beta_times_x**2))* r1[:,k]
            

    # Prevent underflow by using a transformed method to get to normalized product space:
    # (e^x) / (e^x + e^y) = 1 / (1 + e^(y-x))
    # instead of trying to calculate e^x and e^y directly, just use e^(y-x).
    for k in range(K):
        old_settings = np.seterr(over='ignore')
        r1_new[:,k] = 1 / (1 + np.sum(np.exp(np.delete(log_r1,k,axis=1)-np.tile(log_r1[:,k].reshape(len(P),1),(1,K-1))),axis=1))
        np.seterr(**old_settings) 
        
    # Prevent underflow by using a transformed method to get to normalized product space:
    # (e^x) / (e^x + e^y) = 1 / (1 + e^(y-x))
    # instead of trying to calculate e^x and e^y directly, just use e^(y-x).
    for l in range(L):
        old_settings = np.seterr(over='ignore')
        r2_new[:,l] = 1 / (1 + np.sum(np.exp(np.delete(log_r2,l,axis=1)-np.tile(log_r2[:,l].reshape(len(Q),1),(1,L-1))),axis=1))
        np.seterr(**old_settings)

#    # r2 update
#    for l in range(L):
#        # p(y|beta*x)
#        for k in range(K):
#            beta_times_x = np.dot(X,beta[k,l,:]) # (|Yobs| x 1)
#            log_r2[:,l] += digamma(gamma2[Q,l])# - digamma(np.sum(gamma2[Q,:]))            
#            expected_log_prob = -.5*np.log(2*np.pi*sigmaY[k,l]) - (.5/sigmaY[k,l])*(Yobs**2 - 2*Yobs*beta_times_x + beta_times_x**2)
#            log_r2[:,l] += expected_log_prob * r1[:,k]
        

    r_new = [r1_new,r2_new]
  
    return r_new

""" Function to update s 
Inputs - gammas, X, thetas
Outputs - Updated s """
def update_s(gammas,X,thetas,s,M,N,K,L):
    # Initializations
    gamma1 = gammas[0] # M x K
    gamma2 = gammas[1] # N x L
    X1 = X[0] # M x D1
    X2 = X[1] # N x D2
    theta1 = thetas[0]
    theta2 = thetas[1]
    m1 = theta1[0] # K x D1
    sigma1 = theta1[1] # K x D1
    m2 = theta2[0] # L x D2
    sigma2 = theta2[1] # L x D2
    s1_new = np.zeros((s[0].shape)) # M x K
    s2_new = np.zeros((s[1].shape)) # N x L
    log_s1_new = np.zeros((s1_new.shape))
    log_s2_new = np.zeros((s2_new.shape))    
    D1 = m1.shape[1]
    D2 = m2.shape[1]

    # Update eqns
    for k in range(K):
        log_s1_new[:,k] = (digamma(gamma1[:,k]))# - digamma(np.sum(gamma1,1))).reshape(M,)
        for d in range(D1):
            x1_m_d = X1[:,d]
            m1_k_d = m1[k,d]
            sigma1_k_d = sigma1[k,d]
            log_prob_x1m_d = -.5*np.log(2*np.pi*sigma1_k_d) - (.5/sigma1_k_d)*((x1_m_d - m1_k_d)**2)
            log_s1_new[:,k] += log_prob_x1m_d

    # Prevent underflow by using a transformed method to get to normalized product space:
    # (e^x) / (e^x + e^y) = 1 / (1 + e^(y-x))
    # instead of trying to calculate e^x and e^y directly, just use e^(y-x).
    s1_new = np.zeros(log_s1_new.shape)
    for k in range(K):
        old_settings = np.seterr(over='ignore')
        s1_new[:,k] = 1 / (1 + np.sum(np.exp(np.delete(log_s1_new,k,axis=1)-np.tile(log_s1_new[:,k].reshape(M,1),(1,K-1))),axis=1)).reshape(M,)
        np.seterr(**old_settings)     

    for l in range(L):
        log_s2_new[:,l] = (digamma(gamma2[:,l]))# - digamma(np.sum(gamma2,1))).reshape(N,)
        for d in range(D2):
            x2_n_d = X2[:,d]
            m2_l_d = m2[l,d]
            sigma2_l_d = sigma2[l,d]
            log_prob_x2n_d = -.5*np.log(2*np.pi*sigma2_l_d) - (.5/sigma2_l_d)*((x2_n_d - m2_l_d)**2)
            log_s2_new[:,l] += log_prob_x2n_d

    # Prevent underflow by using a transformed method to get to normalized product space:
    # (e^x) / (e^x + e^y) = 1 / (1 + e^(y-x))
    # instead of trying to calculate e^x and e^y directly, just use e^(y-x).
    s2_new = np.zeros(log_s2_new.shape)
    for k in range(K):
        old_settings = np.seterr(over='ignore')
        s2_new[:,l] = 1 / (1 + np.sum(np.exp(np.delete(log_s2_new,l,axis=1)-np.tile(log_s2_new[:,l].reshape(N,1),(1,L-1))),axis=1)).reshape(N,)
        np.seterr(**old_settings)         

    s_new = [s1_new,s2_new]
    return s_new


""" Helper Function to build the objective function to the maximized for updating the
alphas. Code built using the notations for alpha1, works for alpha2 too since
the operations on alpha2 are the same with different notation 
Inputs -
    Alpha - K/L x 1
    Gamma - M/N x K/L """
def objective_fn(alpha, gamma):
    # TO DO: vectorize this
    try:
        M,K = gamma.shape
        assert(alpha.shape == (K,))

        fn = M*gammaln(np.sum(alpha))
        grad = np.zeros((K,))    
        # Denominator of B(alpha)
        for k in range(K):
            fn -= M*gammaln(alpha[k])
            grad[k] += M*(digamma(np.sum(alpha)) - digamma(alpha[k]))
            for m in range(M):
                fn += (alpha[k] - 1)*(digamma(gamma[m,k]) - digamma(np.sum(gamma[m,:])))
                grad[k] += digamma(gamma[m,k]) - digamma(np.sum(gamma[m,:]))
    except RuntimeWarning:
        file_name = '.\Exceptions\exceptions.txt'
        outfile = open(file_name,'a')
        print >>outfile, "Runtime Warning in Objective_fn\n"
        print >>outfile, "Alpha\n"
        print >>outfile, alpha
        print >>outfile, "Gamma\n"
        print >>outfile, gamma
    
    return -fn, -grad        

'''    M,K = gamma.shape
    assert(alpha.shape == (K,))
    # Numerator of B(alpha) in the arg max
    fn = M*gammaln(np.sum(alpha))
    
    # Denominator of B(alpha)
    for k in range(K):
        fn -= M*gammaln(alpha[k])
        
    # The digamma terms in the arg max
    for m in range(M):
        for k in range(K):
            fn += (alpha[k] - 1)*(digamma(gamma[m,k]) - digamma(np.sum(gamma[m,:])))
    
    grad = np.zeros((K,))
    # The gradient of the function w.r.t alpha to feed to the L-BFGS optimizer
    for k in range(K):
        grad[k] += digamma(np.sum(alpha)) - digamma(alpha[k])
    grad = M*grad

    for m in range(M):
        for k in range(K):
            grad[k] += digamma(gamma[m,k]) - digamma(np.sum(gamma[m,:]))

    return -fn, -grad'''

""" Function to Calulate the current lower bound on the Log-likelihood given by
the EM
Inputs - 
Outputs - """
# TO DO: FINISH THE COMMENTS AND RE-WRITE THIS ON YOUR OWN
def get_lower_bound(Y, Xs, alphas, thetas, beta, r_mnk, r_nml, r_m0k, r_n0l,gammas,I,J, reg_lambda):

    theta1 = thetas[0]
    theta2 = thetas[1]
    m1 = theta1[0]
    m2 = theta2[0]
    mus_X = [m1,m2]
    sigma1 = theta1[1]
    sigma2 = theta2[1]
    sigmas_X = [sigma1,sigma2]
    
    betas = beta[0]
    sigmas_Y = beta[1]

    gamma_mk = gammas[0]
    gamma_nl = gammas[1]
 
    (M,K)=r_m0k.shape
    (N,L)=r_n0l.shape
    try:
        D1 = Xs[0].shape[1]
    except:
        D1 = 0
    try:
        D2 = Xs[1].shape[1]
    except:
        D2 = 0 
        
    log_likelihood_lower_bound = 0;

    # Terms involving summation over M  
    # Expectation of log[p(pi_1m)]
    # Mean of a diriclet with parameter gamma_mk
    log_of_gamma_of_sum = gammaln(np.sum(alphas[0])) #log_e(Gamma(sum(alpha_1k)))
    log_likelihood_lower_bound += M*log_of_gamma_of_sum
    for k in range(K):
        log_of_gamma = gammaln(alphas[0][k])
        log_likelihood_lower_bound -= M*log_of_gamma

    for m in range(M):
        psi_gamma_helpers = np.array([psi_gamma_helper(gamma_mk[m,:], k) for k in range(K)]).reshape(K,)
        # Expectation of log[p(pi_1m)] continued
        alpha_times_psi_stuff = (alphas[0]-1)*psi_gamma_helpers
        assert (alpha_times_psi_stuff.shape==(K,))         
        log_likelihood_lower_bound += np.sum(alpha_times_psi_stuff)

        # E_q[log(p(z_1m))]
        log_likelihood_lower_bound += np.sum(r_m0k[m,:]*psi_gamma_helpers)

        # H[q(pi_1m)]
        log_of_gamma_of_sum = gammaln(np.sum(gamma_mk[m,:]))
        log_likelihood_lower_bound -= log_of_gamma_of_sum
        logs_of_gammas = np.array([gammaln(gamma_mk[m,k]) for k in range(K)])
        assert (logs_of_gammas.shape==(K,))        
        log_likelihood_lower_bound += np.sum(logs_of_gammas)
        gamma_times_psi_stuff = (gamma_mk[m,:]-1)*psi_gamma_helpers
        assert (gamma_times_psi_stuff.shape==(K,))  
        log_likelihood_lower_bound -= np.sum(gamma_times_psi_stuff)

    # H[q(z_1m)]
    weighted_log=r_m0k[r_m0k > 1e-10]* np.log(r_m0k[r_m0k > 1e-10]) 
    log_likelihood_lower_bound -= np.sum(weighted_log)

    # log(p(x_1m))
    for k in range(K):
        x_minus_mu_square =(Xs[0]**2-2*(Xs[0]*np.tile(mus_X[0][k,:],(M,1)))+np.tile(mus_X[0][k,:],(M,1))**2)
        for d1 in range(D1):
            expectation_logx=(r_m0k[:,k]*(((-.5)*np.log(2*np.pi*(sigmas_X[0][k,d1]))-(.5/sigmas_X[0][k,d1])*x_minus_mu_square[:,d1]))).reshape(M,1)
            log_likelihood_lower_bound += np.sum(expectation_logx)
        
    # Tems involving summation over N   
    # Expectation of log[p(pi_2n)]
    # Mean of a diriclet with parameter gamma_nl
    log_of_gamma_of_sum = gammaln(np.sum(alphas[1])) #log_e(Gamma(sum(alpha_2l)))
    log_likelihood_lower_bound += N*log_of_gamma_of_sum
    for l in range(L):
        log_of_gamma = gammaln(alphas[1][l])
        log_likelihood_lower_bound -= N*log_of_gamma

    for n in range(N):
        psi_gamma_helpers = np.array([psi_gamma_helper(gamma_nl[n,:], l) for l in range(L)]).reshape(L,)
        # Expectation of log[p(pi_2n)] continued
        alpha_times_psi_stuff = (alphas[1]-1)*psi_gamma_helpers
        assert (alpha_times_psi_stuff.shape==(L,)) 
        log_likelihood_lower_bound += np.sum(alpha_times_psi_stuff)

        # E_q[log(p(z_2n))]
        log_likelihood_lower_bound += np.sum(r_n0l[n,:]*psi_gamma_helpers)

        # H[q(pi_2)]
        gamma_of_sum = gammaln(np.sum(gamma_nl[n,:]))
        log_likelihood_lower_bound -= gamma_of_sum
        logs_of_gammas = np.array([gammaln(gamma_nl[n,l]) for l in range(L)])
        assert (logs_of_gammas.shape==(L,))        
        log_likelihood_lower_bound += np.sum(logs_of_gammas)
        gamma_times_psi_stuff = (gamma_nl[n,:]-1)*psi_gamma_helpers
        assert (gamma_times_psi_stuff.shape==(L,))  
        log_likelihood_lower_bound -= np.sum(gamma_times_psi_stuff)

    # H[q(z_2n)]
    weighted_log=r_n0l[r_n0l > 1e-10]* np.log(r_n0l[r_n0l > 1e-10])
    log_likelihood_lower_bound -= np.sum(weighted_log)
        
    # log(p(x_2n))
    for l in range(L):
        x_minus_mu_square=(Xs[1]**2-2*(Xs[1]*np.tile(mus_X[1][l,:],(N,1)))+np.tile(mus_X[1][l,:],(N,1))**2)
        for d2 in range(D2):
            expectation_logx=(r_n0l[:,l]*(((-.5)*np.log(2*np.pi*(sigmas_X[1][l,d2]))-(.5/sigmas_X[1][l,d2])*x_minus_mu_square[:,d2]))).reshape(N,1)
            log_likelihood_lower_bound += np.sum(expectation_logx)

    # E_q[log(p(z_1mn))]
    # E_q[log(p(z_2nm))]
    # Expectation of log[p(y|B'x)]
    lower_bound_from_ys = 0
    X, num_nonzero = build_test_X(Xs, I, J)
    Y_values = np.array(Y[(I,J)]).reshape((num_nonzero,))
    for l in range(L):
        psi_gamma_helpers_nnz = np.array([psi_gamma_helper(gamma_nl[J[nnz],:], l) for nnz in range(num_nonzero)]).reshape(num_nonzero,)
        log_likelihood_lower_bound += np.sum(r_nml[:,l]*psi_gamma_helpers_nnz)

    #regularization on beta
    reg_beta=0
    for k in range(K):
        psi_gamma_helpers_nnz = np.array([psi_gamma_helper(gamma_mk[I[nnz],:], k) for nnz in range(num_nonzero)]).reshape(num_nonzero,)
        log_likelihood_lower_bound += np.sum(r_mnk[:,k]*psi_gamma_helpers_nnz)
        for l in range(L):
            beta_times_x = np.dot(X,betas[k,l,:])
            weights = r_mnk[:,k]*r_nml[:,l] # nnz X 1 matrix
            Ey_term=weights * ((-.5)*np.log(2*np.pi*sigmas_Y[k,l]) - (.5/sigmas_Y[k,l])*(Y_values**2-2*Y_values*beta_times_x+beta_times_x**2))
            assert (Ey_term.shape==(num_nonzero,))
            lower_bound_from_ys += np.sum(Ey_term)    
            reg_beta-=reg_lambda*np.sum(betas[k,l,:]**2)/(2*sigmas_Y[k,l])
    log_likelihood_lower_bound += lower_bound_from_ys
    del X, num_nonzero
    
    # H[q(z_1mn)q_z2nm]
    weighted_log=r_mnk[r_mnk > 1e-10]* np.log(r_mnk[r_mnk > 1e-10])
    log_likelihood_lower_bound -= np.sum(weighted_log)
    weighted_log=r_nml[r_nml > 1e-10]* np.log(r_nml[r_nml > 1e-10])
    log_likelihood_lower_bound -= np.sum(weighted_log)
    
    # regularization on beta
    log_likelihood_lower_bound+=reg_beta

    return log_likelihood_lower_bound

# General Helpers
def psi_gamma_helper(gamma, k):
    return digamma(gamma[k])-digamma(np.sum(gamma))

def build_test_X(Xs, I, J):
    '''
    Returns the number of values to be predicted and tfor k in range(K):he corresponding attribute vectors stacked together
    '''
    num_nonzero = len(I)
    Xbias = np.ones((num_nonzero,1))
    Xusers = Xs[0][I]
    Xitems = Xs[1][J]
    X = np.hstack([Xbias, Xusers, Xitems])
    # num_nonzero X (1+D1+D2)

    # THIS IS FOR CELL FEATURES, FORGET IT FOR NOW
    try:
        Xcell_data = Xs[2]
        Xcells_I = Xcell_data[0]
        Xcells_J = Xcell_data[1]
        Xcells_vals = Xcell_data[2]
        assert np.all(I == Xcells_I)
        assert np.all(J == Xcells_J)
        X = np.hstack([X, Xcells_vals])
    except:
        pass    # No (or invalid) cell features.  Just keep going

    return X, num_nonzero

# Helpers in M step updates
def build_X(Xs, sample_W):
    '''
    Similar to build_test_X but takes W as hte input and returns I, J in addition to X and num_nonzero
    '''
    I, J, throwaway_W = sp.find(sample_W)
    num_nonzero = len(I)
    Xbias = np.ones((num_nonzero,1))
    Xusers = Xs[0][I]
    Xitems = Xs[1][J]
    X = np.hstack([Xbias, Xusers, Xitems])
    try:
        Xcell_data = Xs[2]
        Xcells_I = Xcell_data[0]
        Xcells_J = Xcell_data[1]
        Xcells_vals = Xcell_data[2]
        assert np.all(I == Xcells_I)
        assert np.all(J == Xcells_J)
        X = np.hstack([X, Xcells_vals])
    except:
        pass    # No (or invalid) cell features.  Just keep going
    return X, I, J, num_nonzero


# TO DO : CHECK PREDICTION OF Y AND SEE IF SIGMA Y CAN BE USED TO PROVIDE BETTER QUALITY PREDICTIONS

# Predicting Ys
def predict_Ys(test_I, test_J, test_Y, Xs, r_mnk, r_nml, r_m0k, r_n0l, betas, print_mean_pred_err=False, indent_level=0, label="Train"):
    '''
    Predicts the response in the test_Y, Given the attribute values, cluster assignments for entities 
    test_I and test_J represent the i,j indices such that Y(test_I(k),test_J(k))=test_Y(k)
    in each dyad and the GLM parameters beta for each cluster
    returns the predicted_Y, mse and se
    '''
    M=Xs[0].shape[0]
    N=Xs[1].shape[0]
    K=r_mnk.shape[1]
    L=r_nml.shape[1]

    X, num_nonzero = build_test_X(Xs, test_I, test_J)
    
    # num_nonzero is the number of predictions to be made, X(num_nonzero.(1+D1+D2)) is the list of {X_a} 
    # where X_a is the concatenated feature vector the dyad indexed by a
    if (label!="Test"):
        # Here r_mnk, r_nml are the actual values learnt for the prediction set
        predicted_Y = np.zeros((num_nonzero,))
        for k in range(K):
            for l in range(L):
                beta_times_x = np.dot(X,betas[k,l,:])
                #the above variable is an (nnz,1) vector of responses if (k,l) is the cluster id
                weights = np.array(r_mnk[:,k]* r_nml[:,l])
                predicted_Y = predicted_Y + (weights * beta_times_x)
    else:
        print "Use predict_Ys_test"

    observed_Y = np.asarray(test_Y)
    se = np.ravel((observed_Y-predicted_Y)**2)
    mse = np.sum(se) / num_nonzero
    rmse = np.sqrt(mse)

    indentation = "".join(["\t" for i in range(indent_level)])
    if print_mean_pred_err:
        print "%s%s MSE: %f" % (indentation, label, mse)
        print "%s%s RMSE: %f" % (indentation, label, rmse)
        print "%s%s Sum Y values %f"  % (indentation, label, np.sum(observed_Y))
        print "%s%s Root squared error %f" % (indentation, label, np.sqrt(np.sum(se))) 
        
def predict_Ys_test(test_I, test_J, test_Y, Xs, pi_m, pi_n, betas, print_mean_pred_err=False, indent_level=0, label="Test"):        
    M=Xs[0].shape[0]
    N=Xs[1].shape[0]
    K=betas.shape[0]
    L=betas.shape[1]

    X, num_nonzero = build_test_X(Xs, test_I, test_J)
    predicted_Y = np.zeros((num_nonzero,))
    for k in range(K):
        for l in range(L):
            beta_times_x = np.dot(X,betas[k,l,:])
            #the above variable is an (nnz,1) vector of responses if (k,l) is the cluster id
            weights = np.array(pi_m[test_I,k]*pi_n[test_J,l])
            predicted_Y = predicted_Y + (weights * beta_times_x)     
  
    observed_Y = np.asarray(test_Y)
    se = np.ravel((observed_Y-predicted_Y)**2)
    mse = np.sum(se) / num_nonzero
    rmse = np.sqrt(mse)

    indentation = "".join(["\t" for i in range(indent_level)])
    if print_mean_pred_err:
        print "%s%s MSE: %f" % (indentation, label, mse)
        print "%s%s RMSE: %f" % (indentation, label, rmse)
        print "%s%s Sum Y values %f"  % (indentation, label, np.sum(observed_Y))
        print "%s%s Root squared error %f" % (indentation, label, np.sqrt(np.sum(se)))
