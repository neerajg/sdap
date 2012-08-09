
'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

# TO DO : check all TO DOs here and in misc
# TO DO : Try damped updates
# TO DO : here and in misc also change update of r condition to check for very low value to make it work for all K and L
from MMBAE_CPLD.misc import initialize
import numpy as np
from scipy.special import psi as digamma
from scipy.special import gammaln
import scipy.sparse as sp
from MMBAE_CPLD.misc import objective_fn
import scipy.optimize as opt

try:
    from scikits.learn.linear_model import Ridge
except ImportError:
    from sklearn.linear_model import Ridge

sigma_tolerance = 1e-4

def train_mmbae_cpld_linear(K, L, X1, X2, train_I, train_J, train_Y, reg_beta, num_iter, delta_convg, reg_alpha1, reg_alpha2):
    
    Xs = [X1,X2]
    
    # Initializations
    delta_likelihood = 1e99
    log_likelihood = []
    log_likelihood.append(-1e10)
    alphas, gammas, betas, r, X_composite, M, N = initialize(X1, X2, train_I, train_J, train_Y, K, L)
    
    ones = np.ones((len(train_I),))
    mu = sp.csr_matrix((ones, (train_I,train_J)), shape=(M,N)).sum(1)
    mv = sp.csr_matrix((ones, (train_I,train_J)), shape=(M,N)).sum(0).transpose()
    mu[mu<1] = 1
    mv[mv<1] = 1    
        
    # EM
    em_convg = False
    t = 0
    while em_convg == False and t < num_iter:
        # M-Step
        alphas, betas = m_step(K, L, X_composite, train_I, train_J, train_Y, alphas, betas, gammas, r, reg_beta, M, N, reg_alpha1, reg_alpha2,mu,mv)
        # E - Step
        gammas, r = e_step(K, L, X_composite, train_I, train_J, train_Y, alphas, betas, gammas, r, M, N, num_iter,delta_convg,mu,mv)
                
        # Test for Convergence
        t +=1        
        log_likelihood.append(get_log_likelihood_lower_bound(K,L,M,N,alphas,betas,gammas,r,train_I,train_J, train_Y,X_composite,mu,mv))
        delta_likelihood = log_likelihood[t]-log_likelihood[t-1]
        delta_likelihood = delta_likelihood*100/abs(log_likelihood[t-1])
        print delta_likelihood, t, log_likelihood[t]
        if (delta_likelihood) < delta_convg:
            em_convg = True
        
    # Return whatever needs to be returned
    params = {'r':r,
              'Xs':Xs,
              'betas':betas,
              'alphas':alphas,
              'gammas':gammas
              }
    obj = log_likelihood
    return params, obj

###################################################################################################################################
# E STEP
###################################################################################################################################
def e_step(K, L, X_composite, train_I, train_J, train_Y, alphas, betas, gammas, r, M, N, num_iter,delta_convg,mu,mv):
    # Until Convergence
    e_convg = False
    t = 0
    delta_likelihood = 1e99
    log_likelihood_e_step = -1e99*np.ones((num_iter+1))

    while e_convg == False and t < num_iter :
        r = update_r(gammas, r, betas, train_Y, train_I, train_J, K, L, M, N, X_composite,mu,mv)
        gammas = update_gammas(gammas, alphas, r, betas, train_Y, train_I, train_J, K, L, M, N, X_composite,mu,mv)        
        # Test for Convergence
        t +=1
        log_likelihood_e_step[t] = get_log_likelihood_lower_bound(K,L,M,N,alphas,betas,gammas,r,train_I,train_J, train_Y,X_composite,mu,mv)
        delta_likelihood = log_likelihood_e_step[t]-log_likelihood_e_step[t-1]
        delta_likelihood = delta_likelihood*100/abs(log_likelihood_e_step[t-1])
        if delta_likelihood < delta_convg:
            e_convg = True
    return gammas, r

def update_gammas(gammas, alphas, r, betas, train_Y, train_I, train_J, K, L, M, N, X_composite,mu,mv):
    gamma1 = gammas[0]
    gamma2 = gammas[1]
    ones = np.ones((len(train_I),))
    mu = sp.csr_matrix((ones, (train_I,train_J)), shape=(M,N)).sum(1)
    mv = sp.csr_matrix((ones, (train_I,train_J)), shape=(M,N)).sum(0)
    mu[mu<1] = 1
    mv[mv<1] = 1
    gammas_new = [np.zeros(gamma1.shape),np.zeros(gamma2.shape)]
    temp1 = get_log_likelihood_lower_bound(K,L,M,N,alphas,betas,gammas,r,train_I,train_J, train_Y,X_composite,mu,mv)
    gammas_new[0] = np.tile(alphas[0].reshape(1,K), (M,1)) +  np.multiply(r[0],mu)# M x K
    temp2 = get_log_likelihood_lower_bound(K,L,M,N,alphas,betas,[gammas_new[0],gammas[1]],r,train_I,train_J, train_Y,X_composite,mu,mv) 
    if temp1>=temp2:
        gammas_new[0] = gamma1
    
    temp1 = get_log_likelihood_lower_bound(K,L,M,N,alphas,betas,[gammas_new[0],gammas[1]],r,train_I,train_J, train_Y,X_composite,mu,mv)
    gammas_new[1] = np.tile(alphas[1].reshape(1,L), (N,1)) + np.multiply(r[1],mv.transpose()) # N x L    
    temp2 = get_log_likelihood_lower_bound(K,L,M,N,alphas,betas,gammas_new,r,train_I,train_J, train_Y,X_composite,mu,mv) 
    if temp1>=temp2:
        gammas_new[1] = gamma2
    
    gammas = gammas_new
    return gammas

def update_r(gammas, r, betas, train_Y, train_I, train_J, K, L, M, N, X_composite,mu,mv):
    # TO DO : check out the simulated annealing bit here and vectorize/matricize
    #r1_mk \prop to exp(psi(gamma1_mk) - psi(sum(gamma1_mk)) + gaussian probability)
    # Initializations
    X = X_composite # |Yobs| x (1 + D1 + D2)
    r1 = r[0] # M x K
    r2 = r[1] # N x L
    gamma1 = gammas[0] # M x K
    gamma2 = gammas[1] # N x L
    beta = betas[0] # K x L x (1 + D1 + D2)
    sigmaY = betas[1] # K x L

    log_r1 = np.zeros((r1.shape))
    log_r2 = np.zeros((r2.shape))
    r1_new = np.zeros((r1.shape))
    r2_new = np.zeros((r2.shape))

    Yobs = train_Y
    temp1 = np.zeros((Yobs.shape[0],K))
    temp2 = np.zeros((Yobs.shape[0],L))
    
    # r1,r2 update
    log_r1 += digamma(gamma1)# - digamma(np.sum(gamma1[P,:]))            
    log_r2 += digamma(gamma2)# - digamma(np.sum(gamma2[Q,:]))
    for k in range(K):
        # p(y|beta*x)
        for l in range(L):
            beta_times_x = np.dot(X,beta[k,l,:]) # (|Yobs| x 1)
            r2_temp = np.tile(np.transpose(r2[:,l]).reshape(1,N), (M,1))[train_I, train_J] # |Yobs| x 1            
            temp1[:,k] += (-.5*np.log(2*np.pi*sigmaY[k,l]) - (.5/(sigmaY[k,l]))*((Yobs- beta_times_x)**2) )* r2_temp
            r1_temp = np.tile(r1[:,k].reshape(M,1), (1,N))[train_I, train_J] # |Yobs| x 1             
            temp2[:,l] += (-.5*np.log(2*np.pi*sigmaY[k,l]) - (.5/(sigmaY[k,l]))*((Yobs- beta_times_x)**2) )* r1_temp
    
    for k in range(K):
        log_r1[:,k] += np.array(np.divide(sp.csr_matrix((temp1[:,k],(train_I, train_J)),shape=(M,N)).sum(1),mu).flatten())[0]
    for l in range(L):
        log_r2[:,l] += np.array(np.divide(sp.csr_matrix((temp2[:,l],(train_I, train_J)),shape=(M,N)).sum(0).transpose(),mv).reshape(N,).flatten())[0]
            

    # Prevent underflow by using a transformed method to get to normalized product space:
    # (e^x) / (e^x + e^y) = 1 / (1 + e^(y-x))
    # instead of trying to calculate e^x and e^y directly, just use e^(y-x).
    for k in range(K):
        old_settings = np.seterr(over='ignore')
        r1_new[:,k] = 1 / (1 + np.sum(np.exp(np.delete(log_r1,k,axis=1)-np.tile(log_r1[:,k].reshape(M,1),(1,K-1))),axis=1))
        np.seterr(**old_settings) 
        
    # Prevent underflow by using a transformed method to get to normalized product space:
    # (e^x) / (e^x + e^y) = 1 / (1 + e^(y-x))
    # instead of trying to calculate e^x and e^y directly, just use e^(y-x).
    for l in range(L):
        old_settings = np.seterr(over='ignore')
        r2_new[:,l] = 1 / (1 + np.sum(np.exp(np.delete(log_r2,l,axis=1)-np.tile(log_r2[:,l].reshape(N,1),(1,L-1))),axis=1))
        np.seterr(**old_settings)
    
    r1_new[r1_new<1e-6]=1e-6
    #if K !=1:
        #r1_new[r1_new>0.9]=0.9
    r2_new[r2_new<1e-6]=1e-6
    #if L !=1:
        #r2_new[r2_new>0.9]=0.9
        
    r = [r1_new, r2_new]
    return r

###################################################################################################################################
# M STEP
###################################################################################################################################
def m_step(K, L, X_composite, train_I, train_J, train_Y, alphas, betas, gammas, r, reg_beta, M, N, reg_alpha1,reg_alpha2,mu,mv):
    Yobs = train_Y
    alphas = update_alphas(gammas, alphas, K, L, M, N, betas, r, train_I, train_J, train_Y, X_composite,reg_alpha1,reg_alpha2,mu,mv)
    betas = update_betas(Yobs, X_composite, r, betas, reg_beta, K, L, M, N, train_I, train_J)
    return alphas, betas

def update_alphas(gammas, alphas, K, L, M, N, betas, r, train_I, train_J, train_Y, X_composite, reg_alpha1, reg_alpha2,mu,mv):

    # Initializations
    gamma1 = gammas[0] # M x K
    gamma2 = gammas[1] # N x L
    alpha1 = alphas[0] # K x 1
    alpha2 = alphas[1] # L x 1

    alphas_new = [np.zeros(K,),np.zeros(L,)]
    alphas_new[0] = alpha1
    alphas_new[1] = alpha2
    
    # objective_fn is a function handler which returns the function value and gradient
    temp1 = get_log_likelihood_lower_bound(K,L,M,N,alphas_new,betas,gammas,r,train_I,train_J, train_Y,X_composite,mu,mv)
    alphas_new[0], nfeval, rc = opt.fmin_tnc(func=objective_fn, x0=alpha1, args=(gamma1,reg_alpha1),bounds=[(0,None) for i in range(K)],maxfun=50000, messages = 0)
    #alphas_new[0] = alphas_new[0]/np.sum(alphas_new[0]) 
    #alphas_new[0], f1, d1= opt.fmin_l_bfgs_b(func=objective_fn, x0=alpha1,args=(gamma1,), bounds=[(0,None) for i in range(K)], maxfun=50000)
    temp2 = get_log_likelihood_lower_bound(K,L,M,N,alphas_new,betas,gammas,r,train_I,train_J, train_Y,X_composite,mu,mv)
    
    if temp1>=temp2:
        alphas_new[0] = alpha1
        
    temp2 = get_log_likelihood_lower_bound(K,L,M,N,alphas_new,betas,gammas,r,train_I,train_J, train_Y,X_composite,mu,mv)    
    #alphas_new[1], f2, d2= opt.fmin_l_bfgs_b(func=objective_fn, x0=alpha2,args=(gamma2,), bounds=[(0,None) for i in range(L)], maxfun=50000)
    alphas_new[1], nfeval, rc = opt.fmin_tnc(func=objective_fn, x0=alpha2, args=(gamma2,reg_alpha2),bounds=[(0,None) for i in range(L)],maxfun=50000, messages = 0)
    #alphas_new[1] = alphas_new[1]/np.sum(alphas_new[1])     
    temp3 = get_log_likelihood_lower_bound(K,L,M,N,alphas_new,betas,gammas,r,train_I,train_J, train_Y,X_composite,mu,mv)
    
    if temp2>=temp3:
        alphas_new[1] = alpha2

    alphas = alphas_new
    return alphas

def update_betas(Yobs, X_composite, r, betas, reg_beta, K, L, M, N, train_I, train_J):
    # initializations
    X = X_composite
    Y_obs = Yobs
    betas_new = np.zeros((betas[0].shape)) # K X L x (1 + D1 + D2)
    sigmas_Y_new = np.zeros((betas[1].shape)) # K x L
    # r[0] - |Yobs| x K
    # r[1] - |Yobs| x L
    D = betas_new.shape[2]
    num_nonzero = len(Yobs)
    Ysq = Y_obs**2
    r1_obs = r[0][train_I]
    r2_obs = r[1][train_J]
    
    # TO DO : vectorize this over K
    # Update eqns
    for k in range(K):
        for l in range(L):
            weights = np.multiply(r1_obs[:,k],r2_obs[:,l])
            linear_weights = weights.reshape((num_nonzero,))
            sqrt_weights = np.sqrt(linear_weights).reshape((num_nonzero,))
            nzweight=len(sqrt_weights[linear_weights>1e-10])
            weighted_X = np.tile(sqrt_weights[linear_weights>1e-10].reshape(nzweight,1),(1,D))*X[linear_weights>1e-10]
            weighted_Y_values = Y_obs[linear_weights>1e-10]*sqrt_weights[linear_weights>1e-10]
            
            # Calculate new beta[k][l]
            try:
                # TO DO : Figure how to account for very large size of data while doing ridge regression
                regressor = Ridge(alpha=reg_beta,fit_intercept=False) 
                regressor.fit(weighted_X,weighted_Y_values)
                betas_new[k,l,:] = regressor.coef_[:]
            
                # TO DO : Regularization here too
                # Calculate new sigmas_Y[k][l]
                new_beta_times_x = np.dot(X,betas_new[k,l,:].reshape(D,1)).reshape(num_nonzero,)
                numerator = np.sum(weights*(Ysq + new_beta_times_x**2 - 2*Y_obs*new_beta_times_x))#+reg_beta*np.sum(betas_new[k,l,:]**2)
                denominator = np.sum(weights)
                if denominator > 1e-10:
                    sigmas_Y_new[k,l] = numerator/denominator
                    if sigmas_Y_new[k,l] < sigma_tolerance:
                        sigmas_Y_new[k,l] = sigma_tolerance
            except:
                betas_new[k,l,:]=betas[0][k,l,:]
                sigmas_Y_new[k,l] = betas[1][k,l]

    sigmas_Y_new[sigmas_Y_new<sigma_tolerance] = sigma_tolerance
    sigmas_Y_new[np.isnan(sigmas_Y_new)] = sigma_tolerance
    
    # Update the calculated betas
    betas = [betas_new,sigmas_Y_new]
    
    return betas

###################################################################################################################################
# LOG LIKELIHOOD
###################################################################################################################################
def get_log_likelihood_lower_bound(K,L,M,N,alphas,betas,gammas,r,train_I,train_J, train_Y,X_composite,mu,mv):
    # TO DO : optimize this by placing all operations on a single line (also optimze the E step and M steps which use similar eqns)
    # log likelihood >= H(q) + Expectation wrt q* of complete log probability given parameters
    log_likelihood = 0
    
    ###############
    # Row terms
    ###############    
    digamma_gamma1mk = digamma(gammas[0])
    digamma_gamma1m = digamma(np.sum(gammas[0],1))
    digamma_gamma1k = np.array(np.sum(digamma_gamma1mk,0))[0]
    digamma_gamma1m_sum_m= np.sum(digamma_gamma1m)    
    # M*ln(B(alpha1)
    log_likelihood += M*(gammaln(np.sum(alphas[0])) - np.sum(gammaln(alphas[0])))
    
    # Sum over k (alpha1k -1)*(sum over m(digamma(gamma1mk)))
    log_likelihood += np.dot(alphas[0]-1,digamma_gamma1k)
    # (sum over m(digamma(sum over k(gamma1mk))))*(sum over k(alpha1k - 1))
    log_likelihood -= digamma_gamma1m_sum_m*(np.sum(alphas[0]) - K)
    
    ###############
    # Col terms
    ###############
    digamma_gamma2nl = digamma(gammas[1])
    digamma_gamma2n = digamma(np.sum(gammas[1],1))
    digamma_gamma2l = np.array(np.sum(digamma_gamma2nl,0))[0]
    digamma_gamma2l_sum_n= np.sum(digamma_gamma2n)    
    # N*ln(B(alpha2)
    log_likelihood += N*(gammaln(np.sum(alphas[1])) - np.sum(gammaln(alphas[1])))
    
    # Sum over l (alpha2l -1)*(sum over n(digamma(gamma2nl)))
    log_likelihood += np.dot(alphas[1]-1,digamma_gamma2l)
    # (sum over n(digamma(sum over l(gamma2nl))))*(sum over l(alpha2l - 1))
    log_likelihood -= digamma_gamma2l_sum_n*(np.sum(alphas[0]) - L)
    
    
    ###############
    # Cross terms 
    ###############
    # sum over o and k (r1ok*(digamma(gamma1mk) - digamma(sum over k(gamma1mk))))
    # digamma_gamma1mk M x K
    # digamma_gamma1m M x 1
    log_likelihood += np.dot(np.array(mu.flatten())[0],np.array(np.sum(np.multiply(r[0],digamma_gamma1mk),1).flatten())[0]) - np.sum(np.multiply(mu,digamma_gamma1m))
    
    # sum over o and l (r2ol*(digamma(gamma2nl) - digamma(sum over l(gamma2nl))))
    log_likelihood += np.dot(np.array(mv.flatten())[0],np.array(np.sum(np.multiply(r[1],digamma_gamma2nl),1).flatten())[0]) - np.sum(np.multiply(mv,digamma_gamma2n))
        
    # Expectation over qz1, qz2 of ln(p(Ymn/Beta_kl times X))
    sigmaY_sqrd = betas[1]
    beta = betas[0]
    X = X_composite
    Yobs = train_Y
    r1_obs = r[0][train_I] # |Yobs| x K
    r2_obs = r[1][train_J] # |Yobs| x L
    for k in range(K):
        for l in range(L):
            weight = np.multiply(r1_obs[:,k],r2_obs[:,l])
            beta_times_x = np.dot(X,beta[k,l,:]) # (|Yobs| x 1)
            log_likelihood += np.sum(np.multiply(weight,(-.5*np.log(2*np.pi*sigmaY_sqrd[k,l])-(.5/(sigmaY_sqrd[k,l]))*(Yobs**2 - 2*Yobs*beta_times_x + beta_times_x**2))))
    
    ###############
    # H(q) Row terms
    ###############
    # Sum over m,k (gamma1mk -1)*(digamma(gamma1mk))
    log_likelihood -= np.sum(np.multiply(gammas[0]-1,digamma_gamma1mk))
    # (sum over m(digamma(sum over k(gamma1mk))))*(sum over k(gamma1mk - 1))
    log_likelihood += np.dot(np.array(np.sum(gammas[0]-1,1).flatten())[0],np.array(digamma_gamma1m.flatten())[0])
    
    log_likelihood -= np.sum(gammaln(np.sum(gammas[0],1)))
    log_likelihood += np.sum(gammaln(gammas[0]))    
       
    ###############
    # H(q) Col terms
    ###############
    # Sum over n,l (gamma2nl -1)*(digamma(gamma2nl))
    log_likelihood -= np.sum(np.multiply(gammas[1]-1,digamma_gamma2nl))
    # (sum over n(digamma(sum over l(gamma2nl))))*(sum over l(gamma2nl - 1))
    log_likelihood += np.dot(np.array(np.sum(gammas[1]-1,1).flatten())[0],np.array(digamma_gamma2n.flatten())[0])

    log_likelihood -= np.sum(gammaln(np.sum(gammas[1],1)))
    log_likelihood += np.sum(gammaln(gammas[1]))
    
    ###############
    # H(q) Cross terms
    ###############
    # sum over o,k(r1ok*ln(r1ok)
    for k in range(K):
        log_likelihood -= np.dot(np.array(mu.flatten())[0],np.multiply(r[0][:,k],np.log(r[0][:,k])))
    # sum over o,l(r2ol*ln(r2ol)
    for l in range(L):
        log_likelihood -= np.dot(np.array(mv.flatten())[0],np.multiply(r[1][:,l],np.log(r[1][:,l])))
    
    return log_likelihood