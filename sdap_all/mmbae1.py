'''
Author - Suriya Gunasekar
Title - Functions for learning and prediction in MMBAE models
Ref: Modified from LDBAE code by Clinton Jones
'''

import pdb
import numpy as np
import scipy.sparse as sp
from scipy.special import psi as scipy_psi
from scipy.special import gammaln as scipy_gammaln
#from scikits.learn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
import scipy.optimize as opt
from datetime import datetime
import gc
import numexpr as ne

sigma_tolerance = 1e-3

def learn_bae(Y, W, Xs, K, L, num_runs, num_iter, precision, reg_lambda, initial_T=50, eta=0.5, init_point=None):
    """Learn the parameters of Bayesian affinity estimation model
    Inputs - 
        Y = M x N ratings matrix
        W = M x N weight matrix (0: missing, 1: non-missing)
        Xs = 2 x [N(v) x D(v)] user covariates matrices
            * 2 = number of modes
            * N(v) = number of entities in mode v
            * D(v) = number of covariates for entities in mode v
        K = number of clusters in mode 0
        L = number of clusters in mode 1
        num_runs = Number of runs with random initialization
        num_iter = Max number of EM iterations
        precision = Precision for convergence
        reg_lambda = Regularization constant for model coefficients
        init_point = Pre-specified initialization of responsibilities (and optional alphas)
            r_mnk: M x N x K responsibilities matrix
            r_nml: N x M x L responsibilities matrix
            alphas: ([K x 1], [L x 1]) concentration parameter vectors (1 vector per mode)

    Outputs - 
        bae_parameters = Structure of learned parameters over all runs with fields:
            alphas: ([K x 1], [L x 1]) concentration parameter vectors (1 vector per mode)
            mus_X: ([K x D(0)], [L x D(1)]) covariates means (1 set of means per cluster, per mode)
            sigmas_X: ([K x 1], [L x 1]) covariates variances (1 set per cluster, per mode)
            betas: K x L x [D(1)+...+D(V)+1] model coefficients
            sigmas_Y: K x L ratings variances"""
    # this is just for debugging: forget it for now
    ofile=open('debug.txt','w')
    ofile.close()
    # Indices of observed variables            
    I, J, throwaway_W = sp.find(W)
    del throwaway_W
    num_nonzero = len(I)
    M = Xs[0].shape[0]
    N = Xs[1].shape[0]
    try:
        D1 = Xs[0].shape[1]
    except:
        D1 = 0
    try:
        D2 = Xs[1].shape[1]
    except:
        D2 = 0
    
    try: 
        D3 = Xs[2].shape[1]
    except:
        D3=0
                   
    D=1+D1+D2+D3
    
    if init_point is not None:
        num_runs = 1

    bae_parameters = []
    log_likelihood = np.zeros(num_iter)
    best_log_likelihood = -1e99

    # Initialization
    # r_mnk, r_nml are NNZ X K and NNZ X L dimension matrices
    if init_point is not None:
        r_mnk = init_point['r_mnk']
        r_nml = init_point['r_nml']
        r_m0k = init_point['r_m0k']
        r_n0l = init_point['r_n0l']
        try:
            alphas = init_point['alphas']
        except KeyError:
            alphas = init_alphas(r_mnk, r_nml)
    else:
        r_mnk, r_nml, r_m0k, r_n0l = init_rs(M, K, N, L, num_nonzero)
        alphas = init_alphas(r_mnk, r_nml, r_m0k, r_n0l)

    gamma_mk,gamma_nl=init_gammas(r_mnk,r_nml,r_m0k,r_n0l,alphas, I, J)
    
    init_point={'r_mnk':r_mnk,'r_nml':r_nml,'r_m0k':r_m0k,'r_n0l':r_n0l,'alphas':alphas}
    
    betas = np.zeros((K,L,D))
    sigmas_Y = np.zeros((K,L))+sigma_tolerance
    mus_X = [np.zeros((K,D1)), np.zeros((L,D2))]
    sigmas_X = [np.zeros((K,))+ sigma_tolerance, np.zeros((L,))+ sigma_tolerance]
    
    # Learning
    for run in range(num_runs): 
        if __debug__:
            print "Run %d" % run

        # EM Section
        t = 0
        i=0
        delta = 1e99
        first = True
        
        #print_lowerboud(Y, Xs, alphas, mus_X, sigmas_X, betas, sigmas_Y, r_mnk, r_nml, r_m0k, r_n0l, gamma_mk, gamma_nl, I, J, reg_lambda)
             
        while delta > precision*10 and t < num_iter:
            # M-step
            print "M STEP"
            #print "Updating Theta"
            mus_X,sigmas_X = update_theta(r_m0k, r_n0l, Xs, mus_X, sigmas_X)
            #print_lowerboud(Y, Xs, alphas, mus_X, sigmas_X, betas, sigmas_Y, r_mnk, r_nml, r_m0k, r_n0l, gamma_mk, gamma_nl, I, J, reg_lambda)    
            gc.collect()  
            #print "Updating Betas"
            betas,sigmas_Y = update_betas_and_sigmas_Y(Y, I, J, Xs, r_mnk, r_nml, reg_lambda, betas, sigmas_Y)
            #print_lowerboud(Y, Xs, alphas, mus_X, sigmas_X, betas, sigmas_Y, r_mnk, r_nml, r_m0k, r_n0l, gamma_mk, gamma_nl, I, J, reg_lambda)
            gc.collect()
            #print "Updating alpha"
            if not first:
                alphas = update_alphas(gamma_mk, gamma_nl, alphas, I, J)
            gc.collect()
            
            
            #print_lowerboud(Y, Xs, alphas, mus_X, sigmas_X, betas, sigmas_Y, r_mnk, r_nml, r_m0k, r_n0l, gamma_mk, gamma_nl, I, J, reg_lambda)
            if (t%10==0):                
                log_likelihood[i] = get_lower_bound(Y, Xs, alphas, mus_X, sigmas_X, betas, sigmas_Y, r_mnk, r_nml, r_m0k, r_n0l, gamma_mk, gamma_nl, I, J, reg_lambda)
                if log_likelihood[i] == np.nan:
                    print "Invalid likelihood (happens occasionally)!  Dropping to debugger"
                    pdb.set_trace()
                if i>0:
                    delta = log_likelihood[i]-log_likelihood[i-1]
                print "\t%s: Iteration %d(%d), lower bound: %f (delta: %f)" % (datetime.now(), t, i, log_likelihood[i], delta)
                i=i+1
            gc.collect()   
#            log_likelihood[t] = get_lower_bound(Y, Xs, alphas, mus_X, sigmas_X, betas, sigmas_Y, r_mnk, r_nml, r_m0k, r_n0l, gamma_mk, gamma_nl, I, J, reg_lambda)
#            if log_likelihood[t] == np.nan:
#                print "Invalid likelihood (happens occasionally)!  Dropping to debugger"
#                pdb.set_trace()
#            if t>0:
#                delta = log_likelihood[t]-log_likelihood[t-1]
#                print "\t%s: Iteration %d, lower bound: %f (delta: %f)" % (datetime.now(), t, log_likelihood[t], delta)    
#            gc.collect()

            t += 1
                
            # E-step (mean field)
            print "E STEP"  
            r_mnk, r_nml, r_m0k, r_n0l, gamma_mk, gamma_nl = mean_field(Y, I, J, Xs, alphas, mus_X, sigmas_X, betas, sigmas_Y, r_mnk, r_nml, r_m0k, r_n0l, initial_T, eta)
            #print_lowerboud(Y, Xs, alphas, mus_X, sigmas_X, betas, sigmas_Y, r_mnk, r_nml, r_m0k, r_n0l, gamma_mk, gamma_nl, I, J, reg_lambda)
            gc.collect()
            # Storage of iteration data:
            first = False
            
            
            
        bae_parameters.append({})
        bae_parameters[run]['alphas'] = alphas
        bae_parameters[run]['mus_X'] = mus_X
        bae_parameters[run]['sigmas_X'] = sigmas_X
        bae_parameters[run]['betas'] = betas
        bae_parameters[run]['sigmas_Y'] = sigmas_Y
        bae_parameters[run]['log_likelihood'] = log_likelihood[i-1]
        bae_parameters[run]['gamma_mk'] = gamma_mk
        bae_parameters[run]['gamma_nl'] = gamma_nl
        bae_parameters[run]['r_mnk'] = r_mnk
        bae_parameters[run]['r_nml'] = r_nml
        bae_parameters[run]['r_m0k'] = r_m0k
        bae_parameters[run]['r_n0l'] = r_n0l
        
        if log_likelihood[i-1] > best_log_likelihood:
            best_log_likelihood = log_likelihood[i-1]
            best_run = run

    r_mnk=bae_parameters[best_run]['r_mnk']
    r_nml=bae_parameters[best_run]['r_nml']  
    r_m0k=bae_parameters[best_run]['r_m0k']
    r_n0l=bae_parameters[best_run]['r_n0l']    
          
    I, J, throwaway_W = sp.find(W)
    predict_Ys(I, J, np.ravel(Y[I,J]), Xs, r_mnk, r_nml, r_m0k, r_n0l, betas, print_mean_pred_err=True, indent_level=1, label="Final Train")

    if num_runs == 1:
        best_run = 0
    return bae_parameters, best_run, init_point

# Initializers
def init_rs(M, K, N, L, num_nonzero):
    '''
    returns uniformly randomly assigned r_mnk and r_nml values to the MN possible ratings 
    '''
    identity_k = np.eye(K)
    # Generate random row cluster assignments, each with equally populated (but randomly assigned) rows
    p_rows=np.zeros(num_nonzero+M,dtype='int')
    perm=np.random.permutation(num_nonzero+M)
    init_m=(num_nonzero+M)//K
    for g in range(K):
        p_rows[perm[(g*init_m):((g+1)*init_m)]]=g
    p_rows[perm[K*init_m:num_nonzero+M]]=K-1;
    r_m0k=identity_k[p_rows[num_nonzero:num_nonzero+M],:]
    print r_m0k[np.sum(r_m0k,1)!=1.0]

    p_rows=p_rows[0:num_nonzero].reshape(num_nonzero,1)
    r_mnk=identity_k[p_rows,:].reshape(num_nonzero,K)
    print r_mnk[np.sum(r_mnk,1)!=1.0]
    
    identity_l = np.eye(L)
    # Generate random column cluster assignments, each with equally populated (but randomly assigned) columns
    p_cols=np.zeros(num_nonzero+N,dtype='int')
    perm=np.random.permutation(num_nonzero+N)
    init_n=(num_nonzero+N)//L
    for g in range(L):
        p_cols[perm[(g*init_n):((g+1)*init_n)]]=g
    p_cols[perm[L*init_n:num_nonzero+N]]=L-1;
    r_n0l=identity_l[p_cols[num_nonzero:num_nonzero+N],:].reshape(N,L)
    print r_n0l[np.sum(r_n0l,1)!=1.0]
    p_cols=p_cols[0:num_nonzero].reshape(num_nonzero,1)     
    r_nml=identity_l[p_cols,:].reshape(num_nonzero,L)
    print r_nml[np.sum(r_nml,1)!=1.0]
    
    assert(r_m0k.shape==(M,K))
    assert(r_n0l.shape==(N,L))
    assert(r_mnk.shape==(num_nonzero,K))
    assert(r_nml.shape==(num_nonzero,L))
    
    return np.float32(r_mnk), np.float32(r_nml), np.float32(r_m0k), np.float32(r_n0l)

def init_alphas(r_mnk, r_nml, r_m0k, r_n0l):
    '''
    assigns the initialization of alphas determined empirically from the rs
    '''
    K=r_m0k.shape[1]
    L=r_n0l.shape[1]
    alphas = [np.zeros((K,)),np.zeros((L,))]
    alphas[0] = np.sum(r_mnk,0)+np.sum(r_m0k,0)
    alphas[1] = np.sum(r_nml,0)+np.sum(r_n0l,0)
    return alphas
    
def init_gammas(r_mnk,r_nml,r_m0k,r_n0l,alphas, I, J):
    M,K = r_m0k.shape
    N,L = r_n0l.shape
    gamma_mk = r_m0k + np.tile(alphas[0].reshape(1,K),(M,1)) # an MxK matrix
    gamma_nl = r_n0l + np.tile(alphas[1].reshape(1,L),(N,1)) # an NxL matrix
    
    for k in range(K):
        tmp=sp.csr_matrix((r_mnk[:,k],(I,J)),shape=(M,N))
        gamma_mk[:,k] += np.ravel(tmp.sum(1))
    for l in range(L):
        tmp=sp.csr_matrix((r_nml[:,l],(I,J)),shape=(M,N))
        gamma_nl[:,l] += np.ravel(tmp.sum(0))
    
    return gamma_mk,gamma_nl
    

# M Step update equations
def update_theta(r_m0k, r_n0l, Xs, mus_X_old, sigmas_X_old, debug=False):
    ''' 
    Updates the values of mu_X and sigma_X which are matrices of dimentions [(K x D1),(L x D2)] and [(K X 1),(L X 1)] respectively
    '''
    M,K = r_m0k.shape
    N,L = r_n0l.shape
    try:
        D1 = Xs[0].shape[1]
    except:
        D1 = 0
    try:
        D2 = Xs[1].shape[1]
    except:
        D2 = 0

    mus_X = [np.zeros((K,D1)), np.zeros((L,D2))]
    sigmas_X = [np.zeros((K,)), np.zeros((L,))]
    
    assert((r_m0k>=0).all)
    assert((r_n0l>=0).all)

    if D1 > 0:
        for k in range(K):
            if (np.sum(r_m0k[:,k])>1e-10):
                tmp=np.tile(r_m0k[:,k].reshape(M,1),(1,D1))
                tmp1=Xs[0]
                tmp2=r_m0k[:,k]
                mus_X[0][k,:] = ne.evaluate("sum(tmp*tmp1,0)") / np.sum(tmp2)
                
                mu_x_tiled=np.tile(mus_X[0][k,:].reshape(1,D1), (M, 1))
                
                x_minus_mu_square = ne.evaluate("tmp1**2 - 2*tmp1*mu_x_tiled + mu_x_tiled**2")
                sigmas_X[0][k] = np.sum(tmp2*(np.sum(x_minus_mu_square, 1)))/(D1*np.sum(tmp2)) 
                del tmp,tmp1,tmp2
            else:
                mus_X[0][k,:] = mus_X_old[0][k,:]
                sigmas_X[0][k] = sigmas_X_old[0][k]        
        sigmas_X[0][sigmas_X[0]<sigma_tolerance] = sigma_tolerance 
        # If the variance is too small, don't let it underflow -- should only be a problem in very small datasets

    if D2 > 0:
        for l in range(L):
            if (np.sum(r_n0l[:,l])>1e-10):
                tmp=np.tile(r_n0l[:,l].reshape(N,1),(1,D2))
                tmp1=Xs[1]
                tmp2=r_n0l[:,l]
                mus_X[1][l,:] = ne.evaluate("sum(tmp*tmp1,0)") / np.sum(tmp2)
                
                mu_x_tiled=np.tile(mus_X[1][l,:].reshape(1,D2), (N, 1))
                x_minus_mu_square = ne.evaluate("tmp1**2 -2*tmp1*mu_x_tiled+ mu_x_tiled**2")
                sigmas_X[1][l] = np.sum(tmp2*(np.sum(x_minus_mu_square, 1)))/(D2*np.sum(tmp2))
                del tmp,tmp1,tmp2
            else:
                mus_X[1][l,:] = mus_X_old[1][l,:]
                sigmas_X[1][l] = sigmas_X_old[1][l]        
        sigmas_X[1][sigmas_X[1]<sigma_tolerance] = sigma_tolerance    
        # If the variance is too small, don't let it underflow -- should only be a problem in very small datasets
    mus_X_new=mus_X
    sigmas_X_new=sigmas_X   
        
    if debug:
        # log(p(x_1m))                    
        for k in range(K):
            mu_x_tiled=np.tile(mus_X_new[0][k,:].reshape(1,D1), (M, 1))
            x_minus_mu_square = (Xs[0]-mu_x_tiled)**2
            expectation_logx=(r_m0k[:,k]*(((-.5*D1)*np.log(2*np.pi*(sigmas_X_new[0][k])) - (.5/sigmas_X_new[0][k])*np.sum(x_minus_mu_square, axis=1))))
            log_likelihood1 = np.sum(expectation_logx)
            
                   
            mu_x_tiled=np.tile(mus_X_old[0][k,:].reshape(1,D1), (M, 1))
            x_minus_mu_square = (Xs[0]-mu_x_tiled)**2
            expectation_logx=(r_m0k[:,k]*(((-.5*D1)*np.log(2*np.pi*(sigmas_X_old[0][k])) - (.5/sigmas_X_old[0][k])*np.sum(x_minus_mu_square, axis=1))))
            log_likelihood_old1 = np.sum(expectation_logx)
            
            if ((log_likelihood_old1-log_likelihood1) >1e-5):
                print "MU_X[0], old:%f, new %f, diff %f :" %(log_likelihood_old1, log_likelihood1, log_likelihood_old1-log_likelihood1)
                ofile=open('debug.txt','a')
                ofile.write('update in theta[0][{6}] failed:\nmu_x_old={0}\nsigma_x_old={1}\nr_m0k={2}\nr_n0l={3}\nmu_x_new={4}\nsigma_x_new={5}\nX[0]={7}\n\n\n'.format(mus_X_old[0][k,:], sigmas_X_old[0][k], r_m0k, r_n0l, mus_X_new[0][k,:], sigmas_X_new[0][k], k, Xs[0]))
                ofile.close()
                raw_input("Press Enter to continue...")

            
        log_likelihood1=0
        log_likelihood_old1=0        
        # log(p(x_2n))
        for l in range(L):
            mu_x_tiled=np.tile(mus_X_new[1][l,:].reshape(1,D2), (N, 1))
            x_minus_mu_square = (Xs[1]-mu_x_tiled)**2
            expectation_logx=(r_n0l[:,l]*(((-.5*D1)*np.log(2*np.pi*(sigmas_X_new[1][l])) - (.5/sigmas_X_new[0][l])*np.sum(x_minus_mu_square, axis=1))))
            log_likelihood1 = np.sum(expectation_logx)        
            
            mu_x_tiled=np.tile(mus_X_old[1][l,:].reshape(1,D2), (N, 1))
            x_minus_mu_square = (Xs[1]**2-mu_x_tiled)**2
            expectation_logx=(r_n0l[:,l]*(((-.5*D1)*np.log(2*np.pi*(sigmas_X_old[1][l])) - (.5/sigmas_X_old[0][l])*np.sum(x_minus_mu_square, axis=1))))
            log_likelihood_old1 = np.sum(expectation_logx)
            
            if ((log_likelihood_old1-log_likelihood1) >1e-5):
                print "MU_X[0], old:%f, new %f, diff %f :" %(log_likelihood_old1, log_likelihood1, log_likelihood_old1-log_likelihood1)

                ofile=open('debug.txt','a')
                ofile.write('update in theta[1][{6}] failed:\nmu_x_old={0}\nsigma_x_old={1}\nr_m0k={2}\nr_n0l={3}\nmu_x_new={4}\nsigma_x_new={5}\nX[1]={7}\n\n\n'.format(mus_X_old[1][l,:], sigmas_X_old[1][l], r_m0k[1], r_n0l[1], mus_X_new[1][l,:], sigmas_X_new[1][l], l, Xs[1]))
                ofile.close()
                raw_input("Press Enter to continue...")


    return mus_X_new,sigmas_X_new

def update_betas_and_sigmas_Y(Y, I, J, Xs, r_mnk, r_nml, reg_lambda, betas_old, sigmas_Y_old, debug=False):
    ''' 
    Updates the values of betas and sigma_y which are matrices of dimentions [K x L x (1+D1+D2)] and [K x L] respectively
    '''
    K=r_mnk.shape[1]
    L=r_nml.shape[1]
    
    D = 1
    try:
        D += Xs[0].shape[1]
    except:
        pass
    try:
        D += Xs[1].shape[1]
    except:
        pass
    try:
        D += Xs[2][2].shape[1]
    except:
        pass    # No cell features, so don't do anything about them
        
    betas_new = np.zeros((K,L,D))
    sigmas_Y_new = np.zeros((K,L))

    X, num_nonzero = build_test_X(Xs, I, J)
    # X is of dimension [num_nonzero x (1+D1+D2)]

    Y_values = np.array(Y[(I,J)]).reshape((num_nonzero,))
    Ysq_values = Y_values**2
    
    # print "%s: Start of outer loop" % datetime.now()
    for k in range(K):
        # print "%s: Start of inner loop" % datetime.now()
        for l in range(L):
            weights = (r_mnk[:,k]*r_nml[:,l])
#            assert ((r_mnk>=0).all)
#            assert ((r_nml>=0).all)
            linear_weights = weights.reshape((num_nonzero,))
            if ((linear_weights>0.01).any):
                sqrt_weights = ne.evaluate("sqrt(linear_weights)")
                nzweight=len(sqrt_weights[linear_weights>1e-10])
                weighted_X = np.tile(sqrt_weights[linear_weights>1e-10].reshape(nzweight,1),(1,D))*X[linear_weights>1e-10]
                weighted_Y_values = Y_values[linear_weights>1e-10]*sqrt_weights[linear_weights>1e-10]
                # Calculate new beta[k][l]
                regressor = Ridge(alpha=reg_lambda,fit_intercept=False) 
                regressor.fit(weighted_X,weighted_Y_values)
                betas_new[k,l,:] = regressor.coef_[:]
                # Calculate new sigmas_Y[k][l]
                new_beta_times_x = np.dot(X,betas_new[k,l,:].reshape(D,1)).reshape(num_nonzero,)
                tmp=betas_new[k,l,:]
                numerator = np.sum(ne.evaluate("(weights*(Ysq_values + new_beta_times_x**2 - 2*Y_values*new_beta_times_x))"))+reg_lambda*np.sum(tmp**2)
                del tmp
                denominator = np.sum(weights)
                sigmas_Y_new[k,l] = numerator/denominator
            else:
                betas_new[k,l,:]=betas_old[k,l,:]
                sigmas_Y_new[k,l] = sigmas_Y_old[k,l]                           
    sigmas_Y_new[sigmas_Y_new<sigma_tolerance] = sigma_tolerance
    sigmas_Y_new[np.isnan(sigmas_Y_new)] = sigma_tolerance

    
    if debug:
        # log(p(y_mn))
        lower_bound_from_ys = 0
        reg_beta=0
        for k in range(K):
            for l in range(L):
                beta_x = np.dot(X,betas_new[k,l,:].reshape(D,1)).reshape(num_nonzero,)
                weights = r_mnk[:,k]*r_nml[:,l] # nnz X 1 matrix
                Ey_term=weights * ((-.5)*np.log(2*np.pi*sigmas_Y_new[k,l]) - (.5/sigmas_Y_new[k,l])*(Y_values**2-2*Y_values*beta_x+beta_x**2))
                assert (Ey_term.shape==(num_nonzero,))
                lower_bound_from_ys += np.sum(Ey_term)   
                reg_beta-=reg_lambda*np.sum(betas_new[k,l,:]**2)/(2*sigmas_Y_new[k,l])   
                
        lower_bound_from_ys_old = 0
        reg_beta_old=0 
        for k in range(K):
            for l in range(L):
                beta_x = np.dot(X,betas_old[k,l,:].reshape(D,1)).reshape(num_nonzero,)
                weights = r_mnk[:,k]*r_nml[:,l] # nnz X 1 matrix
                Ey_term=weights * ((-.5)*np.log(2*np.pi*sigmas_Y_old[k,l]) - (.5/sigmas_Y_old[k,l])*(Y_values**2-2*Y_values*beta_x+beta_x**2))
                assert (Ey_term.shape==(num_nonzero,))
                lower_bound_from_ys_old += np.sum(Ey_term)    
                reg_beta_old-=reg_lambda*np.sum(betas_old[k,l,:]**2)/(2*sigmas_Y_old[k,l])     
    
        if ((lower_bound_from_ys_old+reg_beta_old-lower_bound_from_ys-reg_beta)>1e-5):
            print "BETA, old: %f, new: %f , diff %f:" %(lower_bound_from_ys_old+reg_beta_old, lower_bound_from_ys+reg_beta, lower_bound_from_ys_old+reg_beta_old-lower_bound_from_ys-reg_beta)
            ofile=open('debug.txt','a')
            ofile.write('update in beta failed:\nbeta_old={0}\nsigma_Y_old={1}\nr_mnk={2}\nr_nml={3}\nbeta_new={4}\nsigma_Y_new{5}\nY={6}\n\n\n'.format(betas_old, sigmas_Y_old, r_mnk, r_nml, betas_new, sigmas_Y_new, Y_values))
            ofile.close()
        del X, num_nonzero    
    
    return betas_new, sigmas_Y_new

def update_alphas(gamma_mk, gamma_nl, alphas_old, I, J, debug=False):
    ''' 
    Updates the values of alpha[0], alpha[1], which are matrices of dimentions [1x K ] and [1 x L] respectively
    '''
    (M,K)=gamma_mk.shape
    (N,L)=gamma_nl.shape

    alphas_new = [np.zeros(K,), np.zeros(L,)]
    del M,N
    #objective_value is a function handler which returns the function value and gradient
    alphas_new[0], f0, d0= opt.fmin_l_bfgs_b(func=objective_value, x0=alphas_old[0], args=(gamma_mk,), bounds=[(0,None) for i in range(K)], maxfun=50000)
    if d0['warnflag']>0 :
        print "WARNING: Update for alpha[0] did not converge"
    del i
    alphas_new[1], f1, d1= opt.fmin_l_bfgs_b(func=objective_value, x0=alphas_old[1], args=(gamma_nl,), bounds=[(0,None) for i in range(L)], maxfun=50000)
    if d1['warnflag']>0 :
        print "WARNING: Update for alpha[1] did not converge"
    del f0,f1,i
    
    #===========================================================================
    # if (debug==True):
    #    v1_old,g=objective_value(alphas_old[0],gamma_mk)[0]
    #    v1,g=objective_value(alphas_new[0],gamma_mk)[0]
    #    v1=-v1
    #    v1_old=-v1_old
    #    if ((v1_old-v1)>1e-5):
    #        print "Alpha[0], old: %f, new: %f , diff %f:" %(v1_old,v1,v1_old-v1)
    #        ofile=open('debug.txt','a')
    #        ofile.write('update in alpha[0] failed:\nalpha_old={0}\ngamma_mk={1}\ngamma_nl={2}\nalpha_new={3}\n\n\n'.format(alphas_old[0], gamma_mk, gamma_nl, alphas_new[0]))
    #        ofile.close()
    #        
    #    v2_old,g=objective_value(alphas_old[1],gamma_nl)
    #    v2,g=objective_value(alphas_new[1],gamma_nl)
    #    v2=-v2
    #    v2_old=-v2_old
    #    if ((v2_old-v2)>1e-5):
    #        print "Alpha[1], old: %f, new: %f , diff %f:" %(v2_old,v2,v2_old-v2)
    #        ofile=open('debug.txt','a')
    #        ofile.write('update in alpha[1] failed:\nalpha_old={0}\ngamma_mk={1}\ngamma_nl={2}\nalpha_new={3}\n\n\n'.format(alphas_old[1], gamma_mk, gamma_nl, alphas_new[1]))
    #        ofile.close()
    #===========================================================================
    return alphas_new

# Helpers in M step updates
def build_X(Xs, sample_W):
    '''
    Similar to build_test_X but takes W as hte input and returns I, J in addition to X and num_nonzero
    '''
    I, J, throwaway_W = sp.find(sample_W)
    del throwaway_W
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

def objective_value(alpha, gamm):
    '''
    Objective function for optimization of alphas
    '''
    M,K = gamm.shape # is actually N,L for alpha2
    assert(alpha.shape==(K,))
    val = M*scipy_gammaln(np.sum(alpha)) # numerator of first term in arg max (log(gamma(sum(alpha))))
    
    grad = np.zeros((K,))    # this is going to be a partial derivative
    for k in range(K):
        val -= M*scipy_gammaln(alpha[k]) # denominator of first term in arg max (-log(sum(gamma(alpha[k]))) for all k)
        grad[k] -= M*psi_gamma_helper(alpha, k) 
    
    for m in range(M):      
        for k in range(K):
            val += (alpha[k] - 1)*(psi_gamma_helper(gamm[m,:],k))
            grad[k] += psi_gamma_helper(gamm[m,:],k)
    return -val, -grad


# E Step
def mean_field(Y, I, J, Xs, alphas, mus_X, sigmas_X, betas, sigmas_Y, r_mnk, r_nml, r_m0k, r_n0l, initial_T, eta):
    '''
    Returns the updated r_mnk,r_nml after the E-Step    
    '''
    M,N = Y.shape
    K = len(alphas[0])
    L = len(alphas[1])

    # Temperature:  need to tune and find a good one (but it shouldn't have to get too big)
    T = initial_T 
    delta = 1e99
    iters_run = 0
    while delta > 1e-1:
        # Update gammas
        gamma_mk = r_m0k + alphas[0] # an MxK matrix
        gamma_nl = r_n0l + alphas[1] # an NxL matrix   
        for k in range(K):
            tmp=sp.csr_matrix((r_mnk[:,k],(I,J)),shape=(M,N))
            gamma_mk[:,k] += np.ravel(tmp.sum(1))
        for l in range(L):
            tmp=sp.csr_matrix((r_nml[:,l],(I,J)),shape=(M,N))
            gamma_nl[:,l] += np.ravel(tmp.sum(0))

        # Update rs
        r_mnk_new, r_nml_new, r_m0k_new, r_n0l_new = update_mean_field_rs(Xs, mus_X, sigmas_X, betas, sigmas_Y, gamma_mk, gamma_nl, Y, r_nml, r_mnk, T, I, J)
        
        delta = np.sum((r_mnk_new-r_mnk)**2)+ np.sum((r_nml_new-r_nml)**2) + np.sum((r_m0k_new-r_m0k)**2)+ np.sum((r_n0l_new-r_n0l)**2)
        r_mnk = r_mnk_new
        r_nml = r_nml_new
        r_m0k = r_m0k_new
        r_n0l = r_n0l_new       
        T = np.max([eta*T, 1])
        iters_run += 1
        if iters_run % 100 == 0:
            print "%d iterations of mean field." % iters_run

    return r_mnk_new, r_nml_new, r_m0k_new, r_n0l_new, gamma_mk, gamma_nl

# Helper function for E Step
def update_mean_field_rs(Xs, mus_X, sigmas_X, betas, sigmas_Y, gamma_mk, gamma_nl, Y, r_nml, r_mnk, T, I, J):
    M = Xs[0].shape[0]
    N = Xs[1].shape[0]
    K,L = sigmas_Y.shape
    X, num_nonzero= build_test_X(Xs, I, J)
    
    log_r_mnk_new = np.zeros((num_nonzero,K));
    log_r_nml_new = np.zeros((num_nonzero,L));
    log_r_m0k_new = np.zeros((M,K));
    log_r_n0l_new = np.zeros((N,L));

    try:
        D1 = Xs[0].shape[1]
    except:
        D1 = 0
    try:
        D2 = Xs[1].shape[1]
    except:
        D2 = 0
    D = D1 + D2 + 1
    try:
        D3 = Xs[2][2].shape[1]
        D += D3
    except:
        pass    # No cell features.  Ignore
    
    pi=np.pi

    # r_m0k update
    for k in range(K):
        tmp=np.tile(mus_X[0][k,:], (M, 1))
        tmp1=Xs[0]
        x_minus_mu_square = ne.evaluate("tmp1**2 -2*tmp1*tmp+ tmp**2")
        # M x D1 (x_1m-mu_1)
        # p(x|theta) is gaussian, mean mus_X[k], variance sigmas_X
        tmp1=sigmas_X[0][k]
        log_r_m0k_new[:,k] -= ne.evaluate("sum((.5/tmp1)*x_minus_mu_square, axis=1)").reshape(M,)
        log_r_m0k_new[:,k] += ne.evaluate("(-.5*D1)*(log(2*pi*tmp1))")
        # psi(gamma_k[k])
        log_r_m0k_new[:,k] += scipy_psi(gamma_mk[:,k]).reshape(M,)
        
        # psi(gamma_k[k])
        log_r_mnk_new[:,k] += scipy_psi(gamma_mk[I,k]).reshape(num_nonzero,)
    
    
    # r_n0l update
    for l in range(L):
        tmp=np.tile(mus_X[1][l,:], (N, 1))
        tmp1=Xs[1]
        x_minus_mu_square = ne.evaluate("tmp1**2 -2*tmp1*tmp+ tmp**2") # N x D2 (x_2n-mu_2)
        # p(x|theta) is gaussian, mean mus_X[k], variance sigmas_X
        tmp1=sigmas_X[1][l]
        log_r_n0l_new[:,l] -= ne.evaluate("sum((.5/tmp1)*x_minus_mu_square, axis=1)").reshape(N,)
        log_r_n0l_new[:,l] += ne.evaluate("(-.5*D2)*(log(2*pi*tmp1))")
        # psi(gamma_k[k])
        log_r_n0l_new[:,l] += scipy_psi(gamma_nl[:,l]).reshape(N,)
        
        # psi(gamma_nl[:,l])
        log_r_nml_new[:,l] += scipy_psi(gamma_nl[J,l]).reshape(num_nonzero,)
   
    # r_nml update
    for k in range(K):
        # p(y|beta*x)
        for l in range(L):
            beta_times_x = np.dot(X,betas[k,l,:]) # nnz x 1
            Y_values = np.array(Y[(I,J)]).reshape((num_nonzero,))
            tmp=sigmas_Y[k,l]
            expected_log_prob = ne.evaluate("-.5*log(2*pi*tmp) - (.5/tmp)*(Y_values**2- 2*Y_values*beta_times_x +beta_times_x**2)")
            log_r_mnk_new[:,k] += expected_log_prob * r_nml[:,l]
            log_r_nml_new[:,l] += expected_log_prob * r_mnk[:,k]
    del r_mnk, r_nml 
    
    log_r_m0k_new = ne.evaluate("log_r_m0k_new/T")
    #log_r_m0k_new = log_r_m0k_new
    
    log_r_mnk_new = ne.evaluate("log_r_mnk_new/T")
    #log_r_mnk_new = log_r_mnk_new
    
    log_r_n0l_new = ne.evaluate("log_r_n0l_new/T")
    #log_r_n0l_new = log_r_n0l_new
    
    log_r_nml_new = ne.evaluate("log_r_nml_new/T")
    #log_r_nml_new = log_r_nml_new
    
    
    # Prevent underflow by using a transformed method to get to normalized product space:
    # (e^x) / (e^x + e^y) = 1 / (1 + e^(y-x))
    # instead of trying to calculate e^x and e^y directly, just use e^(y-x).
    r_m0k_new = np.zeros(log_r_m0k_new.shape)
    r_mnk_new = np.zeros(log_r_mnk_new.shape)
    for k in range(K):
        old_settings = np.seterr(over='warn')
        r_m0k_new[:,k] = 1 / (1 + np.sum(np.exp(np.delete(log_r_m0k_new,k,axis=1)-np.tile(log_r_m0k_new[:,k].reshape(M,1),(1,K-1))),axis=1)).reshape(M,)
        r_mnk_new[:,k] = 1 / (1 + np.sum(np.exp(np.delete(log_r_mnk_new,k,axis=1)-np.tile(log_r_mnk_new[:,k].reshape(num_nonzero,1),(1,K-1))),axis=1))
        np.seterr(**old_settings)
    del log_r_m0k_new, log_r_mnk_new 
    
    r_n0l_new = np.zeros(log_r_n0l_new.shape)
    r_nml_new = np.zeros(log_r_nml_new.shape)

    for l in range(L):
        old_settings = np.seterr(over='warn')
        r_n0l_new[:,l] = 1 / (1 + np.sum(np.exp(np.delete(log_r_n0l_new,l,axis=1)-np.tile(log_r_n0l_new[:,l].reshape(N,1),(1, L-1))),axis=1))
        r_nml_new[:,l] = 1 / (1 + np.sum(np.exp(np.delete(log_r_nml_new,l,axis=1)-np.tile(log_r_nml_new[:,l].reshape(num_nonzero,1),(1,L-1))),axis=1))
        np.seterr(**old_settings)
    del log_r_n0l_new, log_r_nml_new 
 
    
    return r_mnk_new, r_nml_new, r_m0k_new, r_n0l_new


# Getting the lower bound on log likelihood
def print_lowerboud(Y, Xs, alphas, mus_X, sigmas_X, betas, sigmas_Y, r_mnk, r_nml, r_m0k, r_n0l, gamma_mk, gamma_nl, I, J, reg_lambda):
    '''
    Calculates the lowerbound of log likelihood and prints them
    '''
    temp_log_like = get_lower_bound(Y, Xs, alphas, mus_X, sigmas_X, betas, sigmas_Y, r_mnk, r_nml, r_m0k, r_n0l, gamma_mk, gamma_nl, I, J, reg_lambda)
    print "Log likelihood : %f" % temp_log_like
        
def get_lower_bound(Y, Xs, alphas, mus_X, sigmas_X, betas, sigmas_Y, r_mnk, r_nml, r_m0k, r_n0l,gamma_mk, gamma_nl, I,J, reg_lambda):
    '''
    Get the lower bound on the observed log likelihood given the current estimates of the parameters
    '''
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
    try:
        D3 = Xs[2].shape[1]
    except:
        D3 = 0 
        
    log_likelihood_lower_bound = 0;
    pi=np.pi

    # Terms involving summation over M  
    # Expectation of log[p(pi_1m)]
    # Mean of a diriclet with parameter gamma_mk
    log_of_gamma_of_sum = scipy_gammaln(np.sum(alphas[0])) #log_e(Gamma(sum(alpha_1k)))
    log_likelihood_lower_bound += ne.evaluate("M*log_of_gamma_of_sum")
    for k in range(K):
        log_of_gamma = scipy_gammaln(alphas[0][k])
        log_likelihood_lower_bound -= ne.evaluate("M*log_of_gamma")

    for m in range(M):
        psi_gamma_helpers = np.array([psi_gamma_helper(gamma_mk[m,:], k) for k in range(K)]).reshape(K,)
        # Expectation of log[p(pi_1m)] continued
        alpha_times_psi_stuff = (alphas[0]-1)*psi_gamma_helpers
        log_likelihood_lower_bound += np.sum(alpha_times_psi_stuff)

        # E_q[log(p(z_1m))]
        log_likelihood_lower_bound += np.sum(r_m0k[m,:]*psi_gamma_helpers)

        # H[q(pi_1m)]
        log_of_gamma_of_sum = scipy_gammaln(np.sum(gamma_mk[m,:]))
        log_likelihood_lower_bound -= log_of_gamma_of_sum
        logs_of_gammas = np.array([scipy_gammaln(gamma_mk[m,k]) for k in range(K)])
        log_likelihood_lower_bound += np.sum(logs_of_gammas)
        gamma_times_psi_stuff = (gamma_mk[m,:]-1)*psi_gamma_helpers
        log_likelihood_lower_bound -= np.sum(gamma_times_psi_stuff)

    # H[q(z_1m)]
    weighted_log=r_m0k[r_m0k > 1e-10]* np.log(r_m0k[r_m0k > 1e-10])
    log_likelihood_lower_bound -= np.sum(weighted_log)

    # log(p(x_1m))
    for k in range(K):
        tmp=np.tile(mus_X[0][k,:], (M, 1))
        tmp1=Xs[0]
        x_minus_mu_square =ne.evaluate("tmp1**2 -2*(tmp1*tmp)+ tmp**2")
        tmp=r_m0k[:,k]
        tmp1=sigmas_X[0][k]
        expectation_logx=tmp*((-.5*D1)*np.log(2*pi*(tmp1)) - (.5/tmp1)*ne.evaluate("sum(x_minus_mu_square, axis=1)"))
        log_likelihood_lower_bound += np.sum(expectation_logx)
        
    # Tems involving summation over N   
    # Expectation of log[p(pi_2n)]
    # Mean of a diriclet with parameter gamma_nl
    log_of_gamma_of_sum = scipy_gammaln(np.sum(alphas[1])) #log_e(Gamma(sum(alpha_2l)))
    log_likelihood_lower_bound += ne.evaluate("N*log_of_gamma_of_sum")
    for l in range(L):
        log_of_gamma = scipy_gammaln(alphas[1][l])
        log_likelihood_lower_bound -= ne.evaluate("N*log_of_gamma")

    for n in range(N):
        psi_gamma_helpers = np.array([psi_gamma_helper(gamma_nl[n,:], l) for l in range(L)]).reshape(L,)
        # Expectation of log[p(pi_2n)] continued
        alpha_times_psi_stuff = (alphas[1]-1)*psi_gamma_helpers
        log_likelihood_lower_bound += np.sum(alpha_times_psi_stuff)

        # E_q[log(p(z_2n))]
        log_likelihood_lower_bound += np.sum(r_n0l[n,:]*psi_gamma_helpers)

        # H[q(pi_2)]
        gamma_of_sum = scipy_gammaln(np.sum(gamma_nl[n,:]))
        log_likelihood_lower_bound -= gamma_of_sum
        logs_of_gammas = np.array([scipy_gammaln(gamma_nl[n,l]) for l in range(L)])  
        log_likelihood_lower_bound += np.sum(logs_of_gammas)
        gamma_times_psi_stuff = (gamma_nl[n,:]-1)*psi_gamma_helpers
        log_likelihood_lower_bound -= np.sum(gamma_times_psi_stuff)

    # H[q(z_2n)]
    weighted_log=r_n0l[r_n0l > 1e-10]* np.log(r_n0l[r_n0l > 1e-10])
    log_likelihood_lower_bound -= np.sum(weighted_log)
        
    # log(p(x_2n))
    for l in range(L):
        tmp=np.tile(mus_X[1][l,:], (N, 1))
        tmp1=Xs[1]
        tmp2=sigmas_X[1][l]
        x_minus_mu_square =ne.evaluate("tmp1**2 -2*(tmp1*tmp)+ tmp**2")
        tmp1=r_n0l[:,l]
        expectation_logx=tmp1*((-.5*D2)*np.log(2*pi*(tmp2)) - (.5/tmp2)*ne.evaluate("sum(x_minus_mu_square, axis=1)"))
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
            tmp=sigmas_Y[k,l]
            Ey_term=ne.evaluate("weights * ((-.5)*log(2*pi*tmp) - (.5/tmp)*(Y_values**2-2*Y_values*beta_times_x+beta_times_x**2))")
            assert (Ey_term.shape==(num_nonzero,))
            lower_bound_from_ys += np.sum(Ey_term)
            tmp1=betas[k,l,:]
            reg_beta-=reg_lambda*np.sum(tmp1**2)/(2*tmp)
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
    return scipy_psi(gamma[k])-scipy_psi(np.sum(gamma))

# Predicting Ys
def predict_Ys(test_I, test_J, test_Y, Xs, r_mnk, r_nml, r_m0k, r_n0l, betas, print_mean_pred_err=False, indent_level=0, label="Train"):
    '''
    Predicts the response in the test_Y, Given the attribute values, cluster assignments for entities 
    test_I and test_J represent the i,j indices such that Y(test_I(k),test_J(k))=test_Y(k)
    in each dyad and the GLM parameters beta for each cluster
    returns the predicted_Y, mse and se
    '''
 
    K=r_mnk.shape[1]
    L=r_nml.shape[1]

    X, num_nonzero = build_test_X(Xs, test_I, test_J)
    
    # num_nonzero is the number of predictions to be made, X(num_nonzero.(1+D1+D2)) is the list of {X_a} 
    # where X_a is the concatenated feature vector the dyad indexed by a
    if ((label!="Test") and  (label!="Validation")):
        # Here r_mnk, r_nml are the actual values learnt for the prediction set
        predicted_Y = np.zeros((num_nonzero,))
        for k in range(K):
            for l in range(L):
                beta_times_x = np.dot(X,betas[k,l,:])
                #the above variable is an (nnz,1) vector of responses if (k,l) is the cluster id
                weights = np.array(r_mnk[:,k]* r_nml[:,l]).reshape(num_nonzero,)
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
    
    return predicted_Y, mse, se
        
def predict_Ys_test(test_I, test_J, test_Y, Xs, pi_m, pi_n, betas, print_mean_pred_err=False, indent_level=0, minval=None, maxval=None, label="Test"):        

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
    if minval!=None:
        predicted_Y[predicted_Y<minval]=minval
    if maxval!=None:
        predicted_Y[predicted_Y>maxval]=maxval
        
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
    
    return predicted_Y, mse, se
                    
            
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



'''
def learn_model_selection_bae_crossval(Y, W, Xs, initial_K=1, initial_L=1, method='MSE', num_iter=500, precision=1e-1, reg_lambda=0.15, init_point=None):
def learn_model_selection_bae(train_Y, train_W, train_Xs, validation_Y, validation_W, validation_Xs, initial_K=1, initial_L=1, method='MSE', num_iter=500, precision=1e-1, reg_lambda=0.15, init_point=None):
def split_row_cluster(Y, W, Xs, r_mk, r_nl, betas, row_split_evaluation, **kwargs):
def split_col_cluster(Y, W, Xs, r_mk, r_nl, betas, col_split_evaluation, **kwargs):
def get_log_likelihood_Ys(Y, W, Xs, r_mk, r_nl, betas, sigmas_Y):
def create_evaluation_function(method):
def get_log_likelihood_MKs(Y, W, Xs, r_mk, r_nl, betas, sigmas_Y, alphas, mus_X, sigmas_X):
def get_log_likelihood_NLs(Y, W, Xs, r_mk, r_nl, betas, sigmas_Y, alphas, mus_X, sigmas_X):
def create_row_split_evaluation_function(method):
def create_col_split_evaluation_function(method):
'''
