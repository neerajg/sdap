'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

# TO DO : check all TO DOs here and in misc
# TO DO : Try damped updates
# TO DO : here and in misc also change update of r condition to check for very low value to make it work for all K and L
import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt

D = 10
thresh = 1e-6
lambda_u = 1e-4
lambda_v = 1e-4
num_iter = 100

def train_pmf_test(M, N, train_I, train_J, train_Y):
    eta = 1
    U,V = initialize(M,N,D)
    t = 0
    convergence = False
    Y_matrix = sp.csr_matrix((train_Y,(train_I,train_J)), shape = (M,N))
    train_Y = Y_matrix.data
    W_matrix = sp.csr_matrix((np.ones(len(train_I,)),(train_I,train_J)), shape = (M,N))
    tot_error_old = 1e99
    while convergence == False and t<num_iter:
        print tot_error_old
        # Update Rows
        U = update_rowFactors(train_Y,U,V,train_I,train_J,M,N,eta)
        # Update Cols
        V = update_colFactors(train_Y,U,V,train_I,train_J,M,N,eta)
        # Get Error
        error_obs = getErrorObs(train_Y, train_I, train_J, U, V)        
        # Check for convergence
        tot_error_new = np.sum(np.sum(error_obs**2))
        if tot_error_old - tot_error_new < thresh or t>num_iter:
            convergence = True
        else:
            tot_error_old = tot_error_new
            t +=1
    
    obj = []
    params = {'U':U,
              'V':V}
    return params, obj

def update_rowFactors(train_Y,U,V,train_I,train_J,M,N,eta_init):
    for m in range(M):
        error_ud = np.zeros(N,)
        temp_u = U[m,:]  
        
        for d in range(D):
            ud_old = temp_u[d]
            ud_new = temp_u[d]            
            converged = False
            eta = eta_init
            sqrd_err_old = 1e99
            while not converged:
                eta_change = False
                temp_u[d] = ud_new
                error_ud[train_J[train_I == m]] = train_Y[train_I == m] - np.dot(temp_u,V.T)[train_J[train_I == m]]
                sqrd_err_new = np.sum(error_ud**2)
                if sqrd_err_old < sqrd_err_new:
                    temp_u[d] = ud_old
                    eta *= 0.5
                    eta_change = True
                else:
                    ud_old = ud_new
                if abs(sqrd_err_old - sqrd_err_new) < thresh or eta < thresh:
                    converged = True
                # update
                update_dirn = lambda_u*ud_old - np.sum(np.multiply(error_ud,V[:,d]))
                update_dirn = update_dirn/update_dirn
                ud_new = ud_old - eta*(update_dirn)
                if not eta_change:
                    sqrd_err_old = sqrd_err_new
        
        # Update the Row Factor
        U[m,:] = temp_u
    return U

def update_colFactors(train_Y,U,V,train_I,train_J,M,N,eta_init):
    for n in range(N):
        error_vd = np.zeros(M,)
        temp_v = V[n,:]  
        
        for d in range(D):
            vd_old = temp_v[d]
            vd_new = temp_v[d]            
            converged = False
            eta = eta_init
            sqrd_err_old = 1e99
            while not converged:
                eta_change = False
                temp_v[d] = vd_new
                error_vd[train_I[train_J == n]] = train_Y[train_J == n] - np.dot(temp_v,U.T)[train_I[train_J == n]]
                sqrd_err_new = np.sum(error_vd**2)
                if sqrd_err_old < sqrd_err_new:
                    temp_v[d] = vd_old
                    eta *= 0.5
                    eta_change = True
                else:
                    vd_old = vd_new
                if abs(sqrd_err_old - sqrd_err_new) < thresh or eta < thresh:
                    converged = True
                # update
                update_dirn = lambda_v*vd_old - np.sum(np.multiply(error_vd,U[:,d]))
                update_dirn = update_dirn/update_dirn
                vd_new = vd_old - eta*(update_dirn)
                if not eta_change:
                    sqrd_err_old = sqrd_err_new
        
        # Update the Row Factor
        V[n,:] = temp_v
    return V
    
def getErrorObs(train_Y, train_I, train_J, U, V):
    error_obs = train_Y - np.dot(U,V.T)[train_I,train_J]
    return error_obs

def initialize(M,N,D):
    U = np.random.randn(M,D)
    V = np.random.randn(N,D)
    return U,V

def kpmf(R, U, V, D, w_nm, Su, Sv, num_obs, threshold, iters, eta, beta, sigma_sqrd):
    old_e = 1e99
    for step in xrange(iters):
        e_nm = R.toarray() - np.dot(U,V.T)
        t_nm = w_nm.toarray()*e_nm # N x M
        tempu = np.zeros((w_nm.shape[0],D)) # N x D
        tempv = np.zeros((w_nm.shape[1],D)) # M x D
        # TO DO : Optimize to remove this loop and vectorize this too
        # U - N x D
        # V - M x D
        for d in range(D):
            v_md = np.tile(V[:,d],w_nm.shape[0]).reshape([w_nm.shape[0],w_nm.shape[1]]) # N x M
            u_nd = np.tile(U[:,d],w_nm.shape[1]).reshape([w_nm.shape[1],w_nm.shape[0]]).T  # N x M

            tempu_nd = np.sum(t_nm*v_md,1).reshape(w_nm.shape[0],1) # N x 1
            tempv_md = np.sum(t_nm*u_nd,0).reshape(w_nm.shape[1],1) # M x 1

            tempu[:,d] = tempu_nd.reshape(w_nm.shape[0],)
            tempv[:,d] = tempv_md.reshape(w_nm.shape[1],)

        grad_U_nd = np.dot(Su,U)  - (1/sigma_sqrd)*tempu # N x D
        grad_V_md = np.dot(Sv,V)  - (1/sigma_sqrd)*tempv # M x D

        U = U - eta * grad_U_nd
        V = V - eta * grad_V_md
        
        # Calculate change in Objective function
        e = w_nm.toarray()*(R.toarray() - np.dot(U,V.T))
        e = np.sqrt(np.sum(e*e)/num_obs)
        diff = old_e - e
        
        #print diff
        
        old_e = e
        if diff < threshold:
            break
    return U, V