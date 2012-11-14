'''
inference using low rank SDP approach
min f(X) + lam * tr(X) s.t. X is psd

X = [ M  W ] = Z*Z'
    [ W' N ]
sanmi.k@gmail.com
'''

import numpy as np
import numexpr as ne
import numpy.linalg as la
import scipy.sparse as sp
import scipy.optimize as so
from utils import matdot, sparse_matvec, get_grad_matvec
from limits import *
import itertools as it

def get_ab_f_and_g(F, Z, z, \
    rows, cols, offsetr, offsetc, rowsc, colsort, mshape, kfuncs,\
    matdot=matdot, dot=np.dot, tr=np.trace):
    ''' inference function for a, b s.t. X* = a*X+b*x  
    '''
    
    mt  = np.empty
    M, N, T = [mshape[s] for s in "M", "N", "T"]
    Yp  = mt((T, ))
    yp  = mt((T, ))
    E   = mt((T, ))
    
    Uf, Vf   = kfuncs(F)
    Uf1, Vf1 = kfuncs(1)
    skipC = Uf is None
    skipD = Vf is None
    
    #######################################
    # re-set sizes
    #######################################
    A = Z[:M]
    B = Z[M:]
    a = z[:M]
    b = z[M:]
    if skipC: 
        U = A
        u = a
    else: 
        U = mt((M, F))
        u = mt((M, 1))
        Uf(A, U)  # K^(1/2)*A
        Uf1(a, u) # K^(1/2)*A 
    
    if skipD: 
        V = B
        v = b
    else:
        V = mt((N, F))
        v = mt((N, 1))
        Vf(B, V)
        Vf1(b, v)
    
    # compute predictions
    matdot(U, V, rows, cols, Yp)
    matdot(u, v, rows, cols, yp)
    
    # compute norms
    AA = dot(A.T, A) # A'A = A'*A
    BB = dot(B.T, B) # B'B = B'*B
    aa = dot(a.T, a).ravel()[0] # A'A = A'*A
    bb = dot(b.T, b).ravel()[0] # B'B = B'*B
    Aa = dot(A.T, a).ravel() # A'a = A'*a
    Bb = dot(B.T, b).ravel() # B'b = B'*b
    
    # hnorms
    hnorm1 = (AA*BB).sum() # hilbert norm X
    hnorm2 = aa*bb         # hilbert norm x
    hnorm3 = dot(Aa.T, Bb) # hilbert norm <X, x>
    
    # tnorms
    tnorm1 = 0.5*(tr(AA) + tr(BB)) # trace norm X
    tnorm2 = 0.5*(aa + bb)         # trace norm x
    
    def ab_f_and_g(ab, Y, reg1, reg2,\
        E=E, Yp=Yp, yp=yp, iT=1.0/T, \
        hnorm1=hnorm1, hnorm2=hnorm2, hnorm3=hnorm3, tnorm1=tnorm1, tnorm2=tnorm2,\
        dot=np.dot, ev=ne.evaluate, mt=np.empty):
        '''function and gradient for H(A,B) = G(AB') = G(L)
        cost = KL(q||p) + nu*trace(Z)
             =  0.5*|| R(t)-F(W(t)) ||_F^2 + 0.5*reg1*||W||_H^2 + reg2*trace(W)
            
        W = [A;B] = (M+N)*R
        grad = [ dG/dW B; (dG/dW)' A]
        
        Norms:
        Hilbert norm: || AB' ||_H^2= trace[ A'AB'B ] = sum [(A'A).(B'B) ]
        Trace norm:   2*|| AB' ||_* = || A ||_F^2 + || B ||_F^2 = trace[ A'A ] + trace[ B'B ]
        '''
        alpha, beta = ab
        
        ev("(alpha*Yp) + (beta*yp) - Y", out=E)
        err   = ev("sum( E*E, axis=None)") # sum square prediction error
        hnorm = alpha*alpha*hnorm1 + beta*beta*hnorm2 + 2.0*alpha*beta*hnorm3
        tnorm = alpha*tnorm1 + beta*tnorm2
        cost  = 0.5*(err + reg1*hnorm) + reg2*tnorm
        
        #####################################
        # grad of cost = < E, X>, <E, x>
        # grad of Hilbert norm: <aX +bx, X>, <aX +bx, x>
        # grad of Trace norm: a||X||_*, ||x||_*
        #####################################
        grad = mt(2)
        grad[0] = ev("sum( E*Yp, axis=None)") + reg1*(alpha*hnorm1 + beta*hnorm3) + reg2*tnorm1
        grad[1] = ev("sum( E*yp, axis=None)") + reg1*(alpha*hnorm3 + beta*hnorm2) + reg2*tnorm2
        
        return cost, grad
    
    return ab_f_and_g

def get_inference_f_and_g(F, \
    rows, cols, offsetr, offsetc, rowsc, colsort, mshape, kfuncs):
    ''' get inference func 
    '''
    
    mt=np.empty
    M, N, P, Q, T = [mshape[s] for s in "M", "N", "P", "Q", "T"]
    EV  = mt((M, F))
    EU  = mt((N, F))
    
    gEU = mt((M, F))
    gEV = mt((N, F))
    gHU = mt((M, F))
    gHV = mt((N, F))
    
    U   = mt((M, F))
    V   = mt((N, F))
    AA  = mt((F, F))
    BB  = mt((F, F))
    Yp  = mt((T, ))
    E   = mt((T, ))
    
    Uf, Vf = kfuncs(F)
    skipC = Uf is None
    skipD = Vf is None
    
    def inference_f_and_g(w, Y, reg1, reg2,\
        Uf=Uf, Vf=Vf, Yp=Yp, E=E, skipC=skipC, skipD=skipD, \
        U=U, V=V, AA=AA, BB=BB, EU=EU, EV=EV, gEU=gEU, gEV=gEV, gHU=gHU, gHV=gHV,\
        rows=rows, cols=cols, iT=1.0/T, M=M, N=N, F=F,\
        offsetr=offsetr, offsetc=offsetc, rowsc=rowsc,\
        dot=np.dot, norm=la.norm, ev=ne.evaluate, matdot=matdot, tr=np.trace,\
        mt=np.empty, hstack=np.hstack, matvec=sparse_matvec):
        '''function and gradient for H(A,B) = G(AB') = G(L)
        cost = KL(q||p) + nu*trace(Z)
             =  0.5*|| R(t)-F(W(t)) ||_F^2 + 0.5*reg1*||W||_H^2 + reg2*trace(W)
            
        W = [A;B] = (M+N)*R
        grad = [ dG/dW B; (dG/dW)' A]
        
        Norms:
        Hilbert norm: || AB' ||_H^2= trace[ A'AB'B ] = sum [(A'A).(B'B) ]
        Trace norm:   2*|| AB' ||_* = || A ||_F^2 + || B ||_F^2 = trace[ A'A ] + trace[ B'B ]
        '''
        #######################################
        # re-set sizes
        #######################################
        W = w.reshape((M+N, F))
        A = W[:M]
        B = W[M:]
        
        # the following avoids unnecessary memory copy for MF.
        if skipC: U=A
        else: Uf(A, U) # K^(1/2)*A 
        if skipD: V=B
        else: Vf(B, V)
        dot(A.T, A, out=AA) # A'A = A'*A
        dot(B.T, B, out=BB) # B'B = B'*B
        
        matdot(U, V, rows, cols, Yp)
        ev("Yp-Y", {"Yp":Yp, "Y":Y}, out=E)
        err   = ev("sum( E*E, axis=None)") # sum square prediction error
        hnorm = (AA*BB).sum()#, 0.0) # hilbert norm. May be negative when small (why?)
        tnorm = tr(AA) + tr(BB) # tnorm = 2*trace norm
        cost  = 0.5*(err + reg1*hnorm + reg2*tnorm)
        
        # grad of cost = KEV, GE'U
        matvec(V, E, cols, offsetr, EV)
        matvec(U, E[colsort], rowsc, offsetc, EU)
        if skipC: gEU=EV
        else: Uf(EV, gEU) # K^(1/2)*A 
        if skipD: gEV=EU
        else: Vf(EU, gEV)
        
        # gradient of hilbert norm
        dot(A, BB, out=gHU)
        dot(B, AA, out=gHV)
        
        # gradient of trace norm
        gTU = A
        gTV = B
        
        # full gradient
        grad      = mt(W.shape)
        ev("gEU + (reg1*gHU) + (reg2*gTU)", out=grad[:M])
        ev("gEV + (reg1*gHV) + (reg2*gTV)", out=grad[M:])
        
        return cost, grad.ravel()
    
    return inference_f_and_g

def get_inference_f_and_g_full(rows, cols, offsetr, offsetc, rowsc, colsort, mshape, kfuncs):
    ''' get inference func 
    '''
    mt=np.empty
    T  = mshape["T"]
    E  = mt((T, ))
    Yp = mt((T, ))
    gE = mt((T, ))
    Kf = kfuncs(None, "full_rank")[0]
    def inference_f_and_g_full(Z, Y, reg1, reg2,\
        iT=1.0/T, Kf=Kf, Yp=Yp, E=E, gE=gE, dot=np.dot, ev=ne.evaluate):
        '''function and gradient full rank model
        cost = ( R(t)-F(W(t)) ) + reg1*||W||_H^2
        '''
        #######################################
        # re-set sizes
        #######################################
        
        Kf(Z, Yp) # U = K^(1/2)*A 
        ev("Yp-Y", {"Yp":Yp, "Y":Y}, out=E)
        err   = ev("sum( E*E, axis=None)") # sum square prediction error
        hnorm = dot(Z, Z) # hilbert norm. May be negative when small (why?)
        cost  = 0.5*(err + reg1*hnorm)
        
        # gradient of loss
        Kf(E, gE) # U = K^(1/2)*A 
        # gradient of hilbert norm
        gH = Z
        # full gradient
        grad = ev("gE + reg1*gH")
        
        return cost, grad
    
    return inference_f_and_g_full


def get_inference(rows, cols, offsetr, offsetc, rowsc, colsort, \
    mshape, maxfun, maxsvd, cost_func, get_calc_pred, kfuncs, \
    disp=0, verbose=False):
    ''' inference step, estimate r, m '''
    
    M, N, T, F = [mshape[s] for s in "M", "N", "T", "F"]
    inv_machine_precision = 1.0/np.finfo(np.float).eps
    f_params = (rows, cols, offsetr, offsetc, rowsc, colsort, mshape, kfuncs)
    E = np.empty(T)
    
    #########################################################
    # Low rank inference (with trace norm, upper bounded rank)
    #########################################################    
    def low_rank_inference(Y, Yp, Z, reg1, reg2, tol, rankup=True, \
        model_type="low_rank", E=E, f_params=f_params, mshape=mshape, maxfun=maxfun, maxsvd=maxsvd,\
        disp=disp, M=M, N=N, F=F, \
        get_grad_matvec=get_grad_matvec, get_inference_f_and_g=get_inference_f_and_g,\
        get_ab_f_and_g=get_ab_f_and_g, cost_func=cost_func, get_calc_pred=get_calc_pred, kfuncs=kfuncs,\
        rows=rows, cols=cols, rowsc=rowsc, offsetr=offsetr, offsetc=offsetc,\
        bfgs=so.fmin_l_bfgs_b, hstack=np.hstack, iprec=inv_machine_precision,\
        eigsh=sp.linalg.eigsh, ev=ne.evaluate, array=np.array, verbose=verbose, \
        randn=np.random.randn, SMALL=SMALL, real=np.real, maximum=np.maximum,\
        noconverge=sp.linalg.ArpackNoConvergence, bounds=[(0.0, None)]*2, sqrt=np.sqrt):
        '''
        inference using low rank SDP approach
        min f(X) + lam * tr(X) s.t. X is psd
        
        X = [ M  W ] = Z*Z' = [AA' AB']
            [ W' N ]          [BA' BB']
        Z' = [A' B']
        '''
        
        f = Z.shape[1] # current rank
        Uf, Vf = kfuncs(f, model_type)
        cp = get_calc_pred(f, rows, cols, mshape, model_type) # coompute prediction
        f_init = cost_func(Y, Yp, Z, reg1, reg2, Uf, Vf, cp, model_type) # initial function
        Uf, Vf, cp = Uf, Vf, cp
        #########################################################
        # Assuming warm start, attempt BFGS
        #########################################################
        if f > 0: # pre-optimization if f>0
            if verbose:
                print "\t\tInitial inference cost %e"%(f_init)
            f_and_g = get_inference_f_and_g(f, *f_params)
            Z = (bfgs(func=f_and_g, x0=Z.ravel(), args=(Y, reg1, reg2), \
                          disp=disp, maxfun=maxfun)[0]).reshape((M+N, f)) #, maxfun=maxfun, factr=factr)
            f_old = cost_func(Y, Yp, Z, reg1, reg2, Uf, Vf, cp, model_type) # pre-rank update cost
        else:
            f_old = f_init
        
        if verbose:
            print "\t\tPre-f inference cost %e"%(f_old)
            
        #########################################################
        # Try increasing rank to reduce error
        #########################################################
        while f < F and rankup: # incremental rank till F
            ################################
            # Step 1: SVD[ -grad_X (F(X)) ]
            ################################
            ev("Y-Yp", {"Yp":Yp, "Y":Y}, out=E) # -Error
            try: # Try catch and use wherever it stops
                a, z  = eigsh(A=get_grad_matvec(E, Z, f, reg1, reg2, *f_params), \
                              k=1, which="LM", maxiter=maxsvd) # eigenvector, maxiter=?
            except noconverge:
                if verbose:
                    print "\t\t\teig noconverge, use random"
                z = randn(Z.shape[0], 1)
            
            ################################
            # Step 2: 1-D opt min_a,b f(a*X+b*X0) s.t a, b>0 
            ################################
            z = real(array(z, order="C", copy=False))
            if f > 0: # 1-d update unecessary if f=0
                f_and_g = get_ab_f_and_g(f, Z, z, *f_params)
                ab = bfgs(func=f_and_g, x0=np.ones(2), args=(Y, reg1, reg2), \
                              disp=disp, bounds=bounds, maxfun=maxfun)[0] #, maxfun=maxfun, factr=factr)
                a, b = maximum(ab, 1E-4)
                #print a, b
                Zi = hstack(( sqrt(a)*Z, sqrt(b)*z )) # EXPENSIVE!!! as r->large
                if verbose: 
                    print "\t\t\tEig weights: %e, %e"%(a, b)
            else:
                Zi = z
                
            ################################
            # Step 3: BFGS update Zi = min_Z f(ZZ') 
            ################################
            fi = Zi.shape[1]
            Ufi, Vfi = kfuncs(fi, model_type)
            f_and_g = get_inference_f_and_g(fi, *f_params)
            Zi = (bfgs(func=f_and_g, x0=Zi.ravel(), args=(Y, reg1, reg2), \
                          disp=disp, maxfun=maxfun)[0]).reshape((M+N, fi)) #, maxfun=maxfun, factr=factr)
            
            ################################
            # if cost is not reduced, exit with old Z 
            ################################
            cpi = get_calc_pred(fi, rows, cols, mshape, model_type)
            f_new = cost_func(Y, Yp, Zi, reg1, reg2, Ufi, Vfi, cpi, model_type)
            loop_eps = abs(f_old-f_new)/max(f_new, f_old, 1E-20)
            #print "loop_eps is", loop_eps, f_old, f_new
            if f_new > f_old or loop_eps <= tol: # break if function value increases
                break 
            
            if verbose:
                print "\t\t\tf=%d, f_new=%e"%(fi, f_new)
        
            ################################
            # else re-set Z, f
            ################################
            f_old = f_new
            Z, Uf, Vf, f, cp = Zi, Ufi, Vfi, fi, cpi
            
        if verbose:
            print "\t\tFinal-f inference cost %e"%(f_old)
        
        return Z, Uf, Vf, cp
    
    #########################################################
    # Full rank inference (hilbert norm only)
    #########################################################   
    UFF = kfuncs(None, "full_rank")[0]
    calc_pred_full = get_calc_pred(None, rows, cols, mshape, "full_rank") # coompute prediction
    f_and_g_full = get_inference_f_and_g_full(*f_params)
    def full_rank_inference(Y, Yp, Z, reg1, reg2, tol, rankup=None, \
        Uf=UFF, Vf=None, cost_func=cost_func, calc_pred=calc_pred_full, \
        model_type="full_rank", E=E, mshape=mshape, maxfun=maxfun, disp=disp,\
        f_and_g=f_and_g_full, bfgs=so.fmin_l_bfgs_b,\
        iprec=inv_machine_precision,verbose=verbose, \
        randn=np.random.randn, SMALL=SMALL, real=np.real, maximum=np.maximum):
        '''
        inference using full rank approach
        min ||Y-f(W)|| + reg1*||W||_2^2 
        '''
        f_init = cost_func(Y, Yp, Z, reg1, 0.0, Uf, Vf, calc_pred, model_type) # initial function
        if verbose:
            print "\t\tInitial inference cost %e"%(f_init)
        #########################################################
        # optimization
        #########################################################
        Z = bfgs(func=f_and_g, x0=Z, args=(Y, reg1, reg2), disp=disp, maxfun=maxfun)[0] #, maxfun=maxfun, factr=factr)
        f_new = cost_func(Y, Yp, Z, reg1, reg2, Uf, Vf, calc_pred, model_type) # pre-rank update cost
        if verbose:
            print "\t\tFinal-f inference cost %e"%(f_new)
        return Z, Uf, Vf, calc_pred
    
    return low_rank_inference, full_rank_inference
