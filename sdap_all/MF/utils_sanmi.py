''' utility functions, matdot'''
import numpy as np
import numexpr as ne
import scipy.linalg as sla
import itertools as it
from limits import *
import scipy.sparse as sp

learning_ktypes = set(["order", "unorder"])
ktypes = set(["order", "unorder", "expon", "inverse", "uniform", "none"])

#########################################################
# Import and test cython functions
########################################################
try: # test to see if functions work
    #Test matdot
    from matdot import matdot
    # matdot(U, V, rows, cols, Y); Y = <U[rows], V[cols]>
    matdot(np.ones((3, 2)), np.ones((4, 2)), np.array(np.arange(3), dtype=np.int32), \
           np.array(np.arange(3), dtype=np.int32), np.empty(3))
    # sparse_matvec(V, E, cols, offset, out); out_i,j = sum_k <E_i,k, U_k,j>
    from sparse_matvec import sparse_matvec
    sparse_matvec(np.ones((3, 2)), np.empty(3), np.array(np.arange(3), dtype=np.int32), \
                  np.array(np.arange(3), dtype=np.int32), np.ones((2, 2)))  
except: # default to python functioons
    from warnings import warn
    warn("unable to load cython functions. Using default python function")
    def matdot(U, V, rows, cols, Y, ev=ne.evaluate):
        # matdot(U, V, rows, cols, Y); Y = <U[rows], V[cols]>
        p = U.shape[1]
        if p==0:
            Y.fill(0.0)
        elif p==1:
            Y = ev("u*v",{"u":U[rows], "v":V[cols]})[:]
        else:
            ev("sum(u*v, axis=1)",{"u":U[rows], "v":V[cols]}, out=Y)

    def sparse_matvec(V, E, cols, offset, W):
        # sparse_matvec(V, vals, cols, offset, out); out_i,j = sum_k <E_i,k, U_k,j>
        M = len(W)
        N = len(V)
        rows = np.repeat(np.arange(M), np.diff(offset), axis=0)
        Em   = sp.coo_matrix((E, np.vstack((rows, cols))), shape=(M, N), dtype=float).tocsr()
        W[:] = np.array(Em.dot(V), copy=False)
#########################################################
# Computing predictions
########################################################
def get_calc_pred(F, rows, cols, mshape, model_type):
    ''' returns calc_pred function
    calc_pred = get_calc_pred(F, rows, cols, mshape)
    '''
    ###################################################
    # Full rank prediction
    ###################################################
    if model_type=="full_rank":
        def calc_pred(Yp, Z, Uf, Vf):
            ''' estimate prediction 
            compute: Y = K*A
            '''
            if Uf is None: Yp[:] = Z
            else: Uf(Z, Yp) # compute Y = K*A
        return calc_pred
    
    ###################################################
    # low rank prediction
    ###################################################
    else: # model_type==low_rank
        ###################################################
        # default for F = 0
        ###################################################
        if F==0:
            def calc_pred0(Yp, *args):
                Yp.fill(0.0) # set Yp = 0
            return calc_pred0
        
        ###################################################
        # for F > 0
        ###################################################
        mt = np.empty
        M, N = [mshape[s] for s in "M", "N"]
        U = mt((M, F))
        V = mt((N, F))
        def calc_pred(Yp, Z, Uf, Vf, \
            U=U, V=V, rows=rows, cols=cols, M=M, matdot=matdot):
            ''' estimate prediction 
            compute: U = K*A, V = G*B
            Yp = <U[rows], V[cols]>
            '''
            A = Z[:M]
            B = Z[M:]
            if Uf is None: U = A
            else: Uf(A, U) # compute U = K*A
            if Vf is None: V = B
            else: Vf(B, V) # compute V = G*B
            matdot(U, V, rows, cols, Yp) # compute Yp = <U[rows], V[cols]>
    
        return calc_pred

#########################################################
# Cost function
#########################################################
def get_cost_func(mshape, Ch, Dh, mt=np.empty):
    ''' returns cost_func function
    cost_func = cost_func(Y, Yp, Z, reg1, reg2, Uf, Vf, calc_pred)
    '''
    T, M, = [mshape[s] for s in "T", "M"]
    
    def cost_func(Y, Yp, Z, reg1, reg2, Uf, Vf, calc_pred, model_type, \
        iT=1.0/T, M=M, ev=ne.evaluate, dot=np.dot):
        ''' cost = ||Y-F(W)||_F^2 + lam1*||W||_F^2 + lam2*||W||_* '''
        ########################################
        # compute predicion
        ########################################
        calc_pred(Yp, Z, Uf, Vf)
        
        if model_type=="full_rank": # no trace norm
            cost = ev("sum((Y-Yp)**2, axis=None)") + reg1*(Z*Z).sum()
        else: # low rank model
            A = Z[:M]
            B = Z[M:]
            AA = dot(A.T, A)
            BB = dot(B.T, B)        
            ########################################
            # cost function
            ########################################
            cost = ev("sum((Y-Yp)**2, axis=None)") + reg1*(AA*BB).sum() + reg2*((A*A).sum() + (B*B).sum())
        return 0.5*cost*iT
    return cost_func

#########################################################
# kernel Utility functions
#########################################################
def extract_kernel(kernel, Ss, Cls=None, c0s=None, array=np.array):
    if Ss in kernel:
        S  = array(kernel[Ss], copy=False, order='C')
        Cl = None if Cls is None else kernel[Cls]
        c0 = None if c0s is None else kernel[c0s]
        P  = None if Cls is None else len(Cl)
    else:
        S  = None
        Cl = None
        c0 = None
        P = 0
    return S, Cl, c0, P

def kwrap(S, C, F):
    ''' kernel computation
    input S (eigen matrix, (M,P)) 
    F = maximum rank
    K = S*C*S.T + c0*(I-S*S.T) = S*(C-C0)*S.T + c0*I
    inv(K) = S*D*S.T + d0*(I-S*S.T) = S*(D-D0)*S.T + d0*T
    where D = inv(C), d0=1/c0
    '''
    if C is None: return None
    P = S.shape[1]
    W = np.empty((P, F))
    SC = S*(C[:-1]-C[-1])
    c0 = C[-1]
    def kdot(A, out, dot=np.dot):
        ''' KA = [X*C*X.T + c0(I-X*X.T)]A
               = [X*(C-c0)*X.T]A + c0*A '''
        dot(S.T, A, out=W)   # dot(S.T, A)
        dot(SC, W, out=out) # dot(S*C, W)
        out += c0*A
    return kdot

def full_rank_kwrap(S, O, C, D, rows, cols, offsetr, offsetc, rowsc, colsort, mshape):
    
    mt=np.empty
    mul=np.multiply
    M, N, P, Q, T = [mshape[s] for s in "M", "N", "P", "Q", "T"]
    
    ZS  = mt((N,P))
    ZO  = mt((M,Q))
    SZO = mt((P,Q))
    SC  = mt((M,P))
    OD  = mt((N,Q))

    if C is None: 
        c0 = 1.0
        Ck = None 
        nskipC = False
    else:
        c0 = C[-1]
        Ck = C[:-1] - c0
        nskipC = True
        mul(S, Ck, out=SC) # S*C; (M, P)
    
    if D is None: 
        d0 = 1.0
        Dk = None
        nskipD = False
    else: 
        d0 = D[-1]
        Dk = D[:-1] - d0
        nskipD = True
        mul(O, Dk, out=OD) # O*D; (N, Q)
    
    Y1, Y2, Y3 = [np.empty(T) for _x in xrange(3)]
    
    def full_rank_k(Z, Yp, \
        c0=c0, d0=d0, Ck=Ck, Dk=Dk, nskipC=nskipC, nskipD=nskipD,\
        Y1=Y1, Y2=Y2, Y3=Y3,Tf=float(T), S=S, O=O, ZS=ZS, ZO=ZO, SZO=SZO, \
        cols=cols, offsetr=offsetr, rowsc=rowsc, offsetc=offsetc, colsort=colsort,\
        matvec=sparse_matvec, matdot=matdot, mul=np.multiply, ev=ne.evaluate, dot=np.dot):
        '''compute Y = <K, Z> = K*Z*G + K*Z*I + I*Z*G + I*Z*I
        '''
        # 4) Identity term: c0*d0*Z
        Yp[:] = c0*d0*Z
        #mul([c0*d0], Z, out=Y4) 
        if nskipC: # 3) d0*[S*C*S'*Z*I] = [ d0*[I*Z'*S*C*S'] ]'
            matvec(SC, Z[colsort], rowsc, offsetc, ZS) # Z'*S*C; (N,M)x(M,P) -> (N,P) 
            matdot(S, ZS, rows, cols, Y3) # < S, Z'*S*C>; (M,P) * (N, P) -> T
            Yp += d0*Y3
        if nskipD: # 2) c0*[I*Z*O*D*O] = [ c0*[O*D*O'*Z'] ]'
            matvec(OD, Z, cols, offsetr, ZO) # Z*O*D; (M,N)x(N,Q) -> (M,Q) 
            matdot(ZO, O, rows, cols, Y2) # < O, Z*O*D>; (M,Q) * (N, Q) -> T
            Yp += c0*Y2
        if nskipC and nskipD: # 2) [S*C*S'*Z*O*D*O]
            dot(ZS.T, OD, out= SZO) # C*S'*Z*O*D; (P,N)x(N,Q) -> (P,Q)
            dot(O, SZO.T, out= ZS) # (C*S'*Z*O*D)*O; (N,Q)*(Q,P) -> (N,P)
            matdot(S, ZS, rows, cols, Y1) # S*(C*S'*Z*O*D*O); (M,P)*(N,P) -> T  
            Yp += Y1
        
    return full_rank_k

def get_kfuncs(S, O, Ch, Dh, rows, cols, offsetr, offsetc, rowsc, colsort, mshape):
    def kfuncs(f, model_type="low_rank",\
        S=S, O=O, Ch=Ch, Dh=Dh, kwrap=kwrap):
        ''' helpful wrapper for calling kwrap'''
        if model_type=="full_rank":
            Uf = full_rank_kwrap(S, O, Ch, Dh, rows, cols, offsetr, offsetc, rowsc, colsort, mshape)
            Vf = None
        else:
            Uf = kwrap(S, Ch, f)
            Vf = kwrap(O, Dh, f)
        return Uf, Vf
    return kfuncs

def out_kwrap(A, C, M, urows, kernel_info, dot=np.dot, zeros=np.zeros):
    ''' kwrap with expansion '''
    S = extract_kernel(*kernel_info)[0]
    if C is None:
        F = A.shape[1]
        out = zeros((M, F))
        out[urows] = A
    else:
        out = dot( S*(C[:-1]-C[-1]), dot(S[urows].T, A) ) # dot(S*C, W)
        out[urows] += C[-1]*A
    return out

def out_kwrap_full(Z, C, D, traindata, train_index_info, uindex, kernel):
    ''' function computes kernel per prediction given row'''
    
    (offsetr, offsetc, colsr, rowsc, colsort) = train_index_info[:-1]
    (rows, cols) = traindata
    (urows, ucols) = uindex
    
    mt=np.empty
    M, N, P, Q = kernel["shape"]
    S = extract_kernel(kernel, "S")[0]
    O = extract_kernel(kernel, "O")[0]
    Mt, Nt = len(urows), len(ucols)
    
    if C is None: 
        c0 = 1.0
        Ck = None
        nskipC = False
    else:
        c0 = C[-1]
        Ck = C[:-1] - c0
        nskipC = True
    if D is None: 
        d0 = 1.0
        Dk = None
        nskipD = False
    else: 
        d0 = D[-1]
        Dk = D[:-1] - d0
        nskipD = True
        
    Y1, Y2 = [np.empty(N) for _x in xrange(2)]
    Y3 = np.empty(Nt)
    rows_index  = np.zeros(N, dtype=np.int32)
    cols_index  = np.arange(N, dtype=np.int32)
    zeros_index = np.zeros(N, dtype=np.int32)
    xr1 = np.zeros(1, dtype=np.int32)
    Zc = Z[colsort]
    flat_train_index = np.ravel_multi_index((rows, cols), (M, N))
    col_intersecti = np.in1d(cols_index, ucols, assume_unique=True) # yes/no row is in training set
    if np.any(col_intersecti): 
        col_intersect = True
    else: 
        col_intersect = False
    
    # pre-computeations
    ZS  = mt((Nt,P))
    if nskipC:
        sparse_matvec(S[urows]*Ck, Zc, rowsc, offsetc, ZS) # Z'*S*C; (Nt,Mt)x(Mt,P) -> (Nt,P) 
    
    ZO  = mt((Mt,Q))
    OD  = mt((Nt,Q))
    if nskipD:
        np.multiply(O[ucols], Dk, out=OD) # O*D; (Nt, Q)
        sparse_matvec(OD, Z, colsr, offsetr, ZO) # Z*O*D; (Mt,Nt)x(Nt,Q) -> (Mt,Q) 
            
    OSZ = mt((N, P))
    #np.dot(ZS.T, OD, out= SZO) # C*S'*Z*O*D; (P,Nt)x(Nt,Q) -> (P,Q)
    #np.dot(O, SZO.T, out= OSZ) # (C*S'*Z*O*D)*O; (N,Q)*(Q,P) -> (N,P)
    if nskipC and nskipD:
        np.dot(O, np.dot(ZS.T, OD), out= OSZ) # (C*S'*Z*O*D)*O; (N,Q)*(Q,P) -> (N,P)
        
    trange = np.arange(len(Z))
     
    def full_kwrap(xr, out, \
        Z=Z, Zc=Zc, c0=c0, d0=d0, Ck=Ck, Dk=Dk, nskipC=nskipC, nskipD=nskipD,\
        Y1=Y1, Y2=Y2, Y3=Y3,S=S, O=O, ZS=ZS, ZO=ZO, OSZ=OSZ, Nt=Nt, M=M, N=N,\
        zeros_index=zeros_index, rows_index=rows_index, cols_index=cols_index, \
        col_intersect=col_intersect, urows=urows, ucols=ucols, xr1=xr1, 
        flat_train_index=flat_train_index, trange=trange, \
        offsetr=offsetr, offsetc=offsetc, rowsc=rowsc, colsr=colsr, colsort=colsort,\
        matvec=sparse_matvec, matdot=matdot, mul=np.multiply, ev=ne.evaluate, dot=np.dot,
        in1d = np.in1d, ravel_multi_index=np.ravel_multi_index, arange=np.arange):
        '''compute Y = <K, Z> = K*Z*G + K*Z*I + I*Z*G + I*Z*I
        '''
        
        rows_index.fill(xr) # current row index
        xr1[0] = xr
        out.fill(0.0)
        #######################################################
        # figure out intersection points
        #######################################################
        row_intersect = in1d(xr1, urows, assume_unique=True)[0] # yes/no row is in training set
        if row_intersect: 
            zeros_index.fill( (arange(Nt)[urows==xr])[0] )
            
        #######################################################
        # example intersect iff row intersect AND col intersect
        #######################################################
        # ignoring because train is disjoint from test
        '''
        if row_intersect and col_intersect:
            flat_test_index = ravel_multi_index((rows_index, cols_index), (M, N))
            intersect_index = in1d(flat_test_index, flat_train_index, assume_unique=True)
            # convert intersect_index to index range of Z
            # NOTE: assumes both are ordered...
            train_intersect = trange[in1d(flat_train_index, flat_test_index[intersect_index], assume_unique=True)]
            # 4) Identity term: c0*d0*I*Z, only if test intersect train
            out[intersect_index] = c0*d0*Z[train_intersect]
        '''
        
        if nskipC and col_intersect: # 3) d0*[S*C*S'*Z*I] = [ d0*[I*Z'*S*C*S'] ]'
            dot(ZS, S[xr], out=Y3) # < S, Z*S*C> (Nt,P)x(P,)-> (Nt,)
            out[ucols] += d0*Y3
        
        if nskipD and row_intersect: # 2) c0*[I*Z*O*D*O] = [ c0*[O*D*O'*Z'] ]'
            matdot(ZO, O, zeros_index, cols_index, Y2) # < O, Z*O*D> (Q,)x(N,Q)-> (N,) (Mt=1)
            out += c0*Y2
        
        if nskipC and nskipD: # 2) [S*C*S'*Z*O*D*O]
            matdot(S, OSZ, rows_index, cols_index, Y1) # S*(C*S'*Z*O*D*O); (M(t),P)*(N(t),P) -> T  
            out += Y1
        
    return full_kwrap

def setup_kernels(Xl, x0, D, P, ktype):
    ''' define Ch = [ 1/sqrt(C), 1/sqrt(c0)]
    s.t. sum(Ch*2) = M
    '''
    #######################################
    # Setup
    #######################################
    P1 = P+1
    DPf = float(D-P)
    iDPf = 1.0/DPf
    x  = np.ones(P1) 
    if ktype=="mf" or Xl is None: # empty
        return None
    #######################################
    # check types (not mf)
    #######################################
    else: #all others
        if ktype.startswith("laplac"): # inv(L) + p*I
            x[:-1] = Xl
            x[-1]  = x0
            p      = ktype[6:]
        elif ktype.startswith("iter"): # inv(L^m) + p*I
            m      = 3
            np.power(Xl, m, out=x[:-1])
            x[-1]  = ((x0*iDPf)**m)*DPf
            p      = ktype[4:]
        elif ktype.startswith("expon"): # inv(exp(L)) + p*I
            t      = 1.0
            np.exp(t*Xl, out=x[:-1])
            x[-1]  = np.exp(t*x0*iDPf)*DPf
            p      = ktype[5:]
    
    #######################################
    # invert
    #######################################
    index = x>SMALL
    x[index] = 1.0/x[index]
    #######################################
    # ensure tr(K) = M
    #######################################
    x *= D/x.sum()
    x[-1] *= iDPf
    #######################################
    # optional +p
    #######################################
    if len(p)>0:
        p = float(p)
        x += p
        
    #######################################
    # sqrt
    #######################################
    return np.sqrt(x)

#########################################################
# Data setup Utility functions
#########################################################
def get_offset(rows, cols, M, N, sorted=True):
    T = len(rows)
    indexes = np.arange(T, dtype=np.int32)
    
    if sorted:
        sdiff = np.diff(rows)
        k = rows[0]
    else:
        rowsc, colsc, rowsort = sp.find(\
        sp.coo_matrix((indexes+1, np.vstack((rows, cols))), shape=(M, N), dtype=int).tocsr())
        rowsort -=1
        sdiff = np.diff( rowsc )
        k = rowsc[0]
    
    offset = np.zeros(M+1, dtype=np.int32)
    bar = sdiff>0
    start = k+1
    for step, num in it.izip(sdiff[bar], indexes[bar]):
        end = start + step # step is negative
        offset[start:end] = num + 1
        start = end
    offset[end:] = T
    
    if sorted:
        return offset
    else:
        return offset, rowsort

def setup_data(rows, cols, vals, M, N, \
    get_offset=get_offset, np=np):
    ''' extracts utility info from data
    NOTE: assumes input data is sorted by row!!! (from sp.find)
    
    returns: rows, cols, vals, offsetr, offsetc, rowsc, colsort, score_starts
    offsetr-> index where rows changes for each col
    offsetc, (rowsc, colsort,)-> index where col changes for each row
    score_starts -> index where vals change for each row
    NOTE: changes data
    '''
    ###############################################
    # First sort vals for each row
    ###############################################
    T = len(vals)
    score_starts = [[0]] # list of score start index
    resort_index=[]
    index = np.arange(T)
    for mx in xrange(M):
        ind_mx = index[rows==mx]
        if not len(ind_mx): #skip empty rows (should never happen!])
            print "EMPTY ROW, %d"%(mx, ) 
            continue
        start, end = ind_mx[0], ind_mx[-1]
        srt_mx = vals[ind_mx].argsort() + start # arg-sort values for row mx
        resort_index.append(srt_mx) # new sorted order for row mx
        score_starts.append(start)
        temp_start_idx = np.arange(start, end)[np.diff( vals[srt_mx] ) > 0] + 1
        if len(temp_start_idx): # should always happen!!!
            score_starts.append(temp_start_idx)
    score_starts.append([T])
    score_starts = np.unique(np.hstack(score_starts))
    resort_index = np.hstack(resort_index)
    
    ###############################################
    # resort data using vals for row
    ###############################################
    rows = rows[resort_index]
    cols = cols[resort_index]
    vals = vals[resort_index]
    
    ###############################################
    # compute offsetr, offsetc
    ###############################################
    offsetr = get_offset(rows, cols, M, N) # already sorted, so doesn't change idx
    colsr = cols.copy()
    offsetc, colsort = get_offset(cols, rows, N, M, sorted=False) # not sorted
    rowsc = rows[colsort]
    
    return (rows, cols, vals), (offsetr, offsetc, colsr, rowsc, colsort, score_starts)

#########################################################
# Function for eigenvalues
#########################################################
def get_grad_matvec(mE, Z, F, reg1, reg2, \
    rows, cols, offsetr, offsetc, rowsc, colsort, mshape, kfuncs):
    ''' matvec used for EIG's '''
    
    mt = np.empty
    M, N = [mshape[s] for s in "M", "N"]
    fev = 1
    Gx  = mt((N, fev))
    Kx  = mt((M, fev))
    EG  = mt((M, fev))
    EK  = mt((N, fev))
    Vx  = mt((F, fev))
    Ux  = mt((F, fev))
    UVx = mt((M, fev))
    VUx = mt((N, fev))
    mEc = mE[colsort] # minus (Error)
    Uf, Vf = kfuncs(fev, "low_rank") # Uf, Vf with vector 
    skipC = Uf is None
    skipD = Vf is None
    addW = F>0 and reg1>0.0
    
    def grad_matvec(x, \
        skipC=skipC, skipD=skipD, addW=addW,\
        reg1=reg1, reg2=reg2, mE=mE, mEc=mEc, Uf=Uf, Vf=Vf, U=Z[:M], V=Z[M:],\
        Gx=Gx, Kx=Kx, EG=EG, EK=EK, Vx=Vx, Ux=Ux, UVx=UVx, VUx=VUx,\
        cols=cols, offsetr=offsetr, rowsc=rowsc, offsetc=offsetc, M=M, \
        zeros=np.zeros, sparse_matvec=sparse_matvec, array=np.array, dot=np.dot, mt=np.empty):
        ''' defines matvec(v) = A*x
            A = grad_X( F(X) ); where X = ZZ'        
        '''
        # views for easy addr
        x = array(x, ndmin=2, copy=False).T
        #x = array(x, ndmin=2, copy=False)
        out = mt(x.shape) # initialize with nu*x
        outu = out[:M]
        outv = out[M:]
        xu = x[:M]
        xv = x[M:]
        
        #############
        # TOP: Ax = K*E*G*xv + reg1*U*V.T*xv
        #############
        # compute K*E*G*x
        if skipD: Gx=xv
        else: Vf(xv, out=Gx) # G*x
        sparse_matvec(Gx, mE, cols, offsetr, EG) # E*G*x
        if skipC: outu[:]=EG 
        else: Uf(EG, out=outu) # K*E*G*x
        #############
        # hilbert norm, add reg1*V*U.T*x
        #############
        if addW:
            dot(V.T, xv, out=Vx) # V.T*x
            dot(U, Vx, out=UVx)  # U*V.T*x
            outu -= reg1*UVx
        
        #############
        # BOTTOM: Bx = G*E*K*xu + reg1*V*U.T*xu
        #############
        # compute G*E*K*x
        if skipC: Kx=xu
        else: Uf(xu, out=Kx) # K*x
        sparse_matvec(Kx, mEc, rowsc, offsetc, EK) # E*K*x
        if skipD: outv[:]=EK 
        else: Vf(EK, out=outv) # G*E*K*x
        #############
        # hilbert norm, add sigma2*U*V.T*x
        #############
        if addW:
            dot(U.T, xu, out=Ux) # V.T*x
            dot(V, Ux, out=VUx) # V*U.T*x
            outv -= reg1*VUx
        
        #############
        # Trace norm, add 0.5*reg2*x 
        # (value doesnt really matter, ensures non-singularity)
        #############
        #out -= 0.5*reg2*x
        #out += M*N*x # temp hack to keep the largest eig positive
        return out

    return sp.linalg.LinearOperator(shape=(M+N, M+N), matvec=grad_matvec, dtype=float) # , rmatvec=grad_matvec

#########################################################
# Regularizer Utility functions
#########################################################
#def compute_nu_max(trace_func, Ch, Dh, Efull_kwrap=None):
#    ''' approx max_eig[grad_(X=0)(F(X))]
#    using tr[grad_(X=0)(F(X))] >= max_eig[grad_(X=0)(F(X))]
#    (accurate for rank one)
#    '''
#    if Ch is None:
#        Ch = 1.0
#        nskipC = False
#    else:
#        nskipC = True
#    if Dh is None:
#        Dh = 1.0
#        nskipD = False
#    else:
#        nskipD = True
#    return trace_func(Ch, Dh, nskipC, nskipD, E=E)
    
def compute_nu_max(E, rows, cols, offsetr, offsetc, rowsc, colsort, \
    mshape, kfuncs):
    M, N, T = [mshape[s] for s in "M", "N", "T"]
    F, reg1, reg2 = 0, 0.0, 0.0
    Z = np.zeros((M+N, F))
    f_params = (rows, cols, offsetr, offsetc, rowsc, colsort, mshape, kfuncs)
    a  = sp.linalg.eigsh(A=get_grad_matvec(E, Z, F, reg1, reg2, *f_params), k=1, which="LM")[0]
    return 2.0*np.maximum(np.abs(a), 1.0)[0]

def compute_sigma_0(*args):
    return compute_nu_max(*args)