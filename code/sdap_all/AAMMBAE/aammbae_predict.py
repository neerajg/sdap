
'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

import numpy as np

def aammbae_predict(I, J, M, N, Xs, betas, r, K, L):
    # y = weighted*(Beta'x)    

    Xbias = np.ones((len(I),1)) # |Yobs| x 1
    Xusers = Xs[0][I,:] # |Yobs| x D1
    Xitems = Xs[1][J,:] # |Yobs| x D2
    X = np.hstack((Xbias, Xusers, Xitems)) # |Yobs| x (1 + D1 + D2) 
       
    beta = betas[0]
    
    r1_obs = r[0][I,:] # |Yobs| x K
    r2_obs = r[1][J,:] # |Yobs| x L 
        
    predictions = np.zeros((len(I),))
    for k in range(K):
        for l in range(L):
            beta_times_x = np.dot(X,beta[k,l,:])
            weights = np.multiply(r1_obs[:,k],r2_obs[:,l])
            predictions += weights*beta_times_x
    predictions[predictions<0.5] = 0.5
    predictions[predictions>5.0] = 5.0
    return predictions