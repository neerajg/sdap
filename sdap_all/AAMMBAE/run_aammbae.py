
'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

from aammbae_linear import train_aammbae_linear
from aammbae_predict import aammbae_predict
import scipy.sparse as sp
import numpy as np

def run_aammbae(K, L, X1, X2, train_I, train_J, train_Y, learner, reg_lambda, num_iter,delta_convg, reg_alpha1, reg_alpha2):
    
    if learner.upper() == 'LINEAR':
        params, obj = train_aammbae_linear(K, L, X1, X2, train_I, train_J, train_Y, reg_lambda, num_iter, delta_convg, reg_alpha1,reg_alpha2)

    train_op = {'params':params,
                'obj':obj,
                'train_I':train_I,
                'train_J':train_J
                }
    return train_op

def hotStartTrainRMSE(model_name, K, L, X1, X2, train_I, train_J, train_Y, train_op):
    M = X1.shape[0]
    N = X2.shape[0]
    Xs = train_op['params']['Xs']
    betas = train_op['params']['betas']
    r = train_op['params']['r']
    predictions = aammbae_predict(train_I, train_J, M, N, Xs, betas, r, K, L)
    Z = sp.csr_matrix((train_Y, (train_I,train_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(train_I,train_J)]).ravel()
    hotStartTrainRMSE = np.sqrt(np.mean((predictions-nonzero_Z)**2)) 
    return hotStartTrainRMSE

def hotStartValRMSE(model_name, K, L, X1, X2, val_I, val_J, val_Y, train_op):
    M = X1.shape[0]
    N = X2.shape[0]
    Xs = train_op['params']['Xs']
    betas = train_op['params']['betas']
    r = train_op['params']['r']
    predictions = aammbae_predict(val_I, val_J, M, N, Xs, betas, r, K, L)
    Z = sp.csr_matrix((val_Y, (val_I,val_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(val_I,val_J)]).ravel()
    hotStartTrainRMSE = np.sqrt(np.mean((predictions-nonzero_Z)**2)) 
    return hotStartTrainRMSE