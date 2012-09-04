
'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

from mf_pmf import train_pmf_test
from bae_predict import bae_linear_predict
import scipy.sparse as sp
import numpy as np

def run_mf(K, L, X1, X2, train_I, train_J, train_Y, reg_lambda, sub_model):
    M = X1.shape[0]
    N = X2.shape[0]
    if sub_model == 'PMF':
        params, obj = train_pmf_test(M, N, train_I, train_J, train_Y)

    train_op = {'params':params,
                'obj':obj
                }
    return train_op

def hotStartTrainRMSE(model_name, K, L, X1, X2, train_I, train_J, train_Y, train_op):
    M = X1.shape[0]
    N = X2.shape[0]
    Xs = train_op['params']['Xs']
    betas = train_op['params']['betas']
    r = train_op['params']['r']
    if model_name.upper().split('_')[1] == 'LINEAR':
        predictions = bae_linear_predict(train_I, train_J, M, N, Xs, betas, r, K, L)
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
    if model_name.upper().split('_')[1] == 'LINEAR':
        predictions = bae_linear_predict(val_I, val_J, M, N, Xs, betas, r, K, L)
    Z = sp.csr_matrix((val_Y, (val_I,val_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(val_I,val_J)]).ravel()
    hotStartTrainRMSE = np.sqrt(np.mean((predictions-nonzero_Z)**2)) 
    return hotStartTrainRMSE