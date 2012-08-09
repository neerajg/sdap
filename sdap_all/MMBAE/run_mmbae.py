
'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

from mmbae_linear import train_mmbae_linear
from mmbae_predict import mmbae_linear_predict
import scipy.sparse as sp
import numpy as np

def run_mmbae(K, L, X1, X2, train_I, train_J, train_Y, learner, reg_lambda, num_iter,delta_convg, reg_alpha1, reg_alpha2):
    
    if learner.upper() == 'LINEAR':
        params, obj = train_mmbae_linear(K, L, X1, X2, train_I, train_J, train_Y, reg_lambda, num_iter, delta_convg, reg_alpha1,reg_alpha2)

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
    if model_name.upper().split('_')[1] == 'LINEAR':
        predictions = mmbae_linear_predict(train_I, train_J, M, N, Xs, betas, r, K, L)
    Z = sp.csr_matrix((train_Y, (train_I,train_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(train_I,train_J)]).ravel()
    hotStartTrainRMSE = np.sqrt(np.mean((predictions-nonzero_Z)**2)) 
    return hotStartTrainRMSE

def hotStartValRMSE(model_name, K, L, X1, X2, val_I, val_J, val_Y, train_op):
    M = X1.shape[0]
    N = X2.shape[0]
    Xs = train_op['params']['Xs']
    betas = train_op['params']['betas']
    r = [np.zeros((M,K)),np.zeros((N,L))]
    ones = np.ones((len(train_op['train_I']),))
    mu = sp.csr_matrix((ones, (train_op['train_I'],train_op['train_J'])), shape=(M,N)).sum(1)
    mv = sp.csr_matrix((ones, (train_op['train_I'],train_op['train_J'])), shape=(M,N)).sum(0)
    mu[mu<1] = 1
    mv[mv<1] = 1
    for k in range(K):
        r[0][:,k] = np.array(np.divide(sp.csr_matrix((train_op['params']['r'][0][:,k], (train_op['train_I'],train_op['train_J'])), shape=(M,N)).sum(1),mu).flatten())[0]
    for l in range(L):
        r[1][:,l] = np.array(np.divide(sp.csr_matrix((train_op['params']['r'][1][:,l], (train_op['train_I'],train_op['train_J'])), shape=(M,N)).sum(0),mv).flatten())[0]        
    
    if model_name.upper().split('_')[1] == 'LINEAR':
        r1_obs = r[0][val_I,:]
        r2_obs = r[1][val_J,:]
        predictions = mmbae_linear_predict(val_I, val_J, M, N, Xs, betas, [r1_obs,r2_obs], K, L)
    Z = sp.csr_matrix((val_Y, (val_I,val_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(val_I,val_J)]).ravel()
    hotStartTrainRMSE = np.sqrt(np.mean((predictions-nonzero_Z)**2)) 
    return hotStartTrainRMSE