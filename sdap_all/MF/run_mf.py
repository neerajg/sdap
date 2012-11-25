
'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

from mf_scoalnamstyle import train_mfscoalnamstyle
from mfscoalnamstyle_predict import mfscoalnamstyle_predict
import scipy.sparse as sp
import numpy as np

def run_mf(K, L, X1, X2, train_I, train_J, train_Y, reg_lambda, sub_model):
    M = X1.shape[0]
    N = X2.shape[0]
    if sub_model == 'ALS':
        implementation = 'graphlab/ALS'
        params, obj = train_mfscoalnamstyle(M, N, train_I, train_J, train_Y, K, L,reg_lambda,implementation)
    if sub_model == 'SELFPMF':
        implementation = 'self/PMF'
        params, obj = train_mfscoalnamstyle(M, N, train_I, train_J, train_Y, K, L,reg_lambda,implementation)
    train_op = {'params':params,
                'obj':obj
                }
    return train_op

def hotStartTrainRMSE(model_name, K, L, X1, X2, train_I, train_J, train_Y, train_op):
    M = X1.shape[0]
    N = X2.shape[0]
    model = train_op['params']['model']
    predictions = mfscoalnamstyle_predict(train_I, train_J, M, N,K,L,model)
    Z = sp.csr_matrix((train_Y, (train_I,train_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(train_I,train_J)]).ravel()
    hotStartTrainRMSE = np.sqrt(np.mean((predictions.flatten()-nonzero_Z)**2)) 
    return hotStartTrainRMSE

def hotStartValRMSE(model_name, K, L, X1, X2, val_I, val_J, val_Y, train_op):
    #print "INSIDE VAL RMSE"
    M = X1.shape[0]
    N = X2.shape[0]
    model = train_op['params']['model']    
    predictions = mfscoalnamstyle_predict(val_I, val_J, M, N,K,L,model)
    #print "VAL PREDICTIONS OBTAINED"
    Z = sp.csr_matrix((val_Y, (val_I,val_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(val_I,val_J)]).ravel()
    hotStartTrainRMSE = np.sqrt(np.mean((predictions.flatten()-nonzero_Z)**2)) 
    #print "VAL RMSE CALCULATED"
    return hotStartTrainRMSE