
'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

import scipy.sparse as sp
import numpy as np
from SCOAL.run_scoal import run_scoal
from SCOAL.run_scoal import predict_scoal
#from MF.mf import run_mf
#from MF.mf import mf_predict
import copy


def run_hybrid(K, L, X1, X2, train_I, train_J, train_Y, reg_1, reg_2, model_1, model_2):
    # Train and predict using model and calculate residue
    # TO DO : Change this to call the two functions dynamically
    thresh = 1e-6
    residue1 = copy.deepcopy(train_Y)
    residue2 = copy.deepcopy(train_Y)
    convergence = False
    t = 0
    num_iter = 10
    old_error = 1e99
    while convergence == False and t<num_iter:
        if model_1 == 'SCOAL':
            reg = reg_1
            op1,residue1 = train_scoal(K, L, X1, X2, train_I, train_J, train_Y, reg, residue2)
        if model_1 == 'MF':
            M = X1.shape[0]
            N = X2.shape[0]
            reg = reg_1
            op1, residue1 = train_mf(M, N, train_I, train_J, train_Y, reg, residue2)
        # Train Model 2
        if model_2 == 'SCOAL':
            reg = reg_2
            op2,residue2 = train_scoal(K, L, X1, X2, train_I, train_J, train_Y, reg, residue1)    
        if model_2 == 'MF':
            M = X1.shape[0]
            N = X2.shape[0]
            reg = reg_2
            op2, residue2 = train_mf(M, N, train_I, train_J, train_Y, reg,residue1)
        train_op = {'op1':op1,
                    'op2':op2}
        new_error = hotStartTrainRMSE(model_1,model_2,K, L, X1, X2, train_I, train_J, train_Y, train_op)
        print "CURRENT OBJECTIVE and ITERATION"
        print old_error, t        
        if old_error - new_error > thresh:
            old_error = new_error
            t += 1
            continue
        else:
            convergence = True
    return train_op

def train_scoal(K, L, X1, X2, train_I, train_J, train_Y, reg,residue):
    # Train SCOAL
    learner = 'ridge'
    scoal_op = run_scoal(K, L, X1, X2, train_I, train_J, residue, learner, reg)
    # Predict using SCOAL
    M = X1.shape[0]
    N = X2.shape[0]
    model = scoal_op['model']    
    predictions = predict_scoal(train_I, train_J, residue, M, N, model)
    # Calculate residue
    residue = train_Y - predictions
    return scoal_op, residue

def train_mf(M, N, train_I, train_J, train_Y, reg, residue):
    # Train MF
    learner = 'ALS'
    mf_op = run_mf(M, N, train_I, train_J, residue, learner, reg)
    # Predict using MF
    model = mf_op['model']    
    predictions = mf_predict(train_I,train_J,M,N,model)
    # Calculate residue
    residue = train_Y - predictions
    return mf_op, residue

def hotStartTrainRMSE(model_1,model_2,K, L, X1, X2, train_I, train_J, train_Y, train_op):
    M = X1.shape[0]
    N = X2.shape[0]
    op1 = train_op['op1']
    op2 = train_op['op2']
    if model_1 == 'SCOAL':
        model = op1['model']        
        predictions = predict_scoal(train_I, train_J, train_Y, M, N, model)
    if model_1 == 'MF':
        model = op1['model']
        predictions = mf_predict(train_I,train_J,M,N,model)
    if model_2 == 'SCOAL':
        model = op2['model']        
        predictions += predict_scoal(train_I, train_J, train_Y, M, N, model)
    if model_2 == 'MF':
        model = op2['model']
        predictions += mf_predict(train_I,train_J,M,N,model)        
    Z = sp.csr_matrix((train_Y, (train_I,train_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(train_I,train_J)]).ravel()
    hotStartTrainRMSE = np.sqrt(np.mean((predictions.flatten()-nonzero_Z)**2)) 
    return hotStartTrainRMSE

def hotStartValRMSE(model_1,model_2,K, L, X1, X2, train_I, train_J, train_Y, train_op):
    M = X1.shape[0]
    N = X2.shape[0]
    op1 = train_op['op1']
    op2 = train_op['op2']
    if model_1 == 'SCOAL':
        model = op1['model']        
        predictions = predict_scoal(train_I, train_J, train_Y, M, N, model)
    if model_1 == 'MF':
        model = op1['model']
        predictions = mf_predict(train_I,train_J,M,N,model)
    if model_2 == 'SCOAL':
        model = op2['model']        
        predictions += predict_scoal(train_I, train_J, train_Y, M, N, model)
    if model_2 == 'MF':
        model = op2['model']
        predictions += mf_predict(train_I,train_J,M,N,model) 

    #print "VAL PREDICTIONS OBTAINED"
    Z = sp.csr_matrix((train_Y, (train_I,train_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(train_I,train_J)]).ravel()
    hotStartTrainRMSE = np.sqrt(np.mean((predictions.flatten()-nonzero_Z)**2)) 
    #print "VAL RMSE CALCULATED"
    return hotStartTrainRMSE