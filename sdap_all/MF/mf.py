'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

# TO DO : check all TO DOs here and in misc
# TO DO : Try damped updates
# TO DO : here and in misc also change update of r condition to check for very low value to make it work for all K and L
import numpy as np
import sys
from MFModel import MFModel
import gc

D = 20
thresh = 1e-4
#lambda_reg = 0.065
lambda_damping = 0
num_iter = 50

def run_mf(M, N, train_I, train_J, train_Y, learner, lambda_reg, K=1, L = 1):
    model = MFModel()
    initial_R = np.random.randint(0, K, size=(M,1))
    initial_C = np.random.randint(0, L, size=(N,1))
    model.initialize(M,N,K,L,initial_R,initial_C,D,train_Y,train_I,train_J,lambda_reg)
    model.train()
    params = {'model':model}
    return params

def mf_predict(train_I, train_J, M, N,model,K=1,L=1):
    predictions = np.zeros((len(train_I),))
    indices = np.array(range(len(predictions)))
    for r in range(K):
        for c in range(L):
            row_index = model.R[train_I]==r
            row_index = row_index.flatten()
            filtered_indices = indices[row_index]
            filtered_I = train_I[row_index]
            filtered_J = train_J[row_index]
            col_index  = model.C[filtered_J]==c
            col_index = col_index.flatten()
            filtered_indices = filtered_indices[col_index]
            filtered_I = filtered_I[col_index]
            filtered_J = filtered_J[col_index]
            predictions[filtered_indices] = np.sum(np.multiply(model.U[filtered_I,:,c],model.V[filtered_J,:,r]),1).reshape(len(filtered_indices),1)
    return predictions
 