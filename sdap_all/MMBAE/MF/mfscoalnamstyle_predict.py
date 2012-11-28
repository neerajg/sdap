
'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

import numpy as np

def mfscoalnamstyle_predict(train_I, train_J, M, N,K,L,model):
    predictions = np.zeros((len(train_I),1))
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