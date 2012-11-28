
'''
Created on Apr 5, 2012

Author - Neeraj Gaur
'''

import numpy as np
import scipy.sparse as sp
from learn_scoal import learn_scoal
from scipy.cluster.vq import whiten, vq
import sys
import SCOAL.general_scoal_model as scoal

convergence_threshold = 1e-6

def run_scoal(K, L, X1, X2, train_I, train_J, train_Y, learner, reg_lambda):
    
    model = scoal.GeneralScoalModel()

    Xs = [0,0]   
    Xs[0] = X1
    Xs[1] = X2     
    M = Xs[0].shape[0]
    N = Xs[1].shape[0]

    vals = train_Y
    train_W = sp.csr_matrix((np.ones(len(train_Y)),(train_I,train_J)), shape=(M,N))    
    Z = sp.csr_matrix((vals, (train_I,train_J)), shape=(M,N))

    # Train Hard SCOAL    
    rowAttr = Xs[0]
    colAttr = Xs[1]
    model.set_attributes(rowAttr, colAttr, crossAttr=None)
    obj = learn_scoal(model, Z, train_W, K, L, learner, reg_lambda,None,None, initial_R=None, initial_C=None)
    parameters = np.zeros((K,L,rowAttr.shape[1] + colAttr.shape[1]))
    for k in range(K):
        for l in range(L):
            parameters[k,l,:] = model.models[k,l].model.coef_
    
    params = {'I':model.I,
              'J':model.J,
              'row_assignment':model.R,
              'col_assignment':model.C,
              'parameters':parameters
              }

    train_op = {'params':params,
                'obj':obj,
                'model':model
                }

    return train_op

def hotStartTrainRMSE(K, L, X1, X2, train_I, train_J, train_Y, train_op):
    M = X1.shape[0]
    N = X2.shape[0]
    model = train_op['model']
    predictions = predict_scoal(train_I, train_J, train_Y, M, N, model)
    Z = sp.csr_matrix((train_Y, (train_I,train_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(train_I,train_J)]).ravel()
    hotStartTrainRMSE = np.sqrt(np.mean((predictions-nonzero_Z)**2)) 
    return hotStartTrainRMSE

'''def warmStartTrainRMSE(K, L, X1, X2, train_I, train_J, train_Y, train_op):
    M = X1.shape[0]
    N = X2.shape[0]
    model = train_op['model']
    predictions = predict_scoal(train_I, train_J, train_Y, M, N, model)
    Z = sp.csr_matrix((train_Y, (train_I,train_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(train_I,train_J)]).ravel()
    warmStartTrainRMSE = np.sqrt(np.mean((predictions-nonzero_Z)**2)) 
    return warmStartTrainRMSE

def coldStartTrainRMSE(K, L, X1, X2, train_I, train_J, train_Y, train_op):
    coldStartTrainRMSE = hotStartTrainRMSE(K, L, X1, X2, train_I, train_J, train_Y, train_op)
    return coldStartTrainRMSE'''

def hotStartValRMSE(K, L, X1, X2, val_I, val_J, val_Y, train_op):
    M = X1.shape[0]
    N = X2.shape[0]
    model = train_op['model']
    predictions = predict_scoal(val_I, val_J, val_Y, M, N, model)
    Z = sp.csr_matrix((val_Y, (val_I,val_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(val_I,val_J)]).ravel()
    hotStartValRMSE = np.sqrt(np.mean((predictions-nonzero_Z)**2)) 
    return hotStartValRMSE

def warmStartValRMSE(K, L, X1, X2, val_I, val_J, val_Y, train_op, centroids):
    M = X1.shape[0]
    N = X2.shape[0]
    I = train_op['params']['I']
    J = train_op['params']['J']
    model = train_op['model']
    # Map the warm/cold start Validation set users and movies to their clusters
    # Users
    user_cluster = vq(whiten(X1), centroids['user_centroids'])[0]
    user_cluster_seen = user_cluster[list(set(I))]
    user_cocluster = model.R[list(set(I))]
    mapping = np.zeros((K,1))
    for k_clust in range(K):
        users_in_cluster = [x for x in range(len(user_cluster_seen)) if user_cluster_seen[x] == k_clust]
        users_in_cluster = user_cocluster[users_in_cluster]
        temp = [len(filter(lambda x: x == k_coclust, users_in_cluster)) for k_coclust in range(K)]
        mapping[k_clust] = temp.index(max(temp))
    val_users = np.array(list(set(val_I)))        
    for user in val_users:
        model.R[user] = mapping[user_cluster[user]]
    # Movies
    movie_cluster = vq(whiten(X2), centroids['movie_centroids'])[0]
    movie_cluster_seen = movie_cluster[list(set(J))]
    movie_cocluster = model.C[list(set(J))]
    mapping = np.zeros((L,1))
    for l_clust in range(L):
        movies_in_cluster = [x for x in range(len(movie_cluster_seen)) if movie_cluster_seen[x] == l_clust]
        movies_in_cluster = movie_cocluster[movies_in_cluster]
        temp = [len(filter(lambda x: x == l_coclust, movies_in_cluster)) for l_coclust in range(L)]
        mapping[l_clust] = temp.index(max(temp))
    val_movies = np.array(list(set(val_J)))        
    for movie in val_movies:
        model.C[movie] = mapping[movie_cluster[movie]]        
    
    # Use the same prediction logic from now on
    predictions = predict_scoal(val_I, val_J, val_Y, M, N, model)
    Z = sp.csr_matrix((val_Y, (val_I,val_J)), shape=(M,N))
    nonzero_Z = np.array(Z[(val_I,val_J)]).ravel()
    warmStartValRMSE = np.sqrt(np.mean((predictions-nonzero_Z)**2)) 
    return warmStartValRMSE

def coldStartValRMSE(K, L, X1, X2, val_I, val_J, val_Y, train_op, centroids):
    coldStartValRMSE = warmStartValRMSE(K, L, X1, X2, val_I, val_J, val_Y, train_op, centroids)
    return coldStartValRMSE

def predict_scoal(I, J, Y, M, N, model):
    W = sp.csr_matrix((np.ones(len(Y)),(I,J)), shape=(M,N))
    I,J = sp.find(W)[:2]
    predictions = model.predict(I,J)
    return predictions

if '__main__' == __name__:
    sys.exit()
