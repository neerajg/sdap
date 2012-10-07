'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

# TO DO : check all TO DOs here and in misc
# TO DO : Try damped updates
# TO DO : here and in misc also change update of r condition to check for very low value to make it work for all K and L
import numpy as np
import scipy.sparse as sp
import clr
clr.AddReference("MyMediaLite.dll")
from MyMediaLite import *

D = 20
thresh = 1e-10
lambda_reg = 1e-4
lambda_damping = 0
num_iter = 100

def train_als(M, N, train_I, train_J, train_Y):
    # load the data
    train_data = IO.RatingData.Read("u1.base")
    test_data  = IO.RatingData.Read("u1.test")

    # set up the recommender
    recommender = RatingPrediction.UserItemBaseline() # don't forget ()
    recommender.Ratings = train_data
    recommender.Train()

    # measure the accuracy on the test data set
    print Eval.Ratings.Evaluate(recommender, test_data)
    
    return

# make a prediction for a certain user and item
print recommender.Predict(1, 1)
    
def train_als_test(M, N, train_I, train_J, train_Y):
    U,V = initialize(M,N,D)
    t = 0
    convergence = False
    Y_matrix = sp.csr_matrix((train_Y,(train_I,train_J)), shape = (M,N))
    W_matrix = sp.csr_matrix((np.ones(len(train_I,)),(train_I,train_J)), shape = (M,N))
    tot_error_old = 1e99
    while convergence == False and t<num_iter:
        # Update Rows
        print "updating U"
        U = update_U(U,V,M,N,W_matrix,Y_matrix)
        # Update Cols
        print "updating V"
        V = update_V(U,V,M,N,W_matrix,Y_matrix)
        # Get Error
        error_obs = getErrorObs(train_Y, train_I, train_J, U, V)        
        # Check for convergence
        tot_error_new = np.sqrt(np.sum(np.sum(error_obs**2))/len(train_Y))
        print tot_error_new, t
        if tot_error_old - tot_error_new < thresh or t>num_iter:
            convergence = True
        else:
            tot_error_old = tot_error_new
            t +=1
    
    obj = []
    params = {'U':U,
              'V':V}
    return params, obj

def initialize(M,N,D):
    U = np.random.randn(M,D)
    V = np.random.randn(N,D)
    return U,V

def update_U(U,V,M,N,W,Y):
    for i in range(M):
        # Select the columns of V' which correspond to the observed ratings for user 1
        indices_obs = W.getrow(i).nonzero()[1]
        if len(indices_obs) < 1:
            continue
        indices_obs.sort()
        V_i = V[indices_obs,:]
        Ai = np.dot(np.transpose(V_i),V_i) + np.diagflat(lambda_reg*indices_obs.size*np.ones(D))
        Y_i = np.dot(np.transpose(V_i),Y.getrow(i).sorted_indices().data)
        U_i = np.linalg.solve(Ai,Y_i)
        #U_i = np.linalg.inv(Ai)*Y_i
        # Damped updates
        U[i,:] = U[i,:]*lambda_damping + (1-lambda_damping)*U_i
    return U

def update_V(U,V,M,N,W,Y):
    for j in range(N):
        # Select the columns of V' which correspond to the observed ratings for user 1
        indices_obs = W.getcol(j).nonzero()[1]
        if len(indices_obs) < 1:
            continue        
        indices_obs.sort()
        U_j = U[indices_obs,:]
        Aj = np.dot(np.transpose(U_j),U_j) + np.diagflat(lambda_reg*indices_obs.size*np.ones(D))
        Y_j = np.dot(np.transpose(U_j),Y.getcol(j).sorted_indices().data)
        V_j = np.linalg.solve(Aj,Y_j)
        #V_j = np.linalg.inv(Aj)*Y_j
        # Damped updates
        V[j,:] = V[j,:]*lambda_damping + (1-lambda_damping)*V_j
    return V
    
def getErrorObs(train_Y, train_I, train_J, U, V):
    error_obs = train_Y - np.dot(U,V.T)[train_I,train_J]
    return error_obs


