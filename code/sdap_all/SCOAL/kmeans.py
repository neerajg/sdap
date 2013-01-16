'''
Created on Nov 26, 2012

@author: neeraj
'''
import sys
import numpy as np
from scipy.spatial.distance import cdist

class kMeans():
    def __init__(self,k=None):
        if k is None:
            print 'NEED K'
            sys.exit()
        self.k = k
        self.flag = True # TODO: change to smthn more elegant
    
    # Given the features this returns the label of the closest centroid
    def predict(self,X):
        dist = np.empty((X.shape[0],self.k))
        y = np.empty((X.shape[0],),dtype = 'int')
        for k in range(self.k):
            dist[:,k] =  cdist(X,self.xk[k,:].reshape(1,X.shape[1]),'sqeuclidean').flatten()
        for i in range(len(X)):
            y[i] = np.argmin(dist[i,:])
        return y
    
    # Given the features and the cluster labels finds the cluster centroids
    def fit(self,X,y):
        if self.flag:
            self.xk = np.empty((self.k,X.shape[1]))
        for k in range(self.k):
            self.xk[k,:] = np.mean(X[y==k,:],0)
    
    # TODO:  Use the same name as the logistic method for now (change later)
    # Given features returns the distance from each cluster centroid
    def predict_log_proba(self,X):
        if self.flag:
            self.dist = np.empty((X.shape[0],self.k))
            self.flag = False
        for k in range(self.k):
            self.dist[:,k] =  cdist(X,self.xk[k,:].reshape(1,X.shape[1]),'sqeuclidean').flatten()
        return self.dist
    
    # TODO:  Use the same name as the logistic method for now (change later)
    # Given features returns the distance from each cluster centroid    
    def predict_proba(self,X):
        if self.flag:
            self.dist = np.empty((X.shape[0],self.k))
            self.flag = False
        for k in range(self.k):
            self.dist[:,k] =  cdist(X,self.xk[k,:].reshape(1,X.shape[1]),'sqeuclidean').flatten()
        return self.dist    