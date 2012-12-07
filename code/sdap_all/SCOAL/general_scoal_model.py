import numpy as np
import scipy.sparse as sp
import scipy.io
from scoal_defs import learner_class, train_loss_dict, test_loss_dict
from scipy.cluster.vq import whiten, vq
from multiprocessing import Array
from ctypes import c_double, c_int
from numpy.ctypeslib import as_array
from itertools import repeat, izip
        
class GeneralScoalModel(object):

    def __init__(self, rowAttr_filename=None, colAttr_filename=None, crossAttr_filename=None):
        self.objective = 1e99
        self.rowAttr = None
        self.colAttr = None
        self.crossAttr = None
        if rowAttr_filename is not None:
            try:
                self.rowAttr = scipy.io.mmread(rowAttr_filename)
            except:
                pass
        if colAttr_filename is not None:
            try:
                self.colAttr = scipy.io.mmread(colAttr_filename)
            except:
                pass
        if crossAttr_filename is not None:
            try:
                self.crossAttr = scipy.io.mmread(crossAttr_filename)
            except:
                pass

    def set_attributes(self, rowAttr=None, colAttr=None, crossAttr=None):
        if rowAttr is not None:
            if self.rowAttr is not None:
                self.rowAttr = np.hstack((self.rowAttr, rowAttr))
            else:
                self.rowAttr = rowAttr
        if colAttr is not None:
            if self.colAttr is not None:
                self.colAttr = np.hstack((self.colAttr, colAttr))
            else:
                self.colAttr = colAttr
        if crossAttr is not None:
            if self.crossAttr is not None:
                self.crossAttr = np.hstack((self.crossAttr, crossAttr))
            else:
                self.crossAttr = crossAttr

    def initialize(self, Z, W, M, N, initial_R, initial_C,K,L,semi_supervised=False):
        self.R = initial_R
        self.C = initial_C
        self.M = M
        self.N = N
        self.K = K
        self.L = L
        self.semi_supervised = semi_supervised
        
        ###############################################################
        # Initialize Data
        ###############################################################
        self.I,self.J = sp.find(W)[:2]
        self.Z = np.array(Z[self.I, self.J], copy=False).ravel()
        self.num_observations = len(self.I)
        self.flat_index = (sp.coo_matrix((np.arange(self.num_observations), (self.I, self.J)), shape=(self.M, self.N))).tocsr()
        
        self.D = 0
        if self.rowAttr is None:
            self.rowAttr = np.empty((M, 0))
        else:
            self.D += self.rowAttr.shape[1]
        if self.colAttr is None:
            self.colAttr = np.empty((N, 0))
        else:
            self.D += self.colAttr.shape[1]
        if self.crossAttr is None:
            self.crossAttr = np.empty((self.num_observations, 0))
        else:
            self.D += self.crossAttr.shape[1]
        
        self.cold_start_rows = np.ones(self.M, dtype=int)
        self.cold_start_cols = np.ones(self.N, dtype=int)

        # Identify cold start rows and cols
        bin_count_rows = np.bincount(self.I)
        self.cold_start_rows[bin_count_rows>0] = 0

        bin_count_cols = np.bincount(self.J)
        self.cold_start_cols[bin_count_cols>0] = 0
        
    def init_learner(self, learner=None, params={'alpha':0.1}, train_loss=None, test_loss=None):
        
        ###############################################################
        # Setup Output learning models
        ###############################################################
        if learner is None: # uses scikits.learn internal bias
            if self.D>0:
                learner = "ridge"
            else:
                learner = "mean"
        self.models = np.empty((self.K, self.L), dtype = 'object')
        for r in range(self.K):
            for c in range(self.L):
                self.models[r,c] = learner_class(learner, params)
        
        if train_loss is None:
            train_loss = "sq_err"
        if test_loss is None:
            test_loss = "mse"
        self.train_loss_type = train_loss
        self.train_loss = train_loss_dict[train_loss]
        self.test_loss = test_loss_dict[test_loss]

    def init_ss_learner(self, learner=None, params={'alpha':0.1}):
        
        ###############################################################
        # Setup semi-supervised co-cluster learning models
        ###############################################################
        self.ss_models = np.empty((2,), dtype = 'object') # For Dyadic data
        self.cumsum_obj = np.zeros((2,))
        self.obj_ss = [np.empty((self.M,self.K)),np.empty((self.N,self.L))]
        for i in range(2):
            if type(params) is dict:
                self.ss_models[i] = learner_class(learner, params)
            else:
                self.ss_models[i] = learner_class(learner, params[i])
        
    def build_covariates(self, filtered_I, filtered_J):
        # Extract and Normalize covariates
        flat_index = np.array(self.flat_index[filtered_I, filtered_J], copy=False).ravel()
        covariates = np.hstack((self.rowAttr[filtered_I], self.colAttr[filtered_J] , self.crossAttr[flat_index]))    
        return covariates

    def train(self):
        total_error = 0.0
        for r in range(self.K):
            for c in range(self.L):
                # Filter out irrelevant rows
                row_index  = self.R[self.I]==r
                filtered_I = self.I[row_index]
                filtered_J = self.J[row_index]
                filtered_Z = self.Z[row_index]
                # Filter out irrelevant cols
                col_index  = self.C[filtered_J]==c
                filtered_I = filtered_I[col_index]
                filtered_Z = filtered_Z[col_index]
                filtered_J = filtered_J[col_index]
                if len(filtered_Z)>0:
                    # Build the current covariates matrix
                    covariates = self.build_covariates(filtered_I, filtered_J)
                    # call learner
                    self.models[r,c].fit(covariates, filtered_Z)
                    if self.train_loss_type == 'negloglik':
                        total_error += self.train_loss(filtered_Z, self.models[r,c].model.predict_log_proba(covariates)).sum()
                    else:
                        total_error += self.train_loss(filtered_Z, self.models[r,c].predict(covariates)).sum()
                    del covariates
        
        # Train the Semi_supervised co-clustering model
        if self.semi_supervised:
            for i in range(2):
                if i == 0:
                    train_indices = list(set(self.I))
                    target = self.R[train_indices]
                    number = len(train_indices)
                    training_vector = np.hstack((self.rowAttr[train_indices],np.ones((number,1))))
                else:
                    train_indices = list(set(self.J))
                    target = self.C[train_indices]
                    number = len(train_indices)
                    training_vector = np.hstack((self.colAttr[train_indices],np.ones((number,1))))
                
                # Train the semi-supervised co-clustering model
                self.ss_models[i].fit(training_vector, target)
                # TODO : generalize this for models other than logistic (make it find obj fn or smthn)
                self.obj_ss[i][train_indices] = abs(self.ss_models[i].model.predict_log_proba(training_vector))
                if i == 0:
                    self.cumsum_obj[i] = np.sum([self.obj_ss[i][j,self.R[j]] for j in range(number)])
                else:
                    self.cumsum_obj[i] = np.sum([self.obj_ss[i][j,self.C[j]] for j in range(number)])
                total_error += self.cumsum_obj[i]

        self.objective = total_error / self.num_observations

        if __debug__:
            print "Objective after train: %f" % (self.objective,)

        return

    def update_row_assignments(self):
        if not self.semi_supervised:
            errors = np.zeros((self.M, self.K))
        else:
            errors = self.obj_ss[0]
        row_errors = np.zeros(self.M)
        
        for c in range(self.L):
            # Filter out irrelevant cols
            col_index  = self.C[self.J]==c
            filtered_I = self.I[col_index]
            filtered_J = self.J[col_index]
            filtered_Z = self.Z[col_index]
            if len(filtered_Z)>0 : # check if column cluster has data
                # Build the current covariates matrix
                covariates = self.build_covariates(filtered_I, filtered_J)
                for r in range(self.K):
                    if self.train_loss_type == 'negloglik':
                        # Calculate negative log likelihood
                        current_errors = self.train_loss(filtered_Z, self.models[r,c].model.predict_log_proba(covariates))
                    else:                 
                        # Calculate the (squared) prediction errors
                        current_errors = self.train_loss(filtered_Z, self.models[r,c].predict(covariates))
                    # Sum along the rows
                    current_row_count = np.bincount(filtered_I, minlength=self.M)
                    current_row_total_errors = np.bincount(filtered_I, current_errors, minlength=self.M)
                    # Add row (squared) errors to the corresponding row cluster r
                    errors[current_row_count>0,r] += current_row_total_errors[current_row_count>0]
                del covariates
                    
        # Debugging:    Calculate the current objective function (errors in currently assigned row clusters)
        if __debug__:
            inds = np.arange(self.M)
            print "objective calculated just prior to row assignment: %f" % (np.sum(errors[inds,self.R])/self.num_observations,)
        # New assignments and corresponding (squared) errors
        # TODO :vectorize this
        self.R = np.argmin(errors,axis=1)
        row_errors = errors[xrange(errors.shape[0]),self.R]
        '''for m in range(self.M):
            self.R[m] = np.argmin(errors[m,:])
            row_errors[m] = errors[m,self.R[m]]'''
        # Sum of all (squared) errors / number of observations = objective
        if self.semi_supervised:
            self.objective = (self.cumsum_obj[1] + np.sum(row_errors))/self.num_observations
        else:
            self.objective = np.sum(row_errors)/self.num_observations
        if __debug__:
            print "Objective after update_row_assignments: %f" % (self.objective,)

    def update_col_assignments(self):
        if not self.semi_supervised:
            errors = np.zeros((self.N, self.L))
        else:
            errors = self.obj_ss[1]        
        col_errors = np.zeros(self.N)
        for r in range(self.K):
            # Filter out irrelevant rows
            row_index = self.R[self.I]==r
            filtered_I = self.I[row_index]
            filtered_J = self.J[row_index]
            filtered_Z = self.Z[row_index]
            if len(filtered_Z)>0:
                # Build the current covariates matrix
                covariates = self.build_covariates(filtered_I, filtered_J)
                for c in range(self.L):
                    if self.train_loss_type == 'negloglik':
                        # Calculate negative log likelihood
                        current_errors = self.train_loss(filtered_Z, self.models[r,c].model.predict_log_proba(covariates))
                    else:                    
                        current_errors = self.train_loss(filtered_Z, self.models[r,c].predict(covariates))
                    current_col_count = np.bincount(filtered_J, minlength=self.N)
                    current_col_total_errors = np.bincount(filtered_J, current_errors, minlength=self.N)
                    errors[current_col_count>0,c] += current_col_total_errors[current_col_count>0]
                del covariates
                
        self.C = np.argmin(errors,axis=1)
        col_errors = errors[xrange(errors.shape[0]),self.C]                
        '''for n in range(self.N):
            self.C[n] = np.argmin(errors[n,:])
            col_errors[n] = errors[n,self.C[n]]'''
        if self.semi_supervised:            
            self.objective = (self.cumsum_obj[0] + np.sum(col_errors))/self.num_observations
        else:
            self.objective = np.sum(col_errors)/self.num_observations
        if __debug__:
            print "Objective after update_col_assignments: %f" % (self.objective,)

    def predict(self,I,J,X1,X2,centroids,quantized=False):
        total_I = np.array([])
        total_J = np.array([])
        total_predictions = np.array([])
        # Find the row/col clusters for new I/J
        # Find the I/J which are new
        new_I = list(set(I).difference(set(self.I)))
        new_J = list(set(J).difference(set(self.J)))
        if new_I:
            if self.semi_supervised:
                self.R[new_I] = self.ss_models[0].model.predict(np.hstack((X1[new_I],np.ones((len(new_I),1)))))
            else:
                # Map the cold start Validation set users and movies to their clusters
                # Users
                user_cluster = vq(whiten(X1), centroids['user_centroids'])[0]
                user_cluster_seen = user_cluster[list(set(I))]
                user_cocluster = self.R[list(set(I))]
                mapping = np.zeros((self.K,1))
                for k_clust in range(self.K):
                    users_in_cluster = [x for x in range(len(user_cluster_seen)) if user_cluster_seen[x] == k_clust]
                    users_in_cluster = user_cocluster[users_in_cluster]
                    temp = [len(filter(lambda x: x == k_coclust, users_in_cluster)) for k_coclust in range(self.K)]
                    mapping[k_clust] = temp.index(max(temp))
                val_users = np.array(list(set(I)))        
                for user in val_users:
                    self.R[user] = mapping[user_cluster[user]]
                
        if new_J:
            if self.semi_supervised:
                self.C[new_J] = self.ss_models[1].model.predict(np.hstack((X2[new_J],np.ones((len(new_J),1)))))
            else:
                # Movies
                movie_cluster = vq(whiten(X2), centroids['movie_centroids'])[0]
                movie_cluster_seen = movie_cluster[list(set(J))]
                movie_cocluster = self.C[list(set(J))]
                mapping = np.zeros((self.L,1))
                for l_clust in range(self.L):
                    movies_in_cluster = [x for x in range(len(movie_cluster_seen)) if movie_cluster_seen[x] == l_clust]
                    movies_in_cluster = movie_cocluster[movies_in_cluster]
                    temp = [len(filter(lambda x: x == l_coclust, movies_in_cluster)) for l_coclust in range(self.L)]
                    mapping[l_clust] = temp.index(max(temp))
                val_movies = np.array(list(set(J)))        
                for movie in val_movies:
                    self.C[movie] = mapping[movie_cluster[movie]]        

            
        # Build the current covariates matrix
        for r in range(self.K):
            for c in range(self.L):
                # Filter out irrelevant rows
                filtered_I = I[self.R[I]==r]
                filtered_J = J[self.R[I]==r]
                # Filter out irrelevant cols
                filtered_I = filtered_I[self.C[filtered_J]==c]
                filtered_J = filtered_J[self.C[filtered_J]==c]
                if len(filtered_I)>0:
                    covariates = self.build_covariates(filtered_I, filtered_J)
                    current_predictions = self.models[r,c].predict(covariates)
                    del covariates
                    total_I = np.concatenate((total_I, filtered_I))
                    total_J = np.concatenate((total_J, filtered_J))
                    total_predictions = np.concatenate((total_predictions, current_predictions))
        predictions = sp.coo_matrix((total_predictions, (total_I, total_J)), shape=(self.M, self.N)).tocsr()
        prediction_list = np.array(predictions[I,J]).ravel()
        return prediction_list
