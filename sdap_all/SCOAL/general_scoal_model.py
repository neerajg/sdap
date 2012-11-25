import numpy as np
import scipy.sparse as sp
import scipy.io
import cPickle as pickle
from scoal_defs import learner_class, train_loss_dict, test_loss_dict
        
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

    def initialize(self, Z, W, M, N, initial_R, initial_C,K,L,semi_supervised):
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
        self.train_loss = train_loss_dict[train_loss]
        self.test_loss = test_loss_dict[test_loss]

    def init_ss_learner(self, learner=None, params={'alpha':0.1}):
        
        ###############################################################
        # Setup semi-supervised co-cluster learning models
        ###############################################################
        self.ss_models = np.empty((2, 1), dtype = 'object') # For Dyadic data
        self.cumsum_log_prob = np.zeros((2,1))
        self.log_prob_ss = [np.empty((self.M,self.K)),np.empty((self.N,self.L))]
        for i in range(2):
            self.ss_models[i] = learner_class(learner, params)
        
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
                    total_error += self.train_loss(filtered_Z, self.models[r,c].predict(covariates)).sum()
                    del covariates
        
        # Train the Semi_supervised co-clustering model
        if self.semi_supervised:
            for i in range(2):
                if i == 0:
                    target = self.R
                    number = self.M
                    training_vector = np.hstack((self.rowAttr,np.ones((self.M,1))))
                else:
                    target = self.C
                    number = self.N
                    training_vector = np.hstack((self.colAttr,np.ones((self.N,1))))
                
                # Train the semi-supervised co-clustering model
                self.ss_models[i].fit(training_vector, target)
                self.log_prob_ss[i] = - self.ss_models[i].predict_log_proba(training_vector)
                self.cumsum_log_prob[i] = [self.log_prob_ss[i][j,self.R[j]] for j in range(number)]
                total_error += self.cumsum_log_prob[i]

        self.objective = total_error / self.num_observations

        if __debug__:
            print "Objective after train: %f" % (self.objective,)

        return

    def update_row_assignments(self):
        if not self.semi_supervised:
            errors = np.zeros((self.M, self.K))
        else:
            errors = self.cumsum_log_prob[1] + self.log_prob_ss[0]
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
        for m in range(self.M):
            self.R[m] = np.argmin(errors[m,:])
            row_errors[m] = errors[m,self.R[m]]
        # Sum of all (squared) errors / number of observations = objective
        self.objective = np.sum(row_errors)/self.num_observations
        if __debug__:
            print "Objective after update_row_assignments: %f" % (self.objective,)

    def update_col_assignments(self):
        if not self.semi_supervised:
            errors = np.zeros((self.N, self.L))
        else:
            errors = self.cumsum_log_prob[0] + self.log_prob_ss[1]        
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
                    current_errors = self.train_loss(filtered_Z, self.models[r,c].predict(covariates))
                    current_col_count = np.bincount(filtered_J, minlength=self.N)
                    current_col_total_errors = np.bincount(filtered_J, current_errors, minlength=self.N)
                    errors[current_col_count>0,c] += current_col_total_errors[current_col_count>0]
                del covariates
                
        for n in range(self.N):
            self.C[n] = np.argmin(errors[n,:])
            col_errors[n] = errors[n,self.C[n]]
        self.objective = np.sum(col_errors)/self.num_observations
        if __debug__:
            print "Objective after update_col_assignments: %f" % (self.objective,)

    def predict(self, I, J, quantized=False):
        total_I = np.array([])
        total_J = np.array([])
        total_predictions = np.array([])
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
