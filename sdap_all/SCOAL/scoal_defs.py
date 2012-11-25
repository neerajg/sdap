from scikit.learn import linear_model
from scikit.learn import metrics
from scikit.learn import svm
from numexpr import evaluate
import numpy as np

ones = np.ones
class MeanModel(object):
    ''' mean model stub class '''
    def __init__(self, *args, **kwargs):
        self.mean = 0.0
            
    def fit(self, X, y):
        self.mean = y.mean()
        
    def predict(self, X):
        pred = ones(X.shape[0])
        pred.fill(self.mean)
        return pred

class MeanModelClassifier(object):
    ''' mean model stub class for classification 
    Assumes {-1, 1} 2 class problem
    '''
    def __init__(self, *args, **kwargs):
        self.mean = 0.0
        
    def fit(self, X, y):
        self.mean = y.mean() # assuming {-1,1}
        
    def predict(self, X):
        if self.mean<0.0:
            return -ones(X.shape[0], dtype=int)
        else:
            return ones(X.shape[0], dtype=int)
    
    def decision_function(self, X):
        pred = ones(X.shape[0])
        pred.fill(self.mean)
        return pred
    
# scikits learners
regressors = {# regressors
                "mean":MeanModel,
                "linear":linear_model.LinearRegression,
                "ridge":linear_model.Ridge,
                "lasso":linear_model.Lasso,
                "elastic-net":linear_model.ElasticNet,
                # sparse regressors
                "sp-lasso":linear_model.sparse.Lasso,
                "sp-elastic-net":linear_model.sparse.ElasticNet,
                # sgd regressor
                "sgd-regress":linear_model.sparse.SGDRegressor}
classifiers = {# classifiers
                "mean-class":MeanModelClassifier,
                "logistic":linear_model.LogisticRegression, # l1 regularization
                "svm":svm.LinearSVC, # linear svm
                # sparse classifiers
                "sp-svm":svm.sparse.LinearSVC,
                # sgd classifier
                "sgd-class":linear_model.sparse.SGDClassifier
                }#  add more as as needed        

squeeze = np.squeeze
abs = np.abs
class learner_class(object):
    ''' wrapper for scikits learners so regressor modules may be called the same way 
    as classifer modules using log probabilities
    '''
    
    def __init__(self, model, kwargs):
        if model in regressors:
            self.model = regressors[model](**kwargs)
            self.predict = self.model.predict
        elif model in classifiers:            
            self.model = classifiers[model](**kwargs)
            self.predict = self.squeeze_decision
        else:
            raise NameError("model is %s is not defined" % (model) )
            
        self.fit = self.model.fit
    
    def squeeze_decision(self, *args, **kwargs):
        x = squeeze(self.model.decision_function(*args, **kwargs))
        if len(self.model.label_)<2:# single class
            ll = self.model.label_[0]
            if ll > 0: # positive
                return abs(x)
            else: # negative
                return -abs(x)
        return x
        
# train errors
sign = np.sign
def sq_err(y_true, y_pred): # sq error
    return evaluate("(y_true - y_pred)**2")

def abs_err(y_true, y_pred): # abs error
    return evaluate("abs(y_true - y_pred)")

def class_error(y_true, y_pred): # errors. Assumes probability and 0.5 threshold
    return y_true!=sign(y_pred)

def logloss(y_true, y_pred): # log loss
    return evaluate("log(1+exp(-y*z))",{"y":y_true, "z":y_pred} )

# test errors
def mse(y_true, y_pred): # mean sq error
    return sq_err(y_true, y_pred).mean()

def mae(y_true, y_pred): # mean abs error
    return abs_err(y_true, y_pred).mean()

def fract_error(y_true, y_pred): # fraction of errors
    return class_error(y_true, y_pred).mean()
    
def logloss_avg(y_true, y_pred): # avg log loss
    return logloss(y_true, y_pred).mean()

def auc(y_true, y_pred): # auc
    fpr, tpr = metrics.roc_curve(y_true, y_pred)[:2]
    return -metrics.auc(fpr, tpr)

def recall_score(y_true, y_pred):
    return -metrics.recall_score(y_true, sign(y_pred) )
    
def precision_score(y_true, y_pred):
    return -metrics.precision_score(y_true, sign(y_pred))

def f2_score(y_true, y_pred):
    return -metrics.fbeta_score(y_true, sign(y_pred), 2)

''' train loss decomposes component wise, Used during training SCOAL '''
train_loss_dict = {"sq_err":sq_err,
                   "abs_err":abs_err,
                   "class_err":class_error,
                   "logloss":logloss}

''' test loss to minimize. Used for final test. These may not decompose element wise 
NOTE: auc, recall, precision and f2 use -score so the "minimize" outer loops work!!!!'''
test_loss_dict = {"mse":mse,
                  "mae":mae,
                  "fract_error":fract_error,
                  "logloss":logloss_avg,
                  "nauc":auc,
                  "nrecall":recall_score,
                  "nprecision":precision_score,
                  "nf2":f2_score}

# Constant definitions
#convergence_threshold = 1e-6
maxIterations = 50 # maximum number of splits allowed

# default options
default_cv = 5 
default_K = 1
default_L = 1
default_param_list = [{"alpha":1}]
default_indir = 'data'
default_model_filename = "models/saved_model_file"
