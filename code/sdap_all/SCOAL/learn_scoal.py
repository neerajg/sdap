#! /usr/bin/env python -u
'''
usage 
learn_scoal.py -h
OR
from learn_scoal import model_selector
model_selector(Z, W, rowAttr, colAttr, crossAttr, learner, num_cv, init_K, init_L, alphas)
NOTE:  Currently returns/prints all validation errors
TODO: pick/return best model
OR
from learn_scoal import learn_scoal
learn_scoal(model, Z, W, K, L, learner, param, initial_R, initial_C)
'''
import sys
from optparse import OptionParser
import numpy as np
import numpy.random as random
import scipy.sparse as sp
import scipy.io
from general_scoal_model import GeneralScoalModel
from collections import defaultdict
from operator import itemgetter

from scoal_defs import maxIterations, default_cv, \
        default_K, default_L, default_param_list, default_indir, default_model_filename

def validSets(N, k):
    ''' randomized k fold cross validation from N elements'''
    cvsets = np.array_split(np.random.permutation(N), k)
    for ix in range(k): # test ind = ix
        yield np.hstack([cvsets[jx] for jx in range(k) if jx !=ix]), cvsets[ix]
        
def read_options(*args, **kwargs):
    ''' checks/sets options read from calling module'''
    if len(args)>1:
        Z = args[0]
    else:
        Z = kwargs["Z"]
    W = kwargs.get("W", None)
    if W is None:
        rows, cols = sp.find(Z)[:2]
        W = sp.coo_matrix( (np.ones(len(rows), dtype=int), (rows, cols)), shape=Z.shape, dtype=int).tocsr()
    rowAttr = kwargs.get("rowAttr", None)
    colAttr = kwargs.get("colAttr", None)
    crossAttr = kwargs.get("crossAttr", None)
    
    learner = kwargs.get("learner", "ridge")
    train_loss = kwargs.get("train_loss", "sq_err")
    test_loss = kwargs.get("test_loss", "mse")
    num_cv = kwargs.get("num_cv", default_cv)
    init_K = kwargs.get("K", default_K) 
    init_L = kwargs.get("L", default_L) 
    model_filename = kwargs.get("model_filename", default_model_filename)
    
    if "alphas" in kwargs:
        params = [{"alpha":alpha} for alpha in kwargs["alphas"]]
    else:
        params = kwargs.get("params", default_param_list) 
    
    # kwargs get over-written by args
    arglist = [Z, W, rowAttr, colAttr, crossAttr, learner, params, \
               train_loss, test_loss, \
               num_cv, init_K, init_L, model_filename]
    for ix in range(1, len(args)):
        arglist[ix] = args[ix]
    return arglist

def model_selector(**argv):
    ''' select the best model from cv set and parameter options '''
    # read data from input dict
    Z, W, rowAttr, colAttr, crossAttr, learner, params, \
    train_loss, test_loss, \
    num_cv, init_K, init_L, model_filename = read_options(**argv)
    
    save_validation_loss = defaultdict(list)
    I,J= sp.find(W)[:2]
    num_data = len(I)
    
    for ix, (trainIdx, validationIdx) in enumerate(validSets(num_data, num_cv)):
        if __debug__:
            print '\nValidation set:',ix
        # for each cv split
        model = GeneralScoalModel()
        model.set_attributes(rowAttr, colAttr, crossAttr)
        train_I = I[trainIdx]
        train_J = J[trainIdx]
        validation_I = I[validationIdx]
        validation_J = J[validationIdx]
        
        Z_training = sp.coo_matrix((np.ravel(Z[(train_I, train_J)]), (train_I, train_J)), shape=Z.shape).tocsr()
        W_training = sp.coo_matrix((np.ravel(W[(train_I, train_J)]), (train_I, train_J)), shape=W.shape).tocsr()
        Z_validation = sp.coo_matrix((np.ravel(Z[(validation_I, validation_J)]), (validation_I, validation_J)), shape=Z.shape).tocsr()
        W_validation = sp.coo_matrix((np.ravel(W[(validation_I, validation_J)]), (validation_I, validation_J)), shape=W.shape).tocsr()

        # Do model selection
        for jx, param in enumerate(params): # for each alpha parameter
            if __debug__:
                print 'parameter:', param
            K = init_K
            L = init_L
            learn_scoal(model, Z, W_training, K, L, learner, param, train_loss, test_loss)
            test_Z = np.ravel(Z[(validation_I, validation_J)])
            validation_loss = model.test_loss(test_Z, model.predict(validation_I, validation_J) )
            save_validation_loss[(jx, K, L)].append(validation_loss)
            if __debug__:
                print "Starting validation loss: %f" % (validation_loss,)
            for _ in range(maxIterations):
                row_test_model = model.copy()
                row_split_validation_loss = test_row_split(row_test_model, Z_training, W_training, \
                                                          Z_validation, W_validation, K, \
                                                          L, learner, param, train_loss, test_loss)
                save_validation_loss[(jx, K+1, L)].append(row_split_validation_loss)
                
                col_test_model = model.copy()
                col_split_validation_loss = test_col_split(col_test_model, Z_training, W_training, \
                                                          Z_validation, W_validation, K, \
                                                          L, learner, param, train_loss, test_loss)
                save_validation_loss[(jx, K, L+1)].append(col_split_validation_loss)
                
                if __debug__:
                    print "Row split loss: %f" % (row_split_validation_loss,)
                    print "Col split loss: %f" % (col_split_validation_loss,)
                
                if row_split_validation_loss <= col_split_validation_loss and row_split_validation_loss < validation_loss:
                    K += 1
                    model = row_test_model
                    validation_loss = row_split_validation_loss
                elif col_split_validation_loss < row_split_validation_loss and col_split_validation_loss < validation_loss:
                    L += 1
                    model = col_test_model
                    validation_loss = col_split_validation_loss
                else:
                    break
                if __debug__:
                    print "k: %d, l: %d" % (K, L)
                    print "Current validation loss: %f" % validation_loss
                
            if __debug__:
                print "Final k,l: %d,%d" % (K, L)
                print "Final validation loss: %f" % validation_loss
    
    save_validation_loss = dict(save_validation_loss)
    if __debug__:
        print "\nValidation losses"
        for key, vals in save_validation_loss.iteritems():
            print key, vals
        print "\n"
        
    ''' Compute best parameter: Sort by length, then by -mean() i.e. longest then min(mean(error)) '''
    sorted_loss = sorted( [ [ key, -np.array(val).mean(), len(val) ] for key, val in save_validation_loss.iteritems()] , key=itemgetter(2,1) )
    best_loss = sorted_loss[-1]
    best_loss[1] = -best_loss[1]
    
    print "\nVALIDATION COMPLETE"
    print "Selected parameter", params[best_loss[0][0]]
    print "Selected K:%d, L:%d" % (best_loss[0][1], best_loss[0][2]) 
    print "Mean validation error %f, with %d observations" % (best_loss[1], best_loss[2])
    return best_loss, save_validation_loss

def learn_scoal_wrapper(**argv):
    Z, W, rowAttr, colAttr, crossAttr, learner, params, \
    train_loss, test_loss, \
    num_cv, init_K, init_L, model_filename = read_options(**argv)
    
    Z = Z.tocsr()
    W = W.tocsr()
    param = params[0]
    I,J= sp.find(W)[:2]
    num_data = len(I)
    model = GeneralScoalModel()
    model.set_attributes(rowAttr, colAttr, crossAttr)
    #learn_scoal(model, Z, W, K, L, learner, param, train_loss, test_loss)
    model.save(model_filename)

# TODO : Find a clean way to take in input parameters for the learner classes
def learn_scoal(model, Z, W, K, L, learner,num_iter=100,
                delta_convg = 1e-6, reg_op_learner = None, 
                train_loss = None, test_loss = None,
                ss_learner = None,reg_ss_model = None, 
                semi_supervised=False, initial_R=None, initial_C=None):
    M,N = W.shape
    obj = []
    old_objective = model.objective
    if initial_R is None:
        initial_R = random.randint(0, K, size=Z.shape[0])
    if initial_C is None:
        initial_C = random.randint(0, L, size=Z.shape[1])
    model.initialize(Z, W, M, N, initial_R, initial_C,K,L,semi_supervised)
    if learner.upper()=='LOGISTIC':
        train_loss = 'negloglik'
        if reg_op_learner is None:
            param={'tol':1e-4, 'fit_intercept':True, 'C':1.0}
        else:
            param={'tol':1e-4, 'fit_intercept':True, 'C':reg_op_learner}            
    elif learner.upper()=='SGDLOGISTIC':
        train_loss = 'negloglik'
        if reg_op_learner is None:
            param={'loss':'log','warm_start':True,'n_iter':10,'alpha':0.02}
        else:
            param={'loss':'log','warm_start':True,'n_iter':10,'alpha':reg_op_learner}        
    else:
        if reg_op_learner is None:
            param={'alpha':0.1}
        else:
            param={'alpha':reg_op_learner}
    model.init_learner(learner, param, train_loss, test_loss)
        
    if semi_supervised:
        if ss_learner == 'logistic':
            if reg_ss_model is None:
                param={'tol':1e-4, 'fit_intercept':False}
                model.init_ss_learner(ss_learner, param)
            else:
                param={'tol':1e-4, 'fit_intercept':False, 'C':reg_ss_model}
        elif ss_learner == 'kMeans':
            param = [{'k':K},{'k':L}]
        model.init_ss_learner(ss_learner, param)
    n_convg = 0
    while True:
        model.train() # Added the semi-supervised Training to the train method
        model.update_row_assignments() #Changed cost for update row and col
        model.update_col_assignments()
        if old_objective - model.objective < delta_convg:
            n_convg +=1
        else:
            n_convg = 0
        if n_convg >4:
            break
        old_objective = model.objective
        obj.append(old_objective)
        print old_objective
    if __debug__:
        print "Final objective value: %f" % (model.objective,)
    return obj

def test_row_split(model, Z_training, W_training, Z_validation, W_validation, K, L, learner, param, train_loss, test_loss):
    M,N = W_training.shape
    initial_C = model.C
    initial_R = find_and_split_row_cluster(model)
    model.initialize(Z_training, W_training, M, N, initial_R, initial_C)
    model.init_learner(learner, param, train_loss, test_loss)
    old_objective = model.objective
    while True:
        model.train()
        model.update_row_assignments()
        model.update_col_assignments()
        if old_objective - model.objective < convergence_threshold:
            break
        old_objective = model.objective
    validation_I, validation_J = sp.find(W_validation)[:2]
    test_Z = np.ravel(Z_validation[(validation_I, validation_J)])
    validation_loss = model.test_loss(test_Z, model.predict(validation_I, validation_J) )
    return validation_loss

def test_col_split(model, Z_training, W_training, Z_validation, W_validation, K, L, learner, param, train_loss, test_loss):
    M,N = W_training.shape
    initial_R = model.R
    initial_C = find_and_split_col_cluster(model)
    model.initialize(Z_training, W_training, M, N, initial_R, initial_C)
    model.init_learner(learner, param, train_loss, test_loss)
    old_objective = model.objective
    while True:
        model.train()
        model.update_row_assignments()
        model.update_col_assignments()
        if old_objective - model.objective < convergence_threshold:
            break
        old_objective = model.objective
    validation_I, validation_J = sp.find(W_validation)[:2]
    test_Z = np.ravel(Z_validation[(validation_I, validation_J)])
    validation_loss = model.test_loss(test_Z, model.predict(validation_I, validation_J) )
    return validation_loss

def find_and_split_row_cluster(model):
    cluster_errors = []
    for r in range(model.K):
        filtered_I = model.I[model.R[model.I] == r]
        filtered_J = model.J[model.R[model.I] == r]
        filtered_Z = model.Z[model.R[model.I] == r]
        # Use the mean cluster error, not the total (so divide by len(filtered_I))
        if len(filtered_I)>0:
            loss = model.train_loss(filtered_Z, model.predict(filtered_I, filtered_J))
            cluster_errors.append(loss)
        else: # append 0
            cluster_errors.append(np.zeros(1))
    # This selects the cluster based on the overall (pointwise) average error for the cluster, not the average error for the rows in the cluster
    max_error_cluster = np.argmax(map(np.mean, cluster_errors))
    filtered_I = model.I[model.R[model.I] == max_error_cluster]
    bin_count_rows = np.bincount(filtered_I)
    row_errors = np.zeros(model.M)
    row_errors[bin_count_rows > 0] = np.bincount(filtered_I, cluster_errors[max_error_cluster])[bin_count_rows > 0] / bin_count_rows[bin_count_rows > 0]
    dummy_I, dummy_J, nonzero_row_errors = sp.find(row_errors)
    nonzero_row_errors.sort()
    median_error_value = nonzero_row_errors[len(nonzero_row_errors)/2]
    initial_R = model.R
    initial_R[row_errors > median_error_value] = model.K
    return initial_R

def find_and_split_col_cluster(model):
    cluster_errors = []
    for c in range(model.L):
        filtered_I = model.I[model.C[model.J] == c]
        filtered_J = model.J[model.C[model.J] == c]
        filtered_Z = model.Z[model.C[model.J] == c]
        if len(filtered_I)>0:
            loss = model.train_loss(filtered_Z, model.predict(filtered_I, filtered_J))
            cluster_errors.append(loss)
        else: # append 0
            cluster_errors.append(np.zeros(1))    
    # This selects the cluster based on the overall (pointwise) average error for the cluster, not the average error for the rows in the cluster
    max_error_cluster = np.argmax(map(np.mean, cluster_errors))
    filtered_J = model.J[model.C[model.J] == max_error_cluster]
    bin_count_cols = np.bincount(filtered_J)
    col_errors = np.zeros(model.N)
    col_errors[bin_count_cols > 0] = np.bincount(filtered_J, cluster_errors[max_error_cluster])[bin_count_cols > 0] / bin_count_cols[bin_count_cols > 0]
    dummy_I, dummy_J, nonzero_col_errors = sp.find(col_errors)
    nonzero_col_errors.sort()
    median_error_value = nonzero_col_errors[len(nonzero_col_errors)/2]
    initial_C = model.C
    initial_C[col_errors > median_error_value] = model.L
    return initial_C

def parse_list(option, opt_str, value, parser):
    list_out = [float(x) for x in value.split(',')]
    setattr(parser.values, option.dest, list_out)
    
def read_options_file():
    ''' checks/sets options read from command line'''
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    # input files
    parser.add_option("-i", "--input_directory", dest="indir", help="input data directory (REQUIRED if OBS_FILE not set)", default=default_indir)
    parser.add_option("-o", "--obs_file", dest="obs_file", help="(REQUIRED if not using default) observations file (overrides file location in INDIR)", default=None)
    parser.add_option("-w", "--weight_file", dest="weight_file", help="(Optional) weights file (overrides file location in INDIR)", default=None)
    parser.add_option("-r", "--row_file", dest="row_file", help="(Optional) row attributes file (overrides file location in INDIR)", default=None)
    parser.add_option("-c", "--col_file", dest="col_file", help="(Optional) col attributes file (overrides file location in INDIR)", default=None)
    parser.add_option("-x", "--cross_file", dest="cross_file", help="(Optional) cross attributes file (overrides file location in INDIR)", default=None)
    # learner
    parser.add_option("-m", "--model", dest="learner", help="(Optional) Scikits learner model", default="ridge")
    parser.add_option("-t", "--train_loss", dest="test_loss", help="(Optional) train loss type", default="sq_err")
    parser.add_option("-e", "--test_loss", dest="train_loss", help="(Optional) test loss type", default="mse")
    parser.add_option("-n", "--num_cv", dest="num_cv", help="(Optional) number of cross validation sets", type="int", default=default_cv)
    # TODO: add robust learner options 
    # CV options
    parser.add_option("-k", "--init_K", dest="init_K", help="(Optional) initial K", type="int", default=default_K)
    parser.add_option("-l", "--init_L", dest="init_L", help="(Optional) initial L", type="int", default=default_L)
    parser.add_option("-a", "--alphas", dest="alphas", help="(Optional) list of alphas, comma separated", \
                      type="string", action="callback", callback=parse_list, default=None)
    parser.add_option("--plain", dest="model_sel", action="store_false", default=True)
    parser.add_option("--model_filename", dest="model_filename", help="Filename where to save model", metavar="FILENAME", default=None)
    (options, args) = parser.parse_args()
    
    indir = options.indir
    if options.obs_file is None:
        obs_file = indir+'/observations.mtx'
    else:
        obs_file = options.obs_file
    if options.weight_file is None:
        weight_file = indir+'/weights.mtx'
    else:
        weight_file = options.weight_file
    if options.row_file is None:
        row_file = indir+'/rowAttr.mtx'
    else:
        row_file = options.row_file
    if options.col_file is None:
        col_file = indir+'/colAttr.mtx'
    else:
        col_file = options.col_file
    if options.cross_file is None:
        cross_file = indir+'/crossAttr.mtx'
    else:
        cross_file = options.cross_file
    
    result = {}
    try:
        result["Z"] = scipy.io.mmread(obs_file)
    except:
        print >>sys.stderr, "No observation file found at %s" % obs_file
        sys.exit()
    try:
        result["W"] = scipy.io.mmread(weight_file)
    except:
        pass
    try:
        result["rowAttr"] = scipy.io.mmread(row_file)
    except:
        pass
    try:
        result["colAttr"] = scipy.io.mmread(col_file)
    except:
        pass
    try:
        result["crossAttr"] = scipy.io.mmread(cross_file)
    except:
        pass
         
    result["learner"] = options.learner
    result["train_loss"] = options.train_loss
    result["test_loss"] = options.test_loss
    result["num_cv"]  = options.num_cv
    result["K"]  = options.init_K
    result["L"]  = options.init_L
    if options.model_filename is not None:
        result["model_filename"] = options.model_filename
    result["model_sel"] = options.model_sel
    if options.alphas is not None:
        result["alphas"]  = options.alphas

    return result

if '__main__' == __name__:
    if len(sys.argv) == 1:
        print "Incorrect args.  Please run learn_scoal.py -h for help on arguments"
        sys.exit(1)
    options = read_options_file()
    if options['model_sel']:
        model_selector( **options )
    else:
        learn_scoal_wrapper( **options )
    #sys.exit(model_selector( **read_options_file() ))
