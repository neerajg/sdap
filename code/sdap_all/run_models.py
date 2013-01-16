'''
Created on May 17, 2012

Author - Neeraj Gaur
Title - Runs the models.

'''

import sys,platform
import time
import DATA.getData as data
import writeResults as op
import run_model as runModel
from  analyzeRMSE import analyzeRMSE
from getParameters import getParameters
from multiprocessing import Array
from ctypes import c_double, c_int
from numpy.ctypeslib import as_array
from itertools import repeat, izip, product
from multiprocessing import Pool

# TO DO : for now we are using hard assignment for the attribute clusters for the warm and cold start cases
# Change this to soft assignment for better performance later on

def test_mult(parameter_set):
    a = 2
    print a
    return

def run_parameter_set(parameter_set):
    K_R = {}
    L_C = {}
    K = parameter_set['K']
    L = parameter_set['L']
    datasetName = parameter_set['datasetName']
    test = parameter_set['test']
    model = parameter_set['model']
    submodel = parameter_set['submodel']     
    regs = parameter_set['regs']
    if not regs:
        regs = None             
    model_name = model+'_'+submodel
    t = time.time()                           
    K_R,L_C = run_models(datasetName,test, model_name,K,L,regs=regs,delta_convg = 1e-4,num_iter = 40,k_fold = 10,pctg_users= 100,pctg_movies = 100,K_R=K_R,L_C=L_C)  
    elapsed = time.time() - t
    print 'ELAPSED TIME' + str(elapsed) 

def run_models(datasetName,test, model_name,K,L,delta_convg,
               num_iter,k_fold,pctg_users,pctg_movies,regs = None,
               testtype=3,M = 2000, N = 10000, D1 = 3, D2 = 4, no_obs_mult = 0.04,K_R=None,L_C=None):
    print "START"
    if K_R is None:
        K_R = {}
    if L_C is None:
        L_C = {}
    K_R = {}
    L_C = {}
    # Get the Data
    if datasetName.upper().split('_')[0] != 'TEST':    
        dataSet, centroids = data.getRatingsTestData(k_fold, pctg_users, pctg_movies, K, L, datasetName)
    if datasetName.upper().split('_')[0] == 'TEST':
        no_obs = int(M*N*no_obs_mult)
        dataSet, centroids = data.getRatingsTestData(k_fold, pctg_users, pctg_movies, K, L, datasetName, M, N, D1, D2, no_obs)
    trctd_X1 = dataSet['trctd_X1']
    trctd_X2 = dataSet['trctd_X2']
    
    model_results = op.makeDir(model_name, datasetName)
    set_folds = [0,1,2,3,4,5,6,7,8,9]
            
    if test == 'all' or test == 'hotStart':
        for fold in set_folds:
        #for fold in range(k_fold):            
            # Get Data Folds
            hotStartDataFold = data.getHotStartDataFolds(fold, dataSet)
            train_I = hotStartDataFold['train_I']
            train_J = hotStartDataFold['train_J']
            train_Y = hotStartDataFold['train_Y']
            params = dataSet['params']
            if params is not None:
                alphas = params['alphas']
                pis = params['pis']
                zs = params['zs']
                betas = params['betas']
        
            val_I = hotStartDataFold['val_I']
            val_J = hotStartDataFold['val_J']
            val_Y = hotStartDataFold['val_Y']        
                             
            # Run training for Hot Start
            print "  TRAINING hot start "+model_name+" FOR FOLD "+str(fold+1)
            #if datasetName.upper().split('_')[0] == 'TEST':
                #log_likelihood_art_data = data.get_likelihood_art_data(alphas, pis, zs, betas, train_I, train_J, train_Y, trctd_X1, trctd_X2, datasetName, model_name)    
            train_op = runModel.runModelHotStart(model_name, K, L, trctd_X1, trctd_X2, train_I, train_J, train_Y, num_iter, delta_convg,regs,K_R,L_C)
            train_op['centroids']=[]
            '''try:
                K_R[str(K)] = train_op['model'].R
                L_C[str(L)] = train_op['model'].C
            except NameError:
                K_R = {}
                L_C = {}'''            
            print "  DONE TRAINING "+model_name+" FOR FOLD "+str(fold+1) 
            
            # Calculate Hot Start Training RMSE
            hotStartTrainRMSE = runModel.calcHotStartTrainRMSE(model_name, K, L, trctd_X1, trctd_X2, train_I, train_J, train_Y, train_op)
            op.writeHotStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, hotStartTrainRMSE, 'train_'+str(fold), len(train_Y),train_op['regs'])
        
            # Calculate Hot Start Validation Set RMSE
            hotStartValRMSE = runModel.calcHotStartValRMSE(model_name, K, L, trctd_X1, trctd_X2, val_I, val_J, val_Y, train_op)
            print hotStartValRMSE, hotStartTrainRMSE
            op.writeHotStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, hotStartValRMSE, 'val_'+str(fold), len(val_Y),train_op['regs'])
        
    '''if test == 'all' or test == 'warmStart':
        #for fold in range(k_fold):
        for fold in set_folds:      
            # Get Data Folds
            warmStartDataFold = data.getWarmStartDataFolds(fold, dataSet, datasetName)
            train_I = warmStartDataFold['train_I']
            train_J = warmStartDataFold['train_J']
            train_Y = warmStartDataFold['train_Y']
        
            val_I = warmStartDataFold['val_I']
            val_J = warmStartDataFold['val_J']
            val_Y = warmStartDataFold['val_Y']
            
            # Run training for Warm Start
            print "  TRAINING warm start "+model_name+" FOR FOLD "+str(fold+1)       
            train_op = runModel.runModelWarmStart(model_name, K, L, trctd_X1, trctd_X2, train_I, train_J, train_Y, reg_beta)
            print "  DONE TRAINING "+model_name+" FOR FOLD "+str(fold+1)
            gc.collect()
            
            # Calculate Warm Start Training RMSE
            warmStartTrainRMSE = runModel.calcWarmStartTrainRMSE(model_name, K, L, trctd_X1, trctd_X2, train_I, train_J, train_Y, train_op)
            op.writeWarmStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, warmStartTrainRMSE, 'train_'+str(fold), len(train_Y), datasetName)                          

            # Calculate Warm Start Validation Set RMSE
            warmStartValRMSE = runModel.calcWarmStartValRMSE(model_name, K, L, trctd_X1, trctd_X2, val_I, val_J, val_Y, train_op, centroids)
            op.writeWarmStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, warmStartValRMSE, 'val_'+str(fold), len(val_Y))
            del train_op'''     

    if test == 'all' or test == 'coldStart':
        #for fold in range(k_fold):
        for fold in set_folds:
            # Get Data Folds
            K_R = {}
            L_C = {}
            coldStartDataFold = data.getColdStartDataFolds(fold, dataSet)
            train_I = coldStartDataFold['train_I']
            train_J = coldStartDataFold['train_J']
            train_Y = coldStartDataFold['train_Y']
            train_X1 = coldStartDataFold['train_X1']
            train_X2 = coldStartDataFold['train_X2']
        
            val_I = coldStartDataFold['val_I']
            val_J = coldStartDataFold['val_J']
            val_Y = coldStartDataFold['val_Y']
            val_X1 = coldStartDataFold['val_X1']
            val_X2 = coldStartDataFold['val_X2']            
            
            # Run training for Cold Start
            print "  TRAINING cold start "+model_name+" FOR FOLD "+str(fold+1)       
            train_op = runModel.runModelColdStart(model_name, K, L, trctd_X1, trctd_X2, train_I, train_J, train_Y, num_iter, delta_convg,regs,K_R,L_C)
            train_op['centroids']=centroids            
            print "  DONE TRAINING "+model_name+" FOR FOLD "+str(fold+1)
            
            # Calculate Cold Start Training RMSE
            coldStartTrainRMSE = runModel.calcColdStartTrainRMSE(model_name, K, L, trctd_X1, trctd_X2, train_I, train_J, train_Y, train_op)
            op.writeColdStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, coldStartTrainRMSE, 'train_'+str(fold), len(train_Y),train_op['regs'])                          

            # Calculate Cold Start Validation Set RMSE
            coldStartValRMSE = runModel.calcColdStartValRMSE(model_name, K, L, trctd_X1, trctd_X2, val_I, val_J, val_Y, train_op)
            print coldStartValRMSE, coldStartTrainRMSE            
            op.writeColdStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, coldStartValRMSE, 'val_'+str(fold), len(val_Y),train_op['regs'])
        
        # Code for imputing attributes and testing the performance in that respect
        # Write the parameters, lambda value, and obj function over iterations to file for each of the test cases

    print "  DONE CROSS-VALIDATION FOR CURRENT SET OF PARAMETERS"
    print "END"
    return K_R,L_C

if __name__ == '__main__':
    
    parallel_crossval = True
    # Get Parameters
    if platform.system() == 'Windows':
        param_file = 'D:/sdap/code/run_models_parameters.txt'
    elif platform.system() == 'Linux':
        param_file = '/home/neeraj/sdap/code/run_models_parameters.txt'
        
    parameter_sets = getParameters(param_file)
    if parallel_crossval is True:
        P = Pool(10)
        result = P.map_async(run_parameter_set,parameter_sets)
        #result = P.map(test_mult,parameter_sets)
        P.close()
        P.join() 
    else:
        for parameter_set in parameter_sets:
            run_parameter_set(parameter_set)
            
    '''for i in range(len(Ks)):
        K = Ks[i]
        L = Ls[i]
        for datasetName in datasets:
            for test in testcases:
                for model in models:
                    for submodel in submodels:                    
                        model_name = model+'_'+submodel
                        t = time.time()                               
                        run_models(datasetName,test, model_name,K,L,delta_convg = 1e-4,num_iter = 40,k_fold = 10,pctg_users= 100,pctg_movies = 100)  
                        elapsed = time.time() - t
                        print 'ELAPSED TIME' + str(elapsed)'''        
    
    # Analyze results
    analyzeRMSE()

    sys.exit()