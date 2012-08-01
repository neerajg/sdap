'''
Created on May 17, 2012

Author - Neeraj Gaur
Title - Runs the models.

'''

import sys,platform
import time
import getData as data
import writeResults as op
import run_model as runModel

import os
sys.path.append(os.path.abspath('./BAE/'))
import bae_linear as bae_linear
import numpy as np
from  analyzeRMSE import analyzeRMSE

# TO DO : for now we are using hard assignment for the attribute clusters for the warm and cold start cases
# Change this to soft assignment for better performance later on

# TO DO : Right now we are only using linear models, change that to include others as well

def run_models(datasetName,test, model_name,K,L,reg_lambda,reg_alpha1,reg_alpha2,delta_convg,num_iter,k_fold,pctg_users,pctg_movies,M = 2000, N = 10000, D1 = 3, D2 = 4, no_obs_mult = 0.04):
    print "START"
    
    # Get the Data
    if datasetName.upper().split('_')[0] != 'TEST':    
        dataSet, centroids = data.getRatingsTestData(k_fold, pctg_users, pctg_movies, K, L, datasetName)
    if datasetName.upper().split('_')[0] == 'TEST':
        no_obs = int(M*N*no_obs_mult)
        dataSet, centroids = data.getRatingsTestData(k_fold, pctg_users, pctg_movies, K, L, datasetName, M, N, D1, D2, no_obs)
    trctd_X1 = dataSet['trctd_X1']
    trctd_X2 = dataSet['trctd_X2']
    
    model_results = op.makeDir(model_name, datasetName)
            
    if test == 'all' or test == 'hotStart':
        for fold in range(k_fold):            
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
            train_op = runModel.runModelHotStart(model_name, K, L, trctd_X1, trctd_X2, train_I, train_J, train_Y, reg_lambda, num_iter, delta_convg, reg_alpha1,reg_alpha2,)
            
            #alphas = train_op['params']['alphas']
            #betas = train_op['params']['betas']
            #gammas = train_op['params']['gammas']
            #r = train_op['params']['r']
            #X1 = train_op['params']['Xs'][0]
            #X2 = train_op['params']['Xs'][1]
            #Xbias = np.ones((len(train_I),1)) # |Yobs| x 1
            #Xusers = X1[train_I,:] # |Yobs| x D1
            #Xitems = X2[train_J,:] # |Yobs| x D2
            #X_composite = np.hstack((Xbias, Xusers, Xitems)) # |Yobs| x (1 + D1 + D2)
            #M = X1.shape[0]
            #N = X2.shape[0]
            #bae_linear.get_log_likelihood_lower_bound(K, L, M, N, alphas, betas, gammas, r, train_I, train_J, train_Y, X_composite)
            print "  DONE TRAINING "+model_name+" FOR FOLD "+str(fold+1)   
            
            # Calculate Hot Start Training RMSE
            hotStartTrainRMSE = runModel.calcHotStartTrainRMSE(model_name, K, L, trctd_X1, trctd_X2, train_I, train_J, train_Y, train_op)
            op.writeHotStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, hotStartTrainRMSE, 'train_'+str(fold), len(train_Y), reg_beta, reg_alpha1, reg_alpha2)
        
            # Calculate Hot Start Validation Set RMSE
            hotStartValRMSE = runModel.calcHotStartValRMSE(model_name, K, L, trctd_X1, trctd_X2, val_I, val_J, val_Y, train_op)
            op.writeHotStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, hotStartValRMSE, 'val_'+str(fold), len(val_Y), reg_beta,  reg_alpha1, reg_alpha2)        
        
    if test == 'all' or test == 'warmStart':
        for fold in range(k_fold):      
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
            
            # Calculate Warm Start Training RMSE
            warmStartTrainRMSE = runModel.calcWarmStartTrainRMSE(model_name, K, L, trctd_X1, trctd_X2, train_I, train_J, train_Y, train_op)
            op.writeWarmStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, warmStartTrainRMSE, 'train_'+str(fold), len(train_Y), datasetName)                          

            # Calculate Warm Start Validation Set RMSE
            warmStartValRMSE = runModel.calcWarmStartValRMSE(model_name, K, L, trctd_X1, trctd_X2, val_I, val_J, val_Y, train_op, centroids)
            op.writeWarmStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, warmStartValRMSE, 'val_'+str(fold), len(val_Y))        

    if test == 'all' or test == 'coldStart':
        for fold in range(k_fold):
            # Get Data Folds
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
            train_op = runModel.runModelColdStart(model_name, K, L, trctd_X1, trctd_X2, train_I, train_J, train_Y, reg_beta)
            print "  DONE TRAINING "+model_name+" FOR FOLD "+str(fold+1)
            
            # Calculate Cold Start Training RMSE
            coldStartTrainRMSE = runModel.calcColdStartTrainRMSE(model_name, K, L, trctd_X1, trctd_X2, train_I, train_J, train_Y, train_op)
            op.writeColdStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, coldStartTrainRMSE, 'train_'+str(fold), len(train_Y))                          

            # Calculate Cold Start Validation Set RMSE
            coldStartValRMSE = runModel.calcColdStartValRMSE(model_name, K, L, trctd_X1, trctd_X2, val_I, val_J, val_Y, train_op, centroids)
            op.writeColdStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, coldStartValRMSE, 'val_'+str(fold), len(val_Y))
        
        # Code for imputing attributes and testing the performance in that respect
        # Write the parameters, lambda value, and obj function over iterations to file for each of the test cases

    print "  DONE CROSS-VALIDATION FOR CURRENT SET OF PARAMETERS"
    print "END"

if __name__ == '__main__':
    
    if platform.system() == 'Windows':
        param_file = 'D:/sdap/code/run_models_parameters.txt'
    elif platform.system() == 'Linux':
        param_file = '/workspace/sdap/code/run_models_parameters.txt'    
    parameters = open(param_file,'r')
    testcases = []
    models = []
    submodels = []
    datasets = []
    reg_betas = []
    reg_alphas1 = []
    reg_alphas2 = []    
    for parameter in parameters:
        if parameter.split(' ')[0] == 'test':
            for testcase in parameter.strip().split(' '):
                if testcase == 'test':
                    continue
                else:
                    testcases.append(testcase.strip())
            continue
        if parameter.split(' ')[0] == 'models':
            for model in parameter.strip().split(' '):
                if model == 'models':
                    continue
                else:
                    models.append(model.strip())
            continue
        if parameter.split(' ')[0] == 'submodels':
            for submodel in parameter.strip().split(' '):
                if submodel == 'submodels':
                    continue
                else:
                    submodels.append(submodel.strip())
            continue    
        if parameter.split(' ')[0] == 'datasets':
            for dataset in parameter.strip().split(' '):
                if dataset == 'datasets':
                    continue
                else:
                    datasets.append(dataset.strip())
            continue 
        if parameter.split(' ')[0] == 'K':
            for K in parameter.strip().split(' '):
                if K == 'K':
                    continue
                else:
                    K = int(K)
            continue
        if parameter.split(' ')[0] == 'L':
            for L in parameter.strip().split(' '):
                if L == 'L':
                    continue
                else:
                    L = int(L.strip())
            continue
        if parameter.split(' ')[0] == 'reg_beta':
            for reg_beta in parameter.strip().split(' '):
                if reg_beta == 'reg_beta':
                    continue
                else:
                    reg_betas.append(float(reg_beta.strip()))
            continue
        if parameter.split(' ')[0] == 'reg_alpha1':
            for reg_alpha1 in parameter.strip().split(' '):
                if reg_alpha1 == 'reg_alpha1':
                    continue
                else:
                    reg_alphas1.append(float(reg_alpha1.strip()))
            continue
        if parameter.split(' ')[0] == 'reg_alpha2':
            for reg_alpha2 in parameter.strip().split(' '):
                if reg_alpha2 == 'reg_alpha2':
                    continue
                else:
                    reg_alphas2.append(float(reg_alpha2.strip()))
            continue        
    parameters.close()
    if K == L and L == 1:
        reg_alphas1 = [0]
        reg_alphas2 = [0]
    for datasetName in datasets:
        for test in testcases:
            for model in models:
                if model =='scoal':
                    reg_alphas1 = [0]
                    reg_alphas2 = [0]
                for submodel in submodels:
                    for reg_beta in reg_betas:
                        for reg_alpha1 in reg_alphas1:
                            for reg_alpha2 in reg_alphas2:
                                print reg_alpha1, reg_alpha2
                                model_name = model+'_'+submodel
                                t = time.time()                               
                                run_models(datasetName,test, model_name,K,L,reg_beta,reg_alpha1, reg_alpha2,delta_convg = 1e-2,num_iter = 60,k_fold = 10,pctg_users= 100,pctg_movies = 100)  
                                elapsed = time.time() - t
                                print 'ELAPSED TIME' + str(elapsed)
    
    # Analyze results
    analyzeRMSE()

    sys.exit()