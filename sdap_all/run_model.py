'''
Created on May 19, 2012

Author - Neeraj Gaur
Title - Runs the appropriate model depending on the input string

'''
import sys, os

sys.path.append(os.path.abspath('./SCOAL/'))
sys.path.append(os.path.abspath('./BAE/'))
sys.path.append(os.path.abspath('./MMBAE/'))

import run_scoal as scoal
import run_bae as bae
import run_mmbae as mmbae

def runModelHotStart(model_name, K, L, X1, X2, train_I, train_J, train_Y, reg_lambda, num_iter, delta_convg, reg_alpha1 = None, reg_alpha2 = None):
    # Train the model and return the parameters and objective function
    model = model_name.split('_')[0].upper()
    if model == 'SCOAL':
        if model_name.split('_')[1].upper() == 'LINEAR':
            learner = 'ridge' 
        train_op = scoal.run_scoal(K, L, X1, X2, train_I, train_J, train_Y, learner, reg_lambda)
    if model == 'BAE':
        if model_name.split('_')[1].upper() == 'LINEAR':
            learner = 'linear'
            reg_beta = reg_lambda
        train_op = bae.run_bae(K, L, X1, X2, train_I, train_J, train_Y, learner, reg_beta, num_iter, delta_convg, reg_alpha1, reg_alpha2)
    if model == 'MMBAE':
        if model_name.split('_')[1].upper() == 'LINEAR':
            learner = 'linear'
            reg_beta = reg_lambda
        train_op = mmbae.run_mmbae(K, L, X1, X2, train_I, train_J, train_Y, learner, reg_beta, num_iter, delta_convg, reg_alpha1, reg_alpha2)            

    return train_op

def runModelWarmStart(model_name, K, L, X1, X2, train_I, train_J, train_Y, reg_lambda):
    # Train the model and return the parameters and objective function
    model = model_name.split('_')[0].upper()
    if model == 'SCOAL':
        if model_name.split('_')[1].upper() == 'LINEAR':
            learner = 'ridge'  
        train_op = scoal.run_scoal(K, L, X1, X2, train_I, train_J, train_Y, learner, reg_lambda)       
    return train_op

def runModelColdStart(model_name, K, L, X1, X2, train_I, train_J, train_Y, reg_lambda):
    # Train the model and return the parameters and objective function
    model = model_name.split('_')[0].upper()
    if model == 'SCOAL':
        if model_name.split('_')[1].upper() == 'LINEAR':
            learner = 'ridge'
        train_op = scoal.run_scoal(K, L, X1, X2, train_I, train_J, train_Y, learner, reg_lambda)       
    return train_op

def calcHotStartTrainRMSE(model_name, K, L, X1, X2, train_I, train_J, train_Y, train_op):
    # Predict the Training Set and return the Prediction RMSE
    model_name = model_name.upper()
    if model_name.upper().split('_')[0] == 'SCOAL': 
        hotStartTrainRMSE = scoal.hotStartTrainRMSE(K, L, X1, X2, train_I, train_J, train_Y, train_op)
    if model_name.upper().split('_')[0] == 'BAE':
        hotStartTrainRMSE = bae.hotStartTrainRMSE(model_name,K, L, X1, X2, train_I, train_J, train_Y, train_op)
    return hotStartTrainRMSE

def calcWarmStartTrainRMSE(model_name, K, L, X1, X2, train_I, train_J, train_Y, train_op):
    # Predict the Training Set and return the Prediction RMSE
    model_name = model_name.upper()
    if model_name.upper().split('_')[0] == 'SCOAL': 
        warmStartTrainRMSE = scoal.warmStartTrainRMSE(K, L, X1, X2, train_I, train_J, train_Y, train_op)
    return warmStartTrainRMSE

def calcColdStartTrainRMSE(model_name, K, L, X1, X2, train_I, train_J, train_Y, train_op):
    # Predict the Training Set and return the Prediction RMSE
    model_name = model_name.upper()
    if model_name.upper().split('_')[0] == 'SCOAL':
        coldStartTrainRMSE = scoal.coldStartTrainRMSE(K, L, X1, X2, train_I, train_J, train_Y, train_op)
    return coldStartTrainRMSE

def calcHotStartValRMSE(model_name, K, L, X1, X2, train_I, train_J, train_Y, train_op):
    # Predict the Training Set and return the Prediction RMSE
    model_name = model_name.upper()
    if model_name.upper().split('_')[0] == 'SCOAL': 
        hotStartValRMSE = scoal.hotStartValRMSE(K, L, X1, X2, train_I, train_J, train_Y, train_op)
    if model_name.upper().split('_')[0] == 'BAE':
        hotStartValRMSE = bae.hotStartValRMSE(model_name,K, L, X1, X2, train_I, train_J, train_Y, train_op)        
    return hotStartValRMSE

def calcWarmStartValRMSE(model_name, K, L, X1, X2, train_I, train_J, train_Y, train_op, centroids):
    # Predict the Training Set and return the Prediction RMSE
    model_name = model_name.upper()
    if model_name.upper().split('_')[0] == 'SCOAL': 
        warmStartValRMSE = scoal.warmStartValRMSE(K, L, X1, X2, train_I, train_J, train_Y, train_op, centroids)
    return warmStartValRMSE

def calcColdStartValRMSE(model_name, K, L, X1, X2, train_I, train_J, train_Y, train_op, centroids):
    # Predict the Training Set and return the Prediction RMSE
    model_name = model_name.upper()
    if model_name.upper().split('_')[0] == 'SCOAL': 
        coldStartValRMSE = scoal.coldStartValRMSE(K, L, X1, X2, train_I, train_J, train_Y, train_op, centroids)
    return coldStartValRMSE

if __name__ == '__main__':
    sys.exit()