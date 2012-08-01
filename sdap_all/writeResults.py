'''
Created on May 17, 2012

Author - Neeraj Gaur
Title - Writes the results in the appropriate directory

'''
import sys
import platform
import os

def makeDir(model_name, datasetName):
    
    if platform.system() == 'Windows':
        base_dir = 'D:'
    elif platform.system() == 'Linux':
        base_dir = '/workspace'
            
    results_dir = base_dir + '/sdap/results/'
    
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    
    model_results = results_dir + model_name + '/'
    if not os.path.isdir(model_results):
        os.mkdir(model_results)
    model_results = model_results + datasetName + '/'
    if not os.path.isdir(model_results):
        os.mkdir(model_results)    
                             
    return model_results

def writeHotStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, rmse, data_set, length, reg_beta,  reg_alpha1, reg_alpha2):
    results_dir = model_results + 'hotStart/'
    writeRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, rmse, data_set, length, results_dir, reg_beta,  reg_alpha1, reg_alpha2)
    return

def writeWarmStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, rmse, data_set, length):
    results_dir = model_results + 'warmStart/'
    writeRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, rmse, data_set, length, results_dir)
    return    

def writeColdStartRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, rmse, data_set, length):
    results_dir = model_results + 'coldStart/'
    writeRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, rmse, data_set, length, results_dir)
    return

def writeRMSE(K, L, k_fold, pctg_users, pctg_movies, model_results, rmse, data_set, length, results_dir, reg_beta,  reg_alpha1, reg_alpha2):
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    file_name = results_dir+'K_'+str(K)+'_L_'+str(L)+'_folds_'+str(k_fold)+'_pctg_users_'+str(pctg_users)+'_pctg_movies_'+str(pctg_users)+'_reg_beta_'+str(reg_beta)+'_reg_alpha1_'+str(reg_alpha1)+'_reg_alpha2_'+str(reg_alpha2)+'_.dat'
    if data_set == 'train_0':
        ofile = open(file_name,'w')
    else:
        ofile = open(file_name,'a')
    print>> ofile, data_set +'_'+ str(rmse) + '_'+str(length)
    ofile.close()
    return

if __name__ == '__main__':
    sys.exit()