'''
Created on Jul 25, 2012

@author: neeraj
'''

# TO DO: Convert this to directly write excel files using xlrd, xlwt, xlutils
import os, sys, platform, shutil
import numpy as np

def calc_rmse(rmses,sizes):
    rmses = np.array(rmses)
    sizes = np.array(sizes)
    mult = np.multiply(rmses,sizes)
    rmse = np.sum(mult)/np.sum(sizes) 
    stddev = np.sqrt(np.sum(np.multiply((rmses - rmse)**2,sizes))/np.sum(sizes))
    return rmse, stddev
    
def analyzeRMSE():

    if platform.system() == 'Windows':
        base_dir = 'D:'
    elif platform.system() == 'Linux':
        base_dir = '/workspace'
    
    results_dir = base_dir+'/sdap/results/'
    
    models = os.listdir(results_dir)
    
    for model in models:
        model_dir = results_dir + model +'/'
        datasets = os.listdir(model_dir)
        for dataset in datasets:
            dataset_dir = model_dir + dataset + '/'
            testcases = os.listdir(dataset_dir)
            for testcase in testcases:
                testcase_dir = dataset_dir + testcase + '/'
                analyzed_dir = testcase_dir + 'analyzedResults/'
                if os.path.isdir(analyzed_dir):
                    shutil.rmtree(analyzed_dir)
                os.mkdir(analyzed_dir)
                results_files = os.listdir(testcase_dir)
                for file_current in results_files:
                    fs = file_current.split('_')                    
                    if len(fs)<2:
                        continue
                    file_name = testcase_dir + file_current
                    train_rmses = []
                    val_rmses = []
                    ifile = open(file_name,'r')
                    total_size_train = []
                    total_size_val = []
                    for line in ifile:
                        ls = line.split('_')
                        rmse = float(ls[2])
                        size = float(ls[3])
                        if ls[0] == 'train':
                            total_size_train.append(size)
                            train_rmses.append(rmse)
                            continue
                        if ls[0] == 'val':
                            total_size_val.append(size)
                            val_rmses.append(rmse)
                    ifile.close()
            
                    val_rmse, val_std_dev = calc_rmse(val_rmses,total_size_val)
                    train_rmse, train_std_dev = calc_rmse(train_rmses,total_size_train)

                    ofilename = analyzed_dir + 'K_'+fs[1]+'_L_'+fs[3]+'_folds_'+fs[5]+'.xls'
                    regs = np.empty(((len(fs)-13)/2,))
                    for i in range(len(regs)):
                        regs[i] = fs[13+2*i]
                    if not os.path.isfile(ofilename):
                        ofile = open(ofilename,'w')
                        op_str = ''
                        for i in range(len(regs)):
                            op_str = op_str+'Reg'+str(i+1)+'\t'
                        print>>ofile,op_str+'Train RMSE(Mean)\tTrain RMSE(std dev)\tValidation RMSE(mean)\tValidation RMSE(std dev)'
                    else:
                        ofile = open(ofilename,'a')
                    op_str = ''
                    for i in range(len(regs)):
                        op_str = op_str+str(regs[i])+'\t'                        
                    print>>ofile,op_str+str(train_rmse)+'\t'+str(train_std_dev)+'\t'+str(val_rmse)+'\t'+str(val_std_dev)
                    ofile.close()

if __name__ == '__main__':
    analyzeRMSE()
    sys.exit()