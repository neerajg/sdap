'''
Created on Nov 22, 2012

@author: neeraj
'''

import platform,sys

def getParameters(param_file):  
    parameters = open(param_file,'r')
    parameter_sets = []
    testcases = []
    models = []
    submodels = []
    datasets = []
    Ls = []
    Ks = []
    regularizers = []
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
                    Ks.append(int(K.strip()))
            continue
        if parameter.split(' ')[0] == 'L':
            for L in parameter.strip().split(' '):
                if L == 'L':
                    continue
                else:
                    Ls.append(int(L.strip()))
            continue
        if parameter.split(' ')[0].startswith('reg'):
            reg = []
            for regularizer in parameter.strip().split(' '):
                if regularizer.startswith('reg'):
                    continue
                else:
                    reg.append(float(regularizer.strip()))
            regularizers.append(reg)            
                 
    parameters.close()
    for testcase in testcases:
        for dataset in datasets:
            for K in Ks:
                for L in Ls:
                    for model in models:
                        if model.upper()=='SCOAL':
                            sub_models_to_consider = list(set([x.split('+')[0] for x in submodels]))
                            if regularizers:
                                reg_sets_toconsider = list(set(regularizers[0]))
                        else:
                            sub_models_to_consider = submodels
                            if regularizers:
                                reg_sets_toconsider = [[x,y]for x in regularizers[0] for y in regularizers[1]]                                
                        for submodel in sub_models_to_consider:
                            if model.upper()=='SS-SCOAL'and submodel.upper().split('+')[1]=='KMEANS':
                                if regularizers:
                                    reg_sets = [[x,x]for x in list(set(regularizers[0]))]
                            else:
                                reg_sets = reg_sets_toconsider
                            for reg_set in reg_sets:
                                parameter_set = {'test':testcase,
                                                 'datasetName':dataset,
                                                 'K':K,
                                                 'L':L,
                                                 'model':model,
                                                 'submodel':submodel,
                                                 'regs':reg_set
                                                 }
                                parameter_sets.append(parameter_set)
    return parameter_sets

if __name__ == '__main__':
    # Get Parameters
    if platform.system() == 'Windows':
        param_file = 'D:/sdap/code/run_models_parameters.txt'
    elif platform.system() == 'Linux':
        param_file = '/home/neeraj/sdap/code/run_models_parameters.txt'
        
    parameter_sets = getParameters(param_file)
    sys.exit()