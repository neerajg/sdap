'''
Created on Nov 22, 2012

@author: neeraj
'''
def getParameters(param_file):  
    parameters = open(param_file,'r')
    testcases = []
    models = []
    submodels = []
    datasets = []
    """reg_betas = []
    reg_alphas1 = []
    reg_alphas2 = []"""
    Ls = []
    Ks = []
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
    parameters.close()
    
    return testcases, models, submodels, datasets, Ls, Ks