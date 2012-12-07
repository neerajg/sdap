'''
Author - Neeraj Gaur
Title - Reads the features and ratings for the complete movielens data

'''
import sys
#import recfile as rf
import numpy as np

from DATA.misc_tools import read_glob, sanitize_noncell_data

def geteHarmonyData(dataset):
    return getData(dataset)

def getData(dataset):
    pass  
    '''dir = '/workspace/sdap/data/eHarmony/'
    feature_files = ['EH-slice1-data.csv','EH-slice2-data.csv']
    ratings_files = ['EH-slice1-labels.csv','EH-slice2-labels.csv']
    num_users = [274654,211810]
    if dataset == 1:
        feature_file = feature_files[0]
        ratings_file = ratings_files[0]
        num_users = num_users[0]
    elif dataset == 2:
        feature_file = feature_files[1]
        ratings_file = ratings_files[1]
        num_users = num_users[1]
    
    feature_file = dir+feature_file
    ratings_file = dir+ratings_file
    #X = np.loadtxt(feature_file,dtype=float,delimiter=',')
    col_list = range(0,59)
    dtype = []
    for x in col_list:
        dtype.append((str(x),'f8'))
    robj = rf.Open(feature_file, delim=',', dtype=dtype, nrows=num_users)
    col_list = col_list[1:56]
    data = robj.read(columns=col_list)
    X = np.empty(shape=(num_users,len(col_list)))
    for row_index in range(num_users):
        X[row_index,:] = np.array(list(data[row_index]))
    Xs = [X,X]
    
    robj = rf.Open(ratings_file, delim=',', dtype=[('0',int),('1',int),('2',int)])
    data = robj.read()
    Y = np.empty(shape=(len(data),),dtype=int)
    I = np.empty_like(Y)
    J = np.empty_like(Y)
    for m in range(len(data)):
        Y[m],I[m],J[m] = list(data[m])
    I = I-1
    J = J-1
    #ratings = np.loadtxt(ratings_file,dtype=float,delimiter=',')
    
    return Xs,Y,I,J'''

if __name__ == '__main__':
    geteHarmonyData(1)
    #sys.exit()


'''    parser.add_option('--train_Y', dest='train_y_file', default='ratings_train.dat', help='Affinities training data input file', metavar='TRAIN_Y_FILE')
    parser.add_option('--val_Y', dest='val_y_file',default='ratings_validation.dat', help='Affinities validation data input file', metavar='TRAIN_Y_FILE')    
    parser.add_option('--test_Y', dest='test_y_file', default='ratings_test.dat', help='Affinities testing data input file', metavar='TEST_Y_FILE')'''