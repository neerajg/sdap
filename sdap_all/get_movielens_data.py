'''
Author - Neeraj Gaur
Title - Reads the features and ratings for the complete movielens data

'''
import sys
import os.path
import numpy as np
import platform

from misc_tools import read_glob, sanitize_noncell_data

def getMovieLensData():
    
    Xs = [None, None, None]
    
    delimter = '\t'
    sec_delimiter = ','
    user_pattern = 'user*.dat'
    movie_pattern = 'movie*.dat'
    ratings_file = 'ratings_complete*'
    if platform.system() == 'Windows':
        base_dir = 'D:'
    elif platform.system() == 'Linux':
        base_dir = '/workspace'
            
    f_dir = base_dir + '/sdap/data/movielens/extracted_features/features'
    r_dir = base_dir + '/sdap/data/movielens/extracted_features/ratings'
        
    user_data = read_glob(os.path.join(f_dir, user_pattern), delimter, sec_delimiter)
    movie_data = read_glob(os.path.join(f_dir, movie_pattern), delimter, sec_delimiter)

    #user_filenames = user_data[4]
    #movie_filenames = movie_data[4]
    assert np.all(np.ravel(user_data[0]) == range(1,len(np.ravel(user_data[0]))+1))
    Xs[0] = sanitize_noncell_data(user_data[3]) # This basically puts the mean value to NaN and Infinity values

    assert np.all(np.ravel(movie_data[0]) == range(1,len(np.ravel(movie_data[0]))+1))
    Xs[1] = sanitize_noncell_data(movie_data[3])

    #cell_data = read_glob(os.path.join(options.feature_directory, options.cell_file_pattern), options.delimiter, options.secondary_delimiter, cell_feature=True)
    #feature_filenames.extend(cell_data[4])
    #assert np.all(np.sum(cell_data[2], axis=0)/cell_data[2].shape[0] == cell_data[2][0])
    #Xcells = sanitize_cell_data((np.ravel(cell_data[0])-1, np.ravel(cell_data[1])-1, cell_data[3]))
    #Xcells_inds_matrix = sp.coo_matrix((range(len(Xcells[0])), (Xcells[0], Xcells[1]))).tocsr()
    #M = Xs[0].shape[0]
    #N = Xs[1].shape[0]
    
    Y = read_glob(os.path.join(r_dir, ratings_file), delimter, sec_delimiter, cell_feature=True)  
    I = np.ravel(Y[0])-1
    J = np.ravel(Y[1])-1
    Y = np.ravel(Y[3])# ?? Clint added const 3 here

    return Xs,Y, I, J

''' if train:
        if full:
            train_Y = read_glob(os.path.join(options.rating_directory, options.train_y_file), options.delimiter, options.secondary_delimiter, cell_feature=True)
        else:
            train_Y = read_glob(os.path.join(options.reduced_rating_directory, options.train_y_file), options.delimiter, options.secondary_delimiter, cell_feature=True)
        train_I = np.ravel(train_Y[0])-1
        train_J = np.ravel(train_Y[1])-1
        vals = np.ravel(train_Y[3])# ?? Clint added const 3 here
        train_Y = sp.coo_matrix((vals, (train_I,train_J)),shape=(M,N)).tocsr()
        #Xs[2] = (train_I, train_J, Xcells[2][np.ravel(Xcells_inds_matrix[train_I,train_J])])
        I = train_I
        J = train_J
        #Y = train_Y
        Y=vals

    return (Xs,I,J,Y)  '''  

if __name__ == '__main__':
    getMovieLensData()
    sys.exit()


'''    parser.add_option('--train_Y', dest='train_y_file', default='ratings_train.dat', help='Affinities training data input file', metavar='TRAIN_Y_FILE')
    parser.add_option('--val_Y', dest='val_y_file',default='ratings_validation.dat', help='Affinities validation data input file', metavar='TRAIN_Y_FILE')    
    parser.add_option('--test_Y', dest='test_y_file', default='ratings_test.dat', help='Affinities testing data input file', metavar='TEST_Y_FILE')'''