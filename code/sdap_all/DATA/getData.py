'''
Created on May 17, 2012

Author - Neeraj Gaur
Title - Returns the Data requested

'''
import sys, os
import copy
from DATA.MLENS.get_movielens_data import getMovieLensData
from DATA.EHARMONY.get_eHarmony_data import geteHarmonyData
import numpy as np
import math
import cPickle as pickle
import platform
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans

def getRatingsTestData(k_fold, pctg_users, pctg_movies, K, L, datasetName,M=None,N=None,D1=None,D2=None,no_obs=None):
    
    if platform.system() == 'Windows':
        base_dir = 'D:'
    elif platform.system() == 'Linux':
        base_dir = '/workspace'
    
    if datasetName.upper().split('_')[0] != 'TEST':
        data_dir = base_dir + '/sdap/data/'+datasetName
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        data_dir = base_dir + '/sdap/data/'+datasetName+'/extracted_features'
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        data_dir = base_dir + '/sdap/data/'+datasetName+'/extracted_features/data_files'
    else:
        data_dir = base_dir + '/sdap/data/Artificial Data'
    
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    ratings_test_dir = data_dir + '/ratings_test'
    
    if not os.path.isdir(ratings_test_dir):
        os.mkdir(ratings_test_dir)    

    if datasetName.upper().split('_')[0] != 'TEST':
        ratings_test_data_file = ratings_test_dir+'/k_'+str(k_fold)+'pctg_users_'+str(pctg_users)+'pctg_movies_'+str(pctg_movies)+'.pickle'
    else:
        ratings_test_data_file = ratings_test_dir+'/k_'+str(k_fold)+datasetName+str(M)+str(N)+str(K)+str(L)+str(D1)+str(D2)+'pctg_users_'+str(pctg_users)+'pctg_movies_'+str(pctg_movies)+'.pickle'
    # Check if the data exists for the current setting, else create the data and also write to disk
    if os.path.isfile(ratings_test_data_file):
        data_exist = True
        print "  DATA EXISTS : RETRIEVING DATA"
        
        ifile = open(ratings_test_data_file, "r")
        data = pickle.load(ifile)
        ifile.close()            
         
    else:
        print "  DATA DOES NOT EXIST : MAKING DATA" 
        data_exist = False
        # Get the complete data
        if datasetName.upper().split('_')[0] != 'TEST':
            if datasetName.upper() == 'MOVIELENS':
                Xs, Y, I, J= getMovieLensData()
                params = None
            if datasetName.upper().startswith('EHARMONY'):
                Xs, Y, I, J= geteHarmonyData(int(datasetName.upper().split('EHARMONY')[1]))
                params = None
        else:
            sys.path.append(os.path.abspath('./'+datasetName.upper().split('_')[1]+'/'))
            artdata_module = __import__('make_artificial_data')
            Xs, Y, I, J, params = artdata_module.make_artificial_data(M,N,D1,D2,K,L,no_obs)
            
        X1 = Xs[0]
        X2 = Xs[1]
        I = np.array(I)
        J = np.array(J)
        # Normalize the attributes
        for d1 in range(X1.shape[1]):
            X1[:,d1] = X1[:,d1] - min(X1[:,d1])
            X1[:,d1] = X1[:,d1]/max(X1[:,d1])
        for d2 in range(X2.shape[1]):
            X2[:,d2] = X2[:,d2] - min(X2[:,d2])
            X2[:,d2] = X2[:,d2]/max(X2[:,d2])          
    
        print "  GOT DATA"
    
        # Randomly sample users and movies
        #users = np.random.randint(0,max(I),int(max(I)*(pctg_users/100.0)))
        users = range(max(I)+1)
        np.random.shuffle(users)
        users = users[0:int((max(I)+1)*(pctg_users/100.0))]
        users.sort()
        users = np.array(users)
        trctd_X1 = X1[users]
        #movies = np.random.randint(0,max(J),int(max(J)*(pctg_movies/100.0)))
        movies = range(max(J)+1)
        np.random.shuffle(movies)  
        movies = movies[0:int((max(J)+1)*(pctg_movies/100.0))]    
        movies.sort()
        movies = np.array(movies)
        trctd_X2 = X2[movies]
        print 'before'
        if not (pctg_users==100 and pctg_movies==100):
            indx = [idx for idx in range(len(I)) if (I[idx] in users and J[idx] in movies)]
            trctd_I = I[indx]
            trctd_J = J[indx]
            trctd_Y = Y[indx]
        else:
            trctd_I = I
            trctd_J = J
            trctd_Y = Y                        
        print 'after'
              
        # Map to new row and col values
        users = list(users)
        movies = list(movies)

        if not (pctg_users==100 and pctg_movies==100):
            for idx in range(len(trctd_I)):
                trctd_I[idx] = users.index(trctd_I[idx])
                trctd_J[idx] = movies.index(trctd_J[idx])
         
        # Split Data into k sets for k-fold cross-validation
        print "  SPLITTING DATA INTO " + str(k_fold) + " SETS FOR CROSS-VALIDATION"
        rand_idx = range(len(trctd_Y))
        np.random.shuffle(rand_idx)
        step_size = int(math.floor(len(trctd_Y)/k_fold))
        indices = range(0,len(trctd_Y)-1,step_size)
        rand_users = list(set(trctd_I))
        np.random.shuffle(rand_users)
        rand_movies = list(set(trctd_J))
        np.random.shuffle(rand_movies)
        user_step_size = int(math.floor(len(rand_users)/k_fold))
        user_indices = range(0,len(rand_users)-1,user_step_size)
        movie_step_size = int(math.floor(len(rand_movies)/k_fold))
        movie_indices = range(0,len(rand_movies)-1,movie_step_size)  
        print "  DONE SPLITTING DATA"
                
        data = {'users':np.array(users),
                'movies':np.array(movies),
                'trctd_X1':np.array(trctd_X1),
                'trctd_X2':np.array(trctd_X2),
                'trctd_I':np.array(trctd_I),
                'trctd_J':np.array(trctd_J),
                'trctd_Y':np.array(trctd_Y),
                'indices':np.array(indices),
                'step_size':step_size,
                'rand_idx':np.array(rand_idx),
                'rand_users':np.array(rand_users),
                'rand_movies':np.array(rand_movies),
                'user_step_size':user_step_size,
                'user_indices':user_indices,
                'movie_step_size':movie_step_size,
                'movie_indices':movie_indices,
                'params':params
                }
        
        print 'BEFORE WRITING TO DISK'
        ofile = open(ratings_test_data_file, "wb")
        pickle.dump(data, ofile)
        ofile.close()  
        print"  DONE PREPARING REDUCED DATA"

    print "\t NUMBER OF OBSERVATIONS = "+ str(len(data['trctd_I']))
    print "\t NUMBER OF USERS = " + str(len(data['users']))
    print "\t NUMBER OF MOVIES = " + str(len(data['movies']))
    trctd_X1 = data['trctd_X1']
    trctd_X2 = data['trctd_X2']
    attr_clusters_dir = data_dir + '/attr_clusters/'
    if not os.path.isdir(attr_clusters_dir):
        os.mkdir(attr_clusters_dir)    

    user_clusters_data_file = attr_clusters_dir+'K_'+str(K)+'pctg_users_'+str(pctg_users)+'pctg_movies_'+str(pctg_movies)+'.pickle'
    # Check if the user centroids exist for the current setting
    if os.path.isfile(user_clusters_data_file) and data_exist:
        print "  USER CENTROIDS EXIST : RETRIEVING CENTROIDS"
        ifile = open(user_clusters_data_file, "r")
        user_centroids = pickle.load(ifile)
        ifile.close()            
    else:
        print "  USER CENTROIDS DO NOT EXIST : COMPUTING CENTROIDS"        
        # Cluster the Attributes of the users
        print"  CLUSTERING THE USER ATTRIBUTES FOR WARM AND COLD START"
        # Whiten the features and use k-means to cluster the users
        user_centroids = kmeans(whiten(trctd_X1),K)[0]
        ofile = open(user_clusters_data_file, "wb")
        pickle.dump(user_centroids, ofile)
        ofile.close()
        
    movie_clusters_data_file = attr_clusters_dir+'L_'+str(L)+'pctg_users_'+str(pctg_users)+'pctg_movies_'+str(pctg_movies)+'.pickle'
    # Check if the movie centroids exist for the current setting
    if os.path.isfile(movie_clusters_data_file) and data_exist:
        print "  MOVIE CENTROIDS EXIST : RETRIEVING CENTROIDS"
        ifile = open(movie_clusters_data_file, "r")
        movie_centroids = pickle.load(ifile)
        ifile.close()            
    else:
        print "  MOVIE CENTROIDS DO NOT EXIST : COMPUTING CENTROIDS"        
        # Cluster the Attributes of the users
        print"  CLUSTERING THE MOVIE ATTRIBUTES FOR WARM AND COLD START"
        # Whiten the features and use k-means to cluster the users
        movie_centroids = kmeans(whiten(trctd_X2),L)[0]
        ofile = open(movie_clusters_data_file, "wb")
        pickle.dump(movie_centroids, ofile)
        ofile.close()        
       
    centroids = {'user_centroids':user_centroids,
                 'movie_centroids':movie_centroids}   
          
    return data, centroids

def getHotStartDataFolds(fold, dataSet):
    
    data = dataSet
    users = data['users']
    movies = data['movies']
    trctd_X1 = data['trctd_X1']
    trctd_X2 = data['trctd_X2']
    trctd_I = data['trctd_I']
    trctd_J = data['trctd_J']
    trctd_Y = data['trctd_Y']
    indices = data['indices']
    step_size = data['step_size']
    rand_idx = data['rand_idx']
    
    val_index = indices[fold]

    val_indices = rand_idx[val_index:val_index+step_size-1]
    val_indices = np.sort(val_indices)
       
    train_indices = np.array((list(set(rand_idx).difference(set(val_indices)))))
    train_indices = np.sort(train_indices)      
      
    train_Y = trctd_Y[train_indices]
    train_I = trctd_I[train_indices]
    train_J = trctd_J[train_indices]
    
    # Only consider those user and movies which are present in the training set for calculating the hot start RMSE
    val_Y = trctd_Y[val_indices]
    val_I = trctd_I[val_indices]
    val_J = trctd_J[val_indices]    
    
    hotStart_users = set(train_I).intersection(set(val_I))
    hotStart_user_indices = [idx for idx in range(len(val_I)) if val_I[idx] in hotStart_users] 
    hotStart_movies = set(train_J).intersection(set(val_J))
    hotStart_movie_indices = [idx for idx in range(len(val_J)) if val_J[idx] in hotStart_movies]
    
    hotStart_user_movie_indices = np.sort(np.array(list(set(hotStart_user_indices).intersection(hotStart_movie_indices))))
    val_Y = val_Y[hotStart_user_movie_indices]
    val_I = val_I[hotStart_user_movie_indices]
    val_J = val_J[hotStart_user_movie_indices] 
    
    dataFold = {'train_Y':train_Y,
                'train_I':train_I,
                'train_J':train_J,
                'val_Y':val_Y,
                'val_I':val_I,
                'val_J':val_J
                }
            
    return dataFold

def getWarmStartDataFolds(fold, dataSet):
    
    data = dataSet
    users = data['users']
    movies = data['movies']
    trctd_X1 = data['trctd_X1']
    trctd_X2 = data['trctd_X2']
    trctd_I = data['trctd_I']
    trctd_J = data['trctd_J']
    trctd_Y = data['trctd_Y']
    indices = data['indices']
    step_size = data['step_size']
    rand_idx = data['rand_idx']
    rand_users = data['rand_users']
    rand_movies = data['rand_movies']
    user_step_size = data['user_step_size']
    user_indices = data['user_indices']
    movie_step_size = data['movie_step_size']
    movie_indices = data['movie_indices']    
    
    val_users = user_indices[fold]
    val_movies = movie_indices[fold]
    
    user_val = rand_users[val_users:val_users+user_step_size-1]
    np.sort(user_val)
    movie_val = rand_movies[val_movies:val_movies+movie_step_size-1]
    np.sort(movie_val)
    
    # Add the ratings for the above users and movies in validation and the others in training
    train_indices = range(len(trctd_Y))
    test_indices = [idx for idx in range(len(trctd_Y)) if trctd_I[idx] in user_val or  trctd_J[idx] in movie_val]
    train_indices = np.array(list(set(train_indices).difference(set(test_indices))))
    
    train_Y = trctd_Y[train_indices]
    train_I = trctd_I[train_indices]
    train_J = trctd_J[train_indices]
    
    val_Y = trctd_Y[test_indices]
    val_I = trctd_I[test_indices]
    val_J = trctd_J[test_indices]
    
    dataFold = {'train_Y':train_Y,
                'train_I':train_I,
                'train_J':train_J,
                'val_Y':val_Y,
                'val_I':val_I,
                'val_J':val_J
                }
                                            
    return dataFold

def getColdStartDataFolds(fold, dataSet):
    
    data = dataSet
    users = data['users']
    movies = data['movies']
    trctd_X1 = data['trctd_X1']
    trctd_X2 = data['trctd_X2']
    trctd_I = data['trctd_I']
    trctd_J = data['trctd_J']
    trctd_Y = data['trctd_Y']
    indices = data['indices']
    step_size = data['step_size']
    rand_idx = data['rand_idx']
    rand_users = data['rand_users']
    rand_movies = data['rand_movies']
    user_step_size = data['user_step_size']
    user_indices = data['user_indices']
    movie_step_size = data['movie_step_size']
    movie_indices = data['movie_indices']    
    
    val_users = user_indices[fold]
    val_movies = movie_indices[fold]
    
    user_val = rand_users[val_users:val_users+user_step_size-1]
    user_val = np.sort(user_val)
    movie_val = rand_movies[val_movies:val_movies+movie_step_size-1]
    movie_val = np.sort(movie_val)
    
    # Add the ratings for the above users and movies in validation and the others in training
    train_indices = range(len(trctd_Y))
    test_indices1 = [idx for idx in range(len(trctd_Y)) if trctd_I[idx] in user_val]
    test_indices2 = [idx for idx in range(len(trctd_Y)) if trctd_J[idx] in movie_val]
    test_indices = list(set(test_indices1).union(set(test_indices2)))
    train_indices = np.array(list(set(train_indices).difference(set(test_indices))))
    
    train_Y = trctd_Y[train_indices]
    train_I = trctd_I[train_indices]
    train_J = trctd_J[train_indices]

    val_Y = trctd_Y[test_indices]
    val_I = trctd_I[test_indices]
    val_J = trctd_J[test_indices]
    
    user_val = np.array(list(set(val_I).difference(set(train_I))))
    movie_val = np.array(list(set(val_J).difference(set(train_J))))
    test_indices = [idx for idx in test_indices if trctd_I[idx] in user_val and trctd_J[idx] in movie_val]
    
    #test_indices = [idx for idx in test_indices if val_I[idx] not in train_I and val_J[idx] not in train_J]
    
    val_Y = trctd_Y[test_indices]
    val_I = trctd_I[test_indices]
    val_J = trctd_J[test_indices]     
    
    val_X1 = trctd_X1[list(set(val_I))]
    val_X2 = trctd_X2[list(set(val_J))]
    train_X1 = trctd_X1[list(set(range(len(trctd_X1))).difference(set(val_I)))]
    train_X2 = trctd_X2[list(set(range(len(trctd_X2))).difference(set(val_J)))]
    dataFold = {'train_Y':train_Y,
                'train_I':train_I,
                'train_J':train_J,
                'val_Y':val_Y,
                'val_I':val_I,
                'val_J':val_J,
                'val_X1':val_X1,
                'val_X2':val_X2,
                'train_X1':train_X1,
                'train_X2':train_X2
                }
                                            
    return dataFold

def get_likelihood_art_data(alphas, pis, zs, betas, I, J, Y, X1, X2, datasetName, model_name):
    sys.path.append(os.path.abspath('./'+datasetName.upper().split('_')[1]+'/'))
    artdata_module = __import__('make_artificial_data')
    log_likelihood_art_data = artdata_module.get_likelihood_art_data(alphas, pis, zs, betas, I, J, Y, X1, X2, model_name)
    return log_likelihood_art_data
if __name__ == '__main__':
    sys.exit()