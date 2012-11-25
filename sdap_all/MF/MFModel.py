'''
Created on Oct 5, 2012

@author: neeraj
'''

import numpy as np
import scipy.sparse as sp
import scipy.io.mmio as mm
import glob, shutil, os, platform
import subprocess, shlex

class MFModel(object):
    def __init__(self):
        self.objective = 1e99
        
    def initialize(self, M,N,K,L,initial_R,initial_C,D,train_Y,train_I,train_J,lambda_reg):
        self.R = initial_R
        self.C = initial_C
        self.M = M
        self.N = N
        self.K = K
        self.L = L        
        self.U = np.zeros(shape=(M,D,L))
        self.V = np.zeros(shape=(N,D,K))
        #self.Y_matrix = sp.csr_matrix((train_Y,(train_I,train_J)), shape = (M,N))
        self.Y_matrix = sp.coo_matrix((train_Y,(train_I,train_J)), shape = (M,N))
        self.Y = train_Y
        self.I = train_I
        self.J = train_J
        self.num_observations = len(train_Y)
        self.lambda_reg = lambda_reg
        return
    
    def train(self):
        total_sqrd_error = 0.0
        # For Each Co-cluster
        for r in range(self.K):
        #for r in range(1):            
            for c in range(self.L):
            #for c in range(1):                
                # Find the Rows and Columns Corresponding to the rated items in the co-cluster
                row_index  = self.R[self.I]==r
                row_index = row_index.flatten()
                filtered_I = self.I[row_index]
                filtered_J = self.J[row_index]          
                col_index  = self.C[filtered_J]==c
                col_index = col_index.flatten()
                #print filtered_I,filtered_J
                filtered_I = filtered_I[col_index]
                filtered_J = filtered_J[col_index]
                #print filtered_I,filtered_J                
                # Extract the Training data for those rows and columns
                filtered_Y = self.Y[row_index]
                filtered_Y = filtered_Y[col_index]         
                if len(filtered_Y)<1:
                    outfile = open("/home/neeraj/mf_debug.txt","ab")
                    print >> outfile, r,c
                    continue       
                
                # Find rows and cols of co-cluster sub-matrix
                rows = np.array(list(set(filtered_I)))
                cols = np.array(list(set(filtered_J)))       
                
                # Call the MF for rows and cols of current co-cluster
                self.matrixFactorization(rows,cols,r,c)
                total_sqrd_error += self.calculateSqrdError(filtered_Y,filtered_I,filtered_J,r,c)
        
        # Update the objective
        self.objective = np.sqrt(total_sqrd_error / self.num_observations)
        #print self.objective
        #import sys
        #sys.exit()
        return
        
# TO DO : Update row and col assignments by forming huge block diagonal matrices and then calling one ALS update for each row/col         
    def update_row_assignments(self):
        # For each row
        for i in range(self.M):
            #print self.objective,self.M,i         
            # Extract the Columns corresponding to the rated movies for the user
            rated_movies = self.Y_matrix.getrow(i).nonzero()[1]
            if len(rated_movies) < 1:
                continue
            rated_movies.sort() 
                       
            # Extract the Vjs corresponding to those columns
            Vjs = self.V[rated_movies,:,:]
            usr_ratings = self.Y_matrix.getrow(i).sorted_indices().data
            # Form K length error vector
            error_K = np.zeros((self.K,1))            
            # For each value of K
            for r in range(self.K):
                # For each value of L
                for c in range(self.L):
                    # Do ALS update for U_i[:,:,]
                    col_index = self.C[rated_movies] == c
                    col_index = col_index.flatten()
                    V_ijs = Vjs[col_index,:,r]
                    ratings = usr_ratings[col_index]
                    if len(ratings)<1:
                        continue                      
                    U_i = self.alsUpdate(V_ijs,ratings)
                    #import sys
                    #if self.R[i]==r:
                        #print U_i
                        #print "Now for the Grpahlab ALS Uis"
                        #print self.U[i,:,c]
                        #sys.exit()
                    # Find error and add to error_K
                    sqrd_error = np.sum((ratings - np.dot(U_i,np.transpose(V_ijs)))**2)
                    error_K[r] += sqrd_error
            # Assign self.R[row] = that index which minimizes error vector
            #print [np.argmin(error_K)],self.R[i]
            if np.sum(error_K) == 0.0:
                continue            
            self.R[i] = [int(np.argmin(error_K))]
        return
            
    def update_col_assignments(self):
        # For each col
        for j in range(self.N):
            if j%1000 == 0:
                print j
        #for j in [0]:

            # Extract the Rows corresponding to the Users who have rated this movie
            active_users = self.Y_matrix.getcol(j).nonzero()[0]
            #print self.Y_matrix.getcol(j)
            if len(active_users) < 1:
                continue
            active_users.sort()
            #print active_users, self.R[active_users]
                       
            # Extract the Uis corresponding to those columns
            Uis = self.U[active_users,:,:]
            movie_ratings = self.Y_matrix.getcol(j).sorted_indices().data
            # Form L length error vector
            error_L = np.zeros((self.L,1))            
            # For each value of L
            for c in range(self.L):
                # For each value of K
                for r in range(self.K):
                    # Do ALS update for V_j[:,:,]
                    row_index = self.R[active_users] == r
                    row_index = row_index.flatten()
                    U_jis = Uis[row_index,:,c]
                    ratings = movie_ratings[row_index]
                    #print r, row_index, ratings                    
                    if len(ratings)<1:
                        continue             
                    V_j = self.alsUpdate(U_jis,ratings)
                    #import sys
                    #if self.C[j]==c:
                        #print V_j
                        #print "Now for the Grpahlab ALS Vjs"
                        #print self.V[j,:,r]
                        #sys.exit()
                    # Find error and add to error_L
                    sqrd_error = np.sum((ratings - np.dot(V_j,np.transpose(U_jis)))**2)
                    error_L[c] += sqrd_error
                    #print error_L,r,c,j
            # Assign self.C[col] = that index which minimizes error vector
            if np.sum(error_L) == 0.0:
                continue
            #print  self.N,j,[np.argmin(error_L)],self.C[j]     
            #if (self.C[j]!=int(np.argmin(error_L))):
                #outfile = open('/home/neeraj/change.txt','ab')
                #print >>outfile, j,[np.argmin(error_L)],self.C[j]
            self.C[j] = [int(np.argmin(error_L))]
        return
        
    def matrixFactorization(self,rows,cols,r,c):
        if platform.system() == 'Windows':
            base_dir = 'D:'
        elif platform.system() == 'Linux':
            base_dir = '/workspace'
                            
        # Pull out the sparse sub-matrix for current co-cluster
        #print r,c
        Y_sparse = self.coo_submatrix_pull(rows,cols)
        # Convert Data into Matrix Market format and write the Training data to file
        Y_mm_file_name = base_dir + '/sdap/data/temp/mf'
        # Remove the current files in .../data/temp/
        #cmd = "rm -f" + base_dir + "/sdap/data/temp/*"
        #os.system(cmd)
        if os.path.isdir(base_dir + "/sdap/data/temp/"):
            shutil.rmtree(base_dir + "/sdap/data/temp/")
        os.mkdir(base_dir + "/sdap/data/temp/")
        Y_mm_file = open(Y_mm_file_name,'w')
        mm.mmwrite(Y_mm_file,Y_sparse,comment="")
        
        # Call Matrix Factorization code
        cmd = "~/graphlabapi/release/demoapps/pmf/pmf "+ Y_mm_file_name +" --matrixmarket=true --lambda="+str(self.lambda_reg)+" --minval=0.5 --maxval=5.0 --ncpus=16 --max_iter=35 --threshold=1e-6"
        subprocess.call(cmd,shell=True)#,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) #TO DO replace 
        #os.system(cmd)
    
        # Read all the files for the output Latent Factors and return Us and Vs
        U,V = self.readLatentFactors(base_dir)
        #print U.shape, len(rows)
        #print V.shape, len(cols)
        #import sys
        #sys.exit()
        
        # Update the Latent Factors in the model
        self.U[rows,:,c] = U
        self.V[cols,:,r] = V
          
        return

    def coo_submatrix_pull(self,rows,cols):
        """
        Pull out an arbitrary i.e. non-contiguous submatrix out of
        a sparse.coo_matrix. from :
        git://gist.github.com/828099.git
        """
        #print rows,cols
        matr = self.Y_matrix
        if type(matr) != sp.coo_matrix:
            raise TypeError('Matrix must be sparse COOrdinate format')
    
        gr = -1 * np.ones(matr.shape[0])
        gc = -1 * np.ones(matr.shape[1])
    
        lr = len(rows)
        lc = len(cols)
    
        ar = np.arange(0, lr)
        ac = np.arange(0, lc)
        gr[rows[ar]] = ar
        gc[cols[ac]] = ac
        mrow = matr.row
        mcol = matr.col
        newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
        newrows = mrow[newelem]
        newcols = mcol[newelem]
    
        return sp.coo_matrix((matr.data[newelem], np.array([gr[newrows],gc[newcols]])),(lr, lc))
    
    def readLatentFactors(self,base_dir):
        files = glob.iglob(os.path.join(base_dir + '/sdap/data/temp/','*.U'))
        t = 0
        for file_name in sorted(files):
            if t == 0:
                a = np.loadtxt(file_name,skiprows=3)
                t = 1
            else:
                b = np.loadtxt(file_name,skiprows=3)
                a = np.vstack((a,b))
        U = a
        files = glob.iglob(os.path.join(base_dir + '/sdap/data/temp/','*.V'))
        t = 0
        for file_name in sorted(files):
            if t == 0:
                a = np.loadtxt(file_name,skiprows=3)
                t = 1
            else:
                b = np.loadtxt(file_name,skiprows=3)
                a = np.vstack((a,b))
        V = a        
        return U,V
    
    def calculateSqrdError(self,filtered_Y,filtered_I,filtered_J,r,c):
        # Calculate sqrd error for this co-cluster and return that error       
        sqrd_error = np.sum((filtered_Y - np.sum(np.multiply(self.U[filtered_I,:,c],self.V[filtered_J,:,r]),1))**2)
        return sqrd_error
    
    def alsUpdate(self,V_i,ratings):
        Ai = np.dot(np.transpose(V_i),V_i) + np.diagflat(self.lambda_reg*len(ratings)*np.ones(V_i.shape[1]))
        Y_i = np.dot(np.transpose(V_i),ratings)
        U_i = np.linalg.solve(Ai,Y_i)
        return U_i