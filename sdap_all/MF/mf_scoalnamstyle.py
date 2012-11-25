'''
Created on Jun 15, 2012
Author - Neeraj Gaur
'''

# TO DO : check all TO DOs here and in misc
# TO DO : Try damped updates
# TO DO : here and in misc also change update of r condition to check for very low value to make it work for all K and L
import numpy as np
import sys
from MFModel import MFModel
from PMF_self import mf_solver
import gc

D = 20
thresh = 1e-4
#lambda_reg = 0.065
lambda_damping = 0
num_iter = 50

def train_mfscoalnamstyle(M, N, train_I, train_J, train_Y, K, L,lambda_reg,implementation):
    model = MFModel()
    t = 0
    initial_R = np.random.randint(0, K, size=(M,1))
    initial_C = np.random.randint(0, L, size=(N,1))
    model.initialize(M,N,K,L,initial_R,initial_C,D,train_Y,train_I,train_J,lambda_reg)
    model.train(implementation)
    old_objective = model.objective
    #print model.U[1:10,5:15,:]
    #print "V"
    #print model.V[1:10,5:15,:]

    #while True:
        #model.update_row_assignments()
        #model.train()
        #model.update_col_assignments()
        #outfile = open('/home/neeraj/latent_U.txt','wb')
        #print>>outfile,model.U
        #outfile = open('/home/neeraj/latent_V.txt','wb')
        #print>>outfile,model.V      
        #sys.exit()
        #model.train()
        #if old_objective - model.objective < thresh or t>num_iter:
        #    break
        #old_objective = model.objective
        #t = t+1

    obj = []
    params = {'model':model}
    return params, obj