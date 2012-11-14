'''
Created on Nov 2, 2012

@author: neeraj
'''

import scipy.sparse as sp
import PMF_self

def test_code():
    # create random sparse matrix
    m = 1000
    n = 500
    rank = 10
    reg =  0.001
    data = sp.rand(m, n, density=0.4, format='coo')
    rows,cols,vals = sp.find(data)

    # call Sanmi code
    PMF_kMeans.mf_solver(rows, cols, vals, rank, reg, m, n)

    # check the error it returns

if __name__ == '__main__':
    test_code()