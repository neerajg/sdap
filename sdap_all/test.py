'''
Created on Oct 5, 2012

@author: neeraj
'''
import glob
import sys
import os
import scipy.io.mmio as mm
import numpy as np

def main():
    files = glob.iglob(os.path.join('D:\\sdap\\data\\temp\\','*.U'))
    t = 0
    for file_name in sorted(files):
        if t == 0:
            a = np.loadtxt(file_name,skiprows=3)
            t = 1
        else:
            b = np.loadtxt(file_name,skiprows=3)
            a = np.vstack((a,b))
    return

if __name__ == '__main__':
    main()
    sys.exit()