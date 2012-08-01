'''
Created on May 2, 2012

@author: neeraj
'''
import sys
import pickle

def main():
    parameter_file = open("hetrecFinal_Model_2_5.pickle", "r")
    parameters = pickle.load(parameter_file)
    print 2
    return

if __name__ == '__main__':
    main()
    sys.exit()