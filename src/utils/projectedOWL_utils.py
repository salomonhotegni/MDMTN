import sys
import numpy as np
import torch
from scipy.linalg import orth
from numpy.random import randn, permutation, rand
from numpy.linalg import norm
from numpy import median
from sklearn.preprocessing import normalize
from math import sqrt,exp,atan,cos,sin,pi,ceil 
import time

##########################################################################
#####  Helper functions for the proximity operator required by GrOWL #####
##########################################################################

def proxOWL_segments(A, B):
    modified = True
    k = 0
    max_its = 1000
    while modified and k <= max_its:
        modified = False
        segments = []
        new_start = True
        start = None
        end = None

        for i in range(len(A) - 1):
            if (A[i] - B[i] > 0) and (A[i+1] - B[i+1] > 0):
                if (A[i] - B[i] < A[i+1] - B[i+1]):
                    modified = True 
                    if new_start:
                        start = i
                        new_start = False
                    continue
                elif (A[i] - B[i] >= A[i+1] - B[i+1]):
                    if start is not None:
                        end = i
                        segments.append((start, end))
                    new_start = True
                    start = None
                    end = None

        if start is not None and end is None:
            end = len(A) - 1
            segments.append((start, end))
            
        if len(segments) == 0:
            break;

        for segment in segments:
            start, end = segment
            avg_A = np.mean(A[start:end+1])
            avg_B = np.mean(B[start:end+1])
            A[start:end+1] = avg_A
            B[start:end+1] = avg_B
            modified = True
        k = k + 1  
        
    X = A - B
    X[X < 0] = 0
    return X


def proxOWL(z, mu):
        # Adapted from: https://github.com/Dejiao2018/GrOWL/blob/master/VGG/owl_projection/projectedOWL.py
        
        #Restore the signs of z
        sgn = np.sign(z)
        #Sort z to non-increasing order
        z = abs(z)
        indx = z.argsort()
        indx = indx[::-1]
        z = z[indx]
        # find the index of the last positive entry in vector z - mu  
        flag = 0
        n = z.size
        x = np.zeros((n,))
        diff = z - mu 
        diff = diff[::-1]
        indc = np.argmax(diff>0)
        flag = diff[indc]
        #Apply prox on non-negative subsequences of z - mu
        if flag > 0:
            k = n - indc
            v1 = z[:k]
            v2 = mu[:k]
            v = proxOWL_segments(v1,v2)
            x[indx[:k]] = v
        else:
            pass
        # restore signs
        x = np.multiply(sgn,x)
        return torch.from_numpy(x)
