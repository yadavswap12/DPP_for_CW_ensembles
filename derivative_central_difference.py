# -*- coding: utf-8 -*-
"""
Created on Wed Dec 05 18:19:18 2018

@author: yadav
"""

# code for derivative of function using central difference method
#input x (array) is argument of function f (also an array)
#f should be predefined for each value in x in a code in which derivative_cent_diff_(x,f) is called 

import math
import cmath
import numpy
def derivative_cent_diff_(x,f):
    derivative = numpy.empty(len(f),float)
    for i in range(x.size-1):
        if i==0:
            derivative[i] = (f[i+1]-f[i])/(x[i+1]-x[i])
        elif i==x.size-1:
            #derivative[i] = (f[i-1]-f[i])/(x[i-1]-x[i])
            derivative[i] = (f[i]-f[i-1])/(x[i]-x[i-1])
        else:    
            #derivative[i] = (f[i+1]-f[i-1])/(x[i+1]-x[i-1])    # wrong formula for central difference, note that step sizes are different    
            derivative[i] = (((f[i+1]-f[i])/(x[i+1]-x[i]))+((f[i]-f[i-1])/(x[i]-x[i-1])))/2.0
    
    return derivative         