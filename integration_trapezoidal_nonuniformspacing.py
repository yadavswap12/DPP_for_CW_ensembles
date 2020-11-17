# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:47:04 2018

@author: yadav
"""
# code for numerical integration using trapezoidal rule with variable step size
#input x (array) is argument of function f in  and f are arrays and 
#f should be predefined for each value in x in a code in which int_trap_non_uniform(x,f) is called 

import math
import cmath
import numpy
def int_trap_non_uniform(x,f):
    I=0
    for i in range(x.size-1):
        delta_I = ((x[i+1]-x[i])/2.0)*(f[i]+f[i+1])
        delta_I = numpy.nan_to_num(delta_I)
        #delta_I = numpy.nan_to_num(((x[i+1]-x[i])/2.0)*(f[i]+f[i+1]))
        #if math.isnan(delta_I)==True:    #https://stackoverflow.com/questions/944700/how-can-i-check-for-nan-values
         #  print "delta_I gives nan"
          # delta_I==0
        I += delta_I
    return I         