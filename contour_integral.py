# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:34:08 2018

@author: yadav
"""

#code for contour integral of a complex function on given contour
#inputs c is an array of x,y on contour and f is a complex function (integrand) 
#f should be predefined in a code in which contr_intgl(c,f) is called
import math
import cmath
import numpy
import integration_trapezoidal_nonuniformspacing
def contr_intgl(c,f):
    g = numpy.empty(len(c),float)
    h = numpy.empty(len(c),float)
    x = numpy.empty(len(c),float)
    y = numpy.empty(len(c),float)
    for i in range(len(c)):    # function len() on array gives no. of rows of array
        x[i] = c[i,0]
        y[i] = c[i,1]
        z = complex(x[i],y[i])
        #z = numpy.complex128(complex(x[i],y[i]))
        g[i] = f(z).real
        h[i] = f(z).imag
    #int_trap_non_uniform(x,g)
    I1 = integration_trapezoidal_nonuniformspacing.int_trap_non_uniform(x,g)
    #int_trap_non_uniform(x,h)        
    I2 = integration_trapezoidal_nonuniformspacing.int_trap_non_uniform(x,h)*1j
    #int_trap_non_uniform(y,h)
    I3 = (-1)*integration_trapezoidal_nonuniformspacing.int_trap_non_uniform(y,h)
    #int_trap_non_uniform(y,g)
    I4 = integration_trapezoidal_nonuniformspacing.int_trap_non_uniform(y,g)*1j
    I_contr = I1+I2+I3+I4
    return  I_contr 
    
    