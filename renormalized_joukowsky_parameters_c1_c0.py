# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 10:20:37 2019

@author: yadav
"""

#code for computation of parameters c1,c0 of Joukowsky transform for CR paper self consistently. See notebook for details
import math
import cmath
import numpy
import contour_integral
import contour_function_2
#tau = 1.0
#rho = -4.0    # V(x)=(tau*x**2 + rho*x)
#theta = 2.0
c1 = 0.2   # initializatio for c1(>0), analytical solution is c1=(-2.0/rho), see page number 2439 in C.R.
c0 = 0.1    # initializatio for c0(>0,>c1), analytical solution is c0=(-rho/2.0), see page number 2439 in C.R.        
jacobian = numpy.empty([2,2],float)
ff = numpy.empty(2,float)
c = numpy.empty(2,float)
c[0] = c1
c[1] = c0

b0=0.19595    # parameters of polynomial fit for f1(x), f1(x)=b0 + b1x + b2x^2 + b3x^3 + b4x^4 
b1=1.38475    # gamma = 0.76
b2=-0.00497
b3=0.01199
b4=0.00423
b5=-0.00196
b6=-0.00159
b7=1.02664E-4
b8=2.28081E-4

#gamma=0.7, alpha=8.0 c1=0.73877829336302647 c0=0.81888520779300789    #corrected3 iteration11
#gamma=0.71, alpha=8.0 c1=0.74270348927791463 c0=0.80975277719105032    #corrected3 iteration11
#gamma=0.72, alpha=8.0 c1=0.74600089423876992 c0=0.80064971196452028    #corrected3 iteration11
#gamma=0.73, alpha=8.0 c1=0.74913295880076414 c0=0.79205466041975692    #corrected3 iteration11
#gamma=0.74, alpha=8.0 c1=0.75185885907320193 c0=0.78367896908653689    #corrected3 iteration11
#gamma=0.75, alpha=8.0 c1=0.75431767129565974 c0=0.77559529048582188    #corrected3 iteration11
#gamma=0.76, alpha=8.0 c1=0.75651787191897635 c0=0.76780260554911939    #corrected3 iteration11


while True:
    
    c1 = c[0]
    c0 = c[1]
    
    contr = contour_function_2.contour_CR_prob_soft_edge(c[0],c[1])
#    contr = contour_function.contour_CR_prob_soft_edge(c[0],c[1],theta)    # to make sure output file in contour_function is written properly    
#    contr = contour_function.contour_CR_prob_soft_edge(c[0],c[1],theta)    # to make sure output file in contour_function is written properly

    def J(z):    # joukowsky transformation for hard edge
            return c1*z+c0-cmath.log((z-1.0/2)/(z+1.0/2))
            
    def deriv_c1_J(z):    # derivative of joukowsky transformation w.r.t. c1
            return z
            
    def deriv_c0_J(z):    # derivative of joukowsky transformation w.r.t. c0
            return 1.0            

#    def f1(z):    # See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((b0+b1*J(z)+b2*(J(z))**2.0+b3*(J(z))**3.0+b4*(J(z))**4.0)/(z))            

#    def f1(z):    # See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((b0+b1*J(z)+b2*(J(z))**2.0+b3*(J(z))**3.0+b4*(J(z))**4.0+b5*(J(z))**5.0+b6*(J(z))**6.0)/(z))

    def f1(z):    # See notebook
            return (1.0/(2.0*(math.pi)*1j))*(b0+b1*J(z)+b2*(J(z))**2.0+b3*(J(z))**3.0+b4*(J(z))**4.0+b5*(J(z))**5.0+b6*(J(z))**6.0+b7*(J(z))**7.0+b8*(J(z))**8.0)
            
#    def f2(z):    # See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((b0+b1*J(z)+b2*(J(z))**2.0+b3*(J(z))**3.0+b4*(J(z))**4.0)/(z+1.0))            

#    def f2(z):    # See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((b0+b1*J(z)+b2*(J(z))**2.0+b3*(J(z))**3.0+b4*(J(z))**4.0+b5*(J(z))**5.0+b6*(J(z))**6.0)/(z+1.0))            

    def f2(z):    # See notebook
            return (1.0/(2.0*(math.pi)*1j))*((b0+b1*J(z)+b2*(J(z))**2.0+b3*(J(z))**3.0+b4*(J(z))**4.0+b5*(J(z))**5.0+b6*(J(z))**6.0+b7*(J(z))**7.0+b8*(J(z))**8.0)/(z-0.5))            
                
#    F1 = ((contour_integral.contr_intgl(contr,f1)).real) - (1.0+theta)
#    F2 = ((contour_integral.contr_intgl(contr,f2)).real) - (1.0)
    
    ff[0] = ((contour_integral.contr_intgl(contr,f1)).real) - (1.0/c1)
    ff[1] = ((contour_integral.contr_intgl(contr,f2)).real) - (1.0)

    
#    def deriv_c1_f1(z):    # derivative of f1(z) w.r.t. c1
#            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_c1_J(z)+2.0*b2*J(z)*deriv_c1_J(z)+3.0*b3*((J(z))**2.0)*deriv_c1_J(z)+4.0*b4*((J(z))**3.0)*deriv_c1_J(z))/(z))  
                           
                           
#    def deriv_c1_f1(z):    # derivative of f1(z) w.r.t. c1
#            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_c1_J(z)+2.0*b2*J(z)*deriv_c1_J(z)+3.0*b3*((J(z))**2.0)*deriv_c1_J(z)+4.0*b4*((J(z))**3.0)*deriv_c1_J(z)+5.0*b5*((J(z))**4.0)*deriv_c1_J(z)+6.0*b6*((J(z))**5.0)*deriv_c1_J(z))/(z))

    def deriv_c1_f1(z):    # derivative of f1(z) w.r.t. c1
            return (1.0/(2.0*(math.pi)*1j))*(b1*deriv_c1_J(z)+2.0*b2*J(z)*deriv_c1_J(z)+3.0*b3*((J(z))**2.0)*deriv_c1_J(z)+4.0*b4*((J(z))**3.0)*deriv_c1_J(z)+5.0*b5*((J(z))**4.0)*deriv_c1_J(z)+6.0*b6*((J(z))**5.0)*deriv_c1_J(z)+7.0*b7*((J(z))**6.0)*deriv_c1_J(z)+8.0*b8*((J(z))**7.0)*deriv_c1_J(z))

    jacobian[0,0] = (contour_integral.contr_intgl(contr,deriv_c1_f1)).real + (1.0/(c1**2.0))    # for J(1,1), See notebook 
#    print ('J(1,1) is', jacobian[0,0])
            
#    def deriv_c0_f1(z):    # derivative of f1(z) w.r.t. c0
#            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_c0_J(z)+2.0*b2*J(z)*deriv_c0_J(z)+3.0*b3*((J(z))**2.0)*deriv_c0_J(z)+4.0*b4*((J(z))**3.0)*deriv_c0_J(z))/(z))  
                           
    
#    def deriv_c0_f1(z):    # derivative of f1(z) w.r.t. c0
#            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_c0_J(z)+2.0*b2*J(z)*deriv_c0_J(z)+3.0*b3*((J(z))**2.0)*deriv_c0_J(z)+4.0*b4*((J(z))**3.0)*deriv_c0_J(z)+5.0*b5*((J(z))**4.0)*deriv_c0_J(z)+6.0*b6*((J(z))**5.0)*deriv_c0_J(z))/(z))   

    def deriv_c0_f1(z):    # derivative of f1(z) w.r.t. c0
            return (1.0/(2.0*(math.pi)*1j))*(b1*deriv_c0_J(z)+2.0*b2*J(z)*deriv_c0_J(z)+3.0*b3*((J(z))**2.0)*deriv_c0_J(z)+4.0*b4*((J(z))**3.0)*deriv_c0_J(z)+5.0*b5*((J(z))**4.0)*deriv_c0_J(z)+6.0*b6*((J(z))**5.0)*deriv_c0_J(z)+7.0*b7*((J(z))**6.0)*deriv_c0_J(z)+8.0*b8*((J(z))**7.0)*deriv_c0_J(z))   

    jacobian[0,1] = (contour_integral.contr_intgl(contr,deriv_c0_f1)).real 
#    print ('J(1,2) is', jacobian[0,1])   
            
#    def deriv_c1_f2(z):    # derivative of f2(z) w.r.t. c1
#            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_c1_J(z)+2.0*b2*J(z)*deriv_c1_J(z)+3.0*b3*((J(z))**2.0)*deriv_c1_J(z)+4.0*b4*((J(z))**3.0)*deriv_c1_J(z))/(z+1.0))  
                           
    
#    def deriv_c1_f2(z):    # derivative of f2(z) w.r.t. c1
#            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_c1_J(z)+2.0*b2*J(z)*deriv_c1_J(z)+3.0*b3*((J(z))**2.0)*deriv_c1_J(z)+4.0*b4*((J(z))**3.0)*deriv_c1_J(z)+5.0*b5*((J(z))**4.0)*deriv_c1_J(z)+6.0*b6*((J(z))**5.0)*deriv_c1_J(z))/(z+1.0))

    def deriv_c1_f2(z):    # derivative of f2(z) w.r.t. c1
            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_c1_J(z)+2.0*b2*J(z)*deriv_c1_J(z)+3.0*b3*((J(z))**2.0)*deriv_c1_J(z)+4.0*b4*((J(z))**3.0)*deriv_c1_J(z)+5.0*b5*((J(z))**4.0)*deriv_c1_J(z)+6.0*b6*((J(z))**5.0)*deriv_c1_J(z)+7.0*b7*((J(z))**6.0)*deriv_c1_J(z)+8.0*b8*((J(z))**7.0)*deriv_c1_J(z))/(z-0.5))

    jacobian[1,0] = (contour_integral.contr_intgl(contr,deriv_c1_f2)).real 
#    print ('J(2,1) is', jacobian[1,0])  
            
#    def deriv_c0_f2(z):    # derivative of f2(z) w.r.t. c0
#            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_c0_J(z)+2.0*b2*J(z)*deriv_c0_J(z)+3.0*b3*((J(z))**2.0)*deriv_c0_J(z)+4.0*b4*((J(z))**3.0)*deriv_c0_J(z))/(z+1.0))  
                           
    
#    def deriv_c0_f2(z):    # derivative of f2(z) w.r.t. c0
#            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_c0_J(z)+2.0*b2*J(z)*deriv_c0_J(z)+3.0*b3*((J(z))**2.0)*deriv_c0_J(z)+4.0*b4*((J(z))**3.0)*deriv_c0_J(z)+5.0*b5*((J(z))**4.0)*deriv_c0_J(z)+6.0*b6*((J(z))**5.0)*deriv_c0_J(z))/(z+1.0))             

    def deriv_c0_f2(z):    # derivative of f2(z) w.r.t. c0
            return (1.0/(2.0*(math.pi)*1j))*((b1*deriv_c0_J(z)+2.0*b2*J(z)*deriv_c0_J(z)+3.0*b3*((J(z))**2.0)*deriv_c0_J(z)+4.0*b4*((J(z))**3.0)*deriv_c0_J(z)+5.0*b5*((J(z))**4.0)*deriv_c0_J(z)+6.0*b6*((J(z))**5.0)*deriv_c0_J(z)+7.0*b7*((J(z))**6.0)*deriv_c0_J(z)+8.0*b8*((J(z))**7.0)*deriv_c0_J(z))/(z-0.5))             

    jacobian[1,1] = (contour_integral.contr_intgl(contr,deriv_c0_f2)).real 
#    print ('J(2,2) is', jacobian[1,1])


    
#    def f1(z):    # for J(1,1), See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((4.0*tau*(c[0]*z+c[1])*z*(((z+1)/z)**(2.0/theta))+rho*z*(((z+1.0)/z)**(1.0/theta)))/(z))
#            
#    jacobian[0,0] = (contour_integral.contr_intgl(contr,f1)).real 
#    print ('J(1,1) is', jacobian[0,0])
#       
#    def f2(z):    # for J(1,2), See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((4.0*tau*(c[0]*z+c[1])*(((z+1.0)/z)**(2.0/theta))+rho*(((z+1.0)/z)**(1.0/theta)))/(z))
#
#    jacobian[0,1] = (contour_integral.contr_intgl(contr,f2)).real 
#    print ('J(1,2) is', jacobian[0,1])            
#            
#    def f3(z):    # for J(2,1), See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((4.0*tau*(c[0]*z+c[1])*z*(((z+1.0)/z)**(2.0/theta))+rho*z*(((z+1.0)/z)**(1.0/theta)))/(z+1.0))
#            
#    jacobian[1,0] = (contour_integral.contr_intgl(contr,f3)).real 
#    print ('J(2,1) is', jacobian[1,0])        
#            
#    def f4(z):    # for J(2,2), See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((4.0*tau*(c[0]*z+c[1])*(((z+1.0)/z)**(2.0/theta))+rho*(((z+1.0)/z)**(1.0/theta)))/(z+1.0))
#            
#    jacobian[1,1] = (contour_integral.contr_intgl(contr,f4)).real 
#    print ('J(2,2) is', jacobian[1,1])
#
#    def f5(z):    # for ff1, See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((2.0*tau*((c[0]*z+c[1])**2.0)*(((z+1.0)/z)**(2.0/theta))+rho*(c[0]*z+c[1])*(((z+1.0)/z)**(1.0/theta)))/(z)) 
#
#    ff[0] = (contour_integral.contr_intgl(contr,f5)).real  - (1.0+theta) 
#
#    def f6(z):    # for ff2, See notebook
#            return (1.0/(2.0*(math.pi)*1j))*((2.0*tau*((c[0]*z+c[1])**2.0)*(((z+1.0)/z)**(2.0/theta))+rho*(c[0]*z+c[1])*(((z+1.0)/z)**(1.0/theta)))/(z+1.0)) 
#
#    ff[1] = (contour_integral.contr_intgl(contr,f6)).real  - 1.0

    delta_X = numpy.linalg.solve(jacobian,ff)            
    
    c_next = c - delta_X    
         
    error = c_next - c
    print('error is', error)
#    print('renormalized parameters c1,c0 are', c[0],c[1])    
    
    c = c_next
    
#    print c[0],c[1]
    
#    if(numpy.amax(error)<=1e-2):
    if(abs(error[0])<=(1e-10) and abs(error[1])<=1e-10):
        break
print('renormalized parameters c1,c0 are', c[0],c[1])            