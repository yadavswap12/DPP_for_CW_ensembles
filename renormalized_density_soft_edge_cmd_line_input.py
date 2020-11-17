# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:55:42 2019

@author: yadav
"""

#code for computation of renormalized density psi(x) for soft-edge with V(x)=rho*x in application of NUS paper method to solved example in CR paper. 
import math
import cmath
import numpy
import contour_integral

#theta = float(raw_input("theta is: "))    # asks for user input from command line. To be used in terminal.    
gamma = float(raw_input("gamma is: "))    # asks for user input from command line. To be used in terminal.    
rho = float(raw_input("rho is: "))    # asks for user input from command line. To be used in terminal.    
#alpha = float(raw_input("alpha is: "))    # asks for user input from command line. To be used in terminal.    

c1 = float(raw_input("c1 is: "))    # renormalized c1 for gamma=0.9 computed from python file 'renormalized_joukowsky_parameter_c.py'    # asks for user input from command line. To be used in terminal.    
c0 = float(raw_input("c0 is: "))    # renormalized c for gamma=0.9 computed from python file 'renormalized_joukowsky_parameter_c.py'    # asks for user input from command line. To be used in terminal.    
iteration = float(raw_input("iteration is: ")) 

#c = theta/rho    # eqn.(4.21) in C.R.
#c1 = 0.873607034265    # renormalized c1 for gamma=0.9 computed from python file 'renormalized_joukowsky_parameters_c1_c0.py'
#c0 = 0.91688096976    # renormalized c0 for gamma=0.9 computed from python file 'renormalized_joukowsky_parameters_c1_c0.py' 
 
#b0=-0.09657    # parameters of polynomial fit for f1(x), f1(x)=b0 + b1x + b2x^2 + b3x^3 + b4x^4 
#b1=2.15722    # gamma = 0.9
#b2=0.1613
#b3=-0.08054
#b4=0.01721
#b5=0.29999
#b6=-0.17164

b0 = float(raw_input("b0 is: "))    # parameters of polynomial fit for f1(x), f1(x)=b0 + b1x + b2x^2 + b3x^3 + b4x^4 
b1 = float(raw_input("b1 is: "))    # gamma = 0.9
b2 = float(raw_input("b2 is: "))
b3 = float(raw_input("b3 is: "))
b4 = float(raw_input("b4 is: "))
b5 = float(raw_input("b5 is: "))
b6 = float(raw_input("b6 is: "))
b7 = float(raw_input("b7 is: "))
b8 = float(raw_input("b8 is: "))
 
epsi = 1e-4
data1 = numpy.loadtxt("input_output_files/rho_"+str(rho)+"/gamma_"+str(gamma)+"/contour/nu1_contour_gamma="+str(gamma)+"_soft_edge_18000points_iter"+str(iteration)+".txt",float)
data2 = numpy.loadtxt("input_output_files/rho_"+str(rho)+"/gamma_"+str(gamma)+"/mapping/mapping_output_nu1_gamma="+str(gamma)+"_soft_edge_18000points_iter"+str(iteration)+".txt",float)
data3 = numpy.loadtxt("input_output_files/rho_"+str(rho)+"/gamma_"+str(gamma)+"/contour/nu2_contour_gamma="+str(gamma)+"_soft_edge_18000points_iter"+str(iteration)+".txt",float)
data4 = numpy.loadtxt("input_output_files/rho_"+str(rho)+"/gamma_"+str(gamma)+"/mapping/mapping_output_nu2_gamma="+str(gamma)+"_soft_edge_18000points_iter"+str(iteration)+".txt",float)
contr = numpy.loadtxt("input_output_files/rho_"+str(rho)+"/gamma_"+str(gamma)+"/contour/nu_contour_gamma="+str(gamma)+"_soft_edge_18000points_iter"+str(iteration)+".txt",float)

f_out=file("input_output_files/rho_"+str(rho)+"/gamma_"+str(gamma)+"/density/renormalized_density_psi_method_2_epsi=1e-4_gamma="+str(gamma)+"_rho="+str(rho)+"_soft_edge_18000points_corrected4_iter"+str(iteration)+".txt","w")

for i in range(len(data1)):    # function len() on array gives no. of rows of array
    x_1 = data1[i,0]
    y_1 = data1[i,1]
    x_2 = data3[len(data3)-i-1,0]
    y_2 = data3[len(data3)-i-1,1]
    r1 = math.sqrt(x_1**2.0+y_1**2.0)
    r2 = math.sqrt(x_2**2.0+y_2**2.0)
    x1 = data2[i,0]
#    s = complex(x,y+epsi)
#    s_conj = numpy.conjugate(s)
    s1 = complex(x_1*(1.0+epsi/r1),y_1*(1.0+epsi/r1))
    s_conj1 = numpy.conjugate(s1)
    s2 = complex(x_2*(1.0+epsi/r2),y_2*(1.0+epsi/r2))
    s_conj2 = numpy.conjugate(s2)

    def Jc(z):    # joukowsky transformation for hard edge
            return c1*z+c0-cmath.log((z-1.0/2)/(z+1.0/2))    
    
#    def f(z):    # using first identity in eqn.(3.23) and eqn.(3.21) in C.W.
#        return (-1.0/(4.0*(math.pi**2.0)*x1))*(b0+b1*Jc(z)+b2*(Jc(z))**2.0+b3*(Jc(z))**3.0+b4*(Jc(z))**4.0)*((1.0/(z-s1))-(1.0/(z-s2)))

#    def f(z):    # using first identity in eqn.(3.23) and eqn.(3.21) in C.W.
#        return (-1.0/(4*(math.pi**2)*x1))*(b0+b1*Jc(z)+b2*(Jc(z))**2+b3*(Jc(z))**3+b4*(Jc(z))**4+b5*(Jc(z))**5+b6*(Jc(z))**6)*((1/(z-s1))-(1/(z-s2)))        

    def f(z):    # using first identity in eqn.(3.23) and eqn.(3.21) in C.W.
        return (-1.0/(4*(math.pi**2)))*(b0+b1*Jc(z)+b2*(Jc(z))**2+b3*(Jc(z))**3+b4*(Jc(z))**4+b5*(Jc(z))**5+b6*(Jc(z))**6+b7*(Jc(z))**7+b8*(Jc(z))**8)*((1/(z-s1))-(1/(z-s2)))        
        
# complex(0.667727717428,0.772534506901) in above function lies on contour nu1
# complex(0.667727717428,0.772534506901+epsi) approaches contour nu1 from above
    psi = contour_integral.contr_intgl(contr,f)     
    #contour_integral = contour_integral.contr_intgl(contr,f)
    #psi = contour_integral
    f_out.write(str(x1)+" "+str(psi.real)+'\n')
    print int(i)
f_out.close()    # () at the end is necessary to close the file     
#print numpy.nan_to_num(contour_integral) 
#-cmath.log((z-1.0/2)/(z+1.0/2))