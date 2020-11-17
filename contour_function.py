# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 17:09:12 2018

@author: yadav
"""

# code to generate contour nu, nu1 and nu2 for J(s)=(c1s+c0)((s+1)/s)**(1/theta)
import math
import cmath
import numpy

#rho = 2.0    # V(x)=rho*x
#c = theta/rho    # eqn.(4.21) in C.R.


def contour_CR_prob_soft_edge(c1,c0,theta):
    
    ii = 1    # counter for rows in contour array
    f_out=file("input_output_files/alpha_8.0_rho_2.0/contour/nu_contour_function_soft_edge_18000points.txt","w")
#    f_out=open("input_output_files//contour//nu_contour_function_soft_edge_18000points.txt","w")    # open() can also be used to create file    
    
#    contour = numpy.empty([2*len(range(1,18000,1)),2],float)
    for i in range(1,18000,1):    #only integer step argument is available for range() hence 0 to 18000. to get 180 degrees will later divide by 100
        cos_phi =  math.cos(math.radians(i/100.0))
        sin_phi =  math.sin(math.radians(i/100.0))
        tan_phi =  math.tan(math.radians(i/100.0))
        tan_phibytheta =  math.tan(math.radians(i/100.0)/theta)
        tan_minusphibytheta =  math.tan(math.radians(-i/100.0)/theta)
#        phi = math.atan((-sin_phi_prime)/())
        a = 1.0
#        b = ((c1/c0)*(cos_phi*tan_phibytheta-sin_phi)*tan_phibytheta-2*cos_phi)
#        b = (c1/c0)*(cos_phi-(sin_phi)/tan_phibytheta)-2.0*cos_phi
        b = ((c1*sin_phi)/(c0*tan_minusphibytheta) + (c1*cos_phi)/(c0) - 2.0*cos_phi)
#        c = 1-(c1/c0)*(tan_phibytheta**2)
        c = 1.0-(c1/c0)
#        print i
        if(b**2.0-4.0*a*c>=0):
            r = (-b+(b**2.0-4.0*a*c)**(1.0/2))/(2.0*a)
            x = (r*cos_phi-1)/(r**2.0-2.0*r*cos_phi+1)
            y = (r*sin_phi)/(r**2.0-2.0*r*cos_phi+1)
            f_out.write(str(x)+" "+str(y)+" "+str(ii)+'\n')
#            f_out1.write(str(x)+" "+str(y)+'\n')
#            contour[ii-1,0] = x 
#            contour[ii-1,1] = y
            ii+=1
        else:
            break

#    print i
#    f_out.write(str(r)+" "+str(x)+" "+str(y)+" "+str(i)+'\n')
#        f_out.write(str(x)+" "+str(y)+'\n')
#        f_out1.write(str(x)+" "+str(y)+'\n')
#    f_out.close
#    f_out1.close
    for k in range(i-1,0,-1):    #only integer step argument is available for range() hence 0 to 18000. to get 180 degrees will later divide by 100

        cos_phi =  math.cos(math.radians(k/100.0))
        sin_phi =  math.sin(math.radians(k/100.0))
        tan_phi =  math.tan(math.radians(k/100.0))
        tan_phibytheta =  math.tan(math.radians(k/100.0)/theta)
        tan_minusphibytheta =  math.tan(math.radians(-k/100.0)/theta)
#        phi = math.atan((-sin_phi_prime)/())
        a = 1.0
#        b = ((c1/c0)*(cos_phi*tan_phibytheta-sin_phi)*tan_phibytheta-2*cos_phi)
#        b = (c1/c0)*(cos_phi-(sin_phi)/tan_phibytheta)-2.0*cos_phi
        b = ((c1*sin_phi)/(c0*tan_minusphibytheta) + (c1*cos_phi)/(c0) - 2.0*cos_phi)
#        c = 1-(c1/c0)*(tan_phibytheta**2)
        c = 1.0-(c1/c0)
#        print i
        if(b**2.0-4.0*a*c>=0):
            r = (-b-(b**2.0-4.0*a*c)**(1.0/2))/(2.0*a)
            x = (r*cos_phi-1)/(r**2.0-2.0*r*cos_phi+1)
            y = (r*sin_phi)/(r**2.0-2.0*r*cos_phi+1)
            f_out.write(str(x)+" "+str(y)+" "+str(ii)+'\n')
#            f_out1.write(str(x)+" "+str(y)+'\n')
#            contour[ii-1,0] = x 
#            contour[ii-1,1] = y 
            ii+=1
            
        else:
            break

#    f_out.close        
        
    for i in range(1,18000,1):    #only integer step argument is available for range() hence 0 to 18000. to get 180 degrees will later divide by 100

        cos_phi =  math.cos(math.radians(i/100.0))
        sin_phi =  math.sin(math.radians(i/100.0))
        tan_phi =  math.tan(math.radians(i/100.0))
        tan_phibytheta =  math.tan(math.radians(i/100.0)/theta)
        tan_minusphibytheta =  math.tan(math.radians(-i/100.0)/theta)
#        phi = math.atan((-sin_phi_prime)/())
        a = 1.0
#        b = ((c1/c0)*(cos_phi*tan_phibytheta-sin_phi)*tan_phibytheta-2*cos_phi)
#        b = (c1/c0)*(cos_phi-(sin_phi)/tan_phibytheta)-2.0*cos_phi
        b = ((c1*sin_phi)/(c0*tan_minusphibytheta) + (c1*cos_phi)/(c0) - 2.0*cos_phi)
#        c = 1-(c1/c0)*(tan_phibytheta**2)
        c = 1.0-(c1/c0)
#        print i
        if(b**2.0-4.0*a*c>=0):
            r = (-b-(b**2.0-4.0*a*c)**(1.0/2))/(2.0*a)
            x = (r*cos_phi-1)/(r**2.0-2.0*r*cos_phi+1)
            y = (-r*sin_phi)/(r**2.0-2.0*r*cos_phi+1)
            f_out.write(str(x)+" "+str(y)+" "+str(ii)+'\n')
#        f_out2.write(str(x)+" "+str(y)+'\n')
#            contour[ii-1,0] = x 
#            contour[ii-1,1] = y 
            ii+=1            
        
        else:
            break
        
#    f_out.close        
    
    for k in range(i-1,0,-1):    #only integer step argument is available for range() hence 0 to 18000. to get 180 degrees will later divide by 100

        cos_phi =  math.cos(math.radians(k/100.0))
        sin_phi =  math.sin(math.radians(k/100.0))
        tan_phi =  math.tan(math.radians(k/100.0))
        tan_phibytheta =  math.tan(math.radians(k/100.0)/theta)
        tan_minusphibytheta =  math.tan(math.radians(-k/100.0)/theta)
#        phi = math.atan((-sin_phi_prime)/())
        a = 1.0
#        b = ((c1/c0)*(cos_phi*tan_phibytheta-sin_phi)*tan_phibytheta-2*cos_phi)
#        b = (c1/c0)*(cos_phi-(sin_phi)/tan_phibytheta)-2.0*cos_phi
        b = ((c1*sin_phi)/(c0*tan_minusphibytheta) + (c1*cos_phi)/(c0) - 2.0*cos_phi)
#        c = 1-(c1/c0)*(tan_phibytheta**2)
        c = 1.0-(c1/c0)
#        print i
        if(b**2.0-4.0*a*c>=0):
            r = (-b+(b**2.0-4.0*a*c)**(1.0/2))/(2.0*a)
            x = (r*cos_phi-1)/(r**2.0-2.0*r*cos_phi+1)
            y = (-r*sin_phi)/(r**2.0-2.0*r*cos_phi+1)
            f_out.write(str(x)+" "+str(y)+" "+str(ii)+'\n')
#            f_out2.write(str(x)+" "+str(y)+'\n')
#            contour[ii-1,0] = x 
#            contour[ii-1,1] = y 
            ii+=1               
        
        else:
            break
      
    f_out.close()    # () at the end is necessary to close the file

    data = numpy.loadtxt("input_output_files/alpha_8.0_rho_2.0/contour/nu_contour_function_soft_edge_18000points.txt",float)      
    contour = data[:,0:2] # The plain [:] operator slices from beginning to end-1, Eg. a[:,0:n] gives all the rows from 0th column to n-1 column    
    
    return contour        