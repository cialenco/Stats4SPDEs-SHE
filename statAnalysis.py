# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11, 2020
This version: Tue July 06, 2021

@author: Igor Cialenco
         email: cialenco@gmail.com
         Web: http://cialenco.com


Part of the code compliments the theoretical developments in preprint: 
    Igor Cialenco, Hyun-Jung Kim, Parameter estimation for discretely sampled stochastic 
    heat equation driven by space-only noise, arXiv:2003.08920, 2020.
    
"""

import numpy as np
import math


class SPDEStats():
    
    def __init__(self):
        pass 


    def SHESpaceOnly_sigmacheck(theta, x_mesh, u):
        '''
        Estimate the volatility sigma by (3.21) arXiv:2003.08920

        Parameters
        ----------
        theta : double
            Theta, assumed to be known.
        x_mesh : array double
            uniform mesh of spatial points at which the solution was computed.
        u : 2D array double 
            value of the solution at time-space grid points.

        Returns
        -------
        sigma_est : TYPE array double
            estimated value of sigma at each time point
        
        sigma_estNM : double
            estimated value of sigma as average of estimates at each time, formuala (3.21)
        
        '''
        
        Nx=len(x_mesh)
        mu = 2/3 # correction factor from Theorem 3.6
    
        sigma_est = np.roll(u,1,1) - 2*u + np.roll(u,-1,1)
        sigma_est = np.power(sigma_est[:,1:-1],2)  
        sigma_est = np.sum(sigma_est, axis = 1)*(theta**2)*(Nx**2)/(math.pow(x_mesh[-1]- x_mesh[1],3)*mu)         
        sigma_est = np.power(sigma_est,0.5)               
        sigma_estNM = np.mean(sigma_est)
        return sigma_est, sigma_estNM


    def SHESpaceOnly_sigmatilde(theta, x_mesh, u):
        '''
        Estimate the volatility sigma by (3.23) arXiv:2003.08920
        Take a=1 and b=1, corresponding to central finite difference approximation 

        Parameters
        ----------
        theta : double
            Theta, assumed to be known.
        x_mesh : array double
            uniform mesh of spatial points at which the solution was computed.
        u : 2D array double 
            value of the solution at time-space grid points.

        Returns
        -------
        sigma_est : TYPE array double
            estimated value of sigma at each time point
        
        sigma_estNM : double
            estimated value of sigma as average of estimates at each time, formuala (3.21)
        
        '''
       
        
        Nx=len(x_mesh)
        mu = 5/12 # correction factor from Theorem 3.6
    
        sigma_est = np.roll(u,-1,1) - np.roll(u,1,1) - u + np.roll(u,2,1)
        sigma_est = np.power(sigma_est[:,2:-2],2)  
        sigma_est = np.sum(sigma_est, axis = 1)*(theta**2)*(Nx**2)/(math.pow(x_mesh[-2]- x_mesh[2],3)*4*mu)     
        sigma_est = np.power(sigma_est,0.5)               
        sigma_estNM = np.mean(sigma_est)
        return sigma_est, sigma_estNM


    def SHESpaceOnly_sigmahat(theta, x_mesh, udx):
        '''
        Estimate the volatility sigma by (3.5) arXiv:2003.08920
        by using values of the derivative

        Parameters
        ----------
        theta : double
            Theta, assumed to be known.
        x_mesh : array double
            uniform mesh of spatial points at which the solution was computed.
        udx : 2D array double 
            value of the derivative of solution at time-space grid points.

        Returns
        -------
        sigma_est : TYPE array double
            estimated value of sigma at each time point
        
        sigma_estNM : double
            estimated value of sigma as average of estimates at each time, formuala (3.21)
        
        '''   
        

        sigmaH_est = np.roll(udx,1,1) - udx 
        sigmaH_est = np.power(sigmaH_est[:,1:-1],2) # delete first and last col to elliminate the values at the boundary 
        sigmaH_est = np.sum(sigmaH_est, axis = 1)*(theta**2)/(x_mesh[-1]- x_mesh[1])   
        sigmaH_est = np.power(sigmaH_est,0.5)
    
        sigmaH_estNM = np.mean(sigmaH_est)

        return sigmaH_est, sigmaH_estNM