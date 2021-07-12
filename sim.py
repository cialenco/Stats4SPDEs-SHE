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
#import matplotlib.pyplot as plt
import math
#import scipy 

class SPDESimulation():    
    
    
    def __init__(self, theta, sigma):
        self.theta = 1   # drift 
        self.sigma = 1   # volatility    
       


    def SolSHEspaceOnly(self, t_mesh, x_mesh, NF):
        '''
        sumualte the solution to SPDE (1.1) arXiv:2003.08920 with G=[0,pi]
        using the explicit formula of the Fouries series representation; formula (3.36)

        Parameters
        ----------
        t_mesh : array of doubles
            time mesh of points at which the soluion will be computed.
        x_mesh : array of doubles
            a uniform grid of points at which the solution will be computed.
        NF : int
            number of Fourier modes used in the approximation (3.36).

        Returns
        -------
        u : TYPE 2D array of doubles.
            values of the slution u(t_j,x_i).

        '''
        
        Mt = len(t_mesh)
        Nx = len(x_mesh)
        u = np.zeros((Mt,Nx))
        ### general initial condition can be also included by uncomiting the lines below that have u0    
        #u0term = np.zeros((Mt,Nx)) 
    
        c = self.sigma/self.theta*math.sqrt(2/math.pi) # a constant in front of the sum (3.36)
        xi = np.random.normal(loc=0.0, scale=1.0, size=NF+1) # generate the Gaussian r.v. xi_k
        
#        temp_tu0 = np.exp(-self.theta*t_mesh)
#        temp_xu0 = np.sin(x_mesh)
#        u0term = np.transpose(np.array([temp_tu0]))*np.array([temp_xu0])*math.sqrt(2/math.pi)
         
        for k in range(1,NF):
            temp_t = 1-np.exp(-k**2*self.theta * t_mesh)       
            temp_x = np.sin(k*x_mesh)*xi[k]/(k**2)
            u += np.transpose(np.array([temp_t]))*np.array([temp_x])
            
        u = u*c
 #       u = u+u0term    
        return u
    
    
    def SolDervSHEspaceOnly(self, t_mesh, x_mesh, NF):
        '''
        Same as SolSHEspaceOnly but also outputs the value of the derivative in x of the solution 

        Parameters
        ----------
        t_mesh : array of doubles
            time mesh of points at which the soluion will be computed.
        x_mesh : array of doubles
            a uniform grid of points at which the solution will be computed.
        NF : int
            number of Fourier modes used in the approximation (3.36).

        Returns
        -------
        u : TYPE 2D array of doubles.
            values of the solution u(t_j,x_i).
            
        udx : 2D array of doubles.
            values of the u_x(t_j,x_i).

        '''
        
        
        Mt = len(t_mesh)
        Nx = len(x_mesh)
        u = np.zeros((Mt,Nx))
        udx = np.zeros((Mt,Nx))
        # u0term = np.zeros((Mt,Nx))
    
        c = self.sigma/self.theta*math.sqrt(2/math.pi)
        xi = np.random.normal(loc=0.0, scale=1.0, size=NF+1)
        
        
        # temp_tu0 = np.exp(-theta*t_mesh)
        # temp_xu0 = np.sin(x_mesh)
        # u0term = np.transpose(np.array([temp_tu0]))*np.array([temp_xu0])*math.sqrt(2/math.pi)
         
        for k in range(1,NF):
            temp_t = 1-np.exp(-k**2*self.theta * t_mesh)       
            temp_x = np.sin(k*x_mesh)*xi[k]/(k**2)
            temp_xder = np.cos(k*x_mesh)*xi[k]/(k)
            u += np.transpose(np.array([temp_t]))*np.array([temp_x])
            udx += np.transpose(np.array([temp_t]))*np.array([temp_xder])
            
        u = u*c
        #u = u+u0term
        udx = udx*c
    
        return u,udx


   