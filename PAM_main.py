# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11, 2020
This version: Tue July 06, 2021

@author: Igor Cialenco
         email: cialenco@gmail.com
         Web: http://cialenco.com


The code compliments the theoretical developments in preprint: 
    Igor Cialenco, Hyun-Jung Kim, Parameter estimation for discretely sampled stochastic 
    heat equation driven by space-only noise, arXiv:2003.08920, 2020.

This is the main file. 

"""

import numpy as np
import matplotlib.pyplot as plt
import math
#import scipy 
import time as time

import statPlotFigs 
from sim import SPDESimulation
from statAnalysis import SPDEStats

dateFilesName = '07-10-21'  # used to name saved files and figures


#%%
if __name__ == "__main__":

    ### All reference to numbered formulas are from arXiv:2003.08920

    # simulate the solution

    t = time.time()
    
    # system parameters
    NF = 30000 # Number of Fourier modes in the approximation  
    sigma = 0.1 # see eq (1.1) 
    theta = 0.1 # see eq (1.1)
    
    # [T0,T] = time interval at which the solution will be evaluated
    T0 = 0.2 
    T = 1 
    Mt =  150 #int(T//dt) # number of points on [T0,T] 
    t_mesh = np.linspace(T0,T,Mt) # time mesh
    
    Nx = 1500 # number of points in space grid
    x_mesh = np.linspace(0,math.pi,Nx) # space grid/mesh
    
    u = np.zeros((Mt,Nx,3)) 
    # initiate the solution as 3D np.array. 
    # Third coordinate is paths
    
    spdeSim = SPDESimulation(theta, sigma)
    u[:,:,0] = spdeSim.SolSHEspaceOnly(t_mesh, x_mesh, NF)
    u[:,:,1] = spdeSim.SolSHEspaceOnly(t_mesh, x_mesh, NF)
    u[:,:,2] = spdeSim.SolSHEspaceOnly(t_mesh, x_mesh, NF)
   
     
    elapsed = time.time() - t
    print(elapsed)
    
    # #%%
    # # plot the solution

    # #from mpl_toolkits.mplot3d import Axes3D
    
    # #ax1 = plt.axes(projection='3d')
    # fig2=plt.figure(figsize=(17,7))
    # ax1=fig2.add_subplot(121,projection='3d')
    # ax2=fig2.add_subplot(122)
    
    # # slice a subset of data to be used for plots
    # t_pStep = 1 #Mt; # 80 = number of points in t used for plotting
    # x_pStep = 15 # Nx; # 40 = number of points in x used for plotting 
    
    # t_plot, x_plot = np.meshgrid(t_mesh[0:Mt:t_pStep], x_mesh[0:Nx:x_pStep]) # created a meshgrid
    # #t_plot, x_plot = np.meshgrid(tmesh, xmesh) # created a meshgrid
    
    
    # ax1.plot_surface(t_plot, x_plot, np.transpose(u[0:Mt:t_pStep,0:Nx:x_pStep,0]), 
    #                   rstride=1, cstride=1, cmap=plt.cm.Greys_r, linewidth=0.01, edgecolor='none')
    # # cmap=plt.cm.coolwarm, 
    # # cmap=plt.cm.bone
    
    # ax1.set_title(r'$u(t,x)$', fontsize = 16);
    # ax1.set_xlabel('t', fontsize =14)
    # ax1.set_ylabel('x', fontsize = 14)
    # ax1.set_zlabel('u');
    # ax1.view_init(25, 240)
    
    # #ax2.pcolormesh(t_plot, x_plot, np.transpose(u), cmap='coolwarm', shading='gouraud') # color plot
    # ax2.pcolormesh(t_plot, x_plot, np.transpose(u[0:Mt:t_pStep,0:Nx:x_pStep,0]), cmap='Greys_r', shading='gouraud') # grayscale plot
    # ax2.contour(t_plot, x_plot, np.transpose(u[0:Mt:t_pStep,0:Nx:x_pStep,0]), cmap='bone')
    
    # # # cmap='RdBu, inferno, bone, coolwarm'
    # ax2.set_title(r'$u(t,x)$, heatmap', fontsize=14);
    # ax2.set_xlabel('t', fontsize=14)
    # ax2.set_ylabel('x', fontsize=14)
    
    # fig2.set_rasterized(True)
    # # plt.savefig('solution-'+dateFilesName+'.eps')
    # # plt.savefig('solution-'+dateFilesName+'.jpg')
    
    # plt.show()
    
    
    
#%%

    # Estimate the volatility sigma, using different number of space points
    
    nr_Nmin = 20 # minimum number of points used for estimators is equal to N/nr_Min
    sigmaC_N = np.zeros((Mt+1,nr_Nmin-1,3)) # initiate the estimator, at all time points and all paths (3.21)
    sigmaC_NM = np.zeros((2,nr_Nmin-1,3)) # initiate the average estimator (3.21), at all paths
   
      
    sigmaTilde_N = np.zeros((Mt+1,nr_Nmin-1,3)) # # initiate the estimator, at all time points and all paths (3.23)
    sigmaTilde_NM = np.zeros((2,nr_Nmin-1,3)) # # initiate the average estimator (3.23), at all paths
   

    for j in range(3): # loop through paths               
        for qv_step in range(1,nr_Nmin): # loop using part of space points (to check convergence in N)
            x_mesh_temp = x_mesh[0:Nx:qv_step]
            u_temp = u[:,0:Nx:qv_step,j]
            
            sigmaC_N[0,qv_step-1,j] = len(x_mesh_temp)
            sigmaC_NM[0,qv_step-1,j] = len(x_mesh_temp)

            sigmaTilde_N[0,qv_step-1,j] = len(x_mesh_temp)
            sigmaTilde_NM[0,qv_step-1,j] = len(x_mesh_temp)

    
            sigmaC_N[1:, qv_step-1,j], sigmaC_NM[1,qv_step-1,j] = SPDEStats.SHESpaceOnly_sigmacheck(theta, x_mesh_temp, u_temp)
            sigmaTilde_N[1:, qv_step-1,j], sigmaTilde_NM[1,qv_step-1,j] = SPDEStats.SHESpaceOnly_sigmatilde(theta, x_mesh_temp, u_temp)
     
       
 #%%

    # plot the estimators
       
    plt.figure(figsize=(10,7))
    #### plot the graphs 
    # plot the true parameter sigma
    plt.plot(sigmaC_N[0,:,0], sigma*np.ones(len(sigmaC_N[0,:,0])), label = r'True $\sigma$', color ='black')
    
    # plot the estimated sigma using u at one time point
    plt.plot(sigmaC_N[0,:,0], sigmaC_N[1,:,0], 
              label = r'$\sqrt{\frac{3}{2}}\  \ \check\!\!\!\!\sigma_{N,1}, t = $'+str(round(t_mesh[0],3)), marker = 'x', color ='black', markersize=7)

  # plot the estimated sigma using u at one time point, 2nd path
    plt.plot(sigmaC_N[0,:,0], sigmaC_N[1,:,1], 
              marker = 'x', color ='black', markersize=7)

   # plot the estimated sigma using u at one time point, 3rd path
    plt.plot(sigmaC_N[0,:,0], sigmaC_N[1,:,2], 
               marker = 'x', color ='black', markersize=7)

    
    # plot the estimate sigma_MN using all time points available
    plt.plot(sigmaC_NM[0,:,0], sigmaC_NM[1,:,0], 
              label = r'$\sqrt{\frac{3}{2}} \  \ \check\!\!\!\!\sigma_{N,M}$', marker = '*', color ='black', markersize=8)

    # plot the estimate sigma_MN using all time points available, 2nd path
    plt.plot(sigmaC_NM[0,:,0], sigmaC_NM[1,:,1], 
              marker = '*', color ='black', markersize=8)

    # plot the estimate sigma_MN using all time points available, 3rd path
    plt.plot(sigmaC_NM[0,:,0], sigmaC_NM[1,:,2], 
              marker = '*', color ='black', markersize=8)

    
    plt.xlabel('$N$', fontsize=18);
    plt.title(r'Estimated value of $\sigma$ using $u(t,x)$', fontsize=18)
    plt.tick_params(direction='out', length=6, width=2, colors='black', labelsize=13)
    
    plt.legend(fontsize=18, frameon=False, loc=0)
    plt.savefig('sigmacheck_NM-'+dateFilesName+'.eps')
    plt.savefig('sigmacheck_NM-'+dateFilesName+'.jpg')
    plt.show()
    
    
    #%
    plt.figure(figsize=(10,7))
    #### plot the graphs  for tildeSigma (3.23)
    # plot the true parameter sigma
    plt.plot(sigmaTilde_N[0,:,0], sigma*np.ones(len(sigmaTilde_N[0,:,0])), label = r'True $\sigma$', color ='black')
    
    # plot the estimated sigma using u at one time point
    plt.plot(sigmaTilde_N[0,:,0], sigmaTilde_N[1,:,0], 
              label = r'$\sqrt{\frac{12}{5}}\  \widetilde{\sigma}_{N,1}, t = $'+str(round(t_mesh[0],3)), marker = 'x', color ='black', markersize=7)


  # plot the estimated sigma using u at one time point, 2nd path
    plt.plot(sigmaTilde_N[0,:,0], sigmaTilde_N[1,:,1], 
              marker = 'x', color ='black', markersize=7)

   # plot the estimated sigma using u at one time point, 3rd path
    plt.plot(sigmaTilde_N[0,:,0], sigmaTilde_N[1,:,2], 
               marker = 'x', color ='black', markersize=7)

    
    # plot the estimate sigma_MN using all time points available, 3rd path
    plt.plot(sigmaTilde_NM[0,:,0], sigmaTilde_NM[1,:,0], 
              label = r'$\sqrt{\frac{12}{5}} \  \widetilde{\sigma}_{N,M}$', marker = '*', color ='black', markersize=8)

    # plot the estimate sigma_MN using all time points available
    plt.plot(sigmaTilde_NM[0,:,0], sigmaTilde_NM[1,:,1], 
              marker = '*', color ='black', markersize=8)

    # plot the estimate sigma_MN using all time points available
    plt.plot(sigmaTilde_NM[0,:,0], sigmaTilde_NM[1,:,2], 
              marker = '*', color ='black', markersize=8)

    
    plt.xlabel('$N$', fontsize=18);
    plt.title(r'Estimated value of $\sigma$ using $u(t,x)$', fontsize=18)
    plt.tick_params(direction='out', length=6, width=2, colors='black', labelsize=13)
    
    plt.legend(fontsize=18, frameon=False, loc=0)
    plt.savefig('sigmatilde_NM-'+dateFilesName+'.eps')
    plt.savefig('sigmatilde_NM-'+dateFilesName+'.jpg')
    plt.show()
    
      
        
    #%%
#################################################################################
    # Checking the asymptotic normality of \check\sigma by MC 
#################################################################################
    # system parameters
    NF = 10000 # Number of Fourier modes in the approximation  
    sigma = 0.1
    theta = 0.1
    spdeSim = SPDESimulation(theta, sigma)
    
    T0 = 0.2
    T = 0.4 # terminal time
    Mt =  2 #int(T//dt) # number of points in time 
    t_mesh = np.linspace(T0,T,Mt)    
    
    Nx = 1000 # number of points in space grid
    x_mesh = np.linspace(0,math.pi,Nx) # space grid/mesh
    
    NSim = 10000 # number of MC simulations
    
    sigma_est_MC = np.zeros(shape=(Mt,NSim))
    sigmaNM_est_MC = np.zeros(shape=(1,NSim))

    #sigma_est2_MC = np.zeros(shape=(Mt,NSim))
    #sigmaNM_est2_MC = np.zeros(shape=(1,NSim))


    
    n0 = 0
    n1 = -1
    x_meshUsedEst = x_mesh[n0:n1] # space mesh to be used in estimation

    for n in range(NSim):
        print(n)
        u = spdeSim.SolSHEspaceOnly(t_mesh, x_mesh, NF)
        sigma_est_MC[:,n], sigmaNM_est_MC[0,n] = SPDEStats.SHESpaceOnly_sigmacheck(theta, x_meshUsedEst, u[:,n0:n1])
        #sigma_est2_MC[:,n], sigmaNM_est2_MC[0,n] = SPDEStats.SHESpaceOnly_sigmatilde(theta, x_meshUsedEst, u[:,n0:n1])
     
        
    # import pickle
    # f = open('sigma_est_NF10KNx1KNSim1K-'+dateFilesName+'.pckl', 'wb')
    # pickle.dump(sigma_est_MC, f)
    # f.close()  
    
    #%%
    fileName = 'sigmaCheckN_' + dateFilesName
    histPlot = statPlotFigs.plotFigs()
    normConst_CLT = np.sqrt(2*(len(x_meshUsedEst)-1))/sigma
    histPlot.plotAsymptNorm(theta, sigma_est_MC[-1,:], normConst_CLT, fileName, '')
  
    fileName = 'sigmaCheckNM_' + dateFilesName
    normConst_CLT = np.sqrt(2*(len(x_meshUsedEst)-1))/sigma
    histPlot.plotAsymptNorm(theta, sigmaNM_est_MC[0,:], normConst_CLT, fileName, '')
    

    # fileName = 'sigmaTildeN_' + dateFilesName
    # histPlot = statPlotFigs.plotFigs()
    # normConst_CLT = np.sqrt(2*(len(x_meshUsedEst)-1))/sigma
    # histPlot.plotAsymptNorm(theta, sigma_est2_MC[-1,:], normConst_CLT, fileName, '')
  
    # fileName = 'sigmaTildeNM_' + dateFilesName
    # normConst_CLT = np.sqrt(2*(len(x_meshUsedEst)-1))/sigma
    # histPlot.plotAsymptNorm(theta, sigmaNM_est2_MC[0,:], normConst_CLT, fileName, '')
    
    

    
    #%%
      
    #################################################################################################################
    #################################################################################################################
    ########################################   u_x   ################################################################
    #################################################################################################################
    
    ######################################%%%%%%%%%%%%%%%%
    # estimating sigma by using \hat\sigma and values of u_x(tx)
    ######################################%%%%%%%%%%%%%%%%

    #dateFilesName = '06-25-21'    
    
    
    # simulate one path of the solution and its derivative
    t = time.time()
    
    # system parameters
    NF = 30000 # Number of Fourier modes in the approximation  
    sigma = 0.1
    theta = 0.1
    
    spdeSimDer = SPDESimulation(theta, sigma)
    
    T0 = 0.2
    T = 1 # terminal time
    Mt =  150 #int(T//dt) # number of points in time 
    t_mesh = np.linspace(T0,T,Mt)
    
    Nx = 1500 # number of points in space grid
    x_mesh = np.linspace(0,math.pi,Nx) # space grid/mesh   
    
    u,udx = spdeSimDer.SolDervSHEspaceOnly(t_mesh, x_mesh, NF)    
        
    elapsed = time.time() - t
    print(elapsed)


    #%%
    nr_Nmin = 50 # minimum number of points N used for estimators 
    sigmaH_N = np.zeros((Mt+1,nr_Nmin-1))
    sigmaH_NM = np.zeros((2,nr_Nmin-1))
    
    for qv_step in range(1,nr_Nmin):
        x_mesh_temp = x_mesh[0:Nx:qv_step]
        udx_temp = udx[:,0:Nx:qv_step]
    
        sigmaH_N[0,qv_step-1] = len(x_mesh_temp)
        sigmaH_NM[0,qv_step-1] = len(x_mesh_temp)
        sigmaH_N[1:, qv_step-1], sigmaH_NM[1,qv_step-1] = SPDEStats.SHESpaceOnly_sigmahat(theta, x_mesh_temp, udx_temp)

        
        
    #%%    
    plt.figure(figsize=(10,7))
    #### plot the graphs 
    # plot the true parameter sigma
    plt.plot(sigmaH_N[0,:], sigma*np.ones(len(sigmaH_N[0,:])), label = r'True $\sigma$', color ='black')
    
    # plot the estimated sigma using u at one time point
    plt.plot(sigmaH_N[0,:], sigmaH_N[1,:], 
              label = r'$\widehat\sigma_{N,1}$', marker = 'x', color ='black', markersize=7)
    
    # plot the estimate sigma_MN using all points
    plt.plot(sigmaH_NM[0,:], sigmaH_NM[1,:], 
              label = r'$\widehat\sigma_{N,M}$', marker = 'o', color ='black', markersize=7)
    
    plt.xlabel('$N$', fontsize=18);
    plt.title(r'Estimated value of $\sigma$ using $u_x(t,x)$', fontsize=18)
    plt.tick_params(direction='out', length=6, width=2, colors='black', labelsize=13)
    
    plt.legend(fontsize=17, frameon=False)
    #plt.savefig('sigmabar_NM-'+dateFilesName+'.eps')
    plt.savefig('sigmabar_NM-'+dateFilesName+'.jpg')
    plt.show()

    
    
    
    
    

