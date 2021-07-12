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
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import pylab
    
class plotFigs():
    
    
    def __init__(self):
            pass
        
         
    def plotAsymptNorm(self, theta, estThetaMC, normConst, fileName, title):
        '''
        a) plot normalized histogram superposed on N(0,1) density.
        b) plot the QQ plot of sample quantiles vs impirical quantiles. 
        both plots are saved as JPG and EPS files

    
        Parameters
        ----------
        theta : double
            true value of the parameter.
        estThetaMC : array double
            estimated values of the parameter.
        normConst : double
            normalized constant.
        fileName : str
            main part of the file name, e.g. '07-07-21'.
        title : str
            title of the plots, optional.

        Returns
        -------
        None.

        '''
        
        normedEstThetaMC = [(est - theta)*normConst for est in estThetaMC]
        
        plt.hist(normedEstThetaMC, bins = 100, density=True, color='LightGray', label='Empirical pdf')
        
        bins = np.linspace(-3, 3, 100)
        bin_centers = 0.5*(bins[1:] + bins[:-1])
        
        # Compute the PDF on the bin centers from scipy distribution object
        pdf = stats.norm.pdf(bin_centers)
        plt.plot(bin_centers, pdf, label="Theoretical pdf", color='Black')
        plt.legend(fontsize=12, frameon=False)
        plt.title(title)
        plt.savefig('Hist_'+fileName+'.jpg')
        plt.savefig('Hist_'+fileName+'.eps')
 
        plt.show()
        
        # q-q plot 
        theta_est_norm =  np.array(normedEstThetaMC)
        qq=sm.qqplot(theta_est_norm, marker='.', markerfacecolor='k', markeredgecolor='k', fmt='k--')
        sm.qqline(qq.axes[0], line='45', fmt='k--')
        
        plt.savefig('QQ-'+fileName+'.jpg')
        plt.savefig('QQ-'+fileName+'.eps')

        pylab.show()
    # return
        