#!/usr/bin/env python
# coding: utf-8

# Regression functions


import numpy as np
import pandas as pd
from math import sqrt
from numpy import linalg
import scipy.stats as stats
import matplotlib.pyplot as plt
# Global parameters for plt graphics
plt.rcParams['font.size'] = 15
plt.rcParams['figure.figsize'] = [10,6]




class Linear_Regression:
    """Ordnary least Square regression"""
    def __init__(self):
        # fit attributes
        self._fit = False
        self.y = None
        self.X = None
        self.constant = False
        self.p = None
        self.n = None
        
        # estimation attributes
        self.hat_matrix = None
        self.betas = None
        self.y_hat = None
       
        # Residual analysis attributes
        self.res = None
        self.QMres = None
        self.res_std = None
        self.hat_diag = None
        self.res_stdt = None
        self.res_stdte = None
        self.S2 = None
        self.res_analysis = None
        
    def __str__(self):
        s  = f'Betas:\n {np.round(self.betas,2)}\n'
        return s 
        
    def fit(self, y, X, add_constant = True):
        """Set the variables and returns the estimation"""
        self.y = y
        
        # if simple linear regression and X is unidimension
        # adjusts X shape to bidimensional
        if len(X.shape) == 1:
            X = np.reshape(X,newshape = (len(X),1))
        
        # adds the constant column if True
        if add_constant == True:
            self.contant = True
            self.X = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
        else:
            self.X = X
        # n is the number of observations and is the number of rows in the X matrix
        self.n = X.shape[0]
        # P is the number of regressors and is the number of columns in the X matrix
        self.p = X.shape[1]
            
        return self._estimation()
        
    def _estimation(self):
        """Runs the estimation"""
        
        # semi hat matrix, is useful because can give us the hat matrix
        # and the beta estimations
        self._XX = linalg.inv(self.X.T @ self.X) @ self.X.T
        
        # the semi hat multiplied by right with the y vector returns the beta estimations
        self.betas =  self._XX @ self.y
        
        # the semi hat multiplied by left with the X matrix returns the hat matrix
        # the hat matrix maps the vector of observed independent variables
        # to a vector of estimations of the dependent variable
        self.hat_matrix = self.X @ self._XX
        
        # estimating observations
        self.y_hat = self.hat_matrix @ self.y
        # estimating the residuals
        self.res = self.y - self.y_hat
        self.res = self.res.reshape((self.n,1))
        
        # changing the fit flag
        self._fit = True
        
        return self.betas
        
        
    def residual_analysis(self):
        '''residual analysis'''
        # h: hat_diag is the diagonal of X matrix
        self.hat_diag = np.diagonal(self.hat_matrix).reshape((self.n,1))
        
        # d: res_std standirized residuals
        self.QMres = (self.res**2).sum()/(self.n - self.p-1)
        self.res_std = self.res / self.QMres**(1/2)
        

        # r: res_stdt is the studentized residuals
        self.res_stdt = self.res / (((self.QMres - (1-self.hat_diag))**(1/2)).reshape((self.n,1)))
        
        
        # S2: S2 is the model variance without the ith observation
        # t: res_stdte studentized residuals exterior, considers S2 and standart deviation item by item
        self.S2 = ((self.n- self.p)*self.QMres - self.res**2 / (1 - self.hat_diag)) / (self.n-self.p-1)
        self.res_stdte = self.res / (self.S2*(1-self.hat_diag))**(1/2)
        
        # residual analysis data frame
        self.res_analysis = np.concatenate([self.res,self.hat_diag,self.res_std,self.res_stdt,self.res_stdte], axis = 1)
        self.res_analysis = pd.DataFrame(self.res_analysis, 
                                         columns = ['residuals', 'hat_diag', 'standarized', 'studentized', 'studen_ext'])
        return self.res_analysis.round(4)
        