#!/usr/bin/env python
# coding: utf-8

# Regression functions

# In[1]:


import numpy as np
from numpy import linalg
import scipy.stats as stats


# In[26]:


class Linear_Regression:
    """Ordnary least Square regression"""
    def __init__(self):
        self._fit = False
        self.y = None
        self.X = None
        self.constant = False
        self.hat_matrix = None
        self.betas = None
        self.y_hat = None
        self.res = None
        
    def __str__(self):
        s  = f'Betas: {np.round(self.betas,2)}\n'
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
        
        # changing the fit flag
        self._fit = True
        
        return self.betas
    
    def predict(X):
        ''' returns predict values given new observations'''
        return X@self.betas
        
        
        # changing the fit flag
        self._fit = True
        self._res = self.y - self.y_hat
        


# ### Test Model
# # artificial example for linear regression
# X = stats.uniform.rvs(0,100,size = 100) # x is sample from a uniform distribuiton
# error = stats.norm.rvs(0,5, size = 100) # error has a normal distribution in our case
# b0 = 10
# b1 = 0.5
# # generating predictable variable
# y = b0 + b1*X + error

# model = Linear_Regression()
# model.fit(y = y, X = X)

