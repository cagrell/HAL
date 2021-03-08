from .continuous_marginal import *
import numpy as np 

class MultivariateDistribution():
    """
    Multivariate distribution for use in FORM
    
    -- Only a list of marginals for now, will include copula later -- 
    """
    def __init__(self):
        self.marginals = []
        self.dim = 0
        
    def AddVariable(self, variable):
        """
        Add a new random variable from marginal distribution
        """
        assert isinstance(variable, ContinuousMarginalDistribution)
        self.marginals.append(variable)
        self.dim += 1
        
    def Describe(self):
        """
        Print a description of the multivariate distribution
        """
        if len(self.marginals) == 0:
            print('Empty distribution object')
        else:
            print('{} dim multivariate distribution:'.format(self.dim))
            for i in range(len(self.marginals)):
                print(self.marginals[i])
                
    def X_to_U(self, X):
        """ 
        Transformation from physical to standard normal space 
        
        X = N * self.dim array of N points in X-space
        U = N * self.dim array of N points in U-space
        """
        return np.array([self.marginals[i].x_to_u(X[:,i]) for i in range(self.dim)]).T
        
    def U_to_X(self, U):
        """ 
        Transformation standard normal to physical space 
        
        X = N * self.dim array of N points in X-space
        U = N * self.dim array of N points in U-space
        """
        return np.array([self.marginals[i].u_to_x(U[:,i]) for i in range(self.dim)]).T