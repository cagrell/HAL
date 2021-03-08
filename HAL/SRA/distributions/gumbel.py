from .continuous_marginal import *
from ...util.stats import normal_cdf_approx
from scipy import stats
import numpy as np

class Gumbel(ContinuousMarginalDistribution):
    """
    Gumbel distribution specified by mean and standard deviation
    """
    def __init__(self, name, mean, std):
        
        super().__init__(name, 'Gumbel')
        self.mean = mean
        self.std = std
        self.beta = std*np.sqrt(6)/np.pi 
        self.mu = mean - self.beta*np.euler_gamma

    def __str__(self):
        """ What to show when the object is printed """
        return '{} - {} : mean = {}, std = {}'.format(self.name, self.type, self.mean, self.std)
    
    def CDF(self, x):
        return np.exp(-np.exp(-(x - self.mu)/self.beta))
    
    def CDF_inv(self, x):
        return self.mu -self.beta*np.log(-np.log(x)) 
    
    def x_to_u(self, x):
        """ Transformation from physical to standard normal space """
        return stats.norm.ppf(self.CDF(x))
        
    def u_to_x(self, u):
        """ Transformation from standard normal to physical space """
        # Use approximate normal cdf for speed.. 
        return self.CDF_inv(normal_cdf_approx(u))