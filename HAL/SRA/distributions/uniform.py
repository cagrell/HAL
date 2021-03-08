from .continuous_marginal import *
from ...util.stats import normal_cdf_approx
from scipy import stats

class Uniform(ContinuousMarginalDistribution):
    """
    Uniform distribution specified by lb (lower bound) and ub (upper bound)
    """
    
    def __init__(self, name, lb, ub):
        
        super().__init__(name, 'Uniform')
        self.lb = lb
        self.ub = ub
    
    def __str__(self):
        """ What to show when the object is printed """
        return '{} - {} : lb = {}, ub = {}'.format(self.name, self.type, self.lb, self.ub)
    
    def x_to_u(self, x):
        """ Transformation from physical to standard normal space """
        return stats.norm.ppf((x - self.lb)/(self.ub - self.lb))
        
    def u_to_x(self, u):
        """ Transformation from standard normal to physical space """
        # Use approximate normal cdf for speed.. 
        return normal_cdf_approx(u)*(self.ub - self.lb) + self.lb