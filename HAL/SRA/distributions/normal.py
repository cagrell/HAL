from .continuous_marginal import *

class Normal(ContinuousMarginalDistribution):
    """
    Normal distribution specified by mean and standard deviation
    """
    
    def __init__(self, name, mean, std):
        
        super().__init__(name, 'Normal')
        self.mean = mean
        self.std = std
    
    def __str__(self):
        """ What to show when the object is printed """
        return '{} - {} : Mean = {}, Std = {}'.format(self.name, self.type, self.mean, self.std)
    
    def x_to_u(self, x):
        """ Transformation from physical to standard normal space """
        return (x - self.mean)/self.std
        
    def u_to_x(self, u):
        """ Transformation from standard normal to physical space """
        return self.mean + u*self.std