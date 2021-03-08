class ContinuousMarginalDistribution():
    """
    Superclass for 1d continuous distributions
    """
    
    def __init__(self, name, dist_type):
        self.name = name
        self.type = dist_type
        
    def __str__(self):
        """ What to show when the object is printed """
        return '{} DESCRIPTION NOT AVAILABLE'.format(self.name)
    
    def x_to_u(self, x):
        """ Transformation from physical to standard normal space """
        raise NotImplementedError('Need to implement u_to_x() for distribution type = {}'.format(self.type))
        
    def u_to_x(self, u):
        """ Transformation from standard normal to physical space """
        raise NotImplementedError('Need to implement x_to_u() for distribution type = {}'.format(self.type))