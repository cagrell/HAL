# Div util functions
import numpy as np

# Time 
def formattime(sec):
    """ Format time in seconds to h:m:s depending on sec """
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)

    if h > 0: return '{} hours {} minutes {} seconds'.format('%.0f'%h, '%.0f'%m, '%.0f'%s)
    if m > 0: return '{} minutes {} seconds'.format('%.0f'%m, '%.1f'%s)
    return '{} seconds'.format('%.3f'%s)

# Other random stuff
def len_none(x):
    return 0 if x is None else len(x)

def scale_to_bounds(x, bounds):
    """
    Input: x = points in [0, 1]^n
    Output: Scaled to lie in the n-dim box given by bounds
    """
    x_scaled = np.zeros(shape = x.shape)

    for i in range(x.shape[1]):
        x_scaled[:,i] = x[:,i]*(bounds[i][1] - bounds[i][0]) + bounds[i][0]
        
    return x_scaled
