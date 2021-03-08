# Optimization functions

# ADD TO /Utils/Optimize..
from scipy import random
from scipy.optimize import minimize
from .div import scale_to_bounds

def gopt_min(fun, bounds, n_warmup = 1000, n_local = 10):
    """
    Global optimization (minimization) based on:
    
    1. Sampling 'n_warmup' uniformly random points
    2. Local optimization (L-BFGS-B) at 'n_local' + 1 points
        - 'n_local' uniformly random starting points
        -  best location from step (1) as starting point
        
        
    Input:
        fun : vectorized function
        bounds : list of tuples [(min, max), (min, max), ... ]
        n_warmup : number of warmup steps
        n_local : number of local optimizations
        
    """
    
    # Input dimension
    dim = len(bounds)
    
    # Warm up samples
    x_warmup = scale_to_bounds(random.uniform(size = (n_warmup, dim)), bounds)

    # Best from warmup
    y = fun(x_warmup)
    idx = y.argmin()
    y_best = y[idx]
    x_best = x_warmup[idx]
    
    # Run local optimization
    if n_local > 0:
        
        # Starting points for local optimization
        x_local = scale_to_bounds(random.uniform(size = (n_local, dim)), bounds)
    
        for x in x_local:
            res = minimize(fun = fun, x0 = x, bounds = bounds, method = 'L-BFGS-B')
            if res.success:
                if res.fun[0] < y_best:
                    y_best = res.fun[0]
                    x_best = res.x
        
    return x_best, y_best

    
def gopt_max(fun, bounds, n_warmup = 1000, n_local = 10):
    """
    Global optimization (maximization) using 'gopt_min'
    """
    x_best, y_best = gopt_min(lambda x: -fun(x), bounds, n_warmup, n_local)
    return x_best, -y_best