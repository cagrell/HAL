import numpy as np
import scipy as sp

def norm_cdf_int(mu, std, LB, UB):
    """ Return P(LB < X < UB) for X Normal(mu, std) """
    rv = sp.stats.norm(mu, std)
    return rv.cdf(UB) - rv.cdf(LB)

def norm_cdf_int_approx(mu, std, LB, UB):
    """ 
    Return P(LB < X < UB) for X Normal(mu, std) using approximation of Normal CDF 
    
    Input: All inputs as 1-D arrays
    """
    l = normal_cdf_approx((LB - mu)/std)
    u = normal_cdf_approx((UB - mu)/std)
    return u - l

def normal_cdf_approx(x):
    """ 
    Approximation of standard normal CDF
    
    Input: x = array
    
    Polynomial approximation from Abramowitz and Stegun p. 932
    http://people.math.sfu.ca/~cbm/aands/frameindex.htm
    
    Absolute error < 7.5*10^-8
    """
    p = 0.2316419
    b = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
    
    xx = abs(x) # Approximation only works for x > 0, return 1 - p otherwise
    
    t = 1/(1 + p*xx)
    Z = (1/(np.sqrt(2*np.pi)))*np.exp(-(x*x)/2)
    pol = b[0]*t + b[1]*(t**2) + b[2]*(t**3) + b[3]*(t**4) + b[4]*(t**5)
    
    prob = 1 - Z*pol # For x > 0
    prob[x < 0] = 1 - prob[x < 0] # Change when x < 0
    
    return prob

