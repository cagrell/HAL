#from .distributions import MultivariateDistribution
#import numpy as np
#from scipy import stats, random

from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from scipy import stats, random
from ..sampler import PrunedSampler
from .SRA_baseclass import *

class SRA_epistemic(SRA_baseclass):
    """
    SRA with finite dimensional epistemic and aleatory uncertainties
    """
    
    def __init__(self, X_dist, E_dist, E_conservative):
        """
        Input:
        
        X_dist           -  Aleatory random variable (MultivariateDistribution)
        E_dist           -  Epistemic random variable (MultivariateDistribution)
        E_conservative   -  List [e1, e2, ..] of conservative (bad) values of E for use in MPP search
        """
        super().__init__(X_dist)
        self.E_dist = E_dist
        self.compute_epistemic_sigma() # Sigma points for E_dist
        self.E_conservative = E_conservative
        
        self.pruned_samples_X = None
        self.e = None # a single realization of the epistemic variable E
        
    def G(self, X):
        return self.G_e(X, self.e)
        
    def G_e(self, X, e):
        """
        Limit state 
        
        Input: 
        
        X = N * self.X_dist.dim array of N points in X-space
        e = a single realization of the epistemic variable E
        
        Output: N-dimensional numpy array with G(X) values
        """
        
        raise NotImplementedError('Need to implement custom limit state G(X, e)')

    def UT_pof_moments(self):
        """
        Estimate E[pof] and var[pof] with respect to epistemic uncertainties using UT
        """
        
        # PoF for each sigma point
        pof = np.array([self.pof_MCIS(e) for e in self.E_sigma])
        
        # Estimate moments
        pof_mean, pof_var = unscented_transform(pof.reshape(-1, 1), self.E_UT_points.Wm, self.E_UT_points.Wc)
        
        return pof_mean[0], pof_var[0][0]
    
    def UT_g_moments(self, X):
        """
        Estimate E[g(X)], Var[g(X)] with respect to epistemic uncertainties using UT
        """
        G = np.array([self.G_e(X, e) for e in self.E_sigma])

        m, v = unscented_transform(G, self.E_UT_points.Wm, self.E_UT_points.Wc)
        v = v.diagonal()
        
        return m, v
    
    def pof_MCIS(self, e):
        """
        Estimate pof for given e using MCIS
        """
                
        # Evaluate limit state
        g = self.G_e(self.pruned_samples_X, e)
        I = (g < 0)*1
        
        # Estimate pof
        pof = self.Sampler.compute_expectation(I)
        
        return pof
    
    def compute_epistemic_sigma(self):
        """
        Set sigma points used in UT for epistemic variables
        """
        self.E_UT_points = MerweScaledSigmaPoints(self.E_dist.dim, alpha = 0.9, beta = 2, kappa = 3-self.E_dist.dim)
        self.E_sigma = self.E_UT_points.sigma_points(np.zeros(self.E_dist.dim), np.eye(self.E_dist.dim))
        
    def generate_samples(self, n_MPP = 2, n_MPP_tries = 100, n_warmup = 1000, n_max = 100):
        """
        Generate samples used to estimate acquisition functions
        
        Input:
        
        n_MPP        -  try to find this number of MPP's
        n_MPP_tries  -  max number of tries in search for new MPP 
        n_warmup     -  warm up samples before pruning 
        n_max        -  max number of samples after pruning
        
        """
        
        # Find some center points for Gaussian mixture
        if n_MPP > 0:
            U_MPP = self.find_conservative_MPPS(n_MPP, n_MPP_tries)
            if len(U_MPP) > 0:
                U_MPP = np.append(U_MPP, np.zeros((1, self.X_dist.dim)), axis = 0)
            else:
                U_MPP = np.zeros((1, self.X_dist.dim))
        else:
            U_MPP = np.zeros((1, self.X_dist.dim))
        
        # Define sampling distribution 
        self.Sampler = PrunedSampler(U_MPP, n_warmup)
        
        # Evaluate pruning criterion
        X = self.X_dist.U_to_X(self.Sampler.samples_warmup)
        include_list, default_values = self.pruning_criterion(X)
        
        # Perform pruning
        self.Sampler.prune(include_list, default_values, n_max)
        
        # HOLD: catch this.. 
        if self.Sampler.N_pruned == 0:
            print('No samples generated!!!!')
    
        # Map samples to X-space and store
        self.pruned_samples_X = self.X_dist.U_to_X(self.Sampler.samples_pruned) 
        
    def find_conservative_MPPS(self, n_MPP = 2, n_MPP_tries = 100, err_g_max = 0.2):
        """
        Search for MPPs
        
        Input:
        
        n_MPP        -  try to find this number of MPP's
        n_MPP_tries  -  max number of tries in search for new MPP 
        err_g_max    -  convergence criterion in MPP search
        """
        
        k = 0
        U_MPP = []
        for i in range(n_MPP_tries):
            u0 = random.normal(size = (self.X_dist.dim))
            self.e = self.E_conservative[random.randint(len(self.E_conservative))]
            
            conv, u_MPP = self.MPP_search(u0 = u0, N_max = 100, err_g_max = err_g_max)
            if conv: 
                k += 1
                U_MPP.append(u_MPP)
                if k >= n_MPP: break
        
        return np.array(U_MPP)
    
    def pruning_criterion(self, X):
        """
        Evaluate each input in X with pruning criterion
        """
        
        # Estimate E[g(X)], Var[g(X)] with respect to epistemic
        # uncertainty for each x in X
        m, v = self.UT_g_moments(X)
                
        include_list = np.abs(m)/np.sqrt(v) < 3
        I = (m < 0)*1
       
        return include_list, I
    
    def bernoulli_var(self):
        """
        Estimate criteria based on Bernoulli variance 
        
        E[gamma] and E[sqrt(gamma)]**2
        """
        gamma = self.gamma(self.pruned_samples_X)
        
        # E[gamma]
        expectation_1 = self.Sampler.ratio_pruned*((gamma*self.Sampler.q_pruned).sum()/self.Sampler.N_pruned)
        
        # E[sqrt(gamma)]
        expectation_2 = self.Sampler.ratio_pruned*((np.sqrt(gamma)*self.Sampler.q_pruned).sum()/self.Sampler.N_pruned)
        
        return expectation_1, expectation_2**2
    
    def gamma(self, X):
        """
        gamma = p*(1-p) for p = P[g(x) < 0 | x]        
        """
        m, v = self.UT_g_moments(X)
            
        z = m/np.sqrt(v)
        phi = stats.norm.pdf(z)
        gamma = phi*(1-phi)
        
        return gamma