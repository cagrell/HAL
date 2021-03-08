from scipy import random, stats
import numpy as np

class GMM():
    """
    Simple GMM of n multivariate Gaussians, all with unit variance and equal weights
    """
    def __init__(self, means):
        """
        Input: 
        means = (N, d) shaped array of N center points in the d-dimensional standard normal space  
        """
        self.means = means
        self.dim = means.shape[1]
        self.m = means.shape[0]    # Will create a mixture of m multivariate Gaussians
        
    def sample(self, N):
        """
        Generate N samples
        """
        
        # Standard normal samples
        U = random.normal(size = (N, self.dim))
        
        # Sample random means with equal probability
        random_means = self.means[list(random.randint(self.m, size = N)),:]
        
        return U + random_means
    
    def pdf(self, X):
        """
        Density of samples X
        """
        pdf_U = np.array([np.prod(stats.norm.pdf(X-mu), 1) for mu in self.means])
        pdf_U = pdf_U.sum(axis = 0)/self.m        
        return pdf_U
    
class PrunedSampler():
    """
    Multivariate Gaussian mixture with pruning based on how likely samples are 
    to effect MCIS estimates of the expectation E[f(X)] for some given function f(X)
    """
    def __init__(self, means, N_warmup = 1000):
        """
        Input: 
        means = (N, d) shaped array of N center points in the d-dimensional standard normal space  
        """
        self.N_warmup = N_warmup
        self.means = means
        self.dim = means.shape[1]
        self.m = means.shape[0]    # Will create a mixture of m multivariate Gaussians
        
        # Generate a (large) initial sample of size N_warmup
        self.generate_warmup_samples(N_warmup)
        
    def generate_warmup_samples(self, N_warmup):
        """
        Generate N_warmup samples from Gaussian mixture
        """
        
        self.GMM_model = GMM(self.means)
        self.samples_warmup = self.GMM_model.sample(N_warmup)
        self.pdf_warmup = self.GMM_model.pdf(self.samples_warmup)

    def prune(self, include_list, default_values, N_max):
        """
        Input: 
        include_list    -  boolean list of length N_warmup
        default_values  -  array of length N_warmup with default values
        
        Creates a reduced sample of size <= N_max as follows:
        1. Remove samples where include_list = False
        2. If the number of remaining samples is > N_max, generate N_max subsamples
        """
        
        # The samples to include
        self.samples_pruned = self.samples_warmup[include_list]
        self.pdf_pruned = self.pdf_warmup[include_list]
        self.N_pruned = self.samples_pruned.shape[0]
        self.ratio_pruned = self.N_pruned / self.N_warmup 
        
        # Downsample if necessary 
        if self.N_pruned > N_max:
            
            idx = list(random.randint(self.N_pruned, size = N_max))
            self.samples_pruned = self.samples_pruned[idx, :]
            self.pdf_pruned = self.pdf_pruned[idx]
            self.N_pruned = N_max
           
        # When MCIS is run on the pruned subset, we must also include
        # the assumed expectation on the left-out samples
        
        # The excluded samples
        exclude_list = np.logical_not(include_list)
        samples_excluded = self.samples_warmup[exclude_list]
        pdf_excluded = self.pdf_warmup[exclude_list]
        default_excluded = default_values[exclude_list]
        
        normal_pdf_excluded = np.prod(stats.norm.pdf(samples_excluded), 1) 
        q_excluded = normal_pdf_excluded / pdf_excluded

        if len(default_excluded) == 0:
            self.expectation_excluded = 0
        else:    
            self.expectation_excluded = (default_excluded*q_excluded).mean() # The assumed expectation of excluded samples
        
        # The density of each pruned sample in the standard normal space
        self.normal_pdf_pruned = np.prod(stats.norm.pdf(self.samples_pruned), 1) 
        
        # The density fractions used in MCIS
        self.q_pruned = self.normal_pdf_pruned / self.pdf_pruned

    def compute_expectation(self, f_x):
        """
        Estimate E[f(X)] using pruned MCIS
        
        Input:
        f_x = array containing [f(x_1), f(x_2), ... ] for each sample x_i in self.samples_pruned
        """
        
        # Expectation of pruned samples
        self.expectation_pruned = (f_x*self.q_pruned).sum()/self.N_pruned 
        
        # Return complete expectation of all samples
        return (1 - self.ratio_pruned)*self.expectation_excluded + self.ratio_pruned*self.expectation_pruned
        
        