from .distributions import MultivariateDistribution
import numpy as np
from scipy import stats, random

class SRA_baseclass():
    """
    Superclass for SRA with MC, MC_IS and design point search
    """
    
    def __init__(self, X_dist, LimitState = None):
        assert isinstance(X_dist, MultivariateDistribution)
        self.X_dist = X_dist 
        self.LimitState = LimitState
        self.Set_gradient_delta(0.000001) # step size for gradient estimation
        
        self._results_MPP = None
        
    def G(self, X):
        """
        Overload this when the limit state is not a simple python function 
        
        Input: X = N * self.X_dist.dim array of N points in X-space
        Output: N-dimensional numpy array with G(X) values
        """
        return np.apply_along_axis(self.LimitState, 1, X)
    
    def Run_MC(self, N_MC):
        """
        Crude Monte Carlo
        
        TODO: if N_MC = None -> run until cov is acceptable
        
        Output:
        beta     -   Reliability index (for comparison with FORM)
        pof      -   Probability of failure
        cov      -   Coefficient of variation
        """
        # Sample in U space
        U = random.normal(size = (N_MC, self.X_dist.dim))
        
        # Trasform to X space
        X = self.X_dist.U_to_X(U)
        
        # Compute limit state and pof
        g = self.G(X)
        I = (g < 0)*1
        pof = I.mean()
        
        # Coefficient of variation
        if pof > 0:
            var = pof*(1-pof)/N_MC
            cov = np.sqrt(var)/pof
        else:
            cov = 1
        
        # Reliability index
        beta = -stats.norm.ppf(pof)
        
        return beta, pof, cov
    
    def Run_MCIS(self, N_MC, u0 = None):
        """
        MC with importance sampling
        
        TODO: if N_MC = None -> run until cov is acceptable
        
        Output:
        beta     -   Reliability index (for comparison with FORM)
        pof      -   Probability of failure
        cov      -   Coefficient of variation
        """
        
        # Get MPP in U-space
        if u0 is None:
            conv, MPP_u = self.MPP_search()
        else:
            conv, MPP_u = True, u0
        
        # Sample in U space
        U = random.normal(size = (N_MC, self.X_dist.dim))
        pdf_U = np.prod(stats.norm.pdf(U), 1) 
        
        # Shift samples to design point
        U_shifted = U + MPP_u
        pdf_U_shifted = np.prod(stats.norm.pdf(U_shifted), 1) 
        
        # Trasform to X space
        X = self.X_dist.U_to_X(U_shifted)
        
        # Evaluate limit state
        g = self.G(X).flatten()
        I = (g < 0)*1
        
        # Estimate pof
        q = I*(pdf_U_shifted / pdf_U)
        pof = (1/N_MC)*q.sum()
        
        # .. and CoV
        if pof > 0:
            var = (1/N_MC)*(1/(N_MC-1))*((q - pof)**2).sum()
            cov = np.sqrt(var)/pof
        else:
            cov = 1
        
        # Reliability index
        beta = -stats.norm.ppf(pof)
        
        return beta, pof, cov
    
    def Run_FORM(self, u0 = None, N_max = 1000, err_g_max = 0.001, err_u_max = 0.001):
        """
        First Order Reliability Analysis
        
        Input: ** see input to MPP_search() **
        
        Output:
        conv     -   Convergence true/false
        beta     -   Reliability index
        pof      -   Probability of failure
        
        Intermediate:
        Stored in self._results_MPP
        """
        
        # Run MPP search 
        conv, u = self.MPP_search(u0, N_max, err_g_max, err_u_max)
        
        # PoF
        alpha = self._results_MPP['alpha'] 
        beta = alpha.dot(u)
        pof = stats.norm.cdf(-beta)
        
        # Store internally
        self._results_FORM = {
            'beta':beta,
            'conv': conv,
            'pof': pof        
        }
        
        return conv, beta, pof
    
    def MPP_search(self, u0 = None, N_max = 1000, err_g_max = 0.001, err_u_max = 0.001):
        """
        Iterative algorithm to find MPP
        
        Input:
        u0         -   Initial guess, if None then the zero vector is used 
        N_max      -   Maximum number of iterations
        err_g_max  -   Convergence criterion on g(u)
        err_u_max  -   Convergence criterion on u
        
        Output:
        u          -   The MPP
        conv       -   Convergence true/false
        
        Intermediate:
        Stored in self._results_MPP
        """
        
        # Initial guess
        if u0 is None:
            u = np.zeros(self.X_dist.dim)
        else:
            u = u0.copy()
        
        g0 = self._G_U(u.reshape(1, -1))[0]
        
        # Run search
        for i in range(N_max):
                
            # Find u_new
            #alpha, u_new = self._MPP_step_DNV(u)
            #alpha, u_new = self._MPP_step_HLRF(u)
            alpha, u_new = self._MPP_step_iHLRF(u)
            
            g_u = self._G_U(u_new.reshape(1, -1))[0]
            
            # Compute error
            err_g = np.abs(g_u/g0)
            err_u = np.linalg.norm(u-alpha.dot(u)*alpha) #np.linalg.norm(u - u_new)
            
            # Update u
            u = u_new
            
            # Check convergence
            conv = (err_g < err_g_max) & (err_u < err_u_max)
            if conv: break
        
        # Store some intermediate results
        if not np.isscalar(conv): conv = conv[0]
        self._results_MPP = {
            'MPP_u':u,
            'MPP_x':self.X_dist.U_to_X(u.reshape(1, -1))[0],
            'conv': conv,
            'alpha': alpha,
            'err_g': err_g, 
            'err_u': err_u,
            'num_iter': i+1,
            'num_limitstate_calls': (self.X_dist.dim + 2)*(i+1) + 1 + i*6 # The last i*6 from iHLRF step size computation
        }
        
        return conv, u
            
    def Set_gradient_delta(self, delta):
        """
        Set step size for gradient estimation
        """
        
        # step size
        self.gradient_delta = delta 
        
        # matrix used to add delta to multiple values
        self._add_delta = np.zeros((self.X_dist.dim + 1, self.X_dist.dim))
        np.fill_diagonal(self._add_delta, delta)
        
    def _MPP_step_DNV(self, u):
        """
        Run one step of the MPP search starting at u
        
        Old DNV method - similar to HL-RF but different formulation
        """
        
        # Compute G(u) and its gradient
        g_u, grad_g_u = self._G_gradient_u(u)
        grad_norm = np.linalg.norm(grad_g_u) 
        
        # Compute alpha and u_new
        alpha = -grad_g_u/grad_norm
        beta = np.linalg.norm(u) + g_u/grad_norm
        u_new = alpha*beta
        
        return alpha, u_new
    
    def _MPP_step_HLRF(self, u):
        """
        Run one step of the MPP search starting at u

        HL-RF method
        """

        # Compute G(u) and its gradient
        g_u, grad_g_u = self._G_gradient_u(u)
        grad_norm = np.linalg.norm(grad_g_u) 

        # Compute alpha and u_new
        alpha = -grad_g_u/grad_norm
        u_new = grad_g_u*(np.dot(grad_g_u, u) - g_u)/(grad_norm**2)
        
        return alpha, u_new
    
    def _MPP_step_iHLRF(self, u):
        """
        Run one step of the MPP search starting at u

        iHL-RF method from:
        Zhang, Y. and Der Kiureghian, A., 1995, 'Two Improved Algorithms for Reliability Analysis'
        https://doi.org/10.1007/978-0-387-34866-7_32
        """
        
        ### Here u_new = u + lambda*d where 
        ### lambda = step size
        ### d = search direction
        
        n_trial = 6 # number of trials in the Armijo rule for step size calculation
        
        # Compute G(u) and its gradient
        g_u, grad_g_u = self._G_gradient_u(u)
        grad_norm = np.linalg.norm(grad_g_u) 
        u_norm = np.linalg.norm(u)
        
        # Compute alpha and search direction
        alpha = -grad_g_u/grad_norm
        d = (g_u/grad_norm + alpha.dot(u))*alpha - u # Search direction
        
        # Compute step size
        step_size = self._iHLRF_step_size(n_trial, u, u_norm, grad_norm, g_u, d)
        
        # Compute next u
        u_new = u + step_size*d

        return alpha, u_new
    
    def _iHLRF_step_size(self, n_trial, u, u_norm, grad_norm, g_u, d):
        """
        Step size used in iHLRF algorithm 
        
        u_new = u + lambda*d
        """
        # Current merit function value
        c = (u_norm/grad_norm)*2 + 10
        merit = 0.5 * u_norm**2 + c * np.abs(g_u)
        
        # Test size candidates and corresponding merit 
        step_size_test = np.array([0.5**np.arange(0, n_trial)]) # Find the largest acceptable from these step sizes
        U_test = np.array([u for _ in range(n_trial)]) + np.dot(d.reshape(-1, 1), step_size_test).T
        
        G_test = self._G_U(U_test).flatten()
        merit_test = 0.5*np.linalg.norm(U_test, axis = 1)**2 + c*np.abs(G_test)
        
        # First occurence where merit_test < merit
        idx = np.argmax(merit_test < merit)
        step_size = step_size_test[0][idx]
        
        return step_size
        
    def _G_U(self, U):
        """
        Limit state in U-space
        """
        return self.G(self.X_dist.U_to_X(U))
    
    def _G_gradient_u(self, u):
        """ 
        Input: u = 1d numpy array of length self.X_dist.dim 
        Output: G(u) and gradient of G(u)
        """
        # Set of u-values to use
        U = np.array([u for _ in range(self.X_dist.dim + 1)]) + self._add_delta

        # Compute G(U)
        g = self._G_U(U)
        g_u = g[-1] # G(u)
        grad_g_u = ((g[0:-1] - g_u)/self.gradient_delta).flatten() # gradient
        
        return g_u, grad_g_u