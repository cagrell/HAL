### Dependent packages ###
import time
import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from scipy import optimize

from ..util.div import formattime, len_none
from ..util.linalg import jitchol, try_jitchol, triang_solve, mulinv_solve, chol_inv, traceprod, nearestPD
from ..util.stats import norm_cdf_int, norm_cdf_int_approx, normal_cdf_approx
    
class GPmodel():
    """ GP model """
    
    def __init__(self, kernel, likelihood = 1, mean = 0, constr_likelihood = 1E-6, verbatim = True):
        
        ### Prior model input ##################################
        
        # GP parameters
        self.kernel = kernel # Object containing kernel function and its derivatives
        self.mean = mean # Constant mean function
        self.likelihood = likelihood
        
        # Design data
        self.X_training = None
        self.Y_training = None
              
        ### Cached data from intermediate calculations ###########
        
        # Depending only on X
        self.K_w = None # K_w = K_x_x + sigma^2*I
        self.K_w_chol = None # Cholesky factor L s.t. L*L.T = K_w
        
        # Depending only on Y
        self.Y_centered = None
        
        # Depending on (X, Y)
        self.LLY = None # Only used in the unconstrained calculations

        ### For new observation ################################
        self.x_new = None
        self.y_new = None

        self.new_X_training = None
        self.new_Kw = None
        self.new_K_w_chol = None
        self.new_Y_centered = None
        self.new_LLY = None

        ### Other ##############################################
        self.verbatim = verbatim # Print info during execution
        
    # Parameters that need calculation reset
    @property
    def X_training(self): return self.__X_training

    @property
    def Y_training(self): return self.__Y_training

    @X_training.setter
    def X_training(self, value):
        self.K_w = None
        self.K_w_chol = None
        self.LLY = None
        
        self.__X_training = value

    @Y_training.setter
    def Y_training(self, value):
        self.Y_centered = None
        self.LLY = None
        self.__Y_training = value

    def __str__(self):
        """ What to show when the object is printed """
        txt = '----- GP model ----- \n mean = {} \n likelihood = {} \n '.format(self.mean, self.likelihood)
        txt += 'kernel: \n {} \n'.format(self.kernel.__str__())
        txt += ' constraint: \n' 
            
        txt += '---------------------'
        return txt
    
    def reset(self):
        """ Reset model. I.e. forget all older calculations """
        self.K_w = None
        self.K_w_chol = None
        self.Y_centered = None
        self.LLY = None

        self.x_new = None
        self.y_new = None

        self.new_X_training = None
        self.new_Kw = None
        self.new_K_w_chol = None
        self.new_Y_centered = None
        self.new_LLY = None

    def set_x_new(self, x_new, lik = None):
        """
        Set new observation location x_new
        """
        self._prep_K_w()

        self.x_new = x_new
        self.y_new = None
        
        ### Update GP matrices
        
        # New training input 
        self.new_X_training = np.append(self.X_training, x_new.reshape(1, -1), axis = 0)
        
        # Update Kw
        new_Kw_col = self.kernel.K(self.X_training, x_new.reshape(1, -1)) # Define new column
        self.new_Kw = np.append(self.K_w, new_Kw_col, axis = 1) # Add column

        likelihood = lik if lik is not None else self.likelihood
        
        self.new_Kw = np.append(self.new_Kw, np.append(new_Kw_col, self.kernel.K(x_new.reshape(1, -1), x_new.reshape(1, -1)) + likelihood, 0).T, axis = 0) # Add row            

        # Update Cholesky (can do this much faster!!)
        self.new_K_w_chol = np.matrix(jitchol(self.new_Kw)) 
        
    def set_y_new(self, y_new):
        """
        Set new observation y_new
        """
        self.y_new = y_new
        self.new_Y_centered = np.append(self.Y_centered, y_new.reshape(-1, 1) - self.mean, axis=0) 
        self.new_LLY = mulinv_solve(self.new_K_w_chol, self.new_Y_centered, triang = True)
        
    def calc_posterior(self, XS, full_cov = True):
        """
        Calculate pridictive posterior distribution f* | Y
        
        Returns: mean, cov (full or only diagonal)
        """
        
        # Check input
        self._check_XY_training()
        assert len(XS.shape) == 2, 'Test data XS must be 2d array'
        
        # Start timer
        t0 = time.time()
        
        # Run pre calcs
        self._prep_Y_centered()
        self._prep_K_w(verbatim = self.verbatim)
        self._prep_K_w_factor(verbatim = self.verbatim)
        self._prep_LLY()
        
        if self.verbatim: print("..Calculating f* | Y ...", end = '')
        
        # Kernel matrices needed
        K_x_xs = np.matrix(self.kernel.K(self.X_training, XS))
        
        v2 = triang_solve(self.K_w_chol, K_x_xs) 
        
        # Calculate mean
        mean = self.mean + K_x_xs.T*self.LLY
        
        # Calculate cov
        if full_cov:
            K_xs_xs = np.matrix(self.kernel.K(XS, XS))
            cov = K_xs_xs - v2.T*v2
        else:
            K_xs_xs_diag = self.kernel.K_diag(XS)
            cov = np.matrix(K_xs_xs_diag - np.square(v2).sum(0)).T
                    
        if self.verbatim: print(' DONE - Total time: {}'.format(formattime(time.time() - t0)))
        
        if full_cov:
            return mean, cov
        else:
            return np.array(mean).flatten(), np.array(cov).flatten()

    def calc_posterior_new(self, XS):
        """
        Mean and variance of GP at XS assuming new observation (self.x_new, self.y_new) was added
        """

        # Check input
        self._check_XY_training()
        self._check_new()
        assert len(XS.shape) == 2, 'Test data XS must be 2d array'
        
        K_x_xs = np.matrix(self.kernel.K(self.new_X_training, XS))
        v2 = triang_solve(self.new_K_w_chol, K_x_xs)
        
        mean = self.mean + K_x_xs.T*self.new_LLY
        
        K_xs_xs_diag = self.kernel.K_diag(XS)
        cov = np.matrix(K_xs_xs_diag - np.square(v2).sum(0)).T
        
        return np.array(mean).flatten(), np.array(cov).flatten()

    def optimize(self, method = 'ML', fix_likelihood = False, bound_min = 1e-6):
        """
        Optimize hyperparameters of unconstrained GP
        
        method = 'ML' -> maximum marginal likelihood
        method = 'CV' -> cross validation
        
        fix_likelihood = False -> Don't optimize GP likelihood parameter self.likelihood
        bound_min = minimum value in parameter bounds = (bound_min, ...)
        """
        
        # Start timer
        t0 = time.time()
        if self.verbatim: print("..Running optimization for unconstrained GP ...", end = '')
        
        # Run optimization
        if method == 'ML':
            res = self._optimize_ML(fix_likelihood, bound_min)
        elif method == 'CV':
            print('TODO...')
            raise NotImplementedError 
        else:
            raise NotImplementedError
        
        # Save results
        self.__setparams(res.x, not fix_likelihood)

        if self.verbatim:
            if res.success:
                print(' DONE - Total time: {}'.format(formattime(time.time() - t0)))
            else:
                print('WARNING -- NO CONVERGENCE IN OPTIMIZATION -- Total time: {}'.format(formattime(time.time() - t0)))

    def _optimize_ML(self, fix_likelihood = False, bound_min = 1e-6):
        """
        Optimize hyperparameters of unconstrained GP using ML
        
        fix_likelihood = False -> Don't optimize GP likelihood parameter self.likelihood
        bound_min = minimum value in parameter bounds = (bound_min, ...)
        """
        
        # Check input
        self._check_XY_training()

        # Define wrapper functions for optimization
        def optfun(theta, fix_likelihood):
            self.reset()
            self.__setparams(theta, not fix_likelihood)
            return -self._loglik()
        
        def optfun_grad(theta, fix_likelihood):
            self.reset()
            self.__setparams(theta, not fix_likelihood)
            grad = -np.array(self._loglik_grad())
            if fix_likelihood: 
                return grad[1:]
            else:
                return grad
            
        # Define bounds
        num_params = self.kernel.dim + 2
        if fix_likelihood: num_params -= 1
        bounds = [(bound_min, None)]*num_params
        
        # Initial guess
        if fix_likelihood:
            theta = np.array(self.kernel.get_params())
        else:
            theta = np.array([self.likelihood] + list(self.kernel.get_params()))
        
        # Run optimizer
        res = optimize.minimize(optfun, theta, args=fix_likelihood, jac = optfun_grad, bounds=bounds, method = 'L-BFGS-B')

        return res
    
    def _loglik(self):
        """
        Calculates log marginal likelihood
        
        I.e. log(P(Y_training | X_training))
        """
               
        # Run pre calcs
        self._prep_Y_centered()
        self._prep_K_w(verbatim = False)
        self._prep_K_w_factor(verbatim = False)
        self._prep_LLY()
        
        ### Calculate log marginal likelihood ###
        n = self.X_training.shape[0]
        loglik = -0.5*self.Y_centered.T*self.LLY - np.log(np.diag(self.K_w_chol)).sum() - (n/2)*np.log(2*np.pi)
        loglik = loglik[0,0]
        
        return loglik
    
    def _loglik_grad(self):
        """
        Calculates gradient of log marginal likelihood w.r.t hyperparameters
        """
               
        # Run pre calcs
        self._prep_Y_centered()
        self._prep_K_w(verbatim = False)
        self._prep_K_w_factor(verbatim = False)
        self._prep_LLY()
        
        # Invert K_w using the Cholesky factor
        K_w_inv = chol_inv(self.K_w_chol)
         
        # Partial derivative of K_w w.r.t. likelihood
        n = self.X_training.shape[0]
        dKw_dlik = np.matrix(np.identity(n))
        
        # Partial derivative of K_w w.r.t. kernel parameters
        dK_dpar = self.kernel.K_gradients(self.X_training, self.X_training)
        
        # Calculate gradient
        alpha = K_w_inv*self.Y_centered
        tmp = alpha*alpha.T - K_w_inv
        
        Dloglik_lik = 0.5*traceprod(tmp, dKw_dlik)                # W.r.t. GP likelihood parameter
        Dloglik_ker = [0.5*traceprod(tmp, K) for K in dK_dpar]    # W.r.t. kernel parameters
        
        Dloglik = [Dloglik_lik] + Dloglik_ker
        
        return Dloglik

    def _check_XY_training(self):
        """
        Check that X_training and Y_training are OK
        """
        assert self.X_training is not None, 'Training data not found. Use model.X_training = ...' 
        assert len(self.X_training.shape) == 2, 'Training data X_training must be 2d array'
        assert self.Y_training is not None, 'Training data not found. Use model.Y_training = ...' 
        assert len(self.Y_training.shape) == 1, 'Training data Y_training must be 1d array'
        assert self.X_training.shape[0] == len(self.Y_training), 'Number of points in X_training and Y_training does not match'

    def _check_new(self):
        """
        Check that x_new, y_new has been set
        """
        assert self.x_new is not None, 'x_new not found. Use model.set_x_new()' 
        assert self.y_new is not None, 'x_new not found. Use model.set_y_new()' 
    
    def __setparams(self, theta, includes_likelihood):
        """
        Set model parameters from single array theta
        """
        if includes_likelihood:
            self.likelihood = theta[0]
            self.kernel.set_params(theta[1:])
        else:
            self.kernel.set_params(theta)
    
    def _prep_K_w(self, verbatim = False):
        """ 
        Calculate K_w = K_x_x + likelihood 
        
        *** Need to run this if one of the following arrays are changed : ***
            - X_training
        
        """
        
        if verbatim: print('..Running calculation of K_w ...', end = '')

        if self.K_w is None:
            
            # Start timer
            t0 = time.time()
            
            n = len(self.X_training)

            if np.isscalar(self.likelihood):
                self.K_w = np.matrix(self.kernel.K(self.X_training, self.X_training) + self.likelihood*np.identity(n)) 
            else:
                self.K_w = np.matrix(self.kernel.K(self.X_training, self.X_training) + np.diag(self.likelihood))

            if verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t0)))

        else:
            if verbatim: print(' SKIP - (cached)')
                
    def _prep_K_w_factor(self, verbatim = False):
        """
        Calculate matrix L s.t. L*L.T = K_w 
        
        *** Need to run this if one of the following arrays are changed : ***
            - X_training
        """
        
        if verbatim: print('..Running calculation of Cholesky factor for K_w ...', end = '')

        if self.K_w_chol is None:

            # Start timer
            t0 = time.time()

            # Cholesky
            self.K_w_chol = np.matrix(jitchol(self.K_w)) 

            if verbatim: print(' DONE - time: {}'.format(formattime(time.time() - t0)))

        else:
            if verbatim: print(' SKIP - (cached)')
                    
    def _prep_LLY(self):
        """
        Calculate LLY = L.T \ L \ Y_centered

        *** Need to run this if one of the following arrays are changed : ***
            - X_training
            - Y_training

        """
              
        if self.LLY is None:
            # Run calculation
            self.LLY = mulinv_solve(self.K_w_chol, self.Y_centered, triang = True)
            
    def _prep_Y_centered(self):
        """
        Calculate Y_centered
        """
        if self.Y_centered is None: self.Y_centered = self.Y_training.reshape(-1, 1) - self.mean
            
    