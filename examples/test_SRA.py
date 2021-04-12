import numpy as np

# Path to HAL module
# can be downloaded at https://github.com/cagrell/HAL
import os, sys
sys.path.append('..\\HAL\\')

from HAL.SRA.distributions import Normal, Uniform, MultivariateDistribution
from HAL.SRA import SRA_baseclass

# Define a model and print it
def main():
    print(' *** Create SRA model ***')

    # Create multivariate distribution (3-dim normal in this case)
    X_dist = MultivariateDistribution()
    X_dist.AddVariable(Normal('x1', 500,100))
    X_dist.AddVariable(Normal('x2', 2000,400))
    X_dist.AddVariable(Uniform('x3', 4.13, 5.86))

    # Print distribution
    print('\n *** Input distribution ***')
    X_dist.Describe()

    # The limit state
    def limit_state(x):
        return 1 - x[1]*(1000*x[2])**(-1) - (x[0]*(200*x[2])**(-1))**2 

    # Define model
    SRA_MODEL = SRA_baseclass(X_dist, limit_state)

    # Run crude MC
    beta_MC, pof_MC, cov_MC = SRA_MODEL.Run_MC(10000)
    print('\n *** Crude MC ***')
    print('MC beta:', beta_MC)
    print('MC pof:', pof_MC)

    # Run FORM
    conv, beta_FORM, pof_FORM = SRA_MODEL.Run_FORM()
    print('\n *** FORM ***')
    print('FORM convergence:', conv)
    print('FORM beta:', beta_FORM)
    print('FORM pof', pof_FORM)

    # ..some intermediate results from FORM
    print('\n * Sensitivity vector etc. from MPP search *')
    for key, val in SRA_MODEL._results_MPP.items():
        print(key + ':', val)

    print('\n done')

if __name__ == "__main__":
    main()