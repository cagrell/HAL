import numpy as np
import matplotlib.pyplot as plt

# Path to HAL module
# can be downloaded at https://github.com/cagrell/HAL
import os, sys
sys.path.append('..\\HAL\\')

from HAL.GP.model import GPmodel
from HAL.GP.kern import kernel_Matern52

def fun_x(x):
    """
    f(x) function to emulate
    """
    xs = x
    f = (0.4*xs - 0.3)**2 + np.exp(-11.534*np.abs(xs)**(1.95)) + np.exp(-5*(xs - 0.8)**2) # From Becht & Ginsbourger paper
    return -(f - 1)


def get_initial_model(x_design, y_design):
    """
    Initial GP model
    """
    ker = kernel_Matern52(variance = 0.1, lengthscale = [0.5])
    model = GPmodel(kernel = ker, likelihood = 1e-5, mean = 0) 
    model.verbatim = False

    # Training data
    model.X_training = x_design
    model.Y_training = y_design

    #model.optimize(fix_likelihood = True)
    return model

# Define a model and print it
def main():
    print(' *** Create GP model ***')

    x_design = np.array([[-1.5], [0], [1]])
    y_design = fun_x(x_design).flatten()
    model = get_initial_model(x_design, y_design)
    print(model)

    x = np.linspace(-2.2, 2, 100)
    f = fun_x(x)
    m, v = model.calc_posterior(x.reshape(-1, 1), full_cov = False)

    x_new = np.array([[-0.5]])
    y_new = fun_x(x_new)[0][0]

    print('\n *** Test fast computation of posterior with new observation (x_new, y_new) ***')

    model.set_x_new(x_new)
    model.set_y_new(y_new)
    print('(x_new, y_new) = ({}, {})'.format(x_new, y_new))

    m, v = model.calc_posterior_new(x.reshape(-1, 1))

    print('\n *** Create plot ***')

    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots()
    ax.plot(x, f, label = 'f(x)') 
    ax.scatter(x_design.flatten(), y_design, color = 'k', label = 'obs')
    ax.plot(x, m, color = 'k', linewidth = 0.5, label = 'GP mean')
    ax.scatter(x_new[0], [y_new], color = 'r', label = 'new')
    ax.fill_between(x, m - 2*np.sqrt(v), m + 2*np.sqrt(v), color = 'k', alpha = 0.1, label = 'GP $\pm$ 2 std')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

    legend = ax.legend(loc='upper right', shadow=False, fontsize=10, frameon = True)
    legend.get_frame().set_facecolor('white')
    
    save_dir = 'C:\\Data\\tmp\\'
    figname = '1dGPfig.png'
    print('Save figure', save_dir+figname)
    fig.savefig(save_dir+figname)
    
    print('\n done')

if __name__ == "__main__":
    main()