# HAL - Hierarchical Active Learning
Python module for active learning in hierarchical (Bayesian network type of) structural reliability models.

Code based on the paper [_C. Agrell and K. R. Dahl (2021) Sequential Bayesian optimal experimental design for structural reliability analysis_](https://arxiv.org/abs/2007.00402). 

HAL contains classes for defining SRA models with epistemic uncertainty, including GP emulators, making use of the following submodules:

__HAL/SRA__: For Structural Reliability Analysis (SRA) with Monte Carlo simulation, importance sampling, design point search and FORM (First Order Reliability Methods)

__HAL/GP__: Gaussian Process (GP) module with fast evaluation of posterior when a single observation is added at some specified new input location __x__<sub>new</sub>

### Prerequisites
The code has been tested with the following requirements: 

__Python 3 (3.9.1 64bit)__
- __numpy (1.19.5)__
- __scipy (1.6.0)__
- __sklearn (0.24.1)__ _Only uses the function sklearn.metrics.pairwise.euclidean_distances from this package for fast computation of Gram matrices (and could easily be replaced by custom code if needed)_
- __filterpy (1.4.5)__ _Currently using the UT implementation from filterpy for UT calculations (unscented_transform and MerweScaledSigmaPoints from filterpy.kalman). Might replace this later with a custom UT package for other sigma-point selection methods_

### Examples
Some examples are given in jupyter notebooks.
