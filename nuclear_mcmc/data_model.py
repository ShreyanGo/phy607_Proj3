# ============================================================================
'''
Module for Data Generation
--------------------------
functions:
- exponential_decay
- generate_synthetic_data
'''
# ============================================================================

import numpy as np

def exponential_decay(t, N0, lambda_decay):
    """
    Exponential decay model: N(t) = N_0 * exp(-lamda*t)
    
    Parameters:
    -----------
    t : array-like
        Time points
    N0 : float
        Initial activity
    lambda_decay : float
        Decay constant
    
    Returns:
    --------
    N : array-like
        Activity at time t
    """
    return N0 * np.exp(-lambda_decay * t)

def generate_synthetic_data(t, N0_true, lambda_true, noise_level=0.05):
    """
    Generate synthetic radioactive decay data with Gaussian noise
    
    Parameters:
    -----------
    t : array-like
        Time points for measurements
    N0_true : float
        True initial activity
    lambda_true : float
        True decay constant
    noise_level : float
        Relative noise level (fraction of signal)
    
    Returns:
    --------
    N_obs : array-like
        Observed activity with noise
    sigma : array-like
        Uncertainty for each measurement
    """
    N_true = exponential_decay(t, N0_true, lambda_true)
    sigma = noise_level * N_true
    N_obs = N_true + np.random.normal(0, sigma)
    return N_obs, sigma
