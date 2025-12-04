# ============================================================================
'''
Module for Probability and Statistics
--------------------------------------
functions:
- log_likelihood
- log_prior
- log_posterior
'''
# ============================================================================

import numpy as np
from nuclear_mcmc.data_model import exponential_decay

def log_likelihood(params, t, N_obs, sigma):
    """
    Log-likelihood function for Gaussian noise model
    
    Parameters:
    -----------
    params : tuple
        (N0, lambda_decay) parameters
    t : array-like
        Time points
    N_obs : array-like
        Observed activity
    sigma : array-like
        Measurement uncertainties
    
    Returns:
    --------
    log_L : float
        Log-likelihood value
    """
    N0, lambda_decay = params
    
    # Physical constraints
    if N0 <= 0 or lambda_decay <= 0:
        return -np.inf
    
    # Model prediction
    N_model = exponential_decay(t, N0, lambda_decay)
    residuals = N_obs - N_model
    
    # Gaussian log-likelihood
    log_L = -0.5 * np.sum((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return log_L

def log_prior(params, N0_range=(0, 200), lambda_range=(0, 2)):
    """
    Log-prior (uniform priors for both parameters)
    
    Parameters:
    -----------
    params : tuple
        (N0, lambda_decay) parameters
    N0_range : tuple
        Range for N0 prior
    lambda_range : tuple
        Range for lambda prior
    
    Returns:
    --------
    log_p : float
        Log-prior value
    """
    N0, lambda_decay = params
    
    # Uniform priors
    if (N0_range[0] < N0 < N0_range[1] and 
        lambda_range[0] < lambda_decay < lambda_range[1]):
        return 0.0
    else:
        return -np.inf

def log_posterior(params, t, N_obs, sigma):
    """
    Log-posterior = log-likelihood + log-prior
    
    Parameters:
    -----------
    params : tuple(float, float)
        Model parameters
    t : array-like
        Time points
    N_obs : array-like
        Observed activity
    sigma : array-like
        Measurement uncertainties

    Returns:
    --------
    log_L + log_p: float
        Log-posterior value
    """
   
    log_p = log_prior(params)
    if np.isinf(log_p):
        return -np.inf
    
    log_L = log_likelihood(params, t, N_obs, sigma)
    return log_L + log_p
