# ============================================================================
'''
Metropolis-Hastings MCMC Implementation Module
----------------------------------------------
functions:
-metropolis_hastings_2d
'''
# ============================================================================

import numpy as np
from nuclear_mcmc.probabilities import log_posterior

def metropolis_hastings_2d(t, N_obs, sigma, initial_params, proposal_width, 
                            n_iterations=50000, burn_in=10000):
    """
    Metropolis-Hastings MCMC for 2D parameter estimation
    
    Parameters:
    -----------
    t : array-like
        Time points
    N_obs : array-like
        Observed activity
    sigma : array-like
        Measurement uncertainties
    initial_params : tuple
        Initial guess for (N0, lambda_decay)
    proposal_width : tuple
        Standard deviations for proposal distribution (σ_N0, σ_λ)
    n_iterations : int
        Number of MCMC iterations
    burn_in : int
        Number of burn-in samples to discard
    
    Returns:
    --------
    chain : ndarray
        MCMC chain (n_iterations x 2)
    acceptance_rate : float
        Fraction of accepted proposals
    """
    # Initialize
    n_params = 2
    chain = np.zeros((n_iterations, n_params))
    current_params = np.array(initial_params)
    current_log_post = log_posterior(current_params, t, N_obs, sigma)
    
    n_accepted = 0
    
    # MCMC loop
    for i in range(n_iterations):
        # Propose new parameters (Gaussian proposal)
        proposed_params = current_params + np.random.normal(0, proposal_width, size=n_params)
        proposed_log_post = log_posterior(proposed_params, t, N_obs, sigma)
        
        # Metropolis-Hastings acceptance criterion
        log_ratio = proposed_log_post - current_log_post
        
        if np.log(np.random.rand()) < log_ratio:
            # Accept proposal
            current_params = proposed_params
            current_log_post = proposed_log_post
            n_accepted += 1
        
        # Store current state
        chain[i] = current_params
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"Iteration {i + 1}/{n_iterations}")
    
    acceptance_rate = n_accepted / n_iterations
    print(f"\nAcceptance rate: {acceptance_rate:.2%}")
    
    # Remove burn-in
    chain_burned = chain[burn_in:]
    
    return chain, chain_burned, acceptance_rate
