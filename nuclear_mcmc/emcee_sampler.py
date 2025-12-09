"""
emcee_sampler.py
----------------
MCMC sampling using the emcee affine-invariant ensemble sampler
"""

import numpy as np
import emcee

from .probabilities import log_posterior


def run_emcee(
    t,
    N_obs,
    sigma,
    n_walkers=32,
    n_steps=5000,
    initial_guess=(100, 0.5),
    init_spread=(1.0, 0.05),
    burn_in=1000,
):
    """
    Run the emcee ensemble sampler

    Parameters
    ----------
    t : array-like
        Time points
    N_obs : array-like
        Observed decay counts
    sigma : array-like
        Measurement uncertainty
    n_walkers : int
        Number of ensemble walkers (should be >= 2×ndim)
    n_steps : int
        Total number of steps to run
    initial_guess : tuple
        Starting guess for (N0, lambda)
    init_spread : tuple
        Std dev of Gaussian noise added to initialize walkers
    burn_in : int
        Number of steps to discard as burn-in

    Returns
    -------
    samples : ndarray
        Flattened MCMC samples after burn-in
    chain : ndarray
        Full emcee chain of shape (n_walkers, n_steps, ndim)
    acceptance_fraction : float
        Mean walker acceptance fraction
    """

    ndim = 2
    N0_guess, lambda_guess = initial_guess

    # Initialize walkers in a small Gaussian ball around initial_guess
    p0 = np.vstack([
        [
            np.random.normal(N0_guess, init_spread[0]),
            np.random.normal(lambda_guess, init_spread[1]),
        ]
        for _ in range(n_walkers)
    ])

    # Log-posterior wrapper for emcee (needs only theta)
    def log_prob_emcee(theta):
        return log_posterior(theta, t, N_obs, sigma)

    # Initialize sampler
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob_emcee)

    # Run sampler
    sampler.run_mcmc(p0, n_steps, progress=True)

    # Extract raw chain
    full_chain = sampler.get_chain()

    # Burn-in removal
    burned_chain = full_chain[:, burn_in:, :]

    # Flatten (n_walkers * (n_steps - burn_in), ndim)
    flat_samples = burned_chain.reshape((-1, ndim))

    # Acceptance statistics
    acc_frac = np.mean(sampler.acceptance_fraction)

    return flat_samples, full_chain, acc_frac
