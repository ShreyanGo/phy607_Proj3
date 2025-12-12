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
    Run the emcee ensemble sampler.

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

    # Initialize walkers near the guess
    p0 = np.vstack([
        [
            np.random.normal(N0_guess, init_spread[0]),
            np.random.normal(lambda_guess, init_spread[1]),
        ]
        for _ in range(n_walkers)
    ])

    # Log-posterior wrapper
    def log_prob_emcee(theta):
        return log_posterior(theta, t, N_obs, sigma)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob_emcee)

    sampler.run_mcmc(p0, n_steps, progress=True)

    # Full chain (emcee returns shape: (n_steps, n_walkers, ndim))
    full_chain_raw = sampler.get_chain()

    # Convert to (n_walkers, n_steps, ndim)
    full_chain = np.transpose(full_chain_raw, (1, 0, 2))

    # Burn-in removal along the steps dimension
    burned_chain = full_chain[:, burn_in:, :]

    # Flatten (walkers × steps, ndim)
    flat_samples = burned_chain.reshape((-1, ndim))

    acc_frac = np.mean(sampler.acceptance_fraction)

    return flat_samples, full_chain, acc_frac

