# ============================================================================
"""
Diagnostics and Statistical Analysis Module
-------------------------------------------
functions:
- print_statistics
- gelman_rubin_rhat
- autocorrelation
- autocorrelation_time
- effective_sample_size
- geweke_test
"""
# ============================================================================

import numpy as np

def print_statistics(chain_burned, param_names=['N₀', 'λ'], true_values=None):
    """
    Print summary statistics of the posterior distribution

    Parameters:
    -----------
    chain_burned : ndarray
        MCMC samples after burn-in, shape (N_samples, n_params)
    param_names : list of str
        Parameter names
    true_values : tuple or list, optional
        True parameter values

    Returns:
    --------
    None
    """
    print("\n" + "=" * 60)
    print("MCMC Parameter Estimation Results")
    print("=" * 60)

    for i, name in enumerate(param_names):
        samples = chain_burned[:, i]

        mean_val = np.mean(samples)
        median_val = np.median(samples)
        std_val = np.std(samples)
        lower = np.percentile(samples, 2.5)
        upper = np.percentile(samples, 97.5)

        print(f"\n{name}:")
        print(f"  Mean:     {mean_val:.4f}")
        print(f"  Median:   {median_val:.4f}")
        print(f"  Std:      {std_val:.4f}")
        print(f"  95% CI:   [{lower:.4f}, {upper:.4f}]")

        if true_values is not None:
            true = true_values[i]
            error = mean_val - true
            rel_error = 100 * error / true
            print(f"  True:     {true:.4f}")
            print(f"  Error:    {error:.4f}  ({rel_error:.2f}%)")

    print("\n" + "=" * 60)


def gelman_rubin_rhat(chains):
    """
    Compute Gelman–Rubin R-hat for each parameter

    Parameters:
    -----------
    chains : ndarray
        Shape (n_chains, n_samples, n_params)

    Returns:
    --------
    rhat : ndarray
        R-hat for each parameter
    """
    chains = np.asarray(chains)
    n_chains, n_samples, n_params = chains.shape

    # Mean per chain, variance per chain, overall mean
    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)
    grand_mean = np.mean(chain_means, axis=0)

    # Between-chain variance B
    B = n_samples * np.var(chain_means, axis=0, ddof=1)

    # Within-chain variance W
    W = np.mean(chain_vars, axis=0)

    # Estimate of marginal posterior variance
    var_hat = (n_samples - 1) / n_samples * W + (1 / n_samples) * B

    R_hat = np.sqrt(var_hat / W)
    return R_hat


def autocorrelation(x, max_lag):
    """
    Compute the autocorrelation function up to a maximum lag

    Parameters:
    -----------
    x : ndarray
        One-dimensional array of MCMC samples
    max_lag : int
        Maximum lag value for which the autocorrelation is computed

    Returns:
    --------
    ndarray
        Autocorrelation values for lags from 0 to ``max_lag - 1``
    """
    x = x - np.mean(x)
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[autocorr.size // 2:]      # keep positive lags
    autocorr /= autocorr[0]                       # normalize
    return autocorr[:max_lag]


def autocorrelation_time(samples, max_lag=200):
    """
    Estimate integrated autocorrelation time τ

    Parameters:
    -----------
    samples : ndarray
        1D MCMC samples
    max_lag : int
        Maximum lag

    Returns:
    --------
    tau : float
        Autocorrelation time
    """
    ac = autocorrelation(samples, max_lag)

    # Gayer cutoff: stop when ac becomes negative
    tau = 1 + 2 * np.sum(ac[1:][ac[1:] > 0])
    return tau


def effective_sample_size(samples, max_lag=200):
    """
    Compute effective sample size (ESS)

    Parameters:
    -----------
    samples : ndarray
        1D samples
    max_lag : int
        Max lag for autocorrelation

    Returns:
    --------
    ess : float
        Effective sample size
    """
    n = len(samples)
    tau = autocorrelation_time(samples, max_lag)
    ess = n / tau
    return ess


def geweke_test(samples, first=0.1, last=0.5):
    """
    Geweke diagnostic: compare early and late means

    Parameters:
    -----------
    samples : ndarray
        1D MCMC samples
    first : float
        Fraction for early window
    last : float
        Fraction for late window

    Returns:
    --------
    z_score : float
        Geweke Z-score - |Z| < 2 indicates convergence
    """
    n = len(samples)

    # Early and late segments
    A = samples[: int(first * n)]
    B = samples[int((1 - last) * n):]

    meanA, varA = np.mean(A), np.var(A)
    meanB, varB = np.mean(B), np.var(B)

    z = (meanA - meanB) / np.sqrt(varA / len(A) + varB / len(B))
    return z
