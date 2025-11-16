"""
Radioactive Decay Parameter Estimation using MCMC
===================================================
This script uses Metropolis-Hastings MCMC to estimate the decay constant (λ)
and initial activity (N₀) from radioactive decay data following N(t) = N₀*exp(-λ*t)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import corner

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Data Generation
# ============================================================================

def exponential_decay(t, N0, lambda_decay):
    """
    Exponential decay model: N(t) = N_0 * exp(-lamda t)
    
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

# ============================================================================
# 2. Likelihood and Prior Functions
# ============================================================================

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
    """
    log_p = log_prior(params)
    if np.isinf(log_p):
        return -np.inf
    
    log_L = log_likelihood(params, t, N_obs, sigma)
    return log_L + log_p

# ============================================================================
# 3. Metropolis-Hastings MCMC Implementation
# ============================================================================

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

# ============================================================================
# 4. Analysis and Visualization
# ============================================================================

def plot_trace(chain, param_names=['N₀', 'λ'], true_values=None):
    """
    Plot MCMC trace plots
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(chain[:, i], alpha=0.7, linewidth=0.5)
        ax.set_ylabel(name, fontsize=12)
        ax.set_xlabel('Iteration', fontsize=11)
        
        if true_values is not None:
            ax.axhline(true_values[i], color='r', linestyle='--', 
                      linewidth=2, label=f'True {name}')
            ax.legend()
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/trace_plots.png', dpi=300, bbox_inches='tight')
    print("Trace plots saved to trace_plots.png")
    plt.close()

def plot_corner(chain_burned, param_names=['N₀', 'λ'], true_values=None):
    """
    Create corner plot for posterior distributions
    """
    fig = corner.corner(chain_burned, 
                       labels=param_names,
                       truths=true_values,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True,
                       title_kwargs={"fontsize": 12})
    
    plt.savefig('/mnt/user-data/outputs/corner_plot.png', dpi=300, bbox_inches='tight')
    print("Corner plot saved to corner_plot.png")
    plt.close()

def plot_posterior_histograms(chain_burned, param_names=['N₀', 'λ'], true_values=None):
    """
    Plot 1D posterior histograms
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(chain_burned[:, i], bins=50, density=True, alpha=0.7, 
               edgecolor='black')
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Posterior Distribution of {name}', fontsize=13)
        
        if true_values is not None:
            ax.axvline(true_values[i], color='r', linestyle='--', 
                      linewidth=2, label=f'True {name}')
            ax.legend()
        
        # Add statistics
        mean_val = np.mean(chain_burned[:, i])
        std_val = np.std(chain_burned[:, i])
        ax.axvline(mean_val, color='blue', linestyle='-', 
                  linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.3f}')
        ax.text(0.95, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               horizontalalignment='right', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/posterior_histograms.png', dpi=300, bbox_inches='tight')
    print("Posterior histograms saved to posterior_histograms.png")
    plt.close()

def plot_fit_results(t, N_obs, sigma, chain_burned, N0_true, lambda_true):
    """
    Plot data with fitted model and uncertainty bands
    """
    # Extract parameter estimates
    N0_samples = chain_burned[:, 0]
    lambda_samples = chain_burned[:, 1]
    
    N0_mean = np.mean(N0_samples)
    lambda_mean = np.mean(lambda_samples)
    
    # Generate predictions for plotting
    t_fine = np.linspace(0, np.max(t), 500)
    
    # True model
    N_true = exponential_decay(t_fine, N0_true, lambda_true)
    
    # Best fit model (mean parameters)
    N_fit = exponential_decay(t_fine, N0_mean, lambda_mean)
    
    # Uncertainty bands (sample from posterior)
    n_samples = 200
    sample_indices = np.random.choice(len(chain_burned), n_samples, replace=False)
    N_samples = np.zeros((n_samples, len(t_fine)))
    
    for i, idx in enumerate(sample_indices):
        N_samples[i] = exponential_decay(t_fine, 
                                        chain_burned[idx, 0], 
                                        chain_burned[idx, 1])
    
    # Calculate percentiles for uncertainty bands
    N_lower = np.percentile(N_samples, 2.5, axis=0)
    N_upper = np.percentile(N_samples, 97.5, axis=0)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(t, N_obs, yerr=sigma, fmt='o', color='black', 
                label='Observed Data', alpha=0.7, markersize=6)
    plt.plot(t_fine, N_true, 'r--', linewidth=2, label='True Model')
    plt.plot(t_fine, N_fit, 'b-', linewidth=2, label='MCMC Fit (Mean)')
    plt.fill_between(t_fine, N_lower, N_upper, color='blue', alpha=0.2,
                    label='95% Credible Interval')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Activity N(t)', fontsize=12)
    plt.title('Radioactive Decay: Data and MCMC Fit', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text box with parameters
    textstr = f'True: N₀={N0_true:.1f}, λ={lambda_true:.3f}\n'
    textstr += f'Estimated: N₀={N0_mean:.2f}±{np.std(N0_samples):.2f}\n'
    textstr += f'           λ={lambda_mean:.3f}±{np.std(lambda_samples):.3f}'
    
    plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    plt.savefig('/mnt/user-data/outputs/decay_fit.png', dpi=300, bbox_inches='tight')
    print("Fit results saved to decay_fit.png")
    plt.close()

def print_statistics(chain_burned, param_names=['N₀', 'λ'], true_values=None):
    """
    Print summary statistics of the posterior
    """
    print("\n" + "="*60)
    print("MCMC Parameter Estimation Results")
    print("="*60)
    
    for i, name in enumerate(param_names):
        mean_val = np.mean(chain_burned[:, i])
        median_val = np.median(chain_burned[:, i])
        std_val = np.std(chain_burned[:, i])
        
        # 95% credible interval
        lower = np.percentile(chain_burned[:, i], 2.5)
        upper = np.percentile(chain_burned[:, i], 97.5)
        
        print(f"\n{name}:")
        print(f"  Mean:   {mean_val:.4f}")
        print(f"  Median: {median_val:.4f}")
        print(f"  Std:    {std_val:.4f}")
        print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
        
        if true_values is not None:
            error = mean_val - true_values[i]
            rel_error = 100 * error / true_values[i]
            print(f"  True:   {true_values[i]:.4f}")
            print(f"  Error:  {error:.4f} ({rel_error:.2f}%)")
    
    print("\n" + "="*60)

# ============================================================================
# 5. Main Execution
# ============================================================================

def main():
    """
    Main function to run the complete MCMC analysis
    """
    print("Radioactive Decay Parameter Estimation using MCMC")
    print("="*60)
    
    # ========================================================================
    # Setup: Define true parameters and generate synthetic data
    # ========================================================================
    
    N0_true = 100.0        # True initial activity
    lambda_true = 0.5      # True decay constant (e.g., half-life = ln(2)/λ)
    
    # Time points for measurements
    t_max = 10.0
    n_points = 30
    t = np.linspace(0, t_max, n_points)
    
    # Generate synthetic data with noise
    noise_level = 0.05  # 5% relative noise
    N_obs, sigma = generate_synthetic_data(t, N0_true, lambda_true, noise_level)
    
    print(f"\nTrue Parameters:")
    print(f"  N₀ = {N0_true}")
    print(f"  λ  = {lambda_true}")
    print(f"  Half-life = {np.log(2)/lambda_true:.3f}")
    print(f"\nData points: {n_points}")
    print(f"Noise level: {noise_level*100}%")
    
    # ========================================================================
    # MCMC Setup
    # ========================================================================
    
    # Initial guess (can be different from true values)
    initial_params = (80.0, 0.4)
    
    # Proposal distribution widths (tune for good acceptance rate: 20-40%)
    proposal_width = (2.0, 0.02)
    
    # MCMC parameters
    n_iterations = 50000
    burn_in = 10000
    
    print(f"\nMCMC Configuration:")
    print(f"  Initial guess: N₀={initial_params[0]}, λ={initial_params[1]}")
    print(f"  Proposal widths: σ_N₀={proposal_width[0]}, σ_λ={proposal_width[1]}")
    print(f"  Total iterations: {n_iterations}")
    print(f"  Burn-in: {burn_in}")
    
    # ========================================================================
    # Run MCMC
    # ========================================================================
    
    print("\nRunning MCMC...")
    chain, chain_burned, acceptance_rate = metropolis_hastings_2d(
        t, N_obs, sigma, initial_params, proposal_width, 
        n_iterations, burn_in
    )
    
    # ========================================================================
    # Analysis and Visualization
    # ========================================================================
    
    print("\nGenerating plots and analysis...")
    
    true_values = [N0_true, lambda_true]
    
    # Print statistics
    print_statistics(chain_burned, true_values=true_values)
    
    # Generate all plots
    plot_trace(chain, true_values=true_values)
    plot_posterior_histograms(chain_burned, true_values=true_values)
    plot_corner(chain_burned, true_values=true_values)
    plot_fit_results(t, N_obs, sigma, chain_burned, N0_true, lambda_true)
    
    print("\nAnalysis complete! All results saved to /mnt/user-data/outputs/")
    
    # ========================================================================
    # Calculate derived quantities
    # ========================================================================
    
    # Half-life from posterior samples
    lambda_samples = chain_burned[:, 1]
    half_life_samples = np.log(2) / lambda_samples
    half_life_mean = np.mean(half_life_samples)
    half_life_std = np.std(half_life_samples)
    half_life_true = np.log(2) / lambda_true
    
    print(f"\nDerived Quantity - Half-life:")
    print(f"  Estimated: {half_life_mean:.3f} ± {half_life_std:.3f}")
    print(f"  True:      {half_life_true:.3f}")
    
    return chain, chain_burned

if __name__ == "__main__":
    chain, chain_burned = main()