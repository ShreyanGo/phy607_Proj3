# ============================================================================
'''
Visualization Module
---------------------------------
functions:
- plot_trace
- plot_corner
- plot_posterior_histograms
- plot_fit_results
'''
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import corner

from nuclear_mcmc.data_model import exponential_decay

def plot_trace(chain, param_names=['N₀', 'λ'], true_values=None):
    """
    Plot MCMC trace plots
    
    Parameters:
    -----------
    chain : ndarray
        Full MCMC chain, shape (n_iter, 2)
    param_names : list of str
        Names of the parameters
    true_values : list or tuple, optional
        True parameter values for reference

    Saves:
    ------
    trace_plots.png
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
    plt.savefig('trace_plots.png', dpi=300, bbox_inches='tight')
    print("Trace plots saved to trace_plots.png")
    plt.close()

def plot_corner(chain_burned, param_names=['N₀', 'λ'], true_values=None):
    """
    Create corner plot for posterior distributions
    
    Parameters:
    -----------
    chain_burned : ndarray
        MCMC samples after burn-in
    param_names : list of str
        Parameter labels
    true_values : list or tuple, optional
        True parameter values

    Saves:
    ------
    corner_plot.png
    """
    fig = corner.corner(chain_burned, 
                       labels=param_names,
                       truths=true_values,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True,
                       title_kwargs={"fontsize": 12})
    
    plt.savefig('corner_plot.png', dpi=300, bbox_inches='tight')
    print("Corner plot saved to corner_plot.png")
    plt.close()

def plot_corner_emcee(samples, param_names=['N₀', 'λ'], true_values=None,
                      filename='corner_plot_emcee.png'):
    """
    Create corner plot for emcee posterior distributions

    Parameters
    ----------
    samples : ndarray
        Flattened emcee samples after burn-in
    param_names : list of str
        Parameter labels
    true_values : list or tuple, optional
        True parameter values
    filename : str
        Output filename for the saved plot

    Saves
    -----
    corner_plot_emcee.png (default)
    """
    fig = corner.corner(samples,
                        labels=param_names,
                        truths=true_values,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12})

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"emcee corner plot saved to {filename}")
    plt.close()

def plot_posterior_histograms(chain_burned, param_names=['N₀', 'λ'], true_values=None):
    """
    Plot 1D posterior histograms
    
    Parameters:
    -----------
    chain_burned : ndarray
        Samples after burn-in
    param_names : list of str
        Parameter labels
    true_values : list or tuple, optional
        True parameter values

    Saves:
    ------
    posterior_histograms.png
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
    plt.savefig('posterior_histograms.png', dpi=300, bbox_inches='tight')
    print("Posterior histograms saved to posterior_histograms.png")
    plt.close()

def plot_fit_results(t, N_obs, sigma, chain_burned, N0_true, lambda_true):
    """
    Plot data with fitted model and uncertainty bands
    
    Parameters:
    -----------
    t : ndarray
        Time points
    N_obs : ndarray
        Observed data
    sigma : ndarray
        Measurement uncertainties
    chain_burned : ndarray
        Posterior samples after burn-in
    N0_true : float
        True initial activity
    lambda_true : float
        True decay constant

    Saves:
    ------
    decay_fit.png
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
    
    plt.savefig('decay_fit.png', dpi=300, bbox_inches='tight')
    print("Fit results saved to decay_fit.png")
    plt.close()
