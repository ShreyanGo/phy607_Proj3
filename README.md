# Radioactive Decay Parameter Estimation Using MCMC Methods

## Overview

This project implements a Bayesian parameter estimation framework for analyzing radioactive decay processes using Markov Chain Monte Carlo (MCMC) methods. The code employs the Metropolis-Hastings algorithm to simultaneously estimate the decay constant (λ) and initial activity (N₀) from experimental or simulated decay data, providing full posterior distributions and uncertainty quantification for both parameters.

## Physical Model

Radioactive decay follows the exponential decay law:

```
N(t) = N₀ exp(-λt)
```

where:
- N(t) is the activity at time t
- N₀ is the initial activity at t = 0
- λ is the decay constant (units: time⁻¹)

The half-life is related to the decay constant by:

```
t₁/₂ = ln(2) / λ
```

## Theoretical Background

### Bayesian Framework

The posterior distribution of the parameters given the data is obtained via Bayes' theorem:

```
P(θ|D) ∝ P(D|θ) × P(θ)
```

where:
- θ = (N₀, λ) are the model parameters
- D represents the observed data
- P(D|θ) is the likelihood function
- P(θ) is the prior distribution
- P(θ|D) is the posterior distribution

### Likelihood Function

Assuming Gaussian measurement uncertainties, the likelihood function is:

```
P(D|θ) = ∏ᵢ (1/√(2πσᵢ²)) exp[-(Nᵢ - N(tᵢ; θ))² / (2σᵢ²)]
```

where Nᵢ are the observed activities at times tᵢ with uncertainties σᵢ.

### Prior Distributions

The code implements uniform (flat) priors over physically reasonable ranges:
- N₀ ∈ (0, 200) - initial activity must be positive
- λ ∈ (0, 2) - decay constant must be positive

These can be modified based on prior knowledge or experimental constraints.

### Metropolis-Hastings Algorithm

The MCMC sampling proceeds as follows:

1. Initialize parameters θ⁽⁰⁾
2. For iteration i = 1, 2, ..., N:
   - Propose new parameters: θ* ~ q(θ*|θ⁽ⁱ⁻¹⁾)
   - Calculate acceptance probability: α = min(1, P(θ*|D) / P(θ⁽ⁱ⁻¹⁾|D))
   - Accept with probability α, otherwise retain θ⁽ⁱ⁻¹⁾
3. Discard burn-in samples and analyze posterior

The proposal distribution q is a symmetric Gaussian centered at the current state, with tunable standard deviations for each parameter.

## Implementation Details

### Code Structure

The implementation is organized into five main sections:

1. **Data Generation**: Functions for creating synthetic decay data with realistic noise characteristics
2. **Statistical Functions**: Log-likelihood, log-prior, and log-posterior calculations
3. **MCMC Sampler**: Metropolis-Hastings algorithm implementation with progress tracking
4. **Analysis Tools**: Statistical summary functions and convergence diagnostics
5. **Visualization**: Comprehensive plotting routines for results interpretation

### Key Functions

#### `exponential_decay(t, N0, lambda_decay)`
Evaluates the decay model at specified time points.

#### `log_likelihood(params, t, N_obs, sigma)`
Computes the log-likelihood for Gaussian noise model. Returns - infinite for physically invalid parameter values.

#### `metropolis_hastings_2d(t, N_obs, sigma, initial_params, proposal_width, n_iterations, burn_in)`
Main MCMC sampler. Returns the full chain, burned-in chain, and acceptance rate.

### Tuning Parameters

The acceptance rate should ideally be in the range 20-40% for optimal efficiency. This is controlled by the `proposal_width` parameter:
- Larger widths → more exploration, lower acceptance
- Smaller widths → less exploration, higher acceptance

For the current implementation with proposal_width = (2.0, 0.02), the acceptance rate is approximately 7.8%, which may benefit from adjustment.

## Usage

### Basic Usage

```python
# Run the complete analysis with default parameters
python radioactive_decay_mcmc.py
```

### Customizing Parameters

Modify the following variables in the `main()` function:

```python
# True parameters (for synthetic data generation)
N0_true = 100.0        # Initial activity
lambda_true = 0.5      # Decay constant

# Data generation
n_points = 30          # Number of measurements
noise_level = 0.05     # Relative noise (5%)

# MCMC configuration
initial_params = (80.0, 0.4)      # Initial guess
proposal_width = (2.0, 0.02)      # Proposal std deviations
n_iterations = 50000              # Total iterations
burn_in = 10000                   # Burn-in period
```

### Using Experimental Data

To analyze experimental data, replace the data generation section with:

```python
# Load experimental data
t = np.loadtxt('time_data.txt')
N_obs = np.loadtxt('activity_data.txt')
sigma = np.loadtxt('uncertainty_data.txt')
```

Ensure that t, N_obs, and sigma are 1D arrays of equal length.

## Output Files


1. **trace_plots.png**: MCMC chain evolution showing convergence behavior for both parameters
2. **posterior_histograms.png**: 1D marginal posterior distributions with statistics
3. **corner_plot.png**: Joint 2D posterior distribution showing parameter correlations
4. **decay_fit.png**: Data with fitted model and 95% credible interval bands

## Validation Results

Using synthetic data with known parameters:

| Parameter | True Value | Estimated (Mean ± Std) | Relative Error |
|-----------|-----------|------------------------|----------------|
| N₀        | 100.0     | 101.70 ± 1.82         | 1.70%         |
| λ         | 0.500     | 0.505 ± 0.003         | 1.05%         |
| t₁/₂      | 1.386     | 1.372 ± 0.008         | 1.01%         |

The posterior distributions successfully recovered the true parameters within their 95% credible intervals, demonstrating proper implementation of the MCMC algorithm.

## Convergence Diagnostics

### Trace Plots
Visual inspection of trace plots should show:
- Random walk behavior after burn-in (indicating proper mixing)
- Stable mean value (no systematic drift)
- Consistent variance throughout the chain

### Acceptance Rate
The acceptance rate provides insight into proposal distribution tuning:
- Too low (<10%): proposals are too large, increase rejection
- Optimal (20-40%): good balance of exploration and acceptance
- Too high (>60%): proposals are too small, inefficient sampling

### Autocorrelation
High autocorrelation indicates slow mixing. Consider:
- Increasing proposal width
- Running longer chains
- Implementing more sophisticated samplers (e.g., Hamiltonian Monte Carlo)

## Dependencies

```
numpy >= 1.20.0
matplotlib >= 3.3.0
scipy >= 1.6.0
corner >= 2.2.0
```

Install via:
```bash
pip install numpy matplotlib scipy corner --break-system-packages
```

## Extensions and Future Work

Potential enhancements to this implementation:

1. **Alternative Priors**: Implement informative priors (e.g., log-normal for λ based on isotope properties)
2. **Parallel Tempering**: Add temperature ladder for improved sampling of multimodal posteriors
3. **Adaptive Proposals**: Implement adaptive MCMC to automatically tune proposal widths
4. **Multiple Decay Chains**: Extend model to handle decay series (parent → daughter → granddaughter)
5. **Background Subtraction**: Include background activity as an additional parameter
6. **Systematic Uncertainties**: Propagate systematic errors in addition to statistical uncertainties

## References

1. Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd Edition. Chapman and Hall/CRC.
2. Metropolis, N., et al. (1953). "Equation of State Calculations by Fast Computing Machines." *Journal of Chemical Physics* 21(6): 1087-1092.
3. Hastings, W.K. (1970). "Monte Carlo Sampling Methods Using Markov Chains and Their Applications." *Biometrika* 57(1): 97-109.
4. Knoll, G.F. (2010). *Radiation Detection and Measurement*, 4th Edition. Wiley.

## Author

Amelia and Shreyan

## License

This code is provided for educational and research purposes. Feel free to modify and distribute with appropriate attribution.

## Contact

For questions, suggestions, or bug reports, please contact via Syracuse University email or submit an issue through the repository.