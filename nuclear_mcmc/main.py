# =============================================================================
# Main Script for Nuclear Decay MCMC Analysis
# =============================================================================

import numpy as np

from nuclear_mcmc.data_model import generate_synthetic_data
from nuclear_mcmc.mh_sampler import metropolis_hastings_2d
from nuclear_mcmc.plotting import (
    plot_trace,
    plot_corner,
    plot_posterior_histograms,
    plot_fit_results,
)
from nuclear_mcmc.diagnostics import (
    print_statistics,
    effective_sample_size,
    geweke_test,
)


def main():
    # =========================================================================
    # Step 1: Generate Synthetic Data
    # =========================================================================
    print("\nGenerating synthetic decay data...")

    np.random.seed(42)
    N0_true = 100.0
    lambda_true = 0.5

    t = np.linspace(0, 10, 50)
    N_obs, sigma = generate_synthetic_data(t, N0_true, lambda_true)

    print("Data generated successfully.")

    # =========================================================================
    # Step 2: Run Metropolis–Hastings MCMC
    # =========================================================================
    print("\nRunning Metropolis–Hastings sampler...")

    initial_params = (80.0, 0.3)
    proposal_width = (2.0, 0.05)

    chain, chain_burned, acceptance_rate = metropolis_hastings_2d(
        t,
        N_obs,
        sigma,
        initial_params,
        proposal_width,
        n_iterations=50000,
        burn_in=10000,
    )

    print(f"Sampler finished. Acceptance rate = {acceptance_rate:.2%}")

    # =========================================================================
    # Step 2.5: Run emcee sampler
    # =========================================================================
    print("\nRunning emcee ensemble sampler...")

    from nuclear_mcmc.emcee_sampler import run_emcee

    # Adjust n_steps > burn_in to ensure non-empty samples
    n_steps = 5000
    burn_in = 1000

    samples_emcee, chain_emcee, acc_emcee = run_emcee(
        t, N_obs, sigma,
        n_walkers=32,
        n_steps=n_steps,
        initial_guess=(100, 0.5),
        init_spread=(1.0, 0.05),
        burn_in=burn_in,
    )

    print(f"emcee Acceptance Fraction: {acc_emcee:.3f}")

    # Debug: check shapes
    print("samples_emcee shape:", samples_emcee.shape)
    print("chain_emcee shape:", chain_emcee.shape)

    # =========================================================================
    # Step 3: Summary statistics for emcee samples (safeguard empty array)
    # =========================================================================
    print("\nSummary statistics for emcee sampler:")

    if samples_emcee.size == 0:
        print("Warning: emcee returned no samples!")
    else:
        print("\nN₀ (emcee):")
        print_statistics(samples_emcee[:, 0].reshape(-1, 1))

        print("\nλ (emcee):")
        print_statistics(samples_emcee[:, 1].reshape(-1, 1))

    # =========================================================================
    # Step 4: Diagnostic Statistics for MH sampler
    # =========================================================================
    print("\nSummary statistics for Metropolis–Hastings sampler:")
    print_statistics(
        chain_burned,
        param_names=["N₀", "λ"],
        true_values=[N0_true, lambda_true],
    )

    print("Computing effective sample sizes...")
    ess_N0 = effective_sample_size(chain_burned[:, 0])
    ess_lambda = effective_sample_size(chain_burned[:, 1])
    print(f"  ESS(N₀) = {ess_N0:.1f}")
    print(f"  ESS(λ)  = {ess_lambda:.1f}")

    print("\nGeweke diagnostic:")
    print(f"  N₀ z-score: {geweke_test(chain_burned[:, 0]):.2f}")
    print(f"  λ z-score:  {geweke_test(chain_burned[:, 1]):.2f}")

    # =========================================================================
    # Step 5: Visualization
    # =========================================================================
    print("\nGenerating plots...")

    plot_trace(chain, param_names=["N₀", "λ"], true_values=[N0_true, lambda_true])
    plot_corner(chain_burned, param_names=["N₀", "λ"], true_values=[N0_true, lambda_true])
    plot_posterior_histograms(chain_burned, param_names=["N₀", "λ"], true_values=[N0_true, lambda_true])

    plot_fit_results(
        t,
        N_obs,
        sigma,
        chain_burned,
        N0_true,
        lambda_true,
    )

    print("Plots saved successfully.")
    print("\nAll tasks completed. MCMC analysis finished!\n")


if __name__ == "__main__":
    main()

