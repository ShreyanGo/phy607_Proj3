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
    plot_corner_emcee
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

    samples_emcee, chain_emcee, acc_emcee = run_emcee(
        t, N_obs, sigma,
        n_walkers=32,
        n_steps=5000,
        initial_guess=(100, 0.5),
        init_spread=(1.0, 0.05),
        burn_in=1000,
    )

    print(f"emcee Acceptance Fraction: {acc_emcee:.3f}")

    # =========================================================================
    # Step 3: Summary statistics for emcee samples
    # =========================================================================

    print("\n============================================================")
    print("Summary statistics for emcee sampler")
    print("============================================================")

    print_statistics(
        samples_emcee,
        param_names=["N₀", "λ"],
        true_values=[N0_true, lambda_true],
    )

    # Effective sample sizes for emcee
    print("\nComputing effective sample sizes for emcee...")
    ess_N0_e = effective_sample_size(samples_emcee[:, 0])
    ess_lambda_e = effective_sample_size(samples_emcee[:, 1])
    print(f"  ESS(N₀) = {ess_N0_e:.1f}")
    print(f"  ESS(λ)  = {ess_lambda_e:.1f}")

    print("\nGeweke diagnostic (emcee):")
    print(f"  N₀ z-score: {geweke_test(samples_emcee[:, 0]):.2f}")
    print(f"  λ z-score:  {geweke_test(samples_emcee[:, 1]):.2f}")

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

    # MH sampler plots
    plot_trace(chain, param_names=["N₀", "λ"], true_values=[N0_true, lambda_true])
    plot_corner(chain_burned, param_names=["N₀", "λ"], true_values=[N0_true, lambda_true])
    plot_posterior_histograms(chain_burned, param_names=["N₀", "λ"], true_values=[N0_true, lambda_true])

    # NEW: emcee corner plot
    plot_corner_emcee(
        samples_emcee,
        param_names=["N₀", "λ"],
        true_values=[N0_true, lambda_true],
        filename="corner_plot_emcee.png"
    )

    # MH model fit plot
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

