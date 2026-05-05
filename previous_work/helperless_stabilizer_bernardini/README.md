Helperless Stabilizer (Bernardini–Rinaldo) utilities

Contents:
- formulas.py: f_Q(x; λ), Δ(K, η), P[UNREL] exact, and NEW variance-based delta formulas (equation 23).
- lambda_estimation.py: estimate λ per chip from SRAM readouts using the small-window density method; performs a posteriori validation; saves JSON under results/.
- compute_unreliable.py: compute exact P[UNREL] over a grid of K for a given η using saved λ; saves JSON/CSV under results/.
- evaluate_ber.py: BER evaluation with multiple modes (grid, target-BER, fixed-Δ) plus NEW variance-based delta evaluation and dual-threshold functions.
- evaluate_dual_threshold_ber.py: Example script demonstrating dual-threshold BER evaluation.
- plot_reliability_histogram.py: Plot empirical reliability distribution with theoretical PDF overlay.
- plot_local_stability.py: Plot theoretical local stability ε(q) vs reliability q.
- test_variance_delta.py: Test script for NEW variance-based delta calculation functionality.

Reference:
- Bernardini, Rinaldo et al., “Theoretical Limits of Helperless Stabilizers for Physically Unclonable Constants.” This directory implements and evaluates the helperless stabilizer ideas described in that work.

Lambda estimation (window method):
- For a window Δ_window around 0.5, compute the empirical fraction p_emp = #{x_i ∈ [0.5−Δ_window, 0.5+Δ_window]}/N.
- Estimate λ via λ̂ = p_emp / (2Δ_window) using the identity f_Q(1/2; λ) = λ.

A posteriori validation (recorded in results JSON under "validation"):
- Empirical vs exact: Compare p_emp to p_exact = ∫[0.5−Δ_window, 0.5+Δ_window] f_Q(x; λ̂) dx; report relative error and pass/fail vs tolerance (default 2%).
- Exact vs linear approx: Compare p_exact to 2Δ_window·λ̂; report relative error and pass/fail.
- Stability on halved window: Re-estimate λ with Δ_window/2 and report relative change and pass/fail.
- Overall flags: empirical_within_tol, approx_within_tol, stability_within_tol, all_validation_checks_passed, validation_summary (PASS/FAIL).

NEW: Variance-based delta calculation (equation 23):
- delta_from_K_maxvar_eta(K, max_variance, eta): Variance-based formula from equation 23
- This is an ADDITIONAL formula that does NOT replace the existing delta_from_K_eta formula
- Both traditional and variance-based formulas can be used independently

Dual-threshold BER evaluation:
- Base mask: Uses K reads with threshold δ (delta) for initial cell selection
- Regeneration mask: Uses N reads with threshold D (D > δ) for regeneration BER evaluation
- Regeneration BER: Computed on remaining (1000-N) readouts using D-threshold mask
- Enrollment BER: Compares base mask (δ,K) vs subsequent masks (D,N) built from non-overlapping N-sized blocks

Usage:
1) Estimate λ and validate window  
   `python -m previous_work.helperless_stabilizer_bernardini.lambda_estimation`  
   Output: `previous_work/helperless_stabilizer_bernardini/results/lambda_estimates.json`

2) Compute exact P[UNREL] vs K for η (default η = 1−1e−6)  
   `python -m previous_work.helperless_stabilizer_bernardini.compute_unreliable`  
   Outputs: `results/p_unreliable_results.json` and `results/p_unreliable_results.csv`

3) Evaluate BER with dual thresholds (δ,K vs D,N)  
   `python -m previous_work.helperless_stabilizer_bernardini.evaluate_dual_threshold_ber`  
   Output: `results/dual_threshold_ber_results.csv`

4) Test variance-based delta calculation  
   `python -m previous_work.helperless_stabilizer_bernardini.test_variance_delta`  
   Outputs: `results/ber_eval_variance_based_eq23_eta*.csv`

5) Run full helperless experiments (helper masks + BER)  
   `python -m previous_work.helperless_stabilizer_bernardini.experiments.main`  
   Saves HDF5 caches under `experiments/cache/` and JSON/CSV summaries under `results/`.

6) Plotting:
   - Reliability histogram: `python -m previous_work.helperless_stabilizer_bernardini.plot_reliability_histogram`
   - Local stability ε(q): `python -m previous_work.helperless_stabilizer_bernardini.plot_local_stability`
   - Interactive plotting menu (aggregated helper/BER plots):
     `python -m previous_work.helperless_stabilizer_bernardini.plotting.plotting_main`

Plotting directory:
- `plotting/plotting_main.py`: Interactive menu for selecting and generating plots from cached results (BER vs threshold, BSR vs threshold, BER vs readouts for multiple K values).
- `plotting/plotting_configuration.py`: Plot functions including `_load_bernardini_iterative_data_per_chip` used by nvm_free_tmvs overlay plots for Dark Bit comparison.

Experiments directory:
- `experiments/aggregate_results.py`: Aggregates per-chip results. `load_incremental_enrollment_ber_per_chip()` computes `ber_all_over_selected = errors / selected_cells` for comparability with the ODHD pipeline.
- `experiments/key_error_rate_processor.py`: Key error rate (KER) computation.
- See `experiments/README.md` for full documentation.