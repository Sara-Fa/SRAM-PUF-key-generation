"""Main entry point for the two-stage SRAM PUF analysis.
This script runs the full analysis and finds optimal configurations for the two-stage TMVS."""
import argparse

from two_stage_tmvs.analysis.analysis_functions import find_optimal_configurations
from two_stage_tmvs.analysis.analysis_functions import run_full_analysis

# use 'python main.py' to load existing results
# use 'python main.py --recompute' to force recomputation of results

def main():
    """Main entry point for the two-stage SRAM PUF analysis."""
    parser = argparse.ArgumentParser(description='SRAM PUF Two-Stage Analysis')
    parser.add_argument('--recompute', action='store_true',
                       help='Force recomputation of results')
    args = parser.parse_args()

    print("Starting SRAM PUF two-stage analysis...")
    full_results, optimal_configs = run_full_analysis(force_recompute=args.recompute)

    if optimal_configs is None:
        # We loaded existing results, need to find optimal configs
        optimal_configs = find_optimal_configurations(full_results)

    print("\nTop 10 Optimal Configurations:")
    for i, config in enumerate(optimal_configs[:10], 1):
        print(f"\nConfiguration #{i}:")
        print(f"Flipping Probability: {config['p_flip']:.2f}")
        print(f"Stage 1: Code Length={config['code1'][0]},"
              f"Thresholds={config['code1'][1]}-{config['code1'][2]}")
        print(f"Stage 2: Code Length={config['code2'][0]},"
              f" Thresholds={config['code2'][1]}-{config['code2'][2]}")
        print(f"Error Probability: {config['error_prob']:.2e}")
        print(f"SRAM Size: {config['sram_size']:.2f} kB")
        print(f"Helper Data Size: {config['helper_data_size']:.2f} kB")

    print("\nAnalysis complete. Results saved in 'results/' directory.")

if __name__ == "__main__":
    main()
