""" Analysis functions for Two-Stage TMVS"""
import pickle
from itertools import product
from pathlib import Path
from tqdm import tqdm
from two_stage_tmvs.analysis.visualization import generate_analysis_plots
import two_stage_tmvs.analysis_constants as const
from two_stage_tmvs.algorithm.base_code import BaseCode
from two_stage_tmvs.algorithm.concatenated_code import ConcatenatedCode

def run_full_analysis(force_recompute=False):
    """Main function to run or load analyses"""
    results_file = const.RESULTS_DIR / "full_analysis.pkl"
    simple_results_file = const.RESULTS_DIR / "simple_analysis.pkl"

    results = None
    simple_results = None

    # Load or compute full (concatenated) analysis
    if not force_recompute and results_file.exists():
        print("Loading previously computed full analysis results...")
        results = load_results(results_file)
    else:
        print("Computing new full analysis results...")
        results = analyze_parameter_space()
        save_results(results, results_file)

    # Load or compute simple (single code) analysis
    if not force_recompute and simple_results_file.exists():
        print("Loading previously computed simple code results...")
        simple_results = load_results(simple_results_file)
    else:
        print("Computing new simple code results...")
        simple_results = analyze_simple_codes()
        save_results(simple_results, simple_results_file)

    # Ensure results directory exists
    const.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("Generating visualizations...")
    generate_analysis_plots(results, const.TARGET_ERROR_PROB, simple_results)

    # Find optimal configurations for full analysis
    optimal_configs = find_optimal_configurations(results)
    save_results(optimal_configs, const.RESULTS_DIR / "optimal_configs.pkl")

    return results, optimal_configs

def analyze_parameter_space():
    """Analyze performance across parameter space"""
    results = []
    # UNCOMMENT the following lines if you want to analyze all combinations of codes
    # total_combinations = len(const.FLIPPING_PROBS) * len(const.CODE_PARAMS)**2
    
    # with tqdm(total=total_combinations, desc="Analyzing parameter space") as pbar:
    #     for p_flip in const.FLIPPING_PROBS:
    #         for code1_params, code2_params in product(const.CODE_PARAMS, repeat=2):

    # COMMENT the following lines if you want to analyze the combinations of codes
    #            where code2 is shorter than code1 (comment until "try:")
    total_combinations = len(const.FLIPPING_PROBS) * sum(
        1 for c1, c2 in product(const.CODE_PARAMS, repeat=2)
        if c1[0] >= c2[0]
    )

    with tqdm(total=total_combinations, desc="Analyzing parameter space") as pbar:
        for p_flip in const.FLIPPING_PROBS:
            for code1_params, code2_params in product(const.CODE_PARAMS, repeat=2):
                if code1_params[0] < code2_params[0]:
                    continue  # Skip if code1 would have smaller length
                try:
                    code1 = BaseCode(code1_params[0], [code1_params[1], code1_params[2]])
                    code2 = BaseCode(code2_params[0], [code2_params[1], code2_params[2]])
                    concat_code = ConcatenatedCode(code1, code2)

                    result = {
                        "p_flip": p_flip,
                        "code1": code1_params,
                        "code2": code2_params,
                        "error_prob": concat_code.two_stage_error_probability(p_flip),
                        "sram_size": concat_code.two_stage_theoretical_required_sram_size(),
                        "helper_data_size": concat_code.two_stage_theoretical_helper_data_size(const.KEY_LENGTH),
                        "codebook_size": concat_code.derive_codebook_memory_size()
                    }
                    results.append(result)
                except ValueError as e:
                    print(f"\nError with {code1_params}→{code2_params} at p_flip={p_flip}: {str(e)}")
                finally:
                    pbar.update(1)
    return results

def analyze_simple_codes():
    """Analyze single-code configurations using code1=(1, [0, 1])"""
    results = []
    total_combinations = len(const.FLIPPING_PROBS) * len(const.CODE_PARAMS)

    simple_code1 = BaseCode(1, [0, 1])

    with tqdm(total=total_combinations, desc="Analyzing simple codes") as pbar:
        for p_flip in const.FLIPPING_PROBS:
            for code2_params in const.CODE_PARAMS:
                try:
                    code2 = BaseCode(code2_params[0], [code2_params[1], code2_params[2]])
                    concat_code = ConcatenatedCode(simple_code1, code2)

                    result = {
                        "p_flip": p_flip,
                        "code1": (1, 0, 1),  # dummy placeholder
                        "code2": code2_params,
                        "error_prob": concat_code.two_stage_error_probability(p_flip),
                        "sram_size": concat_code.two_stage_theoretical_required_sram_size(),
                        "helper_data_size": concat_code.two_stage_theoretical_helper_data_size(const.KEY_LENGTH),
                        "codebook_size": concat_code.derive_codebook_memory_size()
                    }
                    results.append(result)
                except ValueError as e:
                    print(f"\nError with code2={code2_params} at p_flip={p_flip}: {str(e)}")
                finally:
                    pbar.update(1)

    return results


def find_optimal_configurations(results, target_error=const.TARGET_ERROR_PROB):
    """Find configurations meeting error target with minimal resources"""
    filtered = [r for r in results if r['error_prob'] <= target_error]
    # Sort by SRAM size, then helper data size
    return sorted(filtered, key=lambda x: (x['sram_size'], x['helper_data_size']))

def save_results(data, filename):
    """Save results to file"""
    Path(filename).parent.mkdir(exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_results(filename):
    """Load saved results"""
    with open(filename, 'rb') as f:
        return pickle.load(f)
