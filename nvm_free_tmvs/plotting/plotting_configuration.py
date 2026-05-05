""" Plotting configuration module.
This module contains functions to plot various evaluation metrics and results
from the NVM-free TMVS experiments."""
import numpy as np
from nvm_free_tmvs.experiments.aggregated_data_reader import AggregatedDataReader
from nvm_free_tmvs.core.hamming_processor import HammingProcessor
from nvm_free_tmvs.plotting.plotting_functions import Plotting
from nvm_free_tmvs.experiments.optimal_parameters import calculate_failure_vs_memory_tradeoff
from nvm_free_tmvs.utils.analysis_utils import get_shifted_selection_threshold
from nvm_free_tmvs.utils.file_manager  import ReadoutList, get_files, read_readouts
from nvm_free_tmvs.utils.file_manager import enroll_comparator_dir, ber_comparator_dir
from nvm_free_tmvs.utils.file_manager import read_codebook
import nvm_free_tmvs.analysis_constants as const
import pathlib
from previous_work.dark_bit import GenerateBitsMask
from previous_work.helperless_stabilizer_bernardini.plotting.plotting_configuration import _load_bernardini_iterative_data_per_chip
from nvm_free_tmvs.experiments.averaging_data_processor import AveragingDataProcessor
from nvm_free_tmvs.experiments.helper_data_comparator import HelperDataComparator
from nvm_free_tmvs.experiments.global_ber_processor import GlobalBERProcessor
from previous_work.helperless_stabilizer_bernardini.experiments.aggregate_results import (
    load_incremental_enrollment_ber_per_chip,
)


def calculate_and_plot_failure_rate_vs_memory(parameters, nb_enroll_reading_list):
    """ Calculate and plot failure rate vs memory tradeoff. """
    print("Calculating failure rate vs memory tradeoff:")
    print("\tfailure rate list:", const.TEST_FAILURE_RATE_TARGET)
    # nb_enroll_reading_list = const.MAX_ENROLLMENT_READINGS
    failure_rates_list = []
    required_memory_size_list = []
    for nb_enroll_reading in nb_enroll_reading_list:
        print("\tNb. of Enrollment Readings:", nb_enroll_reading)
        (failure_rates_axis, ber_results, required_memory_size_axis, discarding_rates,
        resulting_parameters) = calculate_failure_vs_memory_tradeoff(parameters, nb_enroll_reading)
        print("\tfailure_rates:", failure_rates_axis)
        print("\tber_results:", ber_results)
        print("\trequired_memory_size:", required_memory_size_axis)
        print("\tdiscarding_rates:", discarding_rates)
        print("\tselected_parameters:", resulting_parameters)
        failure_rates_list.append(failure_rates_axis)
        required_memory_size_list.append(required_memory_size_axis)

    Plotting.plot_2d_plots_with_horizontal_line(x_list=required_memory_size_list,
					    y_list=failure_rates_list,
					    xlabel='SRAM Memory Size (kB)',
                        ylabel='Failure Rate',
                        title='Failure Rate vs Memory Size',
                        horizontal_line=1e-6,
                        nb_enroll_reading_list=nb_enroll_reading_list)

def plot_3d_evaluation_vs_threshold_and_num_readings(parameters, dir_name, trivial=None):
    """  Plot 3D evaluation vs threshold and number of readings. """
    print("Plotting 3D evaluation vs threshold and number of readings:")
    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    coeff = [0,0]
    for code_len, coeff[0], coeff[1] in parameters:
        select_th = get_shifted_selection_threshold(code_len, coeff)
        print("\tcode_length=", code_len, ", threshold=",coeff,
              "shifted_th=", select_th)
        cb = read_codebook(code_len, coeff[0], coeff[1])
        codebook_length = len(cb) if cb is not None else 1
        print("\tcodebook_length=", codebook_length)
        target_threshold = select_th

        # Initialize the reader
        reader = AggregatedDataReader(code_len, coeff, nb_enroll_reading, dir_name, trivial=trivial)
        x_axis_val, y_axis_val, z_axis_val = reader.read_aggregated_data()

        # Find the index where y_axis (enroll_select_threshold) matches (-3.0, 3.0)
        threshold_index = np.where((y_axis_val[:, 0] == target_threshold[0]) &
                            (y_axis_val[:, 1] == target_threshold[1]))[0]
        y_axis_val = y_axis_val[:,1] # take only higher threshold

        z_axis_mean = np.array(z_axis_val["mean"])

        # Extract threshold high values (y_axis_val may be 1D or 2D)
        if y_axis_val.ndim == 2:
            th_high_vals = y_axis_val[:, 1]
        else:
            th_high_vals = y_axis_val

        # Print BER at first and last readout for all thresholds
        metric_label = "BER_Reg" if dir_name == ber_comparator_dir else "BER_Enr"
        key_length = 16
        has_psr = z_axis_mean.shape[1] > 1
        num_readings = z_axis_mean.shape[2]
        for nr_idx, nr_label in [(0, 1), (num_readings - 1, num_readings)]:
            hdr = (f"\n\t{'TH*_high':>8s}  {metric_label + f'(NrRead={nr_label})':>24s}  "
                   f"{'KER_est(K=' + str(key_length) + ')':>16s}")
            if has_psr:
                hdr += f"  {'PSR':>8s}"
            print(hdr)
            print(f"\t{'-'*8}  {'-'*24}  {'-'*16}" + (f"  {'-'*8}" if has_psr else ""))
            for ti in range(len(th_high_vals)):
                th_high = th_high_vals[ti]
                ber_val = z_axis_mean[ti, 0, nr_idx]
                ker_val = 1.0 - (1.0 - ber_val) ** key_length
                line = f"\t{th_high:8.1f}  {ber_val:24.6f}  {ker_val:16.6e}"
                if has_psr:
                    psr_val = 1 - z_axis_mean[ti, 1, nr_idx]
                    line += f"  {psr_val:8.4f}"
                print(line)

        y_axis_val = th_high_vals

        if dir_name == ber_comparator_dir:
            reader.plot_3d_results(x_axis_val, y_axis_val, z_axis_mean[:,0,:],
                                   r'Nb. of Readings $N_{\mathrm{res}}$', r'Selection Threshold $\mathrm{TH}^*_{\mathrm{High}}$',
                                   r'$\mathrm{BER}_\mathrm{Reg}$',
                                   None)
        else:
            reader.plot_3d_results(x_axis_val, y_axis_val, z_axis_mean[:,0,:],
                                      r'Nb. of Readings $N_{\mathrm{res}}$', r'Selection Threshold $\mathrm{TH}^*_{\mathrm{High}}$',
                                      r'$\mathrm{BER}_\mathrm{Enr}$',
                                   1 - z_axis_mean[:,1,:], select_th)

def plot_2d_evaluation_vs_num_readings(parameters, dir_name, trivial=None):
    """  Plot 2D evaluation vs number of readings. """
    print("Plotting 2D evaluation vs number of readings:")
    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    coeff = [0,0]
    for code_len, coeff[0], coeff[1] in parameters:
        select_th = get_shifted_selection_threshold(code_len, coeff)
        print("\tcode_length=", code_len, ", threshold=",coeff,
              "shifted_th=", select_th)
        cb = read_codebook(code_len, coeff[0], coeff[1])
        codebook_length = len(cb) if cb is not None else 1
        print("\tcodebook_length=", codebook_length)
        target_threshold = select_th

        # Initialize the reader
        reader = AggregatedDataReader(code_len, coeff, nb_enroll_reading, dir_name, trivial=trivial)
        num_readings_list, thresholds_list, results_list = reader.read_aggregated_data()

        # Find the index where y_axis (enroll_select_threshold) matches (-3.0, 3.0)
        threshold_index = np.where((thresholds_list[:, 0] == target_threshold[0]) &
                            (thresholds_list[:, 1] == target_threshold[1]))[0]

        z_axis_mean = np.array(results_list["mean"])
        # z_axis_min = np.array(results_list["min"])
        # z_axis_max = np.array(results_list["max"])
            
        ylabel = None
        if dir_name == ber_comparator_dir:
            ylabel=r'$\mathrm{BER}_\mathrm{Reg}$'
            print("\tMean BER at select_th:", z_axis_mean[threshold_index,0,:])
        else:
            ylabel=r'$\mathrm{BER}_\mathrm{Enr}$'
            print("\terror_count z_axis[\"mean\"]:", z_axis_mean[threshold_index,0,:])
            print("\tdiscarded_patterns_count z_axis[\"mean\"]:", z_axis_mean[threshold_index,1,:])
            # # print("\terror_count z_axis[\"min\"]:", z_axis_min[threshold_index,0,:])
            # # print("\terror_count z_axis[\"max\"]:", z_axis_max[threshold_index,0,:])
            # print("\tMax Extraction rate at select_th:", z_axis_max[threshold_index,4,:])

        second_y = None
        second_ylabel = None
        if dir_name == enroll_comparator_dir:
            second_y = 1- z_axis_mean[:,1,:].T
            second_ylabel = 'PSR'
        print("shape of z_axis data:", z_axis_mean[:,0,:].T.shape)
        Plotting.plot_2d_line_graphs_with_second_yaxis(x=num_readings_list,
                                    y=z_axis_mean[:,0,:].T,
                                    z=[threshold_index+1], #num_readings_list,
                                    xlabel=r'$N_{\mathrm{res}, \max}$',
                                    ylabel=ylabel,
                                    title='BER vs Thresholds for Different Number of Readings',
                                    legend_label=r'Selection Threshold $\mathrm{TH}^*_{\mathrm{enr}}$',
                                    second_y=second_y, second_ylabel=second_ylabel)

def plot_2d_evaluation_vs_threshold(parameters, target_num_readings, dir_name,
                                    min_keys=100, trivial=None):
    """  Plot 2D evaluation vs threshold. Filters out thresholds with fewer
    than min_keys extractable keys (based on #selected / key_length). """
    print("Plotting 2D evaluation vs threshold:")
    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    coeff = [0,0]
    for code_len, coeff[0], coeff[1] in parameters:
        select_th = get_shifted_selection_threshold(code_len, coeff)
        print("\n\tcode_length=", code_len, ", threshold=",coeff,
              "shifted_th=", select_th)
        cb = read_codebook(code_len, coeff[0], coeff[1])
        codebook_length = len(cb) if cb is not None else 1  # trivial: 1 key bit per pattern
        print("\tcodebook_length=", codebook_length)
        target_threshold = [-1.1,1.1] # select_th

        # Initialize the reader
        reader = AggregatedDataReader(code_len, coeff, nb_enroll_reading, dir_name, trivial=trivial)
        _, thresholds_list, results_list = reader.read_aggregated_data()

        # Find the index where y_axis (enroll_select_threshold) matches (-3.0, 3.0)
        threshold_index = np.where((thresholds_list[:, 0] == target_threshold[0]) &
                            (thresholds_list[:, 1] == target_threshold[1]))[0]

        z_axis_mean = np.array(results_list["mean"])
        # z_axis_min = np.array(results_list["min"])
        # z_axis_max = np.array(results_list["max"])

        # Print BER, PSR, and #selected at each target NrRead for all thresholds
        from common.data_reading_utils import get_num_sram_patterns
        num_patterns = get_num_sram_patterns(code_len)
        metric_label = "BER_Reg" if dir_name == ber_comparator_dir else "BER_Enr"
        key_length = 16
        has_psr = z_axis_mean.shape[1] > 1

        # Filter: keep only thresholds with #Selected >= min_keys at last NrRead
        if has_psr and min_keys > 0:
            last_idx = z_axis_mean.shape[2] - 1
            n_sel_last = (1 - z_axis_mean[:, 1, last_idx]) * num_patterns
            keep = n_sel_last >= min_keys
            n_dropped = int(np.sum(~keep))
            if n_dropped > 0:
                print(f"\tFiltered out {n_dropped} thresholds "
                      f"with #Selected < {min_keys}")
            thresholds_list = thresholds_list[keep]
            z_axis_mean = z_axis_mean[keep]

        for nr in target_num_readings:
            nr_idx = min(nr - 1, z_axis_mean.shape[2] - 1)
            hdr = f"\n\t{'TH*_high':>8s}  {metric_label + f'(NrRead={nr})':>24s}  {'KER_est(K=' + str(key_length) + ')':>16s}"
            sep = f"\t{'-'*8}  {'-'*24}  {'-'*16}"
            if has_psr:
                hdr += f"  {'PSR':>8s}  {'#Selected':>9s}"
                sep += f"  {'-'*8}  {'-'*9}"
            print(hdr)
            print(sep)
            for ti in range(len(thresholds_list)):
                th_high = thresholds_list[ti, 1]
                ber_val = z_axis_mean[ti, 0, nr_idx]
                ker_val = 1.0 - (1.0 - ber_val) ** key_length
                line = f"\t{th_high:8.1f}  {ber_val:24.6f}  {ker_val:16.6e}"
                if has_psr:
                    psr_val = 1 - z_axis_mean[ti, 1, nr_idx]
                    n_sel = psr_val * num_patterns
                    line += f"  {psr_val:8.4f}  {n_sel:9.1f}"
                print(line)

        ylabel = None
        if dir_name == ber_comparator_dir:
            ylabel=r'$\mathrm{BER}_\mathrm{Reg}$'
        else:
            ylabel=r'$\mathrm{BER}_\mathrm{Enr}$'

        second_y = None
        second_ylabel = None
        if dir_name == enroll_comparator_dir:
            second_y = 1- z_axis_mean[:,1,:]
            second_ylabel = 'PSR'
        Plotting.plot_2d_line_graphs_with_second_yaxis(x=thresholds_list[:, 1],
                                    y=z_axis_mean[:,0,:],
                                    z=target_num_readings, #num_readings_list,
                                    xlabel=r'Selection Threshold $\mathrm{TH}^*_{\mathrm{High}}$',
                                    ylabel=ylabel,
                                    title='BER vs Thresholds for Different Number of Readings',
                                    legend_label=r'$N_{\mathrm{res}}$',
                                    second_y=second_y, second_ylabel=second_ylabel)

def plot_ber_vs_n_trivial(parameters, target_th_high, target_nr_read_list, dir_name):
    """Plot BER_Enr vs n for trivial codebook at a fixed threshold.

    All NrRead curves on one figure. Uses aggregated .h5 files (fast).
    """
    import matplotlib.pyplot as plt

    is_enroll = (dir_name == enroll_comparator_dir)
    metric_label = "BER_Enr" if is_enroll else "BER_Reg"
    class_instance = HelperDataComparator if is_enroll else GlobalBERProcessor

    key_length = 16
    if isinstance(target_nr_read_list, int):
        target_nr_read_list = [target_nr_read_list]

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    # Collect data for all NrRead values
    all_series = {}
    all_n_union = set()

    for target_nr_read in target_nr_read_list:
        n_values, ber_values = [], []

        print(f"\n{'='*60}")
        print(f"N_res,max={target_nr_read}:")
        print(f"\t{'n':>4s}  {metric_label:>12s}  {'KER_est(K=' + str(key_length) + ')':>16s}  {'PSR':>8s}")
        print(f"\t{'-'*4}  {'-'*12}  {'-'*16}  {'-'*8}")

        for code_len, cb_low, cb_high in parameters:
            # Use _load_per_threshold_ber which reads aggregated files
            data = _load_per_threshold_ber(
                dir_name, code_len, cb_low, cb_high,
                target_nr_read, trivial=True, class_instance=class_instance,
                min_selected=0)
            # Find the entry matching target_th_high
            match = [d for d in data if abs(d['th_high'] - target_th_high) < 0.05]
            if match:
                ber = match[0]['ber']
                psr = match[0]['psr']
                ker = 1.0 - (1.0 - ber) ** key_length
                n_values.append(code_len)
                ber_values.append(ber)
                all_n_union.add(code_len)
                print(f"\t{code_len:4d}  {ber:12.6f}  {ker:16.6e}  {psr:8.4f}")
            else:
                print(f"\t{code_len:4d}  (no TH*={target_th_high} data)")

        all_series[target_nr_read] = {'n': n_values, 'ber': ber_values}

    if not all_n_union:
        print("No data found.")
        return

    # Single figure with all NrRead curves
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, nr in enumerate(target_nr_read_list):
        s = all_series.get(nr, {'n': [], 'ber': []})
        if s['n']:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            ax.plot(s['n'], s['ber'], marker=marker, linestyle='-', color=color,
                    linewidth=1.5, markersize=5, label=f'$N_{{\\mathrm{{res}}}}$ = {round(nr, -1) if nr >= 5 else nr}')

    ax.set_xlabel(r'Code Length $n$', fontsize=14, weight='bold')
    ax.set_ylabel(r'$\mathbf{' + metric_label.replace('_', r'}_\mathbf{') + r'}$',
                   rotation=0, labelpad=20, fontsize=16, weight='bold')
    all_n_sorted = sorted(all_n_union)
    ax.set_xticks(all_n_sorted)
    ax.set_xticklabels([str(v) for v in all_n_sorted], fontsize=8)
    ax.set_yscale('log')
    ax.legend(fontsize=12, loc='center right')
    ax.grid(True, alpha=0.3)
    title_th = f"TH*={target_th_high}" if target_th_high > 0 else "no threshold"
    # plt.title(f'{metric_label} vs n (trivial, {title_th})')
    plt.tight_layout()
    nr_str = '_'.join(str(nr) for nr in target_nr_read_list)
    fname = f"./nvm_free_tmvs/figures/{metric_label.lower()}_vs_n_trivial_TH{target_th_high}_NrRead{nr_str}.pdf"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\nSaved to {fname}")
    plt.show()


def plot_2d_evaluation_vs_num_readings_overlay(parameters, chip_id, primary_dir, overlay_label="Dark bits"):
    """Deprecated: use dedicated single-axis overlay plots below."""
    print("Plotting 2D evaluation vs number of readings with overlay (Dark Bits):")
    if not isinstance(primary_dir, pathlib.Path):
        primary_dir = pathlib.Path(primary_dir)
    # dark_bits_dir kept for future use if precomputed metrics are stored there

    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    coeff = [0,0]
    for code_len, coeff[0], coeff[1] in parameters:
        select_th = get_shifted_selection_threshold(code_len, coeff)
        print("\tcode_length=", code_len, ", threshold=",coeff,
              "shifted_th=", select_th)
        cb = read_codebook(code_len, coeff[0], coeff[1])
        _ = len(cb) if cb is not None else 1
        target_threshold = select_th

        # Enrollment comparator dataset (primary)
        reader = AggregatedDataReader(code_len, coeff, nb_enroll_reading, primary_dir)
        num_readings_list, thresholds_list, results_list = reader.read_aggregated_data()
        threshold_index = np.where((thresholds_list[:, 0] == target_threshold[0]) &
                            (thresholds_list[:, 1] == target_threshold[1]))[0]
        z_axis_mean = np.array(results_list["mean"])  # (num_thresholds, criteria, num_readings)
        z_axis_min = np.array(results_list["min"])    # (num_thresholds, criteria, num_readings)
        z_axis_max = np.array(results_list["max"])    # (num_thresholds, criteria, num_readings)

        # Labels and PSR from enrollment comparator
        ylabel=r'$\mathrm{BER}_\mathrm{Enr}$'
        second_y = 1 - z_axis_mean[:,1,:].T
        second_ylabel = 'Bits Selection (%)'

        # Compute Dark Bits aggregate metrics across all chips for fair comparison
        x_dark, dark_results = GenerateBitsMask.compute_aggregate_metrics_over_chips(nb_enroll_readings=nb_enroll_reading)
        

        # Align lengths (truncate to min length)
        L = int(min(len(num_readings_list), len(x_dark)))
        x_main = num_readings_list[:L]

        # Select threshold column for our approach
        col = int(threshold_index[0]) if threshold_index.size > 0 else 0
        # Our approach BER bands from aggregated stats
        ber_mean = z_axis_mean[col, 0, :][:L]  # criteria 0 = BER
        ber_min = z_axis_min[col, 0, :][:L]
        ber_max = z_axis_max[col, 0, :][:L]

        # PSR = 1 - discarded; bands invert min/max accordingly
        psr_mean = (1 - z_axis_mean[col, 1, :])[:L]
        psr_min = (1 - z_axis_max[col, 1, :])[:L]
        psr_max = (1 - z_axis_min[col, 1, :])[:L]

        # Dark Bits aggregate BER and discarded metrics across chips
        dark_ber = dark_results['ber']['mean'][:L]
        dark_min = dark_results['ber']['min'][:L]
        dark_max = dark_results['ber']['max'][:L]
        dark_disc_mean = 1 - dark_results['discarded']['mean'][:L]
        dark_disc_min = 1 - dark_results['discarded']['min'][:L]
        dark_disc_max = 1 - dark_results['discarded']['max'][:L]

        # Keep old combined plot for backward compatibility
        Plotting.plot_2d_overlay_with_bands(
            x=x_main,
            a_mean=ber_mean, a_min=ber_min, a_max=ber_max, a_label='ODHD',
            b_mean=dark_ber, b_min=dark_min, b_max=dark_max, b_label=overlay_label,
            xlabel=r'$N_{\mathrm{res}, \max}$', ylabel=ylabel,
            second_y_a_mean=psr_mean, second_y_a_min=psr_min, second_y_a_max=psr_max,
            second_y_b_mean=dark_disc_mean, second_y_b_min=dark_disc_min, second_y_b_max=dark_disc_max,
            second_ylabel=second_ylabel,
        )

def plot_2d_evaluation_vs_num_readings_overlay_ber(parameters, primary_dir,
                                                   overlay_label="Dark bits",
                                                   K=500, reference_delta=0.499, test_D=0.4991,
                                                   th_high_star=None):
    """Single-axis overlay plot for BER_Enr only (ODHD vs Bernardini Dark Bit).

    Uses Bernardini incremental enrollment BER cache (ber_all_over_selected)
    for the overlay, NOT the raw dark-bit per-readout BER.
    """
    import pickle
    from common.data_reading_utils import get_files as get_files_common, read_readouts as read_readouts_common
    print("Plotting 2D evaluation vs number of readings with overlay (BER only):")
    if not isinstance(primary_dir, pathlib.Path):
        primary_dir = pathlib.Path(primary_dir)

    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    coeff = [0,0]
    for code_len, coeff[0], coeff[1] in parameters:
        select_th = get_shifted_selection_threshold(code_len, coeff)
        target_threshold = select_th

        reader = AggregatedDataReader(code_len, coeff, nb_enroll_reading, primary_dir)
        num_readings_list, thresholds_list, results_list = reader.read_aggregated_data()
        threshold_index = np.where((thresholds_list[:, 0] == target_threshold[0]) &
                            (thresholds_list[:, 1] == target_threshold[1]))[0]
        z_axis_mean = np.array(results_list["mean"])
        z_axis_min = np.array(results_list["min"])
        z_axis_max = np.array(results_list["max"])

        ylabel = r'$\mathbf{BER}_\mathbf{Enr}$'

        # Load per-chip Bernardini incremental enrollment BER (ber_all_over_selected)
        cache_dir = pathlib.Path(
            "previous_work/helperless_stabilizer_bernardini/experiments/cache")
        d_str = f"{float(reference_delta):.3f}".replace('.', 'p')
        all_files_common = get_files_common()

        dark_ber_chips = []
        for p in sorted(cache_dir.glob(f"incremental_enroll_ber_*_K{K}_delta{d_str}_D*.pkl")):
            chip_id = p.name.split("_")[3]
            with open(p, "rb") as f:
                obj = pickle.load(f)
            err = np.asarray(obj["error_count"], dtype=np.float64)
            disc = np.asarray(obj["discarded_patterns_count"], dtype=np.float64)
            if chip_id not in all_files_common:
                continue
            readouts_c = read_readouts_common(all_files_common[chip_id])
            num_cells = int(readouts_c[0].data.size)
            selected = num_cells - disc
            safe_sel = np.where(selected == 0, 1, selected)
            ber_per_range = err / safe_sel
            ber_per_range[selected == 0] = 0
            dark_ber_chips.append(np.mean(ber_per_range, axis=0))

        # Load per-chip ODHD enrollment BER for each n in parameters
        import h5py
        import matplotlib.pyplot as plt

        min_nr_read = 3
        L = nb_enroll_reading
        # Use only odd NrRead indices to avoid even/odd zigzag at TH*=0
        odd_indices = np.arange(min_nr_read - 1, L, 2)  # 0-based: indices 2,4,6,...
        x_plot_odd = odd_indices + 1  # NrRead values: 3,5,7,...

        odhd_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'brown']
        odhd_data_per_n = {}   # {n: array of per-chip BER series}
        odhd_th_per_n = {}     # {n: th_high value used}

        for pi, (cl, cbl, cbh) in enumerate(parameters):
            # Use explicit threshold if provided, otherwise codebook threshold
            if isinstance(th_high_star, (list, tuple)):
                th_val = th_high_star[pi]
                target_th = [-th_val, th_val]
            elif th_high_star is not None:
                th_val = th_high_star
                target_th = [-th_val, th_val]
            else:
                target_th = get_shifted_selection_threshold(cl, [cbl, cbh])
                th_val = target_th[1]
            suffix = (f"_N{cl}_Threshold_{cbl}_{cbh}"
                      f"_MaxEnrollReadings_{nb_enroll_reading}")
            chips_ber = []
            for fpath in sorted(primary_dir.glob(f"*{suffix}*.h5")):
                if fpath.name.startswith("aggregated"):
                    continue
                with h5py.File(fpath, "r") as hf:
                    th_low, th_high = target_th
                    th_group = None
                    for gname in hf.keys():
                        parts = gname.split("_")[1:]
                        if len(parts) >= 4:
                            gl = float(parts[0] + '.' + parts[1])
                            gh = float(parts[2] + '.' + parts[3])
                            if abs(gl - th_low) < 0.05 and abs(gh - th_high) < 0.05:
                                th_group = gname
                                break
                    if th_group is None or "combined_data" not in hf[th_group]:
                        continue
                    data = hf[th_group]["combined_data"][:]
                    rates = HelperDataComparator.get_rates_given_counts_single_threshold(
                        data, cl, [cbl, cbh])
                    chips_ber.append(np.mean(rates[0, 1:], axis=0))
            if chips_ber:
                odhd_data_per_n[cl] = np.array(chips_ber)
                odhd_th_per_n[cl] = th_val

        # Print summary at key NrRead values
        print_nrs = [nr for nr in [3, 10, 50, 100, 499] if nr <= L]
        print(f"\n\t{'Method':<24s}", end="")
        for nr in print_nrs:
            print(f"  {'NrRead=' + str(nr):>14s}", end="")
        print()
        print(f"\t{'-'*24}", end="")
        for _ in print_nrs:
            print(f"  {'-'*14}", end="")
        print()
        for n_val, arr in sorted(odhd_data_per_n.items()):
            th = odhd_th_per_n.get(n_val, '?')
            lbl = f"ODHD(n={n_val},TH={th})"
            print(f"\t{lbl:<24s}", end="")
            for nr in print_nrs:
                idx = nr - 1
                if idx < arr.shape[1]:
                    print(f"  {np.mean(arr[:, idx]):14.6f}", end="")
                else:
                    print(f"  {'N/A':>14s}", end="")
            print()
        if dark_ber_chips:
            dark_arr_full = np.array(dark_ber_chips)
            print(f"\t{overlay_label:<24s}", end="")
            for nr in print_nrs:
                idx = nr - 1
                if idx < dark_arr_full.shape[1]:
                    print(f"  {np.mean(dark_arr_full[:, idx]):14.6f}", end="")
                else:
                    print(f"  {'N/A':>14s}", end="")
            print()

        # Plot (odd NrRead only to avoid even/odd zigzag at TH*=0)
        fig, ax1 = plt.subplots(figsize=(10, 5))

        for i, (n_val, arr) in enumerate(sorted(odhd_data_per_n.items())):
            color = odhd_colors[i % len(odhd_colors)]
            sampled = arr[:, odd_indices]
            mean = np.mean(sampled, axis=0)
            mn = np.min(sampled, axis=0)
            mx = np.max(sampled, axis=0)
            if len(odhd_data_per_n) == 1:
                lbl = 'ODHD'
            else:
                th = odhd_th_per_n.get(n_val, 0)
                lbl = f'ODHD (n={n_val}, $\\mathrm{{TH}}^*_{{\\mathrm{{high}}}}$={th})'
            ax1.plot(x_plot_odd, mean, color=color, linewidth=1.5, label=lbl)
            ax1.fill_between(x_plot_odd, mn, mx, color=color, alpha=0.15)

        if dark_ber_chips:
            dark_arr = np.array(dark_ber_chips)
            dark_sampled = dark_arr[:, odd_indices[:dark_arr.shape[1]]]
            dark_x = x_plot_odd[:dark_sampled.shape[1]]
            dark_mean = np.mean(dark_sampled, axis=0)
            dark_min = np.min(dark_sampled, axis=0)
            dark_max = np.max(dark_sampled, axis=0)
            ax1.plot(dark_x, dark_mean, color='red', linewidth=1.5, label=overlay_label)
            ax1.fill_between(dark_x, dark_min, dark_max, color='red', alpha=0.15)

        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_xticks([3, 10, 100, 500])
        ax1.set_xticklabels(['3', '10', '100', '500'])
        ax1.xaxis.set_minor_formatter(plt.NullFormatter())
        # Force y-axis to include 10^-1
        y_lo, y_hi = ax1.get_ylim()
        ax1.set_ylim(min(y_lo, 5e-3), max(y_hi, 0.15))
        import matplotlib.ticker as mticker
        ax1.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=15))
        ax1.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        ax1.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=15))
        ax1.set_xlabel(r'Number of Redaouts $\mathbf{K}$', fontsize=16, weight='bold')
        ax1.set_ylabel(ylabel, rotation=0, fontsize=16, weight='bold', labelpad=30)
        ax1.legend(loc='upper right', fontsize=14)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./nvm_free_tmvs/figures/ber_enr_vs_readouts_overlay.pdf',
                    dpi=300, bbox_inches='tight')
        print("Saved to ./nvm_free_tmvs/figures/ber_enr_vs_readouts_overlay.pdf")
        plt.show()

def plot_2d_evaluation_vs_num_readings_overlay_psr(parameters, primary_dir,
                                                   overlay_label="Dark Bit",
                                                   K=500, reference_delta=0.499,
                                                   th_high_star=None):
    """Single-axis overlay plot for BSR/PSR only (ODHD vs Dark Bit).

    Uses per-chip caches for ODHD and Bernardini incremental cache for Dark Bit.
    Supports multiple n values and explicit threshold via th_high_star.
    """
    import pickle, h5py
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from common.data_reading_utils import get_files as get_files_common, read_readouts as read_readouts_common
    print("Plotting 2D evaluation vs number of readings with overlay (BSR only):")
    if not isinstance(primary_dir, pathlib.Path):
        primary_dir = pathlib.Path(primary_dir)

    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS

    # Load Dark Bit BSR from Bernardini cache
    cache_dir = pathlib.Path(
        "previous_work/helperless_stabilizer_bernardini/experiments/cache")
    d_str = f"{float(reference_delta):.3f}".replace('.', 'p')
    all_files_common = get_files_common()

    dark_bsr_chips = []
    for p in sorted(cache_dir.glob(f"incremental_enroll_ber_*_K{K}_delta{d_str}_D*.pkl")):
        chip_id = p.name.split("_")[3]
        with open(p, "rb") as f:
            obj = pickle.load(f)
        disc = np.asarray(obj["discarded_patterns_count"], dtype=np.float64)
        if chip_id not in all_files_common:
            continue
        readouts_c = read_readouts_common(all_files_common[chip_id])
        num_cells = int(readouts_c[0].data.size)
        bsr_per_range = 1 - disc / num_cells
        dark_bsr_chips.append(np.mean(bsr_per_range, axis=0))

    # Load per-chip ODHD PSR
    odhd_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'brown']
    odhd_data_per_n = {}
    odhd_th_per_n = {}

    coeff = [0, 0]
    for pi, (cl, cbl, cbh) in enumerate(parameters):
        if isinstance(th_high_star, (list, tuple)):
            th_val = th_high_star[pi]
            target_th = [-th_val, th_val]
        elif th_high_star is not None:
            th_val = th_high_star
            target_th = [-th_val, th_val]
        else:
            target_th = get_shifted_selection_threshold(cl, [cbl, cbh])
            th_val = target_th[1]
        suffix = f"_N{cl}_Threshold_{cbl}_{cbh}_MaxEnrollReadings_{nb_enroll_reading}"
        chips_psr = []
        for fpath in sorted(primary_dir.glob(f"*{suffix}*.h5")):
            if fpath.name.startswith("aggregated"):
                continue
            with h5py.File(fpath, "r") as hf:
                th_low, th_high = target_th
                th_group = None
                for gname in hf.keys():
                    parts = gname.split("_")[1:]
                    if len(parts) >= 4:
                        gl = float(parts[0] + '.' + parts[1])
                        gh = float(parts[2] + '.' + parts[3])
                        if abs(gl - th_low) < 0.05 and abs(gh - th_high) < 0.05:
                            th_group = gname
                            break
                if th_group is None or "combined_data" not in hf[th_group]:
                    continue
                data = hf[th_group]["combined_data"][:]
                rates = HelperDataComparator.get_rates_given_counts_single_threshold(
                    data, cl, [cbl, cbh])
                # BSR = PSR / n (each n-bit pattern gives 1 key bit)
                psr = 1 - np.mean(rates[1], axis=0)
                chips_psr.append(psr / cl)
        if chips_psr:
            odhd_data_per_n[cl] = np.array(chips_psr)
            odhd_th_per_n[cl] = th_val

    # Plot (odd NrRead only)
    min_nr_read = 3
    L = nb_enroll_reading
    odd_indices = np.arange(min_nr_read - 1, L, 2)
    x_plot_odd = odd_indices + 1

    # Print summary at key NrRead values
    print_nrs = [nr for nr in [3, 10, 50, 100, 499] if nr <= L]
    print(f"\n\t{'Method':<24s}", end="")
    for nr in print_nrs:
        print(f"  {'NrRead=' + str(nr):>14s}", end="")
    print()
    print(f"\t{'-'*24}", end="")
    for _ in print_nrs:
        print(f"  {'-'*14}", end="")
    print()
    for n_val, arr in sorted(odhd_data_per_n.items()):
        th = odhd_th_per_n.get(n_val, '?')
        lbl = f"ODHD(n={n_val},TH={th})"
        print(f"\t{lbl:<24s}", end="")
        for nr in print_nrs:
            idx = nr - 1
            if idx < arr.shape[1]:
                print(f"  {np.mean(arr[:, idx]):14.6f}", end="")
            else:
                print(f"  {'N/A':>14s}", end="")
        print()
    if dark_bsr_chips:
        dark_arr_full = np.array(dark_bsr_chips)
        print(f"\t{overlay_label:<24s}", end="")
        for nr in print_nrs:
            idx = nr - 1
            if idx < dark_arr_full.shape[1]:
                print(f"  {np.mean(dark_arr_full[:, idx]):14.6f}", end="")
            else:
                print(f"  {'N/A':>14s}", end="")
        print()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    for i, (n_val, arr) in enumerate(sorted(odhd_data_per_n.items())):
        color = odhd_colors[i % len(odhd_colors)]
        sampled = arr[:, odd_indices]
        mean = np.mean(sampled, axis=0)
        mn = np.min(sampled, axis=0)
        mx = np.max(sampled, axis=0)
        if len(odhd_data_per_n) == 1:
            lbl = 'ODHD'
        else:
            th = odhd_th_per_n.get(n_val, 0)
            lbl = f'ODHD (n={n_val}, $\\mathrm{{TH}}^*_{{\\mathrm{{high}}}}$={th})'
        ax1.plot(x_plot_odd, mean, color=color, linewidth=1.5, label=lbl)
        ax1.fill_between(x_plot_odd, mn, mx, color=color, alpha=0.15)

    if dark_bsr_chips:
        dark_arr = np.array(dark_bsr_chips)
        dark_sampled = dark_arr[:, odd_indices[:dark_arr.shape[1]]]
        dark_x = x_plot_odd[:dark_sampled.shape[1]]
        dark_mean = np.mean(dark_sampled, axis=0)
        dark_min = np.min(dark_sampled, axis=0)
        dark_max = np.max(dark_sampled, axis=0)
        ax1.plot(dark_x, dark_mean, color='red', linewidth=1.5, label=overlay_label)
        ax1.fill_between(dark_x, dark_min, dark_max, color='red', alpha=0.15)

    ax1.set_xscale('log')
    ax1.set_xticks([3, 10, 100, 500])
    ax1.set_xticklabels(['3', '10', '100', '500'])
    ax1.xaxis.set_minor_formatter(plt.NullFormatter())
    ax1.set_xlabel(r'Number of Redaouts $\mathbf{K}$', fontsize=16, weight='bold')
    ax1.set_ylabel(r'$\mathbf{BSR}$', rotation=0, fontsize=16, weight='bold', labelpad=30)
    ax1.legend(loc='best', fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./nvm_free_tmvs/figures/bsr_vs_readouts_overlay.pdf',
                dpi=300, bbox_inches='tight')
    print("Saved to ./nvm_free_tmvs/figures/bsr_vs_readouts_overlay.pdf")
    plt.show()

def initialize_and_plot_hd_values_histogram(parameters, chip_id=None):
    """ Initialize and plot Hamming Distance
    histogram for a given chip_id or all chips. """
    print("Plotting Hamming Distance histogram:")
    all_files = get_files()
    if chip_id:
        all_readouts: list[ReadoutList] = [read_readouts(all_files[chip_id])]
    else:
        all_readouts: list[ReadoutList] = [read_readouts(all_files[chip_id])
                                        for chip_id in all_files.keys()]
    coeff = [0,0]
    for n, coeff[0], coeff[1] in parameters:
        for readouts_val in all_readouts:
            print(f"\tChip {readouts_val.chip_id} Codebook ({n},{coeff})"
                  f"  ---------------------------------")
            hamming_processor = HammingProcessor(n, readouts_val, coeff, True)
            hamming_distances = hamming_processor.compute_hamming_distances(
                    0, const.MAX_ENROLLMENT_READINGS, None)
            # hamming_distances = hamming_processor.compute_hamming_distances(
            #         1, 1, None)
            target_sram_pattern_idx = 1
            # select_th = [-2, 2]
            select_th = get_shifted_selection_threshold(n, coeff)
            Plotting.plot_hd_values_histogram(target_sram_pattern_idx,
                                              hamming_distances [:,:5,:],
                                              readouts_val.chip_id,
                                              select_th)


def plot_2d_bernardini_iterative_vs_readings_overlay_ber(
    parameters,
    primary_dir,
    K=500,
    reference_delta=0.499,
    test_D=0.4991,
    overlay_label="Bernardini",
    trivial=True,
):
    """
    Plot NVM-free TMVS BER vs number of readings with Bernardini overlay comparison.
    
    Args:
        parameters: List of (code_len, coeff[0], coeff[1]) tuples
        primary_dir: Directory containing NVM-free TMVS results
        K: Number of iterations for Bernardini data (default 500)
        reference_delta: Reference threshold (delta) used for base mask (first K reads)
        test_D: Test threshold D used for ranges after K; if None, equals reference_delta
        overlay_label: Label for Bernardini overlay
    """
    print("Plotting 2D evaluation vs number of readings with Bernardini overlay (BER only):")
    if not isinstance(primary_dir, pathlib.Path):
        primary_dir = pathlib.Path(primary_dir)

    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    coeff = [0,0]
    for code_len, coeff[0], coeff[1] in parameters:
        select_th = get_shifted_selection_threshold(code_len, coeff)
        cb = read_codebook(code_len, coeff[0], coeff[1])
        _ = len(cb) if cb is not None else 1
        target_threshold = select_th

        ylabel=r'$\mathrm{BER}_\mathrm{Enr}$'

        # Load Bernardini data via incremental enrollment BER cache per chip

        # Load per-chip ODHD data (averaged over ranges per chip) for scatter plotting
        try:
            processor = AveragingDataProcessor(
                primary_dir,
                GlobalBERProcessor if primary_dir == ber_comparator_dir else HelperDataComparator
            )
            odhd_data = processor.aggregate_data_per_chip(
                code_len, tuple(coeff), target_threshold, nb_enroll_reading, None,
                trivial=trivial
            )
        except (ImportError, FileNotFoundError, KeyError, ValueError) as e:
            print(f"Warning: Could not load per-chip ODHD data: {e}")
            odhd_data = {}

        # Collect ODHD scatter points (per chip, averaged over ranges)
        odhd_x_points = []
        odhd_y_points = []
        
        for _, data in odhd_data.items():
            num_readings = np.asarray(data['num_readings']).flatten()
            ber_mean = np.asarray(data['ber_mean']).flatten()
            odhd_x_points.extend(num_readings.tolist())
            odhd_y_points.extend(ber_mean.tolist())

        # Compute per-chip first/second/last averages for ODHD
        odhd_first_vals, odhd_second_vals, odhd_last_vals = [], [], []
        for _, data in odhd_data.items():
            arr = np.asarray(data.get('ber_mean'))
            if arr is None or arr.size == 0:
                continue
            odhd_first_vals.append(float(arr[0]))
            if arr.size > 1:
                odhd_second_vals.append(float(arr[1]))
            odhd_last_vals.append(float(arr[-1]))

        # Collect Bernardini scatter points (incremental enrollment BER averaged over ranges per chip)
        bernardini_x_points = []
        bernardini_y_points = []
        try:
            # Resolve thresholds
            bern_delta = float(reference_delta)
            bern_D = float(test_D) if test_D is not None else bern_delta
            bern_data = load_incremental_enrollment_ber_per_chip(
                K=int(K), delta=bern_delta, D=bern_D
            )
            for entry in bern_data.values():
                iters = entry.get('iterations', [])
                ber_mean = entry.get('ber_mean', [])
                if not iters or not ber_mean:
                    continue
                bernardini_x_points.extend(list(iters))
                bernardini_y_points.extend(list(ber_mean))

            # Compute per-chip first/second/last for Bernardini
            bern_first_vals, bern_second_vals, bern_last_vals = [], [], []
            for entry in bern_data.values():
                b = np.asarray(entry.get('ber_mean', []), dtype=float)
                if b.size == 0:
                    continue
                bern_first_vals.append(float(b[0]))
                if b.size > 1:
                    bern_second_vals.append(float(b[1]))
                bern_last_vals.append(float(b[-1]))
        except Exception as e:
            print(f"Warning: Failed to load Bernardini incremental data: {e}")

        # Print averages at first, second and last x points (over chips)
        if odhd_first_vals and odhd_last_vals:
            second_str = f" / {np.mean(odhd_second_vals):.6g}" if odhd_second_vals else ""
            print(f"ODHD BER avg at first/second/last: {np.mean(odhd_first_vals):.6g}{second_str} / {np.mean(odhd_last_vals):.6g}")
        if 'bern_first_vals' in locals() and bern_first_vals and bern_last_vals:
            second_b_str = f" / {np.mean(bern_second_vals):.6g}" if bern_second_vals else ""
            print(f"{overlay_label} BER avg at first/second/last: {np.mean(bern_first_vals):.6g}{second_b_str} / {np.mean(bern_last_vals):.6g}")
        # Plot both as scatter points
        Plotting.plot_2d_dual_scatter(
            scatter1_x=odhd_x_points, scatter1_y=odhd_y_points, scatter1_label='ODHD',
            scatter2_x=bernardini_x_points, scatter2_y=bernardini_y_points, scatter2_label=f'{overlay_label}',
            xlabel=r'$N_{\mathrm{res}, \max}$', ylabel=ylabel,
            x_log=True, jump_tick_value=100, scaling_flag=True,
            scatter1_color='blue', scatter2_color='red', scatter_alpha=0.6, scatter_size=20,
        )


def plot_2d_bernardini_iterative_vs_readings_overlay_psr(
    parameters,
    primary_dir,
    K=500,
    reference_delta=0.499,
    test_D=0.4991,
    enroll_select_threshold_override=None,
    overlay_label="Bernardini",
    trivial=True,
):
    """
    Plot NVM-free TMVS PSR vs number of readings with Bernardini overlay comparison.
    Uses per-chip ODHD data (averaged over ranges per chip) and plots scatter across chips,
    mirroring the BER scatter overlay behavior.
    """
    print("Plotting 2D evaluation vs number of readings with Bernardini overlay (PSR only):")
    if not isinstance(primary_dir, pathlib.Path):
        primary_dir = pathlib.Path(primary_dir)

    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    coeff = [0,0]
    for code_len, coeff[0], coeff[1] in parameters:
        select_th = get_shifted_selection_threshold(code_len, coeff)
        cb = read_codebook(code_len, coeff[0], coeff[1])
        _ = len(cb) if cb is not None else 1
        # Use override enrollment selection threshold if provided (expects shifted values)
        if enroll_select_threshold_override is not None:
            # Round to one decimal to match group naming in aggregated HDF5
            target_threshold = [round(float(enroll_select_threshold_override[0]), 1),
                                round(float(enroll_select_threshold_override[1]), 1)]
        else:
            target_threshold = select_th

        ylabel='BSR'

        # Load per-chip ODHD data (PSR per chip) for scatter plotting
        odhd_x_points = []
        odhd_y_points = []
        try:
            processor = AveragingDataProcessor(
                primary_dir,
                GlobalBERProcessor if primary_dir == ber_comparator_dir else HelperDataComparator
            )
            odhd_data = processor.aggregate_data_per_chip(
                code_len, tuple(coeff), tuple(target_threshold), nb_enroll_reading, None,
                trivial=trivial
            )
            for _, data in odhd_data.items():
                num_readings = np.asarray(data['num_readings']).flatten()
                # Prefer PSR if available; fall back to 1 - discarded from aggregated if not
                psr_mean = data.get('psr_mean')
                if psr_mean is not None:
                    psr_mean = np.asarray(psr_mean).flatten()
                    odhd_x_points.extend(num_readings.tolist())
                    odhd_y_points.extend(psr_mean.tolist())

            # Compute per-chip first/last averages for ODHD PSR
            odhd_psr_first_vals, odhd_psr_last_vals = [], []
            for _, data in odhd_data.items():
                p = data.get('psr_mean')
                if p is None:
                    continue
                arr = np.asarray(p)
                if arr.size == 0:
                    continue
                odhd_psr_first_vals.append(float(arr[0]))
                odhd_psr_last_vals.append(float(arr[-1]))
        except (ImportError, FileNotFoundError, KeyError, ValueError) as e:
            print(f"Warning: Could not load per-chip ODHD data: {e}")

        # Load Bernardini data for comparison (accuracy per chip)
        bernardini_x_points = []
        bernardini_y_points = []
        try:
            bern_delta = float(reference_delta)
            bern_D = float(test_D) if test_D is not None else bern_delta
            bernardini_data = load_incremental_enrollment_ber_per_chip(
                K=int(K), delta=bern_delta, D=bern_D
            )
            if bernardini_data:
                # Use iterations from any chip (same length across chips)
                iters = list(bernardini_data.values())[0]['iterations']
                for _, data in bernardini_data.items():
                    bsr_list = data.get('bsr_mean', [])
                    if not bsr_list:
                        continue
                    L = min(len(iters), len(bsr_list))
                    bernardini_x_points.extend(list(iters[:L]))
                    bernardini_y_points.extend(list(bsr_list[:L]))

                # Compute per-chip first/last averages for Bernardini BSR
                bern_bsr_first_vals, bern_bsr_last_vals = [], []
                for _, data in bernardini_data.items():
                    bsr = np.asarray(data.get('bsr_mean', []))
                    if bsr.size == 0:
                        continue
                    bern_bsr_first_vals.append(float(bsr[0]))
                    bern_bsr_last_vals.append(float(bsr[-1]))
        except Exception as e:
            print(f"Warning: Failed to load Bernardini incremental data: {e}")

        # Print averages at first and last x points (over chips)
        if odhd_psr_first_vals and odhd_psr_last_vals:
            print(f"ODHD PSR avg at first/last: {np.mean(odhd_psr_first_vals):.6g} / {np.mean(odhd_psr_last_vals):.6g}")
        if 'bern_bsr_first_vals' in locals() and bern_bsr_first_vals and bern_bsr_last_vals:
            print(f"{overlay_label} BSR avg at first/last: {np.mean(bern_bsr_first_vals):.6g} / {np.mean(bern_bsr_last_vals):.6g}")
        # Plot both as scatter points
        Plotting.plot_2d_dual_scatter(
            scatter1_x=odhd_x_points, scatter1_y=odhd_y_points, scatter1_label='ODHD',
            scatter2_x=bernardini_x_points, scatter2_y=bernardini_y_points, scatter2_label=f'{overlay_label}',
            xlabel=r'$N_{\mathrm{res}, \max}$', ylabel=ylabel,
            x_log=True, jump_tick_value=100, scaling_flag=True,
            scatter1_color='blue', scatter2_color='red', scatter_alpha=0.6, scatter_size=20,
        )


# ---------------------------------------------------------------------------
# Helper: load BER at a specific NrRead from per-chip caches for all thresholds
# ---------------------------------------------------------------------------
def _load_per_threshold_ber(cache_dir, code_len, cb_low, cb_high, nr_read,
                            trivial, class_instance, min_selected=100,
                            skip_integer_th=False):
    """Load BER at a given NrRead for all thresholds from aggregated cache.

    Uses the aggregated .h5 file (single file read) instead of per-chip caches.
    Returns list of dicts: [{'th_high': float, 'ber': float, 'psr': float, 'n_sel': float}, ...]
    If skip_integer_th=True, excludes integer thresholds (avoids d* boundary spikes).
    """
    import h5py
    from common.data_reading_utils import get_num_sram_patterns
    num_patterns = get_num_sram_patterns(code_len)
    nr_idx = nr_read - 1

    # Find aggregated file
    base = (f"aggregated_code_N{code_len}_Threshold_{cb_low}_{cb_high}"
            f"_MaxEnrollReadings_{const.MAX_ENROLLMENT_READINGS}")
    agg_path = None
    suffixes = ["_trivial.h5", ".h5"] if trivial else [".h5", "_trivial.h5"]
    for suffix in suffixes:
        candidate = cache_dir / f"{base}{suffix}"
        if candidate.exists():
            agg_path = candidate
            break

    if agg_path is None:
        print(f"  Warning: No aggregated file found for N{code_len} in {cache_dir.name}")
        return []

    has_psr = (class_instance == HelperDataComparator)
    results = []

    with h5py.File(agg_path, "r") as hf:
        for gname in hf.keys():
            # Parse threshold from group name
            parts = gname.split("_")[1:]
            if len(parts) < 4:
                continue
            th_high = float(parts[2] + '.' + parts[3])
            if skip_integer_th and th_high == int(th_high) and th_high > 0:
                continue

            mean_data = hf[gname]["mean"][:]  # (criteria, NrRead)
            nr_safe = min(nr_idx, mean_data.shape[1] - 1)

            ber = float(mean_data[0, nr_safe])
            psr = float(1 - mean_data[1, nr_safe]) if has_psr and mean_data.shape[0] > 1 else 1.0
            n_sel = psr * num_patterns

            if n_sel >= min_selected:
                results.append({'th_high': th_high, 'ber': ber, 'psr': psr, 'n_sel': n_sel})

    results.sort(key=lambda r: r['th_high'])
    return results


# ---------------------------------------------------------------------------
# Plot 1: BER_Enr vs n (trivial, TH=0, all n)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Plot: BER_Enr vs BER_Reg (per threshold, one point per threshold)
# ---------------------------------------------------------------------------
def plot_ber_enr_vs_ber_reg(parameters, trivial=True, nr_read=500,
                            min_selected=100, key_length=16, skip_integer_th=False):
    """Scatter plot: BER_Enr vs BER_Reg for each threshold, one subplot per (n, cb)."""
    import matplotlib.pyplot as plt

    print(f"Plotting BER_Enr vs BER_Reg (trivial={trivial}, NrRead={nr_read}, "
          f"min_selected={min_selected}):")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'brown']

    for pi, (code_len, cb_low, cb_high) in enumerate(parameters):
        enr_data = _load_per_threshold_ber(
            enroll_comparator_dir, code_len, cb_low, cb_high,
            nr_read, trivial, HelperDataComparator, min_selected, skip_integer_th)
        reg_data = _load_per_threshold_ber(
            ber_comparator_dir, code_len, cb_low, cb_high,
            nr_read, trivial, GlobalBERProcessor, min_selected, skip_integer_th)

        # Match by threshold
        reg_map = {r['th_high']: r['ber'] for r in reg_data}
        xs, ys, labels = [], [], []
        for e in enr_data:
            if e['th_high'] in reg_map:
                xs.append(e['ber'])
                ys.append(reg_map[e['th_high']])
                labels.append(e['th_high'])

        color = colors[pi % len(colors)]
        lbl = f'n={code_len}' if len(parameters) > 1 else 'ODHD'
        ax.scatter(xs, ys, color=color, s=30, label=lbl, zorder=3)

        # Annotate each point with threshold
        for x, y, th in zip(xs, ys, labels):
            ax.annotate(f'{th:.1f}', (x, y), fontsize=6, alpha=0.7,
                        textcoords='offset points', xytext=(4, 4))

        print(f"\tn={code_len}: {len(xs)} thresholds plotted")
        for x, y, th in zip(xs, ys, labels):
            ker_enr = 1 - (1 - x) ** key_length
            ker_reg = 1 - (1 - y) ** key_length
            print(f"\t  TH*={th:.1f}  BER_Enr={x:.6f}  BER_Reg={y:.6f}  "
                  f"KER_Enr={ker_enr:.6e}  KER_Reg={ker_reg:.6e}")

    ax.set_xlabel(r'$\mathbf{BER}_\mathbf{Enr}$', fontsize=14, weight='bold')
    ax.set_ylabel(r'$\mathbf{BER}_\mathbf{Reg}$', fontsize=14, weight='bold', rotation=0, labelpad=30)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    # plt.title(f'BER_Enr vs BER_Reg ($N_{{\\mathrm{{res}}, \\max}}$={nr_read})')
    plt.tight_layout()
    fname = './nvm_free_tmvs/figures/ber_enr_vs_ber_reg.pdf'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\nSaved to {fname}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3: KER_Enr (left) and KER_Reg (right) vs SRAM size (per threshold)
# ---------------------------------------------------------------------------
def plot_ker_vs_sram_size(parameters, trivial=True, nr_read=500,
                          min_selected=100, key_length=16, skip_integer_th=False):
    """Dual y-axis: KER_Enr (left) and KER_Reg (right) vs SRAM size per threshold."""
    import matplotlib.pyplot as plt

    print(f"Plotting KER vs SRAM size (trivial={trivial}, NrRead={nr_read}, K={key_length}, "
          f"min_selected={min_selected}, skip_integer_th={skip_integer_th}):")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for code_len, cb_low, cb_high in parameters:
        enr_data = _load_per_threshold_ber(
            enroll_comparator_dir, code_len, cb_low, cb_high,
            nr_read, trivial, HelperDataComparator, min_selected, skip_integer_th)
        reg_data = _load_per_threshold_ber(
            ber_comparator_dir, code_len, cb_low, cb_high,
            nr_read, trivial, GlobalBERProcessor, min_selected)

        reg_map = {r['th_high']: r['ber'] for r in reg_data}
        enr_map = {e['th_high']: e for e in enr_data}

        sram_sizes, ker_enrs, ker_regs, th_labels = [], [], [], []
        for th_high, e in sorted(enr_map.items()):
            if th_high not in reg_map:
                continue
            ber_enr = e['ber']
            ber_reg = reg_map[th_high]
            psr = e['psr']
            # SRAM size = KEY_LENGTH * n / psr (non-overlapping windows)
            if psr > 0:
                sram_bits = key_length * code_len / psr
            else:
                continue
            ker_enr = 1 - (1 - ber_enr) ** key_length
            ker_reg = 1 - (1 - ber_reg) ** key_length

            sram_sizes.append(sram_bits)
            ker_enrs.append(ker_enr)
            ker_regs.append(ker_reg)
            th_labels.append(th_high)

        ax1.scatter(sram_sizes, ker_enrs, color='blue', s=30, zorder=3)
        ax2.scatter(sram_sizes, ker_regs, color='green', s=30, zorder=3)

        # Annotate with threshold
        for x, y_enr, y_reg, th in zip(sram_sizes, ker_enrs, ker_regs, th_labels):
            ax1.annotate(f'{th:.1f}', (x, y_enr), fontsize=8, alpha=0.7,
                         textcoords='offset points', xytext=(-4, -10))

        print(f"\tn={code_len}: {len(sram_sizes)} thresholds")
        print(f"\t{'TH*':>6s}  {'SRAM(bits)':>10s}  {'KER_Enr':>12s}  {'KER_Reg':>12s}  {'PSR':>8s}")
        for x, ke, kr, th in zip(sram_sizes, ker_enrs, ker_regs, th_labels):
            psr = enr_map[th]['psr']
            print(f"\t{th:6.1f}  {x:10.0f}  {ke:12.6e}  {kr:12.6e}  {psr:8.4f}")

    ax1.set_xlabel('SRAM size (bits)', fontsize=14, weight='bold')
    ax1.set_ylabel(r'$\mathbf{KER}_\mathbf{Enr}$', fontsize=14, rotation=0, weight='bold',
                   color='blue', labelpad=30)
    ax2.set_ylabel(r'$\mathbf{KER}_\mathbf{Reg}$', fontsize=14, rotation=0, weight='bold',
                   color='green', labelpad=30)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='green')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    ax1.grid(True, alpha=0.3)
    # plt.title(f'KER vs SRAM size (K={key_length}, $N_{{\\mathrm{{res}}, \\max}}$={nr_read})')
    plt.tight_layout()
    fname = './nvm_free_tmvs/figures/ker_vs_sram_size.pdf'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\nSaved to {fname}")
    plt.show()
