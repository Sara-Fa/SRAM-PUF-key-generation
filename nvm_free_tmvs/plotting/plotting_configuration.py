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

def plot_3d_evaluation_vs_threshold_and_num_readings(parameters, dir_name):
    """  Plot 3D evaluation vs threshold and number of readings. """
    print("Plotting 3D evaluation vs threshold and number of readings:")
    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    coeff = [0,0]
    for code_len, coeff[0], coeff[1] in parameters:
        select_th = get_shifted_selection_threshold(code_len, coeff)
        print("\tcode_length=", code_len, ", threshold=",coeff,
              "shifted_th=", select_th)
        codebook_length = len(read_codebook(code_len,
                                 coeff[0],
                                 coeff[1]))
        print("\tcodebook_length=", codebook_length)
        target_threshold = select_th

        # Initialize the reader
        reader = AggregatedDataReader(code_len, coeff, nb_enroll_reading, dir_name)
        x_axis_val, y_axis_val, z_axis_val = reader.read_aggregated_data()

        # Find the index where y_axis (enroll_select_threshold) matches (-3.0, 3.0)
        threshold_index = np.where((y_axis_val[:, 0] == target_threshold[0]) &
                            (y_axis_val[:, 1] == target_threshold[1]))[0]
        y_axis_val = y_axis_val[:,1] # take only higher threshold

        z_axis_mean = np.array(z_axis_val["mean"])
        # z_axis_min = np.array(z_axis_val["min"])
        # z_axis_max = np.array(z_axis_val["max"])

        if dir_name == ber_comparator_dir:
            print(f"Mean BER at target_threshold={target_threshold}: {z_axis_mean[threshold_index,0,:]}")
            reader.plot_3d_results(x_axis_val, y_axis_val, z_axis_mean[:,0,:],
                                   r'Nb. of Readings $N_{\mathrm{res}}$', r'Selection Threshold $\mathrm{TH}^*_{\mathrm{High}}$', 
                                   r'$\mathrm{BER}_\mathrm{Reg}$',
                                   None)
        else:
            print("\terror_count z_axis[\"mean\"]:", z_axis_mean[threshold_index,0,:])
            print("\tdiscarded_patterns_count z_axis[\"mean\"]:", z_axis_mean[threshold_index,1,:])
            # # print("\terror_count z_axis[\"min\"]:", z_axis_min[threshold_index,0,:])
            # # print("\terror_count z_axis[\"max\"]:", z_axis_max[threshold_index,0,:])
            # print("\tMean Extraction rate at select_th:", z_axis_mean[threshold_index,4,:])
            reader.plot_3d_results(x_axis_val, y_axis_val, z_axis_mean[:,0,:],
                                      r'Nb. of Readings $N_{\mathrm{res}}$', r'Selection Threshold $\mathrm{TH}^*_{\mathrm{High}}$', 
                                      r'$\mathrm{BER}_\mathrm{Enr}$',
                                   1 - z_axis_mean[:,1,:], select_th)

def plot_2d_evaluation_vs_num_readings(parameters, dir_name):
    """  Plot 2D evaluation vs number of readings. """
    print("Plotting 2D evaluation vs number of readings:")
    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    coeff = [0,0]
    for code_len, coeff[0], coeff[1] in parameters:
        select_th = get_shifted_selection_threshold(code_len, coeff)
        print("\tcode_length=", code_len, ", threshold=",coeff,
              "shifted_th=", select_th)
        codebook_length = len(read_codebook(code_len,
                                 coeff[0],
                                 coeff[1]))
        print("\tcodebook_length=", codebook_length)
        target_threshold = select_th

        # Initialize the reader
        reader = AggregatedDataReader(code_len, coeff, nb_enroll_reading, dir_name)
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
                                    xlabel='Nb. of Readouts',
                                    ylabel=ylabel,
                                    title='BER vs Thresholds for Different Number of Readings',
                                    legend_label=r'Selection Threshold $\mathrm{TH}^*_{\mathrm{enr}}$',
                                    second_y=second_y, second_ylabel=second_ylabel)

def plot_2d_evaluation_vs_threshold(parameters, target_num_readings, dir_name):
    """  Plot 2D evaluation vs threshold. """
    print("Plotting 2D evaluation vs threshold:")
    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    coeff = [0,0]
    for code_len, coeff[0], coeff[1] in parameters:
        select_th = get_shifted_selection_threshold(code_len, coeff)
        print("\n\tcode_length=", code_len, ", threshold=",coeff,
              "shifted_th=", select_th)
        codebook_length = len(read_codebook(code_len,
                                 coeff[0],
                                 coeff[1]))
        print("\tcodebook_length=", codebook_length)
        target_threshold = [-1.1,1.1] # select_th

        # Initialize the reader
        reader = AggregatedDataReader(code_len, coeff, nb_enroll_reading, dir_name)
        _, thresholds_list, results_list = reader.read_aggregated_data()

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
            # print("\tdiscarded_patterns_count z_axis[\"mean\"]:", z_axis_mean[threshold_index,1,:])
            # # print("\terror_count z_axis[\"min\"]:", z_axis_min[threshold_index,0,:])
            # # print("\terror_count z_axis[\"max\"]:", z_axis_max[threshold_index,0,:])
            # print("\tMax Extraction rate at select_th:", z_axis_max[threshold_index,4,:])

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
        _ = len(read_codebook(code_len, coeff[0], coeff[1]))
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
            xlabel='Nb. of Readouts', ylabel=ylabel,
            second_y_a_mean=psr_mean, second_y_a_min=psr_min, second_y_a_max=psr_max,
            second_y_b_mean=dark_disc_mean, second_y_b_min=dark_disc_min, second_y_b_max=dark_disc_max,
            second_ylabel=second_ylabel,
        )

def plot_2d_evaluation_vs_num_readings_overlay_ber(parameters, primary_dir, overlay_label="Dark bits"):
    """Single-axis overlay plot for BER only (our vs dark bits)."""
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
        z_axis_mean = np.array(results_list["mean"])  # (num_thresholds, criteria, num_readings)
        z_axis_min = np.array(results_list["min"])    # (num_thresholds, criteria, num_readings)
        z_axis_max = np.array(results_list["max"])    # (num_thresholds, criteria, num_readings)

        ylabel=r'$\mathrm{BER}_\mathrm{Enr}$'

        x_dark, dark_results = GenerateBitsMask.compute_aggregate_metrics_over_chips()
        # Align to full Dark Bits length; pad our series with NaN beyond available range
        L = int(len(x_dark))
        x_main = x_dark[:L]

        col = int(threshold_index[0]) if threshold_index.size > 0 else 0
        # Our available length
        avail = min(z_axis_mean.shape[2], len(num_readings_list), L)
        ber_mean_av = z_axis_mean[col, 0, :][:avail]
        ber_min_av = z_axis_min[col, 0, :][:avail]
        ber_max_av = z_axis_max[col, 0, :][:avail]
        # Pad to L with NaN to align with x_dark
        pad_len = L - avail
        if pad_len > 0:
            ber_mean = np.concatenate([ber_mean_av, np.full(pad_len, np.nan)])
            ber_min = np.concatenate([ber_min_av, np.full(pad_len, np.nan)])
            ber_max = np.concatenate([ber_max_av, np.full(pad_len, np.nan)])
        else:
            ber_mean, ber_min, ber_max = ber_mean_av, ber_min_av, ber_max_av

        dark_ber = dark_results['ber']['mean'][:L]
        dark_min = dark_results['ber']['min'][:L]
        dark_max = dark_results['ber']['max'][:L]

        Plotting.plot_2d_overlay_with_bands(
            x=x_main,
            a_mean=ber_mean, a_min=ber_min, a_max=ber_max, a_label='ODHD',
            b_mean=dark_ber, b_min=dark_min, b_max=dark_max, b_label=overlay_label,
            xlabel='Nb. of Readouts', ylabel=ylabel,
            second_y_a_mean=None, second_y_a_min=None, second_y_a_max=None,
            second_y_b_mean=None, second_y_b_min=None, second_y_b_max=None,
            x_log=True, jump_tick_value=100, scaling_flag=True,
        )

def plot_2d_evaluation_vs_num_readings_overlay_psr(parameters, primary_dir, overlay_label="Dark bits"):
    """Single-axis overlay plot for Selection/PSR only (our vs dark bits discarded)."""
    print("Plotting 2D evaluation vs number of readings with overlay (PSR only):")
    if not isinstance(primary_dir, pathlib.Path):
        primary_dir = pathlib.Path(primary_dir)

    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    coeff = [0,0]
    for code_len, coeff[0], coeff[1] in parameters:
        select_th = get_shifted_selection_threshold(code_len, coeff)
        _ = len(read_codebook(code_len, coeff[0], coeff[1]))
        target_threshold = select_th

        reader = AggregatedDataReader(code_len, coeff, nb_enroll_reading, primary_dir)
        num_readings_list, thresholds_list, results_list = reader.read_aggregated_data()
        threshold_index = np.where((thresholds_list[:, 0] == target_threshold[0]) &
                            (thresholds_list[:, 1] == target_threshold[1]))[0]
        z_axis_mean = np.array(results_list["mean"])  # (num_thresholds, criteria, num_readings)
        z_axis_min = np.array(results_list["min"])    # (num_thresholds, criteria, num_readings)
        z_axis_max = np.array(results_list["max"])    # (num_thresholds, criteria, num_readings)

        ylabel='Bit Selection Rate'

        x_dark, dark_results = GenerateBitsMask.compute_aggregate_metrics_over_chips(nb_enroll_readings=nb_enroll_reading)
        L = int(min(len(num_readings_list), len(x_dark)))
        x_main = num_readings_list[:L]

        col = int(threshold_index[0]) if threshold_index.size > 0 else 0
        psr_mean = (1 - z_axis_mean[col, 1, :])[:L]
        psr_min = (1 - z_axis_max[col, 1, :])[:L]
        psr_max = (1 - z_axis_min[col, 1, :])[:L]

        dark_disc_mean = 1 - dark_results['discarded']['mean'][:L]
        dark_disc_min = 1 - dark_results['discarded']['min'][:L]
        dark_disc_max = 1 - dark_results['discarded']['max'][:L]

        Plotting.plot_2d_overlay_with_bands(
            x=x_main,
            a_mean=psr_mean, a_min=psr_min, a_max=psr_max, a_label='ODHD',
            b_mean=dark_disc_mean, b_min=dark_disc_min, b_max=dark_disc_max, b_label=overlay_label,
            xlabel='Nb. of Readouts', ylabel=ylabel,
            second_y_a_mean=None, second_y_a_min=None, second_y_a_max=None,
            second_y_b_mean=None, second_y_b_min=None, second_y_b_max=None,
            ylabel_rotation=90, jump_tick_value=10,
        )

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
        _ = len(read_codebook(code_len, coeff[0], coeff[1]))
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
                code_len, tuple(coeff), target_threshold, nb_enroll_reading, None
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
            xlabel=r'Number of Readouts ($N_{\mathrm{DB}}$)', ylabel=ylabel,
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
        _ = len(read_codebook(code_len, coeff[0], coeff[1]))
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
                code_len, tuple(coeff), tuple(target_threshold), nb_enroll_reading, None
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
            xlabel=r'Number of Readouts ($N_{\mathrm{DB}}$)', ylabel=ylabel,
            x_log=True, jump_tick_value=100, scaling_flag=True,
            scatter1_color='blue', scatter2_color='red', scatter_alpha=0.6, scatter_size=20,
        )
