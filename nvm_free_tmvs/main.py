""" Main file for running experiments and processing readouts. """
# import time
import numpy as np
from nvm_free_tmvs.experiments.aggregated_data_reader import AggregatedDataReader
from nvm_free_tmvs.core.hamming_processor import HammingProcessor
from nvm_free_tmvs.experiments.plotting import Plotting
from nvm_free_tmvs.experiments.optimal_parameters import calculate_failure_vs_memory_tradeoff
from nvm_free_tmvs.utils.analysis_utils import get_shifted_selection_threshold
from nvm_free_tmvs.utils.file_manager  import ReadoutList, get_files, read_readouts
from nvm_free_tmvs.utils.file_manager import ber_comparator_dir
from nvm_free_tmvs.utils.file_manager import enroll_comparator_dir
from nvm_free_tmvs.utils.file_manager import read_codebook
import nvm_free_tmvs.analysis_constants as const



def calculate_and_plot_failure_rate_vs_memory(parameters):
    """ Calculate and plot failure rate vs memory tradeoff. """
    print("Calculating failure rate vs memory tradeoff:")
    print("\tfailure rate list:", const.TEST_FAILURE_RATE_TARGET)
    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    (failure_rates_axis, ber_results, required_memory_size_axis, discarding_rates,
    resulting_parameters) = calculate_failure_vs_memory_tradeoff(parameters, nb_enroll_reading)
    print("\tfailure_rates:", failure_rates_axis)
    print("\tber_results:", ber_results)
    print("\trequired_memory_size:", required_memory_size_axis)
    print("\tdiscarding_rates:", discarding_rates)
    print("\tselected_parameters:", resulting_parameters)

    Plotting.plot_2d_plot_with_horizontal_line(x=required_memory_size_axis,
					    y=failure_rates_axis,
					    xlabel='SRAM Memory Size (kB)',
						    ylabel='Failure Rate',
						    title='Failure Rate vs Memory Size',
						    horizontal_line=1e-6)

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
            print("Mean BER at select_th:", z_axis_mean[threshold_index,0,:])
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
            ylabel=r'$\mathrm{BER}_\mathrm{High}$'
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
                                    legend_label='Nb. of Readings',
                                    second_y=second_y, second_ylabel=second_ylabel)

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

def main():
    """ Main function for running experiments and processing readouts. """

    # calculate and plot failure rate vs memory trade
    # # all_parameters =  [(17, 1, 16)]
    # all_parameters = [(7,1,6), (9,1,8), (11, 1, 10), (11, 2, 9), (13, 1, 12), (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]
    # calculate_and_plot_failure_rate_vs_memory(all_parameters)

    # plot 3D enrollment/regeneration evaluation vs threshold and number of readings
    # all_parameters = [(27, 3, 24)]
    # # dir_name = enroll_comparator_dir
    # dir_name = ber_comparator_dir
    # plot_3d_evaluation_vs_threshold_and_num_readings(all_parameters, dir_name)

    # plot 2D enrollment/regeneration evaluation vs threshold and number of readings
    all_parameters = [(27, 3, 24)] #   #
    dir_name = enroll_comparator_dir # discarding rate is also ploted
    # dir_name = ber_comparator_dir
    target_num_readings = [1, 10]
    plot_2d_evaluation_vs_threshold(all_parameters, target_num_readings, dir_name)
    # plot_2d_evaluation_vs_num_readings(all_parameters, dir_name)

    # plot Hamming Distance histogram
    # parameters = [(7,1,6)] # # add the case 27 to the next list
    # chip_id='M2'
    # initialize_and_plot_hd_values_histogram(parameters, chip_id)
    


if __name__ == "__main__":
    main()
