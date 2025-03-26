import numpy as np
from nvm_free_tmvs.utils.file_manager import ber_comparator_dir
from nvm_free_tmvs.utils.file_manager import enroll_comparator_dir
from nvm_free_tmvs.experiments.aggregated_data_reader import AggregatedDataReader
from nvm_free_tmvs.utils.file_manager import read_codebook
from nvm_free_tmvs.experiments.plotting import Plotting
import nvm_free_tmvs.analysis_constants as const
from tmvs.formulas import theoretical_required_sram_size, theoretical_selection_probability, key_failure_probability
from tmvs.analysis_constants import KEY_LENGTH


def find_optimal_parameters(parameters_list, num_enroll_reading,
                            target_failure_rate=1e-6):
    """
    Find the optimal parameters that satisfy the target failure rate.
    Note: BER=0 sometimes refers to invalid sram patterns. I ignored this case
    because dicarding rate will be 1 and avoided by the algorithm.
    """
    best_parameters = None
    min_ber = None
    min_failure_rate = None
    best_regenerating_error_rate = None
    best_enroll_error_rate = None
    min_discarding_rate = float('inf')
    min_required_sram_bits = float('inf')
    
    for code_len, coeff_0, coeff_1 in parameters_list:
        coeff = [coeff_0, coeff_1]
        # print("Processing: code_length=", code_len, ", threshold=", coeff)
        
        # Initialize the reader
        enroll_reader = AggregatedDataReader(code_len, coeff, num_enroll_reading, enroll_comparator_dir)
        ber_reader = AggregatedDataReader(code_len, coeff, num_enroll_reading, ber_comparator_dir)

        enroll_num_enroll_readings, enroll_threshold_values, enroll_result_values = enroll_reader.read_aggregated_data()
        ber_num_enroll_readings, ber_threshold_values, ber_result_values = ber_reader.read_aggregated_data()
        
        # Ensure matching x and y axis values
        assert (enroll_threshold_values.size == ber_threshold_values.size), \
            "Mismatch between threshold_values size"
        assert (enroll_num_enroll_readings.size == ber_num_enroll_readings.size), \
            "Mismatch between num_enroll_readings size"
        # if not np.any(mask):
        #     continue
        
        # Apply constraint and find minimum value
        ber_result_values_mean = np.array(ber_result_values["mean"])
        enroll_result_values_mean = np.array(enroll_result_values["mean"])
        
        enroll_error_rate = enroll_result_values_mean[:,0,:]
        regenerating_error_rate = ber_result_values_mean[:,0,:]
        # shape of total_error_rate: (num_thresholds, num_enroll_readings)
        total_error_rate = regenerating_error_rate + enroll_error_rate \
            - regenerating_error_rate * enroll_error_rate
        total_failure_rate = np.array([[key_failure_probability(value) for value in row]
                                       for row in total_error_rate], dtype=np.float64)
        # apply constraint
        constraint = total_failure_rate < target_failure_rate
        valid_indices = np.where(constraint) # pairs of indices (num_thresholds, num_enroll_readings)
        if valid_indices[0].size > 0:
            # indices of the smallest value of discarding rate
            valid_discarding_rate = enroll_result_values_mean[valid_indices[0], 1, valid_indices[1]]
            smallest_discarding_rate_idx = np.argmin(valid_discarding_rate)
            chosen_threshold_idx = valid_indices[0][smallest_discarding_rate_idx]
            chosen_enroll_reading_idx = valid_indices[1][smallest_discarding_rate_idx]
            chosen_discarding_rate = enroll_result_values_mean[chosen_threshold_idx,
                                                               1,
                                                               chosen_enroll_reading_idx]
    
            chosen_selection_rate = 1 - chosen_discarding_rate
            if chosen_selection_rate == 0:
                # print("Warning: discarding rate is 1, skipping this case")
                continue
            chosen_required_sram_bits = KEY_LENGTH * (code_len + 1/chosen_selection_rate - 1)
            # print("\nformula results:", formula / (8*1024), "kB")
            
            # if chosen_discarding_rate < min_discarding_rate:
            if chosen_required_sram_bits < min_required_sram_bits:
                min_discarding_rate = chosen_discarding_rate
                best_threshold_value = enroll_threshold_values[chosen_threshold_idx]
                best_num_enroll_readings = enroll_num_enroll_readings[chosen_enroll_reading_idx]
                best_parameters = (code_len, coeff,
                                   best_num_enroll_readings,
                                   best_threshold_value)
                min_ber = total_error_rate[
                    chosen_threshold_idx, chosen_enroll_reading_idx]
                min_failure_rate = total_failure_rate[
                    chosen_threshold_idx, chosen_enroll_reading_idx]
                best_regenerating_error_rate = regenerating_error_rate[
                    chosen_threshold_idx, chosen_enroll_reading_idx]
                best_enroll_error_rate = enroll_error_rate[
                    chosen_threshold_idx, chosen_enroll_reading_idx]
                min_required_sram_bits = chosen_required_sram_bits
    return (best_parameters, min_ber, min_failure_rate,
            min_discarding_rate, min_required_sram_bits,
            best_regenerating_error_rate,
            best_enroll_error_rate)

def calculate_failure_vs_memory_tradeoff(parameters_list, num_enroll_reading):
    """
    Calculate the failure rate vs memory tradeoff given parameters.
    """
    failure_rate_list = const.TEST_FAILURE_RATE_TARGET
    failure_rates = []
    ber_results = []
    required_memory_size = []
    discarding_rates = []
    selected_parameters = []

    for target_failure_rate in failure_rate_list:
        (best_parameters, min_ber, min_failure_rate,
         min_discarding_rate, min_required_sram_bits,
        _, _) = find_optimal_parameters(parameters_list,
                                        num_enroll_reading,
                                        target_failure_rate)
        selected_parameters.append(best_parameters)
        failure_rates.append(min_failure_rate)
        ber_results.append(min_ber)
        discarding_rates.append(min_discarding_rate)
        required_memory_size.append(min_required_sram_bits / (8*1024))
    return (np.array(failure_rates), ber_results, np.array(required_memory_size), 
            discarding_rates, selected_parameters)


# Example usage:
if __name__ == "__main__":
    # Define the enroll_select_threshold value we want
    nb_enroll_reading = const.MAX_ENROLLMENT_READINGS
    all_parameters = [(7,1,6), (9,1,8), (11, 1, 10), (11, 2, 9), (13, 1, 12), (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]

    # testing plot function
    print("Testing plot function")
    print(const.TEST_FAILURE_RATE_TARGET)
    (failure_rates_axis, ber_results, required_memory_size_axis, discarding_rates,
     resulting_parameters) = calculate_failure_vs_memory_tradeoff(all_parameters, nb_enroll_reading)
    print("failure_rates:", failure_rates_axis)
    print("required_memory_size:", required_memory_size_axis)
    print("selected_parameters:", resulting_parameters)

    Plotting.plot_2d_plot_with_horizontal_line(x=required_memory_size_axis,
                                               y=failure_rates_axis,
                                               xlabel='SRAM Memory Size (kB)',
                                                ylabel='Failure Rate',
                                                title='Failure Rate vs Memory Size',
                                                horizontal_line=1e-6)

    # parameters = [(27,3,24)]
    # n_k = 128
    # target_failure = 3e-5
    
    # (best_result_tuple, ber, failure_rate_value,
    #  discarding_rate, required_sram_bits,
    #  min_regenerating_error_rate,
    #  min_enroll_error_rate) = find_optimal_parameters(
    #     all_parameters, nb_enroll_reading, target_failure)
    # print("\noptimal parameters:", best_result_tuple)
    # print("expected number of sram bits (kB) =", required_sram_bits/ (8*1024))
    # print("best discarding rate value:", discarding_rate)
    # print("best helper data ber:", ber)
    # print("best helper data failure:", failure_rate_value)
    # print("min_regenerating_error_rate:", min_regenerating_error_rate)
    # print("min_enroll_error_rate:", min_enroll_error_rate)
    
    # code_length = best_result_tuple[0]
    # coefficient = best_result_tuple[1]
    # new_target_th = np.array(best_result_tuple[-1])
    
    # codebook_length = len(read_codebook(code_length,
    #                             coefficient[0],
    #                             coefficient[1]))

    # # fix new_target_th to un-shift
    # new_target_th = [2,25]
    # #### DIFFERENT THEORETICAL P_SELECT DUE TO AVERAGING OVER READINGS !!! ######
    # unprecise_p_select = theoretical_selection_probability(code_length, new_target_th, codebook_length)
    # print("\nwrong expected discarding probability =", 1 - unprecise_p_select)
    # temp_result = theoretical_required_sram_size(code_length, new_target_th, codebook_length)
    # print("\nwrong expected number of sram bits (kB) =", temp_result)

    # selection_rate = 1 - discarding_rate
    # formula = n_k * (code_length + 1/selection_rate - 1)
    # print("\nformula results:", formula / (8*1024), "kB")

