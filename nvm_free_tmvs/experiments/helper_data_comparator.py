""" Comparator between the helper data across multiple enrollments
using the same chip and codebook. """
import time
import os
from typing import List
from multiprocessing import Pool, cpu_count
import numpy as np
from nvm_free_tmvs.plotting.plotting_functions import Plotting
from nvm_free_tmvs.core.hamming_processor import HammingProcessor
from nvm_free_tmvs.core.chunk_data_processor import ChunkDataProcessor
from nvm_free_tmvs.utils.file_manager import ReadoutList, read_codebook
from nvm_free_tmvs.utils.file_manager import get_files, read_readouts
from nvm_free_tmvs.utils.analysis_utils import get_enrollment_ranges, get_enrollment_threshold_values
from nvm_free_tmvs.algorithm.enroll import Enroll
from nvm_free_tmvs.experiments.comparator_cache_manager import ComparatorCacheManager
import common.data_constants as data_const
from common.data_reading_utils import get_num_sram_patterns
import nvm_free_tmvs.analysis_constants as const

# nvm_free_tmvs
# helperless_tmvs

class HelperDataComparator:
    """ Comparator between the helper data across multiple enrollments. """
    def __init__(self, code_length: int, readouts: ReadoutList, select_threshold: List[float],
                 active_multithreading: bool): # replace select_threshold by margin_ceoff
        """ Initialize the HelperDataComparator. """
        self.code_length = code_length
        self.readouts = readouts
        self.chip_id = readouts.chip_id
        self.select_threshold = select_threshold
        # self.margin_coeff = margin_coeff
        # self.select_threshold, _ = calculate_threshold (code_length, margin_coeff)
        self.active_multithreading = active_multithreading
        self.cache_manager = ComparatorCacheManager()

    @staticmethod
    def _plot_results(range_enroll_readings, select_margin, ber_results, selection_rate_results):
        """ Plot results. """
        # 2D line graphs for BER and validity rates
        Plotting.plot_2d_line_graphs(x=range_enroll_readings,
                                     y=ber_results, z=select_margin,
                                     xlabel='Number of Readings', ylabel='BER',
                                     title='BER vs Number of Readings for Different Thresholds',
                                     legend_label='Threshold')
        Plotting.plot_2d_line_graphs(x=range_enroll_readings,
                                     y=selection_rate_results, z=select_margin,
                                     xlabel='Number of Readings', ylabel='BER',
                                     title='Selection Rate vs Number of Readings for Different Thresholds',
                                     legend_label='Threshold')

        # 3D surface plots for BER and validity rates
        Plotting.plot_3d_surface(x=range_enroll_readings,
                                 y=select_margin, z=ber_results,
                                 xlabel='Number of Readings', ylabel='Threshold Values',
                                 zlabel='BER', title='BER vs Thresholds and Number of Readings',
                                 log_scale=True)
        Plotting.plot_3d_surface(x=range_enroll_readings,
                                 y=select_margin, z=selection_rate_results,
                                 xlabel='Number of Readings', ylabel='Threshold Values',
                                 zlabel='Selection Rate',
                                 title='Selection Rate vs Thresholds and Number of Readings')

    @staticmethod
    def calculate_error_count(enrollment_data_comparator):
        """ Calculate the error count. """
        # sum over num_sram_patterns, num_codewords, resulting shape (num_readings)
        # dtype=uint32
        return np.sum(enrollment_data_comparator, axis=1)

    @staticmethod
    def calculate_discarding_count(enrollment_data):
        """ Calculate the discarding count. """
        # shape (num_readings)
        # dtype=int32
        return np.sum(np.all(enrollment_data == -1, axis=2), axis=1)

    @staticmethod
    def calculate_key_bits_count(enrollment_data):
        """ Calculate the key bits count. """
        # Count elements equal to 0 and 1 for every row (along axis 0)
        # shape (num_readings)
        # dtype=int32
        zero_count = np.sum(enrollment_data == 0, axis=(1, 2))
        one_count = np.sum(enrollment_data == 1, axis=(1, 2))
        return zero_count, one_count

    def process_enroll_range(self, args):
        """
        Worker function for processing a single threshold and test_enroll_ranges.
        """
        (range_idx, start_enroll_idx, hamming_processor, num_enroll_readings,
         incremental_computation, enroll_select_threshold,
         hamming_distances, enrollment_data_reference
         ) = args

        # Create an instance of Enroll and execute for the range
        enroll_instance = Enroll(
            hamming_processor, start_enroll_idx, num_enroll_readings, incremental_computation
        )
        # enrollement_data of shape (num_readings, num_sram_patterns, num_codewords)
        enrollment_data = enroll_instance.execute(enroll_select_threshold, hamming_distances)

        # Compute comparison, sum over codewords
        enrollment_data_comparator = np.sum(enrollment_data != enrollment_data_reference,
                                            axis=-1, dtype=np.uint32)
        sum_uint64 = np.sum(enrollment_data != enrollment_data_reference, axis=-1, dtype=np.uint64)

        overflow_detected = enrollment_data_comparator != sum_uint64
        if np.any(overflow_detected):
            print("Overflow detected!")

        # Compute results
        error_count = self.calculate_error_count(enrollment_data_comparator)
        discarded_patterns_count = self.calculate_discarding_count(enrollment_data)
        zero_count, one_count = self.calculate_key_bits_count(enrollment_data)

        return range_idx, error_count, discarded_patterns_count, zero_count, one_count


    def compare_and_save_helper_data(self):
        """ Compare helper data across multiple enrollments. """
        enroll_ranges = get_enrollment_ranges()

        chunk_data_processor = ChunkDataProcessor(self.code_length,
                                                    self.readouts,
                                                    self.active_multithreading)
        chunked_data = chunk_data_processor.chunk_readouts()

        hamming_processor = HammingProcessor(self.code_length, self.readouts,
                                            self.select_threshold,
                                            self.active_multithreading)

        hamming_distances = None
        num_enroll_readings = const.MAX_ENROLLMENT_READINGS
        incremental_computation = True
        _, threshold_values_list = get_enrollment_threshold_values(self.code_length)
        # threshold_values_list = threshold_values_list[-1:]
        # threshold_values_list = threshold_values_list[-22:]
        
        print("Threshold values list: ", threshold_values_list)
        test_enroll_ranges = enroll_ranges[1:] # skip the first range
        hamming_distance_tag = 0

        # Initialize result arrays
        error_count = np.zeros(
            (len(threshold_values_list), len(enroll_ranges), num_enroll_readings))
        discarded_patterns_count = np.zeros(
            (len(threshold_values_list), len(enroll_ranges), num_enroll_readings))
        zero_key_bits_count = np.zeros(
            (len(threshold_values_list), len(enroll_ranges), num_enroll_readings))
        one_key_bits_count = np.zeros(
            (len(threshold_values_list), len(enroll_ranges), num_enroll_readings))

        for i, enroll_select_threshold in enumerate(threshold_values_list):
            # Check if this threshold exists in the cache
            if self.cache_manager.check_threshold_in_cache(self.chip_id, self.select_threshold,
                                                           self.code_length, const.MAX_ENROLLMENT_READINGS,
                                                           enroll_select_threshold):
                print(f"Skipping computation for threshold {enroll_select_threshold}, already in cache.")
                continue

            # compute hamming distances
            if not hamming_distance_tag:
                hamming_distances = hamming_processor.compute_hamming_distances(
                    0, data_const.READINGS_TO_ANALYZE, chunked_data)
                hamming_distance_tag = 1

            print(f"\nEnrollment select threshold {i}: {enroll_select_threshold}")
            start_time = time.time()

            # Compute reference data for enrollment
            enroll_instance = Enroll(hamming_processor, 0, num_enroll_readings, incremental_computation)
            sliced_hamming_distances = hamming_distances[:, :, 0: num_enroll_readings]
            enrollment_data_reference = enroll_instance.execute(enroll_select_threshold,
                                                                sliced_hamming_distances)
            # num_readings, num_sram_patterns, codebook_length = enrollment_data_reference.shape
            # print("\nkey and helper data for test pattern:",
            #     enrollment_data_reference[-1][test_sram_pattern_idx])
            # Create a NumPy array
            zero_count, one_count = self.calculate_key_bits_count(enrollment_data_reference)
            zero_key_bits_count[i, 0, :] = zero_count
            one_key_bits_count[i, 0, :] = one_count
            # print(f"0 occurs {zero_count} times")
            # print(f"1 occurs {one_count} times")
           
            discarded_patterns_count[i, 0, :] = self.calculate_discarding_count(enrollment_data_reference)
            # total_selected_patterns =  num_sram_patterns * np.ones(num_enroll_readings) - discarded_patterns_count[i, 0, :]
            # print("invalid patterns rate: ", discarded_patterns_count[i, 0, :] / enrollment_data_reference.shape[1])
            # print("extraction rate: ", np.divide(zero_count + one_count, total_selected_patterns) ) 
            # Plotting.plot_hd_values_histogram( test_sram_pattern_idx,
            #                               hamming_distances[:,:,0: num_enroll_readings],
            #                               self.chip_id)

            # print(f"Enrollment data reference shape: {enrollment_data_reference.shape}")

            results = []
            # Prepare arguments for multiprocessing
            args = [
                (
                    range_idx + 1, start_enroll_idx, # skip the first range
                    hamming_processor, num_enroll_readings, incremental_computation,
                    enroll_select_threshold,
                    hamming_distances[:, :, start_enroll_idx: end_enroll_idx],
                    enrollment_data_reference
                )
                for range_idx, (start_enroll_idx, end_enroll_idx) in enumerate(test_enroll_ranges)
            ]

            if 'SLURM_CPUS_ON_NODE' in os.environ:
                num_processes = int(os.environ['SLURM_CPUS_ON_NODE'])
            else:
                num_processes = min(cpu_count() - 1, len(args))

            print(f"Using {num_processes} processes for threshold {i}.")
            # Use multiprocessing pool
            with Pool(processes=num_processes) as pool:  # Adjust processes to your CPU
                results = pool.map(self.process_enroll_range, args)

            # Consolidate results
            for range_idx, e_count, d_count, z_count, o_count in results:
                error_count[i, range_idx, :] = e_count
                discarded_patterns_count[i, range_idx, :] = d_count
                zero_key_bits_count[i, range_idx, :] = z_count
                one_key_bits_count[i, range_idx, :] = o_count

            # for range_idx, (start_enroll_idx, end_enroll_idx) in enumerate(test_enroll_ranges):
            #     print("***********")
            #     print(f"Enrolling from {start_enroll_idx} to {end_enroll_idx}.")
            #     enroll_instance = Enroll(hamming_processor, start_enroll_idx,
            #                             num_enroll_readings, incremental_computation)
            #     sliced_hamming_distances = hamming_distances[:, :, start_enroll_idx: end_enroll_idx]
            #     enrollment_data = enroll_instance.execute(enroll_select_threshold,
            #                                               sliced_hamming_distances)
            #     # print(f"Enrollment data shape: {enrollment_data.shape}")
            #     # print("key and helper data for test pattern:",
            #         # enrollment_data[-1][test_sram_pattern_idx])
            #     # print("invalid patterns rate: ", temp_invalid_count / enrollment_data.shape[1])
            #     # invalid_count += np.sum(np.all(enrollment_data == -1, axis=2), axis=1)
            #     # Plotting.plot_hd_values_histogram( test_sram_pattern_idx,
            #     #                               hamming_distances[:,:,start_enroll_idx: start_enroll_idx + num_enroll_readings],
            #     #                               self.chip_id)
            #     enrollment_data_comparator = np.sum(enrollment_data != enrollment_data_reference,
            #                                 axis=-1, dtype=np.uint8)

            # Cache results
            self.cache_manager.save_incremental_cache(
                chip_id=self.chip_id,
                select_threshold=self.select_threshold,
                code_length=self.code_length,
                num_enroll_readings=num_enroll_readings,
                enroll_select_threshold=enroll_select_threshold,
                error_count=error_count[i],
                discarded_patterns_count=discarded_patterns_count[i],
                zero_key_bits_count=zero_key_bits_count[i],
                one_key_bits_count=one_key_bits_count[i]
            )

            end_time = time.time()
            print(f"Time taken for threshold {i}: {end_time - start_time}")
            print("***********\n")

        # range_enroll_readings = np.arange(1, num_enroll_readings)
        # Find optimal cases
        # self.get_optimal_cases(select_margin, range_enroll_readings,
        #                        error_count, discarded_patterns_count)

        # Plot results
        # self._plot_results(range_enroll_readings, select_margin, error_count,
        #                    discarded_patterns_count)

        # return enroll_ranges, threshold_values_list, error_count, discarded_patterns_count


    @staticmethod
    def get_rates_given_counts_single_threshold(results, code_length, select_threshold):
        """ Get rates given counts for a single threshold. """
        rate_results = []
        # readouts data in bytes
        num_sram_patterns = get_num_sram_patterns(code_length)
        # print("num_sram_patterns: ", num_sram_patterns)
        codebook_length = len(read_codebook(code_length,
                                 select_threshold[0],
                                 select_threshold[1]))
        # print("codebook_length: ", codebook_length)
        error_count = results[0]
        discarded_patterns_count = results[1]
        zero_key_bits_count = results[2]
        one_key_bits_count = results[3]
        
        total_bits = num_sram_patterns * codebook_length
        total_accepted_patterns = num_sram_patterns - discarded_patterns_count
        # extraction_rate = (zero_key_bits_count + one_key_bits_count) / total_accepted_aptterns
        # Ensure no division by zero
        safe_total = np.where(total_accepted_patterns == 0, 1, total_accepted_patterns)  # Replace 0s with 1s for safe division
        extraction_rate = (zero_key_bits_count + one_key_bits_count) / safe_total
        # Set extraction_rate to 0 where total_accepted_patterns was originally zero
        extraction_rate[total_accepted_patterns == 0] = 0

        rate_results.append(error_count / total_bits)
        rate_results.append(discarded_patterns_count / num_sram_patterns)
        rate_results.append(zero_key_bits_count / total_bits)
        rate_results.append(one_key_bits_count / total_bits)
        rate_results.append(extraction_rate)
        return np.array(rate_results)

    @staticmethod
    def get_rates_given_counts(results, code_length, select_threshold):
        """ Get rates given counts. """
        rate_results = {}
        # readouts data in bytes
        num_sram_patterns = get_num_sram_patterns(code_length)
        # print("num_sram_patterns: ", num_sram_patterns)
        codebook_length = len(read_codebook(code_length,
                                 select_threshold[0],
                                 select_threshold[1]))
        # print("codebook_length: ", codebook_length)

        for threshold_key, value in results.items():
            error_count = value["error_count"]
            discarded_patterns_count = value["discarded_patterns_count"]
            zero_key_bits_count = value["zero_key_bits_count"]
            one_key_bits_count = value["one_key_bits_count"]
            print("shape of zero_key_bits_count:", zero_key_bits_count.shape)
            print("shape of discarded_patterns_count:", discarded_patterns_count)

            total_bits = num_sram_patterns * codebook_length
            total_accepted_patterns = num_sram_patterns - discarded_patterns_count
            # extraction_rate = (zero_key_bits_count + one_key_bits_count) / total_accepted_patterns
            # Ensure no division by zero
            safe_total = np.where(total_accepted_patterns == 0, 1, total_accepted_patterns)  # Replace 0s with 1s for safe division
            extraction_rate = (zero_key_bits_count + one_key_bits_count) / safe_total
            # Set extraction_rate to 0 where total_accepted_patterns was originally zero
            extraction_rate[total_accepted_patterns == 0] = 0

            rate_results[threshold_key] = {
                "error_rate": error_count / total_bits,
                "discarding_rate":  discarded_patterns_count / num_sram_patterns,
                "zero_key_bits_rate": zero_key_bits_count / total_bits,
                "one_key_bits_rate": one_key_bits_count / total_bits,
                "extraction_rate": extraction_rate
            }
        return rate_results

    def initialize(self) -> np.ndarray:
        """
        Try to load data from cache. If not found, compute and save it.
        """

        # Try to load from cache
        results = self.cache_manager.load_cache(self.chip_id, self.select_threshold,
                                                self.code_length, const.MAX_ENROLLMENT_READINGS)

        # If data is not found, compute and save it
        if results is None:
            print("Cache not found. You need to compute results...")
            self.compare_and_save_helper_data()
            print("Results shape: ", len(results))
            results = self.cache_manager.load_cache(self.chip_id, self.select_threshold,
                                                self.code_length, const.MAX_ENROLLMENT_READINGS)

        rate_results = self.get_rates_given_counts(results, self.code_length, self.select_threshold)
        # print("Shape of ber rate results: ", rate_results[(-3.0,3.0)]["error_rate"].shape)
        # print("Results loaded: ", rate_results[(-3.0,3.0)])

        return rate_results

if __name__ == "__main__":
    all_files = get_files()
    # parameters = [(7,1,6), (9,1,8), (11, 1, 10), (11, 2, 9), (13, 1, 12), (13, 2, 11), (15, 1, 14), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(45, 10, 35),(47, 8, 39)]
    # parameters = [(27, 3, 24), (41, 6, 35),  (17, 1, 16)] #
    parameters = [(47, 8, 39)]
    # parameters = [(7,1,6), (9,1,8), (11, 1, 10), (11, 2, 9), (13, 1, 12), (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]
    
    chip_ids = list(all_files.keys()) # (['L45', 'M17', 'M2', 'M22', 'M39', 'M42', 'M44', 'M47', 'M49'])
    # all_readouts: list[ReadoutList] = [read_readouts(all_files['L45'])]
    all_readouts: list[ReadoutList] = [read_readouts(all_files[chip_id])
                                       for chip_id in chip_ids]
    coeff = [0,0]
    for n, coeff[0], coeff[1] in parameters:
        print("\n\nn =",n,"sigma=",coeff[1])
        for readouts_val in all_readouts:
            print("---------------------------------")
            helper_data_comparator = HelperDataComparator(n, readouts_val, coeff, True)
            helper_data_comparator.compare_and_save_helper_data()
            # helper_data = helper_data_comparator.initialize()
            # print("helper_data: ", helper_data[(-8,8)]["error_rate"][0:10])
