""" This module contains the GlobalBERProcessor class. """
from typing import List
from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory
import os
import time
import numpy as np
from nvm_free_tmvs.core.hamming_processor import HammingProcessor
from nvm_free_tmvs.core.chunk_data_processor import ChunkDataProcessor
from nvm_free_tmvs.utils.file_manager import ReadoutList, get_files, read_readouts
from nvm_free_tmvs.utils.analysis_utils import get_enrollment_ranges, get_enrollment_threshold_values
from nvm_free_tmvs.algorithm.ber_processor import BERProcessor
from nvm_free_tmvs.experiments.global_ber_cache_manager import BERCacheManager
import common.data_constants as data_const
import nvm_free_tmvs.analysis_constants as const



class GlobalBERProcessor:
    """ Class for processing Bit Error Rate (BER) data for different enrollments. """
    def __init__(self, code_length: int, readouts: ReadoutList, select_threshold: List[float],
                 active_multithreading: bool): # replace select_threshold by margin_ceoff
        """ Initialize the GlobalBERProcessor. """
        self.code_length = code_length
        self.readouts = readouts
        self.chip_id = readouts.chip_id
        self.select_threshold = select_threshold
        # self.margin_coeff = margin_coeff
        # self.select_threshold, _ = calculate_threshold (code_length, margin_coeff)
        self.active_multithreading = active_multithreading
        self.cache_manager = BERCacheManager()


    def process_enroll_range(self, args):
        """
        Perform BER calculation for a single enrollment range.
        """
        range_idx, start_enroll_idx, end_enroll_idx, ber_processor, enroll_select_threshold, \
            mem_name, mem_shape, mem_dtype, bool_mem_name, bool_mem_shape, bool_mem_dtype=  args

        # Reconnect to shared memory
        bool_chunk_sums_shared = SharedMemory(name=bool_mem_name)
        # boolean_hamming_shared = SharedMemory(name=bool_mem_name)
        hamming_shared = SharedMemory(name=mem_name)

        bool_chunk_sum = np.ndarray(bool_mem_shape, dtype=bool_mem_dtype,
                                    buffer=bool_chunk_sums_shared.buf)[:,:,range_idx]
        # boolean_hamming_distances = np.ndarray(bool_mem_shape, dtype=bool_mem_dtype,
        #                                        buffer=boolean_hamming_shared.buf)
        sliced_hamming_distances = np.ndarray(mem_shape, dtype=mem_dtype,
                                       buffer=hamming_shared.buf)[
            :, :, start_enroll_idx: end_enroll_idx]

        # Calculate error and valid bits count
        # !!!!!!!!!!!!!!! you are sending data as parameters ??
        error, valid = ber_processor.execute(
            enroll_select_threshold, sliced_hamming_distances, bool_chunk_sum)
            # enroll_select_threshold, sliced_hamming_distances, boolean_hamming_distances)
        return range_idx, error, valid

    def compute_and_save_global_ber(self):
        """
        Perform computation and save the global BER to the cache.
        """
        enroll_ranges = get_enrollment_ranges()

        chunk_data_processor = ChunkDataProcessor(self.code_length,
                                                self.readouts,
                                                self.active_multithreading)
        chunked_data = chunk_data_processor.chunk_readouts()

        hamming_processor = HammingProcessor(self.code_length, self.readouts,
                                                self.select_threshold,
                                                self.active_multithreading)
        hamming_distances, boolean_hamming_distances = None, None
        num_enroll_readings = const.MAX_ENROLLMENT_READINGS
        incremental_computation = True
        hamming_distance_tag = 0
        _, threshold_values_list = get_enrollment_threshold_values(self.code_length)
        # boolean_hamming_shared = None
        bool_chunk_sums_shared = None
        hamming_shared = None
        bool_mem_name, bool_mem_shape, bool_mem_dtype = None, None, None
        mem_name, mem_shape, mem_dtype = None, None, None

        # Initialize result arrays
        error_count = np.zeros(
            (len(threshold_values_list), len(enroll_ranges), num_enroll_readings))
        valid_patterns_count = np.zeros(
            (len(threshold_values_list), len(enroll_ranges), num_enroll_readings))

        # test_threshold = [[-3,3],[-3.5,3.5]]
        # Iterate over the threshold values
        for i, enroll_select_threshold in enumerate(threshold_values_list):
            # Check if this threshold exists in the cache
            if self.cache_manager.check_threshold_in_cache(self.chip_id, self.select_threshold,
                                                           self.code_length, num_enroll_readings,
                                                           enroll_select_threshold):
                print(f"Skipping computation for threshold {enroll_select_threshold}, "
                      f"already in cache.")
                continue

            # Compute Hamming distances if not already computed
            if not hamming_distance_tag:
                # shape of hamming_distances (num_sram_patterns, num_codewords, num_readings)
                hamming_distances = hamming_processor.compute_hamming_distances(
                    0, data_const.READINGS_TO_ANALYZE, chunked_data)
                print("size of hamming_distances memory", hamming_distances.nbytes/1e9, "GB")
                # give value 1 to all values greater than 0, and 0 to all values less than 0
                boolean_hamming_distances = (hamming_distances >> 7) + 1
                print("size of boolean_hamming_distances memory", boolean_hamming_distances.nbytes/1e9, "GB")
                del chunked_data  # Free up memory

                # Consider boolean_hamming_distances as chunks of num_enroll_readings matrices
                num_chunks = boolean_hamming_distances.shape[2] // num_enroll_readings

                time1 = time.time()
                # calculate this in another way, do all sum then substract
                # shape of partial_chunk_sums (num_sram_patterns, num_codewords, num_chunks)
                partial_chunk_sums = np.zeros((boolean_hamming_distances.shape[0], boolean_hamming_distances.shape[1],
                                       num_chunks), dtype=np.uint16)
                # shape of bool_chunk_sums (num_sram_patterns, num_codewords, num_chunks)
                bool_chunk_sums = np.zeros((boolean_hamming_distances.shape[0], boolean_hamming_distances.shape[1],
                                       num_chunks), dtype=np.uint16)
                
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * num_enroll_readings
                    end_idx = start_idx + num_enroll_readings
                    partial_chunk_sums[:, :, chunk_idx] = np.sum(boolean_hamming_distances[:, :, start_idx:end_idx
                                                                         ], axis=2, dtype=np.uint16)
                    # Sum over all other chunks
                total_sum_chunks = np.sum(partial_chunk_sums, axis=2, dtype=np.uint16)
                
                for chunk_idx in range(num_chunks):
                    bool_chunk_sums[:, :, chunk_idx] = total_sum_chunks - partial_chunk_sums[:,:,chunk_idx]
               
                del boolean_hamming_distances  # Free up memory
                print("size of bool_chunk_sums memory", bool_chunk_sums.nbytes/1e9, "GB")

                time2 = time.time()
                print("Time taken to sum chunks: ", time2 - time1)
                print("chunk_sums shape: ", np.array(bool_chunk_sums).shape)
                
                # Create shared memory block
                bool_chunk_sums_shared = SharedMemory(create=True, size=bool_chunk_sums.nbytes)
                # boolean_hamming_shared = SharedMemory(create=True, size=boolean_hamming_distances.nbytes)
                hamming_shared = SharedMemory(create=True, size=hamming_distances.nbytes)
                
                # Metadata for shared memory
                bool_mem_name = bool_chunk_sums_shared.name
                bool_mem_shape = bool_chunk_sums.shape
                bool_mem_dtype = bool_chunk_sums.dtype
                # bool_mem_name = boolean_hamming_shared.name
                # bool_mem_shape = boolean_hamming_distances.shape
                # bool_mem_dtype = boolean_hamming_distances.dtype
                mem_name = hamming_shared.name
                mem_shape = hamming_distances.shape
                mem_dtype = hamming_distances.dtype

                # Attach NumPy array to shared memory
                np.copyto(np.ndarray(bool_mem_shape, bool_mem_dtype,
                                     buffer=bool_chunk_sums_shared.buf),
                            bool_chunk_sums)
                # np.copyto(np.ndarray(bool_mem_shape, bool_mem_dtype,
                #                      buffer=boolean_hamming_shared.buf),
                #           boolean_hamming_distances)
                np.copyto(np.ndarray(mem_shape, mem_dtype,
                                        buffer=hamming_shared.buf),
                            hamming_distances)
                hamming_distance_tag = 1  # Mark as computed

            print(f"\nEnrollment select threshold {i}: {enroll_select_threshold}")
            start_time = time.time()

            test_enroll_readings = enroll_ranges
            # Use multiprocessing to process enroll ranges
            if self.active_multithreading == 1:
                # Prepare arguments for multiprocessing
                results = []
                args = [(range_idx, start_enroll_idx, end_enroll_idx,
                         BERProcessor(hamming_processor, start_enroll_idx,
                                   num_enroll_readings, incremental_computation),
                      enroll_select_threshold,
                      mem_name, mem_shape, mem_dtype,
                    #   hamming_distances[:, :, start_enroll_idx: end_enroll_idx],
                      bool_mem_name, bool_mem_shape, bool_mem_dtype)
                    #   boolean_hamming_distances)
                     for range_idx, (start_enroll_idx, end_enroll_idx) in enumerate(
                         test_enroll_readings)]

                if 'SLURM_CPUS_ON_NODE' in os.environ:
                    num_processes = int(os.environ['SLURM_CPUS_ON_NODE'])
                else:
                    num_processes = min(cpu_count() - 1, len(args))

                print(f"Using {num_processes} processes for threshold {i}.")

                # Use multiprocessing pool
                with Pool(processes=num_processes) as pool:
                    results = pool.map(self.process_enroll_range, args)

                # Collect results
                for range_idx, error, valid in results:
                    error_count[i][range_idx] = error
                    valid_patterns_count[i][range_idx] = valid

            else:
                # Process sequentially
                for range_idx, (start_enroll_idx, end_enroll_idx) in enumerate(
                    test_enroll_readings):
                    print(f"Enrolling from {start_enroll_idx} to {end_enroll_idx}.")
                    ber_processor = BERProcessor(hamming_processor, start_enroll_idx,
                                   num_enroll_readings, incremental_computation)
                    # sliced_hamming_distances = hamming_distances[
                    #     :, :, start_enroll_idx: end_enroll_idx]
                    _, error, valid = self.process_enroll_range(
                        # (range_idx, ber_processor, enroll_select_threshold,
                        #  sliced_hamming_distances, boolean_hamming_distances)
                        # )
                        (range_idx, start_enroll_idx, end_enroll_idx,
                            ber_processor, enroll_select_threshold,
                            mem_name, mem_shape, mem_dtype,
                            bool_mem_name, bool_mem_shape, bool_mem_dtype)
                        )
                    error_count[i][range_idx] = error
                    valid_patterns_count[i][range_idx] = valid

            # Save the results to the cache
            self.cache_manager.save_incremental_cache(
                chip_id=self.chip_id,
                select_threshold=self.select_threshold,
                code_length=self.code_length,
                num_enroll_readings=num_enroll_readings,
                enroll_select_threshold=enroll_select_threshold,
                error_count=error_count[i],
                valid_patterns_count=valid_patterns_count[i]
                )

            end_time = time.time()
            print(f"Time taken for threshold {i}: {end_time - start_time}")
            print("***********\n")
            
        # Clean up shared memory outside the loop
        if bool_chunk_sums_shared:
            bool_chunk_sums_shared.close()
            bool_chunk_sums_shared.unlink()
        # if boolean_hamming_shared:
        #     boolean_hamming_shared.close()
        #     boolean_hamming_shared.unlink()

        if hamming_shared:
            hamming_shared.close()
            hamming_shared.unlink()

    @staticmethod
    def get_ber_statistics(ber_data):
        """
        Compute statistics for the Bit Error Rate data.
        """
        # Compute statistics for the Bit Error Rate data
        mean_ber = np.mean(ber_data) #, axis=0)
        std_ber = np.std(ber_data) #, axis=0)
        max_ber = np.max(ber_data) #, axis=0)
        min_ber = np.min(ber_data) #, axis=0)
        return mean_ber, std_ber, max_ber, min_ber

    def execute(self):
        """
        TODO
        """

    @staticmethod
    def get_rates_given_counts_single_threshold(results, *_):
        """ Get rates given counts for a single thrshold. """
        test_readings_count = data_const.READINGS_TO_ANALYZE - const.MAX_ENROLLMENT_READINGS

        safe_total = np.where(results[1] == 0, 1, test_readings_count * results[1].astype(np.float64))  # Replace 0s with 1s for safe division
        safe_result = np.divide(results[0], safe_total)
        safe_result[results[1] == 0] = 0
        
        return np.expand_dims(safe_result, axis=0)
        # return np.expand_dims(np.divide(results[0], test_readings_count * results[1].astype(np.float64),
        #                  where=results[1] > 0), axis=0)

    @staticmethod
    def get_rates_given_counts( results):
        """ Get rates given counts. """
        rate_results = {}
        test_readings_count = data_const.READINGS_TO_ANALYZE - const.MAX_ENROLLMENT_READINGS

        for threshold_key, value in results.items():
            error_count = value["error_count"]
            valid_patterns_count = value["valid_patterns_count"]
            rate_results[threshold_key] = {"error_rate": np.divide(error_count,
                                                    test_readings_count * valid_patterns_count,
                                                    where=valid_patterns_count > 0)}
        return rate_results

    def initialize(self) -> np.ndarray:
        """
        Load or compute BER based on the class parameters.
        """
        num_enroll_readings = const.MAX_ENROLLMENT_READINGS
        # Check if data is cached
        results = self.cache_manager.load_cache(self.chip_id, self.select_threshold,
                                                self.code_length, num_enroll_readings)

        # If data is not found, compute and save it
        if results is None:
            print("Cache not found. You need to compute results...")
            self.compute_and_save_global_ber()
            results = self.cache_manager.load_cache(self.chip_id, self.select_threshold,
                                                self.code_length, num_enroll_readings)
        rate_results = self.get_rates_given_counts(results)
        return rate_results

if __name__ == "__main__":
    all_files = get_files()
    parameters =  [(41, 6, 35), (27, 3, 24)] # [ (17, 1, 16), (27, 3, 24), (41, 6, 35)] #
    # parameters = [(7,1,6), (9,1,8), (11, 1, 10), (11, 2, 9), (13, 1, 12), (13, 2, 11), (15, 1, 14), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(45, 10, 35),(47, 8, 39)]
    # parameters = [(17, 1, 16), (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29)]
    # parameters = [(31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]    
    # parameters = [(7,1,6), (9,1,8), (11, 1, 10), (11, 2, 9), (13, 1, 12), (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]
    # all_readouts: list[ReadoutList] = [read_readouts(all_files['L45'])]
    all_readouts: list[ReadoutList] = [read_readouts(all_files[chip_id])
                                       for chip_id in all_files.keys()]
    coeff = [0,0]
    for n, coeff[0], coeff[1] in parameters:
        for readouts_val in all_readouts:
            print(f"Chip {readouts_val.chip_id} Codebook ({n},{coeff}) ---------------------------------")
            ber_comparator = GlobalBERProcessor(n, readouts_val, coeff, True)

            # use this when running on cluster
            ber_comparator.compute_and_save_global_ber()

            # ber = ber_comparator.initialize()
            # print("BER: ", ber[(-7,7)]["error_rate"][0:10])
