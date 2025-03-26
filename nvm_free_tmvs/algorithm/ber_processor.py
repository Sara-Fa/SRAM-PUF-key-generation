""" Module for Bit Error Rate (BER) analysis. """
import time
import numpy as np
import common.data_constants as data_const
from nvm_free_tmvs.algorithm.base import BaseAnalysis
from nvm_free_tmvs.algorithm.enroll import Enroll
# from nvm_free_tmvs.algorithm.ber_cache_manager import BERCacheManager
from nvm_free_tmvs.core.hamming_processor import HammingProcessor
from nvm_free_tmvs.utils.file_manager  import ReadoutList, get_files, read_readouts


class BERProcessor(BaseAnalysis):
    """ Base class for BER analysis. """
    def __init__(self, hamming_processor: HammingProcessor,
                 data_start_idx: int, num_enroll_readings: int,
                 incremental_computation: bool):
        super().__init__(hamming_processor, data_start_idx, num_enroll_readings,
                         incremental_computation)
        # self.enroll_instance = enroll_instance

    def calculate_ber(self, boolean_hamming_distances: np.ndarray,
                      enrollment_data: np.ndarray,) -> np.ndarray:
        """
        Calculate error counts for the test range for each `num_enroll_readings` value 
        using transformed Hamming distances.
        """
        # Get test readings indices and slice Hamming distances
        num_tested_readings = data_const.READINGS_TO_ANALYZE - self.num_enroll_readings

        # Initialize arrays
        num_enroll_readings = enrollment_data.shape[0]
        error_count = np.zeros(num_enroll_readings)
        valid_bits_count = np.zeros(num_enroll_readings, dtype=np.uint32)

        # Vectorized computation for each enrollment reading
        for enroll_readings_idx in range(num_enroll_readings):
            # print("Enroll readings index: ", enroll_readings_idx)
            # shape of enrollment_data is (num_readings, num_sram_patterns, num_codewords)
            secret_key_bits = enrollment_data[enroll_readings_idx]

            # Ensure the matrices have compatible shapes (num_sram_patterns, num_codewords)
            assert boolean_hamming_distances.shape == secret_key_bits.shape, \
                "Shape mismatch between hamming distances and secret key."

            # Mask
            valid_mask = secret_key_bits != -1

            # Count the number of non -1 elements
            valid_bits_count[enroll_readings_idx] = valid_mask.sum()

            # Count mismatched elements (valid only)
            error_count[enroll_readings_idx] = np.sum(
                (boolean_hamming_distances * (secret_key_bits == 0)) +
                ((num_tested_readings - boolean_hamming_distances)
                 * (secret_key_bits == 1)))

        return error_count, valid_bits_count


    @staticmethod
    def compute_error_rates(error_count: np.ndarray, valid_bits_count: np.ndarray,
                            test_readings_count) -> np.ndarray:
        """
        Compute error rates from error counts.
        """
        return  np.divide(error_count, valid_bits_count * test_readings_count,
                          where=valid_bits_count > 0)

    def execute(self, enroll_select_threshold, enroll_hamming_distances = None,
                boolean_hamming_distances = None):
        """
        Execute the BER analysis.
        """
        if boolean_hamming_distances is None or enroll_hamming_distances is None:
            hamming_distances = self.hamming_processor.compute_hamming_distances(
                0, data_const.READINGS_TO_ANALYZE)
            # Convert Hamming distances to binary
            boolean_matrix = np.where(hamming_distances > 0, 1, 0)
            end_idx = self.data_start_idx + self.num_enroll_readings
            boolean_hamming_distances = np.sum(boolean_matrix[:, :, :self.data_start_idx], axis=2, dtype=np.uint16) + \
                            np.sum(boolean_matrix[:, :, end_idx:], axis=2, dtype=np.uint16)
            
            # Slice Hamming distances
            enroll_hamming_distances = hamming_distances[
                :,:,self.data_start_idx:self.data_start_idx+self.num_enroll_readings]

        # assert boolean_hamming_distances.shape[2] == data_const.READINGS_TO_ANALYZE, (
        #     "Hamming distances do not match the number of total readings.")

        # Perform enrollment
        enroll_instance = Enroll(self.hamming_processor, self.data_start_idx,
                                 self.num_enroll_readings, self.incremental_computation)
        enrollment_data = enroll_instance.execute(enroll_select_threshold,
                                                  enroll_hamming_distances)

        # Calculate BER
        error_count, valid_bits_count = self.calculate_ber(
            boolean_hamming_distances, enrollment_data)

        return error_count, valid_bits_count



if __name__ == "__main__":

    all_files = get_files()
    parameters = [(7,1,6)] #, (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]
    # parameters = [(7,1,6), (9,1,8), (11, 1, 10), (11, 2, 9), (13, 1, 12), (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]

    all_readouts: list[ReadoutList] = [read_readouts(all_files['L45'])]
    # all_readouts: list[ReadoutList] = [read_readouts(all_files[chip_id])
    #                                    for chip_id in all_files.keys()]

    coeff = [0,0]
    for n, coeff[0], coeff[1] in parameters:
        print("\n\nn =",n,"sigma=",coeff[1])
        for readouts in all_readouts:
            print("---------------------------------")
            start_idx = 0 
            num_enroll_reading = 10
            processor = HammingProcessor(readouts=readouts, code_length=n,
                                          select_threshold=coeff,
                                          active_multithreading=True)
            ber_processor = BERProcessor(processor, start_idx,
                                         num_enroll_reading, True)
            start_time_1 = time.time()
            error, counts = ber_processor.execute([-3,3])
            end_time_1 = time.time()
            print("Executed BER_count calculation during: ", end_time_1-start_time_1)
            # print("Error count: ", error)
            ber = ber_processor.compute_error_rates(error, counts, 990)
            print("BER results shape:", ber.shape)
            print("BER results: ", ber)