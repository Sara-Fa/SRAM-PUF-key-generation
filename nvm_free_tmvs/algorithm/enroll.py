"""  Enroll class for the nvm_free_tmvs_algo module. """
import numpy as np
from nvm_free_tmvs.algorithm.base import BaseAnalysis
from nvm_free_tmvs.core.hamming_processor import HammingProcessor

class Enroll(BaseAnalysis):
    """ Enroll class for the nvm_free_tmvs_algo module. """
    def __init__(self, hamming_processor: HammingProcessor,
                 data_start_idx: int, num_enroll_readings: int,
                 incremental_computation: bool):
        super().__init__(hamming_processor, data_start_idx, num_enroll_readings,
                         incremental_computation)
        # self.incremental_computation = incremental_computation

    def extract_key_and_helper_data_incrementally (self, enroll_select_threshold,
                                                   hamming_distances: np.ndarray):
        """
        Process the 3D matrix hd with layout [sram_pattern_idx, codeword_idx, reading_idx] to:
        1. Check the values across reading_idx for each sram_pattern_idx and codeword_idx.
        2. Exclude codeword_idx where values are not of the same sign across reading_idx.
        3. Sum the valid values across reading_idx for each codeword_idx.
        4. Exclude codeword_idx where the sum is not within the threshold.
        5. Give values to the remaining codeword_idx based on the threshold.
        """
        num_sram_patterns, num_codewords, num_readings = hamming_distances.shape
        # Initialize with -1 to indicate no valid codewords
        # every index corresponds to the (number of used readings - 1)
        secret_bits = -1 * np.ones((num_readings, num_sram_patterns, num_codewords), dtype=np.int8)

        for sram_pattern_idx in range(num_sram_patterns):
            # Extract submatrix for current sram_pattern_idx and transpose for efficient reading
            submatrix = hamming_distances[sram_pattern_idx, :, :].T # Shape: (num_readings, num_codewords)
            sign_matrix = np.sign(submatrix)  # Shape: (num_readings, num_codewords)

            # Compute secret bits for the first reading (original TMVS)
            zero_bits_mask = submatrix[0] <= enroll_select_threshold[0]
            secret_bits[0, sram_pattern_idx, zero_bits_mask] = 0
            one_bits_mask = submatrix[0] >= enroll_select_threshold[1]
            secret_bits[0, sram_pattern_idx, one_bits_mask] = 1

            # Initialize a boolean mask for consistent signs
            consistent_signs = np.ones(num_codewords, dtype=bool)

            # Initialize the cumulative sums
            cumulative_sums = np.cumsum(submatrix, axis=0)  # Shape: (num_readings, num_codewords)

            for end_reading_idx in range(1, num_readings):

                # Update consistency: Check if the new reading matches the first reading's sign
                new_signs = sign_matrix[end_reading_idx]  # Shape: (num_codewords,)
                consistent_signs &= (new_signs == sign_matrix[0])

                # Calculate the new average values
                new_avg_values = cumulative_sums[end_reading_idx]/(end_reading_idx+1)  # Shape: (num_codewords,)

                # Apply conditions
                zero_bits_mask = consistent_signs & (new_avg_values <= enroll_select_threshold[0])
                secret_bits[end_reading_idx, sram_pattern_idx, zero_bits_mask] = 0
                one_bits_mask = consistent_signs & (new_avg_values >= enroll_select_threshold[1])
                secret_bits[end_reading_idx, sram_pattern_idx, one_bits_mask] = 1

        return secret_bits

    # not incrementally
    def extract_key_and_helper_data(self, enroll_select_threshold, hamming_distances: np.ndarray):
        """
        Using a single number of readings, not from 1 to num_readings.
        Process the 3D matrix hd with layout [sram_pattern_idx, codeword_idx, reading_idx] to:
        1. Check the values across reading_idx for each sram_pattern_idx and codeword_idx.
        2. Exclude codeword_idx where values are not of the same sign across reading_idx.
        3. Sum the valid values across reading_idx for each codeword_idx.
        4. Exclude codeword_idx where the sum is not within the threshold.
        5. Give values to the remaining codeword_idx based on the threshold.
        """

        num_sram_patterns, num_codewords, _ = hamming_distances.shape
        # Initialize with -1 to indicate no valid codewords
        secret_bits = -1 * np.ones((1,num_sram_patterns,num_codewords), dtype=np.int8)
        for sram_pattern_idx in range(num_sram_patterns):
            # Extract submatrix for current sram_pattern_idx
            # with shape (num_codewords, num_readings)
            submatrix = hamming_distances[sram_pattern_idx, :, :]

            # Check for consistent signs across reading_idx for each codeword_idx
            consistent_signs = np.all(submatrix > 0, axis=1) | np.all(submatrix < 0, axis=1)  # Boolean mask

            # Calculate average value for each codeword across readings
            average_values = np.mean(submatrix, axis=1)  # Shape: (num_codewords,)

            # Apply conditions
            secret_bits[0, sram_pattern_idx,
                        consistent_signs & (average_values <= enroll_select_threshold[0])] = 0
            secret_bits[0, sram_pattern_idx,
                        consistent_signs & (average_values >= enroll_select_threshold[1])] = 1

        return secret_bits


    def execute(self, enroll_select_threshold, enroll_hamming_distances = None,
                boolean_hamming_distances=None):
        """ Execute the enrollment process. """
        if enroll_hamming_distances is None:
            enroll_hamming_distances = self.hamming_processor.compute_hamming_distances(
                self.data_start_idx, self.num_enroll_readings)

        assert enroll_hamming_distances.shape[2] == self.num_enroll_readings, (
            "Hamming distances do not match the number of enroll readings.")

        enrollment_data = None
        if self.incremental_computation:
            enrollment_data = self.extract_key_and_helper_data_incrementally(
                enroll_select_threshold,
                enroll_hamming_distances)
        else:
            enrollment_data = self.extract_key_and_helper_data(
                enroll_select_threshold, enroll_hamming_distances)

        return enrollment_data
