""" Test the HammingProcessor class. """
import numpy as np
import unittest
from nvm_free_tmvs.core.hamming_processor import HammingProcessor
from nvm_free_tmvs.core.chunk_data_processor import ChunkDataProcessor
from nvm_free_tmvs.utils.file_manager  import ReadoutList, get_files, read_readouts, read_codebook


def compute_hamming_distance_manual(readouts, n, coeff, lookup_table, actual_hamming_distances,
                                    data_start_idx, enroll_readings_num):
    """
    Manually compute Hamming distances using straightforward bitwise operations.
    """
    chunk_data_processor = ChunkDataProcessor(n, readouts, active_multithreading=1)
    chunked_data = chunk_data_processor.chunk_readouts()[data_start_idx: data_start_idx +enroll_readings_num]
    codebook = read_codebook(n, coeff[0], coeff[1])
    n_codebook = len(codebook)
    # n_chunks, chunk_size = chunked_data.shape
    num_reading, num_sram_pattern = chunked_data.shape

    
    # Initialize hamming distances
    # hamming_distances = np.zeros((n_codebook, num_reading, num_sram_pattern), dtype=np.uint8)
    hamming_distances = np.zeros((num_sram_pattern, n_codebook, num_reading), dtype=np.uint8)  # (sram_pattern_idx, codeword_idx, reading_idx)

    # for i, cb_val in enumerate(codebook):
    #     for j in range(num_reading):
    #         for k in range(num_sram_pattern):
    for j in range(num_reading):  # Iterate over readings
        for k in range(num_sram_pattern):  # Iterate over SRAM patterns
            for i, cb_val in enumerate(codebook):  # Iterate over codewords
                
                xor_result = cb_val ^ chunked_data[j, k]  # XOR operation
                # hamming_distances[i, j, k] = lookup_table[xor_result] # bin(xor_result).count("1")   # Count set bits
                hamming_distances[k, i, j] = lookup_table[xor_result] # bin(xor_result).count("1")   # Count set bits

                # if actual_hamming_distances[i, j, k] != hamming_distances[i, j, k]:
                if actual_hamming_distances[k, i, j] != hamming_distances[k, i, j]:
                    print("-----------------------------------")
                    # print("Mismatch at (i, j, k):", i, j, k)
                    # print("Expected:", actual_hamming_distances[i, j, k])
                    # print("Actual:", hamming_distances[i, j, k])
                    print("Mismatch at (k, i, j):", k, i, j)
                    print("Expected:", actual_hamming_distances[k, i, j])
                    print("Actual:", hamming_distances[k, i, j])
                    print("Codebook value:", cb_val)
                    print("Chunk value:", chunked_data[j, k])
                    print("XOR result:", xor_result)
                    print("Lookup table value:", lookup_table[xor_result])
                    return hamming_distances
    return hamming_distances

def test_hamming_distance_computation(readouts, n, coeff, data_start_idx, enroll_readings_num):
    """
    Test function to verify the correctness of the Hamming distance computation.
    """

    # Compute Hamming distances using the HammingProcessor
    enroll_key = HammingProcessor(readouts=readouts, code_length=n,
                                        select_threshold=coeff,
                                        active_multithreading=False)
    lookup_table = enroll_key.create_lookup_table()
    actual = enroll_key.compute_hamming_distances(data_start_idx, enroll_readings_num)
    print("first few elements of lookup table: ", lookup_table[:10])

    # Compute expected results manually
    expected = compute_hamming_distance_manual(readouts, n, coeff, lookup_table, actual, data_start_idx, enroll_readings_num)
    print("computed expected hamming distances!")
    print("Shape of expected: ", expected.shape)
    
    # Compare the results
    assert np.array_equal(expected, actual), "Hamming distances do not match!"
    print("Hamming distances match!")
    
class TestHammingProcessor(unittest.TestCase):
    def test_compute_hamming_distance_returns_empty_list_if_input_is_empty(self):
        test_input = []
        #self.assert()

if __name__ == "__main__":
    unittest.main()
    exit(0)

    all_files = get_files()
    parameters = [(9,1,8)]
    # parameters = [(7,1,6), (9,1,8), (11, 1, 10), (11, 2, 9), (13, 1, 12), (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]
    all_readouts: list[ReadoutList] = [read_readouts(all_files['L45'])]
    coeff = [0,0]
    data_start_idx=0
    enroll_readings_num=10
    for n, coeff[0], coeff[1] in parameters:
        print("\n\nn =",n,"sigma=",coeff[1])
        for readouts in all_readouts:
            print("\nChip ",readouts.chip_id)
            test_hamming_distance_computation(readouts, n, coeff, data_start_idx, enroll_readings_num)