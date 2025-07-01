""" ConcatenatedCode class for Two-Stage TMVS Analysis
This class manages the concatenation of two codes and provides methods to calculate
theoretical error probabilities, required SRAM sizes, and helper data sizes.
"""
from math import ceil, log2
from two_stage_tmvs.algorithm.base_code import BaseCode
import tmvs.analysis_constants as const
import two_stage_tmvs.analysis_constants as data_const
from tmvs.formulas import theoretical_error_probability
from tmvs.formulas import theoretical_required_sram_size

class ConcatenatedCode:
    """ Class for managing concatenated codes in Two-Stage TMVS analysis.
    This class takes two BaseCode instances and provides methods to calculate
    error probabilities, memory sizes, and required SRAM sizes for the concatenated code.
    """
    def __init__(self, code1: BaseCode, code2: BaseCode):
        self.code1 = code1
        self.code2 = code2

    def is_single_code(self):
        """Check if the first code has code length 1.
        This refers to the special case of a single code code2."""
        return self.code1.code_length == 1

    def derive_codebook_memory_size(self):
        """
        Calculate the memory size required for the concatenated codebook.
        The memory size is calculated based on the codebooks of both codes.
        """
        memory_size = self.code2.codebook_size * self.code2.code_length

        if (not self.is_single_code()) and (self.code1.code_length != self.code2.code_length or
            self.code1.select_threshold != self.code2.select_threshold):
            # If the first code is a single code or both codes have the same parameters,
            # we only need the memory size of the second code.
            memory_size += self.code1.codebook_size * self.code1.code_length
        return memory_size / (8 * 1024)

    def two_stage_error_probability(self, p_flip):
        """ Calculate the theoretical error probability for the concatenated code."""
        first_stage_error_prob = theoretical_error_probability(self.code1.code_length,
                                                               self.code1.select_threshold,
                                                               p_flip)
        second_stage_error_prob = theoretical_error_probability(self.code2.code_length,
                                                                self.code2.select_threshold,
                                                                first_stage_error_prob)
        return second_stage_error_prob

    def two_stage_theoretical_required_sram_size(self):
        """ Calculate the number of SRAM bits required for the concatenated code."""
        key_length = const.KEY_LENGTH
        sram_size_2 = theoretical_required_sram_size(self.code2.code_length,
                                                     self.code2.select_threshold,
                                                     self.code2.codebook_size,
                                                     key_length)
        sram_size_1 = theoretical_required_sram_size(self.code1.code_length,
                                                     self.code1.select_threshold,
                                                     self.code1.codebook_size,
                                                     sram_size_2 * (8 * 1024))
        return sram_size_1

    def two_stage_theoretical_helper_data_size(self, key_length):
        """ Calculate the theoretical required helper data size for the concatenated code."""
        number_sram_bits = self.two_stage_theoretical_required_sram_size() * (8 * 1024)
        patterns_address_size = data_const.ADDRESS_SIZE + (key_length * self.code2.code_length # length of output of code 1
                                      * ceil(log2(number_sram_bits)))
        if self.is_single_code():
            codewords_indices_memory_size = (
                data_const.ADDRESS_SIZE +
                key_length * ceil(log2(self.code2.codebook_size))
            )
        else:
            codewords_indices_memory_size = (
                data_const.ADDRESS_SIZE +
                key_length * self.code2.code_length * ceil(log2(self.code1.codebook_size))
            )
            if (self.code1.code_length != self.code2.code_length) or \
            (self.code1.select_threshold != self.code2.select_threshold):
                codewords_indices_memory_size += data_const.ADDRESS_SIZE
            codewords_indices_memory_size += key_length * ceil(log2(self.code2.codebook_size))
        return (patterns_address_size + codewords_indices_memory_size) / (8 * 1024)


if __name__ == "__main__":
    secret_key_length = const.KEY_LENGTH
    flipping_probability = 0.05
    test_code1 = BaseCode(7, [1, 6])
    # test_code2 = BaseCode(7, [1, 6])
    test_code2 = BaseCode(9, [1, 8])
    # test_code1 = BaseCode(11, [1, 10])
    # test_code1 = BaseCode(13, [1, 12])
    test_concat_code = ConcatenatedCode(test_code1, test_code2)
    print("--"*20)
    print(f"Code Length 1: {test_code1.code_length}")
    print(f"Select Threshold 1: {test_code1.select_threshold}")
    print(f"Codebook Size 1: {test_code1.codebook_size}")
    print(f"Code Length 2: {test_code2.code_length}")
    print(f"Select Threshold 2: {test_code2.select_threshold}")
    print(f"Codebook Size 2: {test_code2.codebook_size}")
    print("--"*20)
    print("p_flip:", flipping_probability)
    print("Error Probability:",
          test_concat_code.two_stage_error_probability(flipping_probability))
    print("Concatenated Codebook Memory Size:", test_concat_code.derive_codebook_memory_size())
    print("Required SRAM Size (kB):", test_concat_code.two_stage_theoretical_required_sram_size())
    print("Required Helper Data Size (kB):", 
          test_concat_code.two_stage_theoretical_helper_data_size(secret_key_length))
