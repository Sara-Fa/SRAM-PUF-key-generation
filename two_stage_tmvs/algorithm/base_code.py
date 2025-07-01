"""
BaseCode class for managing single codes based on code length and selection thresholds.
Handles both Reed-Muller and custom codebook size derivations.
"""
from common.data_reading_utils import read_codebook
from two_stage_tmvs.algorithm.RM_code_generator import derive_RM_codebook_size

class BaseCode:
    """ Base class for codes used in Two-Stage TMVS analysis."""
    def __init__(self, code_length, select_threshold):
        self.code_length = code_length
        self.select_threshold = select_threshold
        if code_length == 1:
            self.codebook_size = 1
        elif code_length % 2 == 0:
            self.codebook_size = derive_RM_codebook_size(self.code_length)
        else:
            self.codebook_size = self.derive_codebook_size()

    def derive_codebook_size(self):
        """ Derives the codebook size."""
        return len(read_codebook(self.code_length,
                                 self.select_threshold[0],
                                 self.select_threshold[1]))

if __name__ == "__main__":
    test_code = BaseCode(7, [1, 6])
    print(f"Code Length: {test_code.code_length}")
    print(f"Select Threshold: {test_code.select_threshold}")
    print(f"Codebook Size: {test_code.codebook_size}")
