""" Constants for helper-less TMVS analysis. """
import numpy as np
MAX_ENROLLMENT_READINGS = 10 #10  # Maximum number of readings per chip used for enrollement 
								# in helper-less TMVS
MAX_CODEWORDS_PER_CHUNK = 50  # Maximum number of codewords derived per chunk for enrollement 
							  # in helper-less TMVS
MAX_NUM_CHUNKS = 200  # Maximum number of chunks used for enrollement in helper-less TMVS 
					  # (above this limit, memory errors at n=17 occur)
THRESHOLD_STEP_SIZE = 0.1  # Step size for threshold values in helper-less TMVS
HD_DATA_TYPE = np.int8  # Data type for Hamming distances in helper-less TMVS
# temporary constants
PRESELECTION_READINGS = 7
CODEWORDS_PER_SRAM_PATTERN = 1  # Number of the best codewords derived per SRAM pattern 
								# for enrollement in helper-less TMVS
# Define the range of exponents
exponents = np.arange(-4, -7, -1, dtype=float)  # -4, -5, -6, -7
# Generate the sequence of failure rate targets
TEST_FAILURE_RATE_TARGET = np.array([x for exp in exponents
                                     for x in (10**exp, 7 * 10**(exp-1), 3 * 10**(exp-1))])[:-1]