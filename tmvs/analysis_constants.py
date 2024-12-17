"""
TMVS analysis constants, to be imported in other modules.
"""
import numpy as np
CONVERGENCE_THRESHOLD = 0.0001
CODEBOOK_SIZES_TO_TEST = np.arange(50,2550,50, dtype=int)
CODE_LENGTH_TO_TEST_FAILURE = np.arange(17, 43, 2, dtype=int)
CODE_LENGTH_TO_TEST_OPTIMALITY = np.arange(7, 100, 2, dtype=int)
SD_COEFFICIENT_TO_TEST_OPTIMALITY = np.arange(1, 8, 0.5)
KEY_LENGTH = 128


# the number of decimals to consider when deriving different BER frequencies
BER_PLOT_RESOLUTION = {
    (7,1,6): 4,
    (9,1,8): 5,
    (11,2,9): 5,
    (11,1,10): 9,
    (13,1,12): 9,
    (13,2,11): 9,
    (15,1,14): 10,
    (17,1,16): 12,
    (27,3,24): 12,
    (29,4,25): 12,
    (31,5,26):12,
    (33,5,28):12,
    (35,6,29):12,
    (37,7,30):15,
    (39,8,31):15,
    (41,6,35):20,
    (45,10,35):20,
    (47,8,39):22
}
