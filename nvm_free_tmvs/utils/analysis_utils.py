""" Helper functions for analysis. """
import numpy as np
import common.data_constants as data_const
import nvm_free_tmvs.analysis_constants as const


def get_enrollment_ranges():
    """ Get the ranges for enrollment. """
    enroll_ranges = [(x, x + const.MAX_ENROLLMENT_READINGS)
                        for x in range(0, data_const.READINGS_TO_ANALYZE,
                                    const.MAX_ENROLLMENT_READINGS)]
    return np.array(enroll_ranges, dtype=np.uint16)

def get_optimal_data_type (num_items: int):
    """
    Get an optimal data type for a variable based on the number of items
    """
    max_items = num_items - 1
    
    if max_items < 2**8:
        return np.uint8
    if max_items < 2**16:
        return np.uint16
    if max_items < 2**32:
        return np.uint32
    return np.uint64

def get_shifted_selection_threshold (code_length: int, select_threshold: list):
    """
    Get the shifted selection threshold based on the margin coefficient
    """
    modified_threshold = select_threshold.copy()
    floor_half = int(code_length * data_const.P_SRAM)
    modified_threshold[0] = round(select_threshold[0] - floor_half - 1,2)
    modified_threshold[1] = round(select_threshold[1] - floor_half,2)
    return modified_threshold

def get_enrollment_threshold_values(code_length: int, cb_coeff=None, trivial=True):
    """Get the enrollment threshold values.

    Parameters
    ----------
    code_length : int
        Pattern width n.
    cb_coeff : list, optional
        Codebook coefficients [cb_low, cb_high].  Only used when
        ``trivial=False`` to enforce the uniqueness lower bound.
    trivial : bool
        If True (default), sweep from 0 (trivial codebook — uniqueness
        guaranteed by complementary codewords).  If False, the lower
        bound is th*_high_code = get_shifted_selection_threshold(n,
        [cb_low, cb_high])[1] to ensure a pattern can match at most one
        codeword.
    """
    threshold_values_list = []
    floor_half = int(code_length * data_const.P_SRAM)

    step_size = const.THRESHOLD_STEP_SIZE

    if trivial or cb_coeff is None:
        # Trivial codebook: sweep full range (uniqueness guaranteed)
        start_range = 0
    else:
        # Non-trivial: start from th*_high_code (codebook uniqueness bound)
        th_high_code_star = get_shifted_selection_threshold(
            code_length, [cb_coeff[0], cb_coeff[1]])[1]
        start_range = max(0, th_high_code_star)

    # For trivial codebooks, extend past floor_half so the sweep reaches
    # threshold [0, 0].  Mean d* is a float (averaged over NrRead), so
    # thresholds between [-1, 1] and [0, 0] are meaningful — they select
    # patterns whose mean d* is very close to zero.
    end_range = floor_half + 1 if (trivial or cb_coeff is None) else floor_half
    select_margin = np.round(np.arange(start_range, end_range + step_size, step_size),
                        decimals=1)

    for i in select_margin:
        thresholds = [i, code_length - i]
        threshold_values_list.append(
            get_shifted_selection_threshold(code_length, thresholds))
    return select_margin, threshold_values_list