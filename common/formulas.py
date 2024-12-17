"""TODO"""
from math import ceil, floor, sqrt
import common.data_constants as data_const


def calculate_threshold (n, margin_coeff):
    """
    Calculate the lower and higher thresholds on the
    Hamming distance between an SRAM pattern and a codeword
    
    Args:
        n (int): codeword length
        margin_coeff (List[float]): [mean_coeff, sd_coeff]

    Returns:
        List[int]: [lower_threshold, higher_threshold]
    """
    mean_coeff = margin_coeff[0]
    sd_coeff = margin_coeff[1]

    # The event of bit flipping within n SRAM cells follows a Binomial distrubution

    # The mean of binom distribution for n cells
    mean_err = n * data_const.P_FLIP
    # The standard deviation for binom distribution for n cells
    sd_err = sqrt(n * data_const.P_FLIP * (1 - data_const.P_FLIP))

    # the margin of bit flips that will determine the threshold
    margin_err = ceil(mean_coeff * mean_err + sd_coeff * sd_err)

    # the lower threshold
    min_threshold = floor(n * data_const.P_SRAM) - margin_err

    # the higher
    max_threshold = ceil(n * data_const.P_SRAM) + margin_err

    # threshold
    threshold = [min_threshold, max_threshold]

    if min_threshold < 0 or max_threshold > n:
        return [-1,-1], -1

    return threshold, margin_err