"""
This script implements the formulas to calculate thresholds, error
probability, selection probability and memory requirements of TMVS.
"""
import logging
from math import comb, ceil, log2
from scipy.stats import binom
import numpy as np
import tmvs.analysis_constants as const
import common.data_constants as data_const


def theoretical_selection_probability (n, select_threshold, codebook_size):
    """
    Calculate the theoretical selection probability of TMVS

    Args:
        n (int): codeword length
        select_threshold (List[int]): [lower_threshold, higher_threshold]
        codebook_size (int): number of codewords

    Returns:
        float: selection probability
    """

    p_sram = data_const.P_SRAM
    min_threshold = select_threshold[0]
    max_threshold = select_threshold[1]

    if n % 2 == 1: # a sequence should have odd length to avoid ties

        # an SRAM pattren is selected with respect to a codeword if
        # [hamming distance <= min_threshold] or [hamming distance >= max_threshold]
        # note: the probability of the two events is equal when
        # the memory is unbiased and p_sram = 0.5
        p_select = codebook_size * (binom.cdf(min_threshold, n, p_sram)
                                      + binom.cdf(n-max_threshold, n, 1-p_sram))
        return p_select

    logging.info("Even codeword length !")


def theoretical_error_probability (n, select_threshold, p_flip):
    """
    Calculate the theoretical error decoding probability of TMVS

    Args:
        n (int): codeword length
        select_threshold (List[float]): [lower_threshold, higher_threshold]

    Returns:
        float: error probability
    """

    p_error = 0
    p_sram = data_const.P_SRAM
    min_threshold = select_threshold[0]
    max_threshold = select_threshold[1]

    if n % 2 == 1: # a sequence should have odd length to avoid ties
        hd_e_start = int(np.ceil(n * p_sram))
        for hd_e in range(hd_e_start, n + 1):
            # using lower threshold
            for hd_0 in range(min_threshold+1):
                p_hd_0 =  binom.pmf(hd_0, n, p_sram) / (binom.cdf(min_threshold, n, p_sram)
                                      + binom.cdf(n-max_threshold, n, 1-p_sram))
                for k in range(int(np.floor((n-hd_e+hd_0)/2))+1):
                    p_error +=  (p_flip**(hd_e-hd_0+2*k) * (1 - p_flip)**(n- (hd_e-hd_0+2*k))
                                * comb(n-hd_0,hd_e-hd_0+k) * comb(hd_0,k) * p_hd_0)
            # using higher threshold
            for hd_0 in range(n-max_threshold+1):
                p_hd_0 =  binom.pmf(hd_0, n, p_sram) / (binom.cdf(min_threshold, n, p_sram)
                                      + binom.cdf(n-max_threshold, n, 1-p_sram))
                for k in range(int(np.floor((n-hd_e+hd_0)/2))+1):
                    p_error +=  (p_flip**(hd_e-hd_0+2*k) * (1 - p_flip)**(n- (hd_e-hd_0+2*k))
                                * comb(n-hd_0,hd_e-hd_0+k) * comb(hd_0,k) * p_hd_0)
        return p_error
    logging.info("Even codeword length !")


def key_failure_probability (p_error):
    """
    Calculate the theoretical key failure probability of TMVS

    Args:
        p_error (float): bit error decoding probability

    Returns:
        float: failure probability
    """

    p_failure = 1 - (1 - p_error) ** const.KEY_LENGTH
    return p_failure


def theoretical_required_sram_size (n, select_threshold, codebook_size):
    """
    Calculate theoretically the amount of SRAM source bits required to extract a secret key

    Args:
        n (int): codeword length
        select_threshold (List[float]): [lower_threshold, higher_threshold]
        codebook_size (int): number of codewords

    Returns:
        float: SRAM memory size (kB)
    """

    # By definition p_select = nb_extracted_bits / (nb_extracted_bits + nb_skipped_patterns)
    p_select = theoretical_selection_probability (n, select_threshold, codebook_size)

    # From the p_select formula we deduce the avg. number of skipped sequence per key bit
    # (nb_skipped_patterns / nb_extracted_bits) = 1 / p_select - 1
    skipping_ratio = 1 / p_select - 1

    # required SRAM size to extract KEY_LENGTH bits is
    # for each bit: the number of skipped bits + pattern length
    number_sram_bits = const.KEY_LENGTH * (n + skipping_ratio)

    return number_sram_bits / (8 * 1024) # output in kB


def theoretical_required_helper_data_size (n,  select_threshold, codebook_size):
    """
    Calculate theoretically the amount of helper data bits required to extract a secret key

    Args:
        n (int): codeword length
        select_threshold (List[float]): [lower_threshold, higher_threshold]
        codebook_size (int): number of codewords

    Returns:
        float: SRAM memory size (kB)
    """

    # memory size required to save addresses of selected SRAM patterns
    number_sram_bits = theoretical_required_sram_size (n, select_threshold, codebook_size)
    number_sram_bits *= (8 * 1024)
    # assuming the pointer requires 32 bits
    patterns_addr_memory_size = 32 + (const.KEY_LENGTH-1) * ceil(log2(number_sram_bits))

    # memory size required to save indices of corresponding codewords
    # assuming the pointer requires 32 bits
    codewords_indices_memory_size = 32 + const.KEY_LENGTH * ceil(log2(codebook_size))

    helper_data_size = codewords_indices_memory_size + patterns_addr_memory_size

    return helper_data_size / (8 * 1024) # output in kB


def required_codebook_size (n, codebook_size):
    """
    Calculate theoretically the amount of codebook bits required to extract a secret key

    Args:
        n (int): codeword length
        codebook_size (int): number of codewords

    Returns:
        float: SRAM memory size (kB)
    """

    return n * codebook_size / (8 * 1024) # output in kB
