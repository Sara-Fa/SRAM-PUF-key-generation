""" 
Generates Reed-Muller (RM) codes and provides utility functions for codeword generation,
filtering, and threshold calculation.
"""
import itertools
import numpy as np

# --------- RM(1, m) Codeword Generation ---------
def generate_rm_1_m_code(m: int) -> np.ndarray:
    """Generates RM(1, m) code of length 2^m."""
    n = 2 ** m
    codewords = []
    for coeffs in itertools.product([0, 1], repeat=m + 1):
        a = np.array(coeffs[1:], dtype=int)
        b = coeffs[0]
        x = np.array([list(map(int, f"{i:0{m}b}")) for i in range(n)], dtype=int)
        y = (np.dot(x, a) + b) % 2
        codewords.append(y)
    return np.array(codewords)

def remove_complements(codewords: np.ndarray) -> np.ndarray:
    """Removes codewords that are complements of others."""
    seen = set()
    unique = []
    for cw in codewords:
        t = tuple(cw)
        comp = tuple(1 - cw)
        if comp not in seen:
            seen.add(t)
            unique.append(cw)
    return np.array(unique)

def remove_zero_and_ones_codeword(codewords: np.ndarray) -> np.ndarray:
    """Remove the all-zero codeword from the codewords array."""
    return np.array([cw for cw in codewords if not (np.all(cw == 0) or np.all(cw == 1))])

def derive_RM_codebook_size(n: int) -> int:
    """Calculates the size of the RM(1, m) codebook."""
    # n = 2 ** m  # Length of the codewords
    return n-1  # Number of unique pairs of codewords

def new_threshold(m):
    """Calculates the new threshold values for RM(1, m) codes."""
    n = 2 ** m  # Length of the codewords
    margin = (n/2-1) // 2  # Number of errors
    TH_low = margin
    TH_high = n - TH_low
    return int(TH_low), int(TH_high)

def RM_codebook(m):
    """Generates the RM(1, m) codebook."""
    codewords = generate_rm_1_m_code(m)
    codewords = remove_zero_and_ones_codeword(codewords)
    list_codewords = remove_complements(codewords)
    TH_low, TH_high = new_threshold(m)
    return TH_low, TH_high, list_codewords
