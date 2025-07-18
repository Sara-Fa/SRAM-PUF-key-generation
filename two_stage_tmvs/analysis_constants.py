""" Constants for Two-Stage TMVS Analysis"""
from pathlib import Path

ADDRESS_SIZE = 32 # Size of the pointer address in bits
CODE_PARAMS = [(7, 1, 6), (9, 1, 8), (11, 1, 10), (11, 2, 9), (13, 1, 12),
                    (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25),
                    (31, 5, 26), (33, 5, 28), (35, 6, 29), (37, 7, 30), (39, 8, 31),
                    (41, 6, 35), (45, 10, 35), (47, 8, 39)] 
# CODE_PARAMS = [(8, 1, 7), (16, 3, 13), (32, 7, 25)] # hamming codes
FLIPPING_PROBS = [0.05, 0.075, 0.085, 0.10, 0.125, 0.15]
KEY_LENGTH = 128
TARGET_ERROR_PROB = 7.81e-9  # Target error probability for optimal configurations
MIN_TARGET_ERROR_PROB = 1e-20  # Minimum target error probability for configurations
MAX_CODE_LENGTH = 35 # Maximum code length for the codes used in the heatmap
PLOT_DIR = Path("two_stage_tmvs/results/plots")
RESULTS_DIR = Path("two_stage_tmvs/results")
