""" Analysis utilities for helperless stabilizer Bernardini experiments. """
import numpy as np
from common.data_constants import READINGS_TO_ANALYZE


def get_enrollment_ranges(max_enrollment_readings: int = 100):
    """ Get the ranges for enrollment. """
    enroll_ranges = [(x, x + max_enrollment_readings)
                    for x in range(0, READINGS_TO_ANALYZE, max_enrollment_readings)]
    return np.array(enroll_ranges, dtype=np.uint16)


def get_enrollment_threshold_values(threshold_min: float = 0.05, 
                                   threshold_max: float = 0.49, 
                                   step_size: float = 0.05):
    """ Get the enrollment threshold values (delta or D values). """
    threshold_values_list = np.round(
        np.arange(threshold_min, threshold_max + step_size, step_size), 
        decimals=2
    )
    return threshold_values_list


def get_enrollment_readings_values(min_readings: int = 10, 
                                 max_readings: int = 200, 
                                 step_size: int = 10):
    """ Get the enrollment readings values (K or N values). """
    readings_values_list = np.arange(min_readings, max_readings + step_size, step_size)
    return readings_values_list
