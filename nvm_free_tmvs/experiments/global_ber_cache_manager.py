""" Module for saving and loading the output of BERAnalysis class. """
import h5py
import numpy as np
from nvm_free_tmvs.utils.file_manager import ber_comparator_dir


class BERCacheManager:
    """
    Class for saving and loading the output of compute_and_save_global_ber function.
    """

    def __init__(self):
        """
        Initialize the BERCacheManager and ensure the directory exists.
        """
        ber_comparator_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    @staticmethod
    def _get_cache_file_path(chip_id, select_threshold, code_length,
                             num_enroll_readings, trivial=False) -> str:
        """
        Generate a filename based on parameters.
        """
        trivial_tag = "_trivial" if trivial else ""
        filename = (
            f"ber_comparator_chip{chip_id}_N{code_length}"
            f"_Threshold_{select_threshold[0]}_{select_threshold[1]}"
            f"_MaxEnrollReadings_{num_enroll_readings}{trivial_tag}.h5"
        )
        return ber_comparator_dir / filename

    @staticmethod
    def _get_group_name(enroll_select_threshold) -> str:
        """
        Generate a group name based on the enroll_select_threshold value.
        """
        return f"threshold_{enroll_select_threshold[0]:.1f}_{enroll_select_threshold[1]:.1f}".replace(".", "_")

    def save_incremental_cache(self, chip_id, select_threshold, code_length,
                               num_enroll_readings, enroll_select_threshold,
                               error_count, valid_patterns_count, trivial=False):
        """
        Save ber results of the compute_and_save_global_ber function.
        """
        file_path = self._get_cache_file_path(chip_id, select_threshold,
                                              code_length, num_enroll_readings,
                                              trivial=trivial)
        group_name = self._get_group_name(enroll_select_threshold)

        # Combine the arrays into a single 3D array
        combined_data = np.stack(
            [error_count, valid_patterns_count], axis=0
            )  # Shape: (2, rows, cols)

        # Open the file in append mode to allow incremental writes
        with h5py.File(file_path, "a") as hdf:
            # Check if the group already exists
            if group_name in hdf:
                print(f"Results for threshold {enroll_select_threshold} already exist. "
                      f"Skipping save.")
                return
            try:
                # Create a new group for this threshold and save results
                group = hdf.create_group(group_name)
                group.create_dataset("combined_data", data=combined_data, dtype=np.uint32,
                                     compression="gzip", compression_opts=9,)
                print(f"Incremental results saved for threshold {enroll_select_threshold} "
                      f"in {file_path}.")
            except (OSError, ValueError, PermissionError) as e:
                print(f"Error saving data: {e}")

    def load_cache(self, chip_id, select_threshold, code_length,
                   num_enroll_readings, trivial=False):
        """
        Load the results of the compute_and_save_global_ber function from an .h5 file.
        Returns None for all datasets if the file is not found.
        """
        file_path = self._get_cache_file_path(chip_id, select_threshold,
                                              code_length, num_enroll_readings,
                                              trivial=trivial)
        # Check if the file exists
        if not file_path.exists():
            # File does not exist, all ranges are missing
            print(f"No cache file found at {file_path}.")
            return None

        # Open the file in read mode and load data
        with h5py.File(file_path, "r") as hf:
            all_groups = list(hf.keys())
            if not all_groups:
                print(f"No valid data found in cache file {file_path}.")
                return None
            # Load all groups (results for each threshold)
            results = {}
            for group_name in all_groups:
                # Split the string and extract the relevant parts from the group name
                parts = group_name.split("_")[1:]
                # Combine the numbers to form floats
                enroll_select_threshold = (float(parts[0] + '.' + parts[1]),
                                           float(parts[2] + '.' + parts[3]))
                try:
                    # Load the combined data and unpack it into separate arrays
                    combined_data = hf[group_name]["combined_data"][:]  # Shape: (2, rows, cols)
                    (error_count, valid_patterns_count) = combined_data
                    results[enroll_select_threshold] = {
                        "error_count": error_count,
                        "valid_patterns_count": valid_patterns_count
                    }
                except (OSError, ValueError) as e:
                    print(f"Error loading data: {e}")
                    return None
            print(f"Cache loaded successfully from {file_path}.")
            return results

    def check_threshold_in_cache(self, chip_id, select_threshold, code_length,
                                 num_enroll_readings, enroll_select_threshold,
                                 trivial=False):
        """
        Check if results for a specific threshold exist in the cache.
        """
        file_path = self._get_cache_file_path(
            chip_id, select_threshold, code_length, num_enroll_readings,
            trivial=trivial)
        group_name = self._get_group_name(enroll_select_threshold)

        if not file_path.exists():
            return False

        with h5py.File(file_path, "r") as hf:
            return group_name in hf

    # def affirm_all_ranges_saved(self, chip_id, select_threshold, code_length,
    #                             num_enroll_readings, enroll_ranges):
    #     """
    #     Check if all ranges in the given list are saved in the HDF5 file.
    #     If the file does not exist, all ranges are returned as missing.
    #     """
    #     file_path = self._get_cache_file_path(chip_id, select_threshold,
    #                                           code_length, num_enroll_readings)
    #      # Check if the file exists
    #     if not file_path.exists():
    #         print(f"File not found: {file_path}")
    #         # File does not exist, all ranges are missing
    #         return {self._get_group_name(start_idx, end_idx)
    #                 for start_idx, end_idx in enroll_ranges}
    #     with h5py.File(file_path, "r") as hdf:
    #         saved_ranges = set(hdf.keys())
    #         required_ranges = {self._get_group_name(start_idx, end_idx)
    #                            for start_idx, end_idx in enroll_ranges}
    #         missing_ranges = required_ranges - saved_ranges
    #         return missing_ranges

    # def get_data_or_missing_ranges(self, chip_id, select_threshold, code_length,
    #                                num_enroll_readings, enroll_ranges):
    #     """
    #     Check if all ranges are saved. If all are saved, return all data;
    #     otherwise, return missing ranges.
    #     """
    #     missing_ranges = self.affirm_all_ranges_saved(chip_id, select_threshold,
    #                                                code_length, num_enroll_readings,
    #                                                enroll_ranges)
    #     if not missing_ranges:  # All ranges are saved
    #         data = {}
    #         for start_idx, end_idx in enroll_ranges:
    #             loaded_data = self.load_cache(chip_id, select_threshold, code_length,
    #                                           start_idx, end_idx)
    #             if loaded_data is not None:
    #                 data[(start_idx, end_idx)] = loaded_data
    #         return data, []

    #     # Convert missing group names back to tuples of (start_idx, end_idx)
    #     missing_ranges_list = []
    #     for group_name in missing_ranges:
    #         _, _, start_idx, end_idx = group_name.split("_")
    #         missing_ranges_list.append((int(start_idx), int(end_idx)))
    #     missing_ranges_list.sort()
    #     return None, missing_ranges_list
