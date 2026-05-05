""" Module for saving and loading the output of BERAnalysis class. """
import h5py
import numpy as np
from nvm_free_tmvs.utils.file_manager import enroll_comparator_dir


class ComparatorCacheManager:
    """
    Class for saving and loading the output of compare_helper_data function.
    """

    def __init__(self):
        """
        Initialize the ComparatorCacheManager and ensure the directory exists.
        """
        enroll_comparator_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    @staticmethod
    def get_cache_file_path(chip_id, select_threshold, code_length,
                            num_enroll_readings, trivial=False) -> str:
        """
        Generate a filename based on parameters.
        """
        trivial_tag = "_trivial" if trivial else ""
        filename = (
            f"enrollment_comparator_chip_{chip_id}_code_N{code_length}"
            f"_Threshold_{select_threshold[0]}_{select_threshold[1]}"
            f"_MaxEnrollReadings_{num_enroll_readings}{trivial_tag}.h5"
        )
        return enroll_comparator_dir / filename

    @staticmethod
    def get_group_name(enroll_select_threshold) -> str:
        """
        Generate a group name based on the enroll_select_threshold value.
        """
        return f"threshold_{enroll_select_threshold[0]:.1f}_{enroll_select_threshold[1]:.1f}".replace(".", "_")

    def save_incremental_cache(self, chip_id, select_threshold, code_length,
                               num_enroll_readings, enroll_select_threshold,
                               error_count, discarded_patterns_count,
                               zero_key_bits_count, one_key_bits_count,
                               trivial=False):
        """
        Save the results of the compare_helper_data function to an .h5 file incrementally.
        """
        file_path = self.get_cache_file_path(
            chip_id, select_threshold, code_length,
            num_enroll_readings, trivial=trivial)
        group_name = self.get_group_name(enroll_select_threshold)

        # Combine the arrays into a single 3D array
        combined_data = np.stack(
            [error_count, discarded_patterns_count, zero_key_bits_count, one_key_bits_count], axis=0
        )  # Shape: (4, rows, cols)

        # Open the file in append mode to allow incremental writes
        with h5py.File(file_path, "a") as hf:
            # Check if the group already exists
            if group_name in hf:
                del hf[group_name]
                print(f"Overwriting existing results for threshold {enroll_select_threshold}.")
                # print(f"Results for threshold {enroll_select_threshold} already exist. "
                #       f"Skipping save.")
                # return
            try:
                # Create a new group for this threshold and save results
                group = hf.create_group(group_name)
                group.create_dataset("combined_data", data=combined_data, dtype=np.uint32,
                                     compression="gzip", compression_opts=9,)
                print(f"Incremental results saved for threshold {enroll_select_threshold} "
                      f"{file_path}.")
            except (OSError, ValueError, PermissionError) as e:
                print(f"Error saving data: {e}")

    # def save_cache(self, chip_id, select_threshold, code_length, num_enroll_readings,
    #                enroll_ranges, enroll_threshold_values, error_count, discarded_patterns_count):
    #     """
    #     Save the results of the compare_helper_data function to an .h5 file.
    #     """
    #     file_path = self.get_cache_file_path(chip_id, select_threshold,
    #                                           code_length, num_enroll_readings)
    #     with h5py.File(file_path, "w") as hf:
    #         try:
    #             hf.create_dataset("enroll_ranges", data=np.array(enroll_ranges))
    #             hf.create_dataset("enroll_threshold_values", data=np.array(enroll_threshold_values))
    #             hf.create_dataset("error_count", data=error_count, dtype=np.uint32)
    #             hf.create_dataset("discarded_patterns_count", data=discarded_patterns_count,
    #                               dtype=np.uint32)
    #         except (OSError, ValueError, PermissionError) as e:
    #             print(f"Error saving data: {e}")
    #     print(f"Results saved to {file_path}")
   
    def load_cache(self, chip_id, select_threshold, code_length,
                   num_enroll_readings, trivial=False):
        """
        Load the results of the compare_helper_data function from an .h5 file.
        Returns None for all datasets if the file is not found.
        """
        file_path = self.get_cache_file_path(
            chip_id, select_threshold, code_length,
            num_enroll_readings, trivial=trivial)

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
                    combined_data = hf[group_name]["combined_data"][:]  # Shape: (4, rows, cols)
                    (error_count, discarded_patterns_count, zero_key_bits_count,
                     one_key_bits_count) = combined_data
                    # results[tuple(enroll_select_threshold)] = {
                    #     "results": hf[group_name]["results"][:],
                    #     # "error_count": hf[group_name]["error_count"][:],
                    #     # "discarded_patterns_count": hf[group_name]["discarded_patterns_count"][:],
                    #     # "zero_key_bits_count": hf[group_name]["zero_key_bits_count"][:],
                    #     # "one_key_bits_count": hf[group_name]["one_key_bits_count"][:],
                    # }
                    results[enroll_select_threshold] = {
                    "error_count": error_count,
                    "discarded_patterns_count": discarded_patterns_count,
                    "zero_key_bits_count": zero_key_bits_count,
                    "one_key_bits_count": one_key_bits_count,
                    }
                except (OSError, ValueError) as e:
                    print(f"Error loading data: {e}")
                    return None
            print(f"Cache loaded successfully from {file_path}.")
            return results

    # def load_cache(self, chip_id, select_threshold, code_length, num_enroll_readings):
    #     """
    #     Load the results of the compare_helper_data function from an .h5 file.
    #     Returns None for all datasets if the file is not found.
    #     """
    #     file_path = self.get_cache_file_path(chip_id, select_threshold,
    #                                           code_length, num_enroll_readings)

    #     # Check if the file exists
    #     if not file_path.exists():
    #         # File does not exist, all ranges are missing
    #         return None, None, None, None

    #     with h5py.File(file_path, "r") as hf:
    #         try:
    #             enroll_ranges = hf["enroll_ranges"][:]
    #             enroll_threshold_values = hf["enroll_threshold_values"][:]
    #             error_count = hf["error_count"][:]
    #             discarded_patterns_count = hf["discarded_patterns_count"][:]
    #         except (OSError, ValueError) as e:
    #             print(f"Error loading data: {e}")
    #             return None, None, None, None
    #     print(f"Results loaded from {file_path}")
    #     return enroll_ranges, enroll_threshold_values, error_count, discarded_patterns_count

    def check_threshold_in_cache(self, chip_id, select_threshold, code_length,
                                 num_enroll_readings, enroll_select_threshold,
                                 trivial=False):
        """
        Check if results for a specific threshold exist in the cache.
        """
        file_path = self.get_cache_file_path(
            chip_id, select_threshold, code_length,
            num_enroll_readings, trivial=trivial)
        group_name = self.get_group_name(enroll_select_threshold)

        if not file_path.exists():
            return False

        with h5py.File(file_path, "r") as hf:
            return group_name in hf