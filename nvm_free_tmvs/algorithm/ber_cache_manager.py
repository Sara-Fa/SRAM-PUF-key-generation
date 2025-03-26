""" Module for saving and loading the output of BERAnalysis class. """
import numpy as np
from nvm_free_tmvs.utils.file_manager import ber_comparator_dir


class BERCacheManager:
    """
    Class for saving and loading the output of calculate_ber function.
    """

    def __init__(self):
        """
        Initialize the BERCacheManager and ensure the directory exists.
        """
        ber_comparator_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    @staticmethod
    def _get_cache_file_path(chip_id, select_threshold,
                             code_length, data_start_idx,
                             num_enroll_readings) -> str:
        """
        Generate a filename based on parameters.
        """
        filename = (
            f"ber_chip{chip_id}_N{code_length}"
            f"_TH_low_{select_threshold[0]}_TH_high_{select_threshold[1]}"
            f"_enroll_{data_start_idx}_{data_start_idx+num_enroll_readings}.npz"
        )
        filepath = ber_comparator_dir / filename
        return filepath

    def save_cache(self, chip_id, select_threshold ,code_length,
                   data_start_idx, num_enroll_readings,
                   error_counts, update_counts):
        """
        Save ber results in .npz file. Checks if the file already exists.
        """
        file_path = self._get_cache_file_path(chip_id, select_threshold ,code_length,
                                              data_start_idx, num_enroll_readings)
        # Check if file exists
        if file_path.exists():
            print(f"Warning: File {file_path} already exists. Skipping save.")
        else:
            try:
                np.savez(file_path,
                        error_counts=error_counts,
                        update_counts=update_counts
                        )
                # print(f"BER results cached at {file_path}")
            except (OSError, ValueError, PermissionError) as e:
                print(f"Error saving cache to {file_path}: {e}")

    def load_cache(self, chip_id, select_threshold ,code_length,
                   data_start_idx, num_enroll_readings):
        """
        Load cached ber results if they exist.
        """
        file_path = self._get_cache_file_path(chip_id, select_threshold ,code_length,
                                                data_start_idx, num_enroll_readings)
        if file_path.exists():
            try:
                with np.load(file_path) as data:
                    error_counts = data['error_counts']
                    update_counts = data['update_counts']
                    # print(f"BER results loaded from {file_path}")
                    return error_counts, update_counts
            except (OSError, ValueError) as e:
                print(f"Error loading cache from {file_path}: {e}")
                return None
        return None
