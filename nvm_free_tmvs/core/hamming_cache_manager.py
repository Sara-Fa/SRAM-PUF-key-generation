""" Module for managing cached Hamming distances. """
import numpy as np
# import h5py
from nvm_free_tmvs.utils.file_manager import hamming_distances_dir

class CacheManager:
    """
    Class for managing cached Hamming distances.
    """
    def __init__(self):
        """
        Initialize the CacheManager and ensure the directory exists.
        """
        hamming_distances_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    @staticmethod
    def _get_cache_file_path(code_length, select_threshold, chip_id):
        """
        Generate a unique file path based on parameters.
        """
        filename = (
            f"hamming_distances_chip{chip_id}_N{code_length}"
            # f"_TH_low_{select_threshold[0]}_TH_high_{select_threshold[1]}.h5"
            f"_TH_low_{select_threshold[0]}_TH_high_{select_threshold[1]}.npz"
        )
        filepath = hamming_distances_dir / filename
        return filepath

    def load_cache(self, code_length, select_threshold, chip_id):
        """
        Load cached Hamming distances if they exist.
        """
        filepath = self._get_cache_file_path(code_length, select_threshold, chip_id)
        if filepath.exists():
            try:
                with np.load(filepath) as data:
                    return data['hamming_distances']
                # with h5py.File(filepath, "r") as h5file:
                #     return h5file["hamming_distances"][:]
            except (OSError, ValueError) as e:
                print(f"Error loading cache from {filepath}: {e}")
                return None
        return None

    # def save_cache_incrementally(self, code_length, select_threshold, chip_id, dataset_shape, chunk_size):
    #         """
    #         Create an HDF5 file for incremental saving of Hamming distances.
    #         """
    #         filepath = self._get_cache_file_path(code_length, select_threshold, chip_id)
    #         h5_file = h5py.File(filepath, "w")
    #         h5_file.create_dataset(
    #             "hamming_distances",
    #             shape=dataset_shape,
    #             dtype="uint8",
    #             compression="gzip",
    #             compression_opts=9,
    #             chunks=chunk_size,
    #         )
    #         print(f"Hamming distances will be incrementally cached at {filepath}")
    #         return h5_file

    def save_cache(self, code_length, select_threshold, chip_id, data):
        """
        Save Hamming distances to cache.
        """
        filepath = self._get_cache_file_path(code_length, select_threshold, chip_id)
        try:
            np.savez_compressed(filepath, hamming_distances=data)
            # with h5py.File(filepath, "w") as h5file:
            #     h5file.create_dataset(
            #         "hamming_distances",
            #         data=data,
            #         compression="gzip",
            #         compression_opts=9
            #     )
            print(f"Hamming distances cached at {filepath}")
        except (OSError, ValueError, PermissionError) as e:
            print(f"Error saving cache to {filepath}: {e}")
