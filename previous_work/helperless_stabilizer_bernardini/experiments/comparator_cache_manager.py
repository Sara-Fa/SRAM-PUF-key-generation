""" Cache manager for helperless stabilizer Bernardini experiments. """
import os
import pickle
from typing import Dict, Optional
import numpy as np


class ComparatorCacheManager:
    """ Cache manager for helper data comparator results (grouped per (K, delta)). """

    def __init__(self, cache_dir: str = "previous_work/helperless_stabilizer_bernardini/experiments/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _file_path(self, chip_id: str, K: int, delta: float, use_equal_ranges: bool) -> str:
        delta_str = f"{float(delta):.3f}".replace('.', 'p')
        mode = "equal" if use_equal_ranges else "keqne_series"
        return os.path.join(self.cache_dir, f"enroll_comparator_{chip_id}_K{int(K)}_delta{delta_str}_{mode}.pkl")

    def _load_all(self, chip_id: str, K: int, delta: float, use_equal_ranges: bool) -> Dict:
        path = self._file_path(chip_id, K, delta, use_equal_ranges)
        if not os.path.exists(path):
            return {}
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _save_all(self, chip_id: str, K: int, delta: float, use_equal_ranges: bool, data: Dict) -> None:
        path = self._file_path(chip_id, K, delta, use_equal_ranges)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def entry_exists(self, chip_id: str, K: int, delta: float, use_equal_ranges: bool, key) -> bool:
        data = self._load_all(chip_id, K, delta, use_equal_ranges)
        return key in data

    def check_threshold_in_cache(self, chip_id: str, threshold: float, num_enroll_readings: int) -> bool:
        """ Check if threshold exists in cache for equal ranges mode. """
        # For equal ranges mode, we check if the file exists
        path = self._file_path(chip_id, num_enroll_readings, threshold, True)
        return os.path.exists(path)

    def save_equal_ranges(self, chip_id: str, K: int, delta: float, D: float,
                          error_count: np.ndarray, accepted_cells_count: np.ndarray,
                          zero_key_bits_count: np.ndarray, one_key_bits_count: np.ndarray) -> None:
        data = self._load_all(chip_id, K, delta, True)
        data[float(D)] = {
            "error_count": error_count,
            "accepted_cells_count": accepted_cells_count,
            "zero_key_bits_count": zero_key_bits_count,
            "one_key_bits_count": one_key_bits_count,
        }
        self._save_all(chip_id, K, delta, True, data)

    def save_keqne_series(self, chip_id: str, K: int, delta: float, N: int, D: float,
                          accepted_cells_ref: float, zero_key_bits_ref: float, one_key_bits_ref: float,
                          enrollment_ber_error_counts: np.ndarray,
                          enrollment_ber_rates: np.ndarray) -> None:
        data = self._load_all(chip_id, K, delta, False)
        data[(int(N), float(D))] = {
            "accepted_cells_ref": float(accepted_cells_ref),
            "zero_key_bits_ref": float(zero_key_bits_ref),
            "one_key_bits_ref": float(one_key_bits_ref),
            "enrollment_ber_error_counts": enrollment_ber_error_counts,
            "enrollment_ber_rates": enrollment_ber_rates,
        }
        self._save_all(chip_id, K, delta, False, data)

    def load_all(self, chip_id: str, K: int, delta: float, use_equal_ranges: bool) -> Optional[Dict]:
        data = self._load_all(chip_id, K, delta, use_equal_ranges)
        return data if data else None

    def load_cache(self, chip_id: str, num_enroll_readings: int) -> Optional[Dict]:
        """ Load all cached data for a chip and enrollment readings (equal ranges mode). """
        results = {}
        cache_files = [f for f in os.listdir(self.cache_dir) 
                      if f.startswith(f"enroll_comparator_{chip_id}_") and f.endswith("_equalTrue.pkl")]
        
        if not cache_files:
            return None
        
        for cache_file in cache_files:
            with open(os.path.join(self.cache_dir, cache_file), 'rb') as f:
                cache_data = pickle.load(f)
                # Extract threshold from filename or use a default key
                for threshold_key, data in cache_data.items():
                    results[threshold_key] = data
        
        return results if results else None

    def _incremental_ber_file_path(self, chip_id: str, K: int, delta: float, D: float) -> str:
        """ Get file path for incremental enrollment BER cache. """
        delta_str = f"{float(delta):.3f}".replace('.', 'p')
        D_str = f"{float(D):.3f}".replace('.', 'p')
        return os.path.join(
            self.cache_dir, 
            f"incremental_enroll_ber_{chip_id}_K{int(K)}_delta{delta_str}_D{D_str}.pkl"
        )

    def save_incremental_enrollment_ber(self, chip_id: str, K: int, delta: float, D: float,
                                       error_count: np.ndarray, discarded_patterns_count: np.ndarray) -> None:
        """
        Save incremental enrollment BER results to cache.
        
        Args:
            chip_id: Chip identifier
            K: Number of readings per range
            delta: Reference threshold
            D: Test threshold
            error_count: (num_ranges, K) array of error counts
            discarded_patterns_count: (num_ranges, K) array of discarded patterns counts
        """
        path = self._incremental_ber_file_path(chip_id, K, delta, D)
        cache_data = {
            "error_count": error_count,
            "discarded_patterns_count": discarded_patterns_count,
            "K": int(K),
            "delta": float(delta),
            "D": float(D)
        }
        with open(path, 'wb') as f:
            pickle.dump(cache_data, f)

    def load_incremental_enrollment_ber(self, chip_id: str, K: int, delta: float, D: float) -> Optional[Dict]:
        """
        Load incremental enrollment BER results from cache.
        
        Returns:
            Dict with 'error_count' and 'discarded_patterns_count' if found, None otherwise
        """
        path = self._incremental_ber_file_path(chip_id, K, delta, D)
        if not os.path.exists(path):
            return None
        
        with open(path, 'rb') as f:
            cache_data = pickle.load(f)
            return {
                "error_count": cache_data["error_count"],
                "discarded_patterns_count": cache_data["discarded_patterns_count"]
            }


class BERCacheManager:
    """ Cache manager for BER processor results. """
    
    def __init__(self, cache_dir: str = "previous_work/helperless_stabilizer_bernardini/experiments/cache"):
        """ Initialize the cache manager. """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_filename(self, chip_id: str, threshold: float, num_enroll_readings: int) -> str:
        """ Get the cache filename for given parameters. """
        threshold_str = f"{threshold:.3f}".replace('.', 'p')
        return os.path.join(self.cache_dir, f"regenerate_ber_{chip_id}_th{threshold_str}_num_readings{num_enroll_readings}.pkl")
    
    def check_threshold_in_cache(self, chip_id: str, threshold: float, 
                               num_enroll_readings: int) -> bool:
        """ Check if threshold exists in cache. """
        cache_file = self._get_cache_filename(chip_id, threshold, num_enroll_readings)
        return os.path.exists(cache_file)
    
    def save_incremental_cache(self, chip_id: str, threshold: float, num_enroll_readings: int,
                             error_count: np.ndarray, valid_patterns_count: np.ndarray):
        """ Save incremental cache for a single threshold. """
        cache_file = self._get_cache_filename(chip_id, threshold, num_enroll_readings)
        
        cache_data = {
            "error_count": error_count,
            "valid_patterns_count": valid_patterns_count,
            "threshold": threshold,
            "num_enroll_readings": num_enroll_readings
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def load_cache(self, chip_id: str, num_enroll_readings: int) -> Optional[Dict]:
        """ Load all cached data for a chip and enrollment readings. """
        results = {}
        cache_files = [f for f in os.listdir(self.cache_dir) 
                      if f.startswith(f"regenerate_ber_{chip_id}_") and f.endswith(f"_num_readings{num_enroll_readings}.pkl")]
        
        if not cache_files:
            return None
        
        for cache_file in cache_files:
            with open(os.path.join(self.cache_dir, cache_file), 'rb') as f:
                cache_data = pickle.load(f)
                threshold = cache_data["threshold"]
                results[threshold] = {
                    "error_count": cache_data["error_count"],
                    "valid_patterns_count": cache_data["valid_patterns_count"]
                }
        
        return results if results else None
