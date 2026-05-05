""" Global BER Processor for helperless stabilizer Bernardini approach.

This module processes Bit Error Rate (BER) data for different enrollments using
bit matrices directly.
"""
import time
from typing import List, Tuple, Dict
import numpy as np
from common.data_reading_utils import get_files, read_readouts, ReadoutList
from ..evaluate_ber import build_bit_matrix, mask_from_threshold, majority_from_matrix
from .analysis_utils import get_enrollment_ranges
from .comparator_cache_manager import BERCacheManager


class GlobalBERProcessor:
    """ Class for processing Bit Error Rate (BER) data for different enrollments. """
    
    def __init__(self, readouts: ReadoutList, threshold_values: List[float], 
                 num_enroll_readings: int, process_all_ranges: bool = True,
                 iterative_ber: bool = True, use_cache: bool = True):
        """ Initialize the GlobalBERProcessor. """
        self.readouts = readouts
        self.chip_id = readouts.chip_id
        self.threshold_values_list = threshold_values
        self.num_enroll_readings = num_enroll_readings
        self.process_all_ranges = process_all_ranges
        self.iterative_ber = iterative_ber
        self.use_cache = use_cache
        self.cache_manager = BERCacheManager()
        
        # Build bit matrix once for efficiency
        self.bit_matrix = build_bit_matrix(readouts)
        self.total_reads, self.num_cells = self.bit_matrix.shape
        # Precompute popcount LUT for packed-bit reductions
        self._popcount_lut = np.unpackbits(
            np.arange(256, dtype=np.uint8)[:, None], axis=1
        ).sum(axis=1).astype(np.uint8)

    def _errors_popcount(self, block_bool: np.ndarray, ref_bool: np.ndarray) -> np.ndarray:
        """Compute per-row Hamming errors using packed bits and a popcount LUT."""
        R = block_bool.shape[0]
        if R == 0:
            return np.array([], dtype=np.int64)
        # Cast to uint8 (0/1) and pack along cells axis
        blk = block_bool.astype(np.uint8, copy=False)
        ref = ref_bool.astype(np.uint8, copy=False)
        ref_packed = np.packbits(ref, axis=0)              # (num_bytes,)
        blk_packed = np.packbits(blk, axis=1)              # (R, num_bytes)
        x = np.bitwise_xor(blk_packed, ref_packed[None, :])
        return self._popcount_lut[x].sum(axis=1, dtype=np.int64)

    def process_enroll_range_single(self, start_enroll_idx: int, end_enroll_idx: int, 
                                  threshold: float) -> Tuple[int, int]:
        """
        Process a single enrollment range for BER calculation (non-iterative version).
        Uses existing mask_from_threshold and majority_from_matrix functions.
        
        Args:
            start_enroll_idx: Start index of enrollment range
            end_enroll_idx: End index of enrollment range  
            threshold: Threshold value (delta or D) for mask generation
            
        Returns:
            Tuple of (total_error_count, total_accepted_count)
        """
        # Create a temporary bit matrix with only the enrollment range for mask/majority computation
        enroll_window = self.bit_matrix[start_enroll_idx:end_enroll_idx]
        
        # Heldout slices
        top_all = self.bit_matrix[:start_enroll_idx]
        bottom_all = self.bit_matrix[end_enroll_idx:]

        heldout_rows = top_all.shape[0] + bottom_all.shape[0]
        if heldout_rows == 0:
            return 0, 0

        # Use existing functions to compute mask and majority from enrollment window
        # Create a temporary matrix with enrollment data for the functions
        temp_matrix = np.vstack([enroll_window, np.zeros((1, enroll_window.shape[1]), dtype=np.uint8)])
        
        # Compute mask and majority using existing functions
        mask = mask_from_threshold(temp_matrix, self.num_enroll_readings, threshold)
        majority = majority_from_matrix(temp_matrix, self.num_enroll_readings)
        
        # Get accepted cells
        accepted_idx = np.where(mask)[0]
        accepted_count = accepted_idx.size
        
        print(f"    Single computation: accepted={accepted_count}", flush=True)
        
        if accepted_count == 0:
            return 0, accepted_count

        # Get reference bits for accepted cells
        ref_bits = majority[accepted_idx]
        
        # Compute errors from top and bottom separately to avoid large vstack allocation
        errors_top = int(np.sum(top_all[:, accepted_idx] ^ ref_bits)) if top_all.shape[0] > 0 else 0
        errors_bottom = int(np.sum(bottom_all[:, accepted_idx] ^ ref_bits)) if bottom_all.shape[0] > 0 else 0
        total_errors = errors_top + errors_bottom

        # Calculate BER
        current_ber = total_errors / (heldout_rows * accepted_count) if accepted_count > 0 else 0.0
        print(f"      avg_ber={current_ber:.6e}", flush=True)

        return total_errors, accepted_count

    def process_enroll_range(self, start_enroll_idx: int, end_enroll_idx: int, 
                            threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single enrollment range for BER calculation.
        
        Args:
            start_enroll_idx: Start index of enrollment range
            end_enroll_idx: End index of enrollment range  
            threshold: Threshold value (delta or D) for mask generation
            
        Returns:
            Tuple of (error_count_per_reading, valid_patterns_count_per_reading)
        """
        if self.iterative_ber:
            return self._process_enroll_range_iterative(start_enroll_idx, end_enroll_idx, threshold)
        else:
            return self._process_enroll_range_single_wrapper(start_enroll_idx, end_enroll_idx, threshold)

    def _process_enroll_range_iterative(self, start_enroll_idx: int, end_enroll_idx: int, 
                                       threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single enrollment range for BER calculation (iterative version).
        
        Args:
            start_enroll_idx: Start index of enrollment range
            end_enroll_idx: End index of enrollment range  
            threshold: Threshold value (delta or D) for mask generation
            
        Returns:
            Tuple of (error_count_per_reading, valid_patterns_count_per_reading)
        """
        # Enrollment window (view, no copy)
        enroll_window = self.bit_matrix[start_enroll_idx:end_enroll_idx]

        # Heldout slices don't depend on iteration; we will slice columns per iteration
        top_all = self.bit_matrix[:start_enroll_idx]
        bottom_all = self.bit_matrix[end_enroll_idx:]

        # Prepare outputs over iterations 1..num_enroll_readings
        errors_per_iter = np.zeros(self.num_enroll_readings, dtype=np.int64)
        accepted_per_iter = np.zeros(self.num_enroll_readings, dtype=np.int64)

        heldout_rows = top_all.shape[0] + bottom_all.shape[0]
        if heldout_rows == 0:
            return errors_per_iter, accepted_per_iter

        # Precompute cumulative ones over the enrollment window for incremental stats
        # cum_ones[t-1] = number of ones across first t reads for each cell
        cum_ones = enroll_window.cumsum(axis=0, dtype=np.int32)
        # Iterate per enrollment iteration t
        for t in range(1, self.num_enroll_readings + 1):
            ones_t = cum_ones[t - 1]  # shape (num_cells,)
            # majority at iteration t: 1 if ones_t*2 >= t else 0
            majority_t = (ones_t * 2 >= t).astype(np.uint8)
            # acceptance mask at iteration t: |p_hat - 0.5| >= threshold -> |2*ones_t - t| >= 2*T*t
            # compute with floats to respect threshold precision
            accepted_t = (np.abs((ones_t / np.float64(t)) - 0.5) >= np.float64(threshold))
            accepted_idx_t = np.where(accepted_t)[0]
            accepted_count = accepted_idx_t.size
            accepted_per_iter[t - 1] = accepted_count
            print(f"    t={t}/{self.num_enroll_readings}, accepted={accepted_count}", flush=True)
            if accepted_count == 0:
                errors_per_iter[t - 1] = 0
                continue

            ref_t = majority_t[accepted_idx_t]
            # Compute errors from top and bottom separately to avoid large vstack allocation
            errors_top = int(np.sum(top_all[:, accepted_idx_t] ^ ref_t)) if top_all.shape[0] > 0 else 0
            errors_bottom = int(np.sum(bottom_all[:, accepted_idx_t] ^ ref_t)) if bottom_all.shape[0] > 0 else 0
            errors_per_iter[t - 1] = errors_top + errors_bottom

            # Calculate current BER for this iteration
            heldout_rows = top_all.shape[0] + bottom_all.shape[0]
            current_ber = errors_per_iter[t - 1] / (heldout_rows * accepted_count) if accepted_count > 0 else 0.0
            print(f"      partial_errors={errors_per_iter[t-1]}, avg_ber={current_ber:.6e}", flush=True)

        return errors_per_iter, accepted_per_iter

    def _process_enroll_range_single_wrapper(self, start_enroll_idx: int, end_enroll_idx: int, 
                                           threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper for single BER computation to match the iterative interface.
        
        Args:
            start_enroll_idx: Start index of enrollment range
            end_enroll_idx: End index of enrollment range  
            threshold: Threshold value (delta or D) for mask generation
            
        Returns:
            Tuple of (error_count_per_reading, valid_patterns_count_per_reading)
        """
        total_errors, total_accepted = self.process_enroll_range_single(start_enroll_idx, end_enroll_idx, threshold)
        
        # Return arrays with single value to match iterative interface
        errors_per_iter = np.array([total_errors], dtype=np.int64)
        accepted_per_iter = np.array([total_accepted], dtype=np.int64)
        
        return errors_per_iter, accepted_per_iter

    def compute_and_save_global_ber(self):
        """ Perform computation and save the global BER to the cache. """
        enroll_ranges = get_enrollment_ranges(self.num_enroll_readings)
        
        # Determine which ranges to process based on the option
        if self.process_all_ranges:
            ranges_to_process = enroll_ranges
        else:
            ranges_to_process = enroll_ranges[:1]  # Only first range

        # Guard against incomplete windows (e.g., when num_enroll_readings is large)
        # Keep only ranges fully contained within total reads
        ranges_to_process = [
            (start_idx, end_idx)
            for (start_idx, end_idx) in ranges_to_process
            if end_idx <= self.total_reads
        ]

        # Print accurate message after filtering
        if self.process_all_ranges:
            print(f"Processing all {len(ranges_to_process)} enrollment ranges")
        else:
            print("Processing only the first enrollment range (for testing/debugging)")
        
        # Initialize result arrays
        num_thresholds = len(self.threshold_values_list)
        num_ranges = len(ranges_to_process)
        
        # Determine array dimensions based on iterative vs non-iterative mode
        if self.iterative_ber:
            # Store totals per enrollment iteration (1..num_enroll_readings)
            error_count = np.zeros((num_thresholds, num_ranges, self.num_enroll_readings), dtype=np.int64)
            valid_patterns_count = np.zeros((num_thresholds, num_ranges, self.num_enroll_readings), dtype=np.int64)
            print(f"Using iterative BER computation (computing BER for each reading 1..{self.num_enroll_readings})")
        else:
            # Store single value per range (non-iterative)
            error_count = np.zeros((num_thresholds, num_ranges, 1), dtype=np.int64)
            valid_patterns_count = np.zeros((num_thresholds, num_ranges, 1), dtype=np.int64)
            print(f"Using single BER computation (computing BER once after {self.num_enroll_readings} readings)")

        results_dict: Dict[float, Dict[str, np.ndarray]] = {}

        for i, threshold in enumerate(self.threshold_values_list):
            # Check if this threshold exists in the cache (only if using cache)
            if self.use_cache and self.cache_manager.check_threshold_in_cache(self.chip_id, threshold, 
                                                         self.num_enroll_readings):
                print(f"Skipping computation for threshold {threshold}, already in cache.")
                continue

            print(f"\nProcessing threshold {i}: {threshold}")
            start_time = time.time()

            # Process each enrollment range (or just the first one)
            for range_idx, (start_enroll_idx, end_enroll_idx) in enumerate(ranges_to_process):
                print(f"Processing enrollment range {range_idx}: {start_enroll_idx} to {end_enroll_idx}")
                
                error_count_per_reading, valid_count_per_reading = self.process_enroll_range(
                    start_enroll_idx, end_enroll_idx, threshold
                )
                
                # Store per-iteration totals
                error_count[i, range_idx] = error_count_per_reading
                valid_patterns_count[i, range_idx] = valid_count_per_reading

            # Compute and print average BER using rate helpers (like p1)
            test_readings_count = self.total_reads - self.num_enroll_readings
            rates_arr = self.get_rates_given_counts_single_threshold(
                (error_count[i], valid_patterns_count[i]), test_readings_count
            )[0]
            avg_ber = float(np.mean(rates_arr)) if rates_arr.size > 0 else 0.0
            print(f"Average BER over {'all' if self.process_all_ranges else 'first'} enrollment range(s) for threshold {threshold}: {avg_ber:.6e}")

            # Store or cache results
            if self.use_cache:
                self.cache_manager.save_incremental_cache(
                    chip_id=self.chip_id,
                    threshold=threshold,
                    num_enroll_readings=self.num_enroll_readings,
                    error_count=error_count[i],
                    valid_patterns_count=valid_patterns_count[i]
                )
            else:
                results_dict[float(threshold)] = {
                    "error_count": error_count[i],
                    "valid_patterns_count": valid_patterns_count[i],
                }

            end_time = time.time()
            print(f"Time taken for threshold {i}: {end_time - start_time}")
            print("***********\n")

        # If not using cache, return in-memory results
        if not self.use_cache:
            return results_dict

    @staticmethod
    def get_ber_statistics(ber_data: np.ndarray):
        """
        Compute statistics for the Bit Error Rate data.
        """
        # Compute statistics for the Bit Error Rate data
        mean_ber = np.mean(ber_data)
        std_ber = np.std(ber_data)
        max_ber = np.max(ber_data)
        min_ber = np.min(ber_data)
        return mean_ber, std_ber, max_ber, min_ber

    @staticmethod
    def get_rates_given_counts_single_threshold(
        results: Tuple,
        test_readings_count: int,
        *,
        return_both: bool = False,
        num_cells: int | None = None,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Get BER (and optionally acceptance) rates for a single threshold.

        Args:
            results: Tuple of (error_count, valid_patterns_count)
            test_readings_count: Number of heldout/test readings used for BER denominator
            return_both: If True, also returns acceptance_rate alongside ber_rate
            num_cells: Required when return_both=True to compute acceptance rate as
                valid_patterns_count / num_cells

        Returns:
            ber_rate array with an added leading axis (1, ...). If return_both=True,
            returns a tuple (ber_rate, acceptance_rate) with the same broadcasting.
        """
        error_count, valid_patterns_count = results

        # Calculate BER rate
        safe_total = np.where(valid_patterns_count == 0, 1,
                              test_readings_count * valid_patterns_count.astype(np.float64))
        ber_rate = np.divide(error_count, safe_total)
        ber_rate[valid_patterns_count == 0] = 0
        ber_rate = np.expand_dims(ber_rate, axis=0)

        if not return_both:
            return ber_rate

        if num_cells is None or num_cells <= 0:
            raise ValueError("num_cells must be provided and > 0 when return_both=True")

        # Acceptance rate independent of test_readings_count
        acceptance_rate = (valid_patterns_count.astype(np.float64)) / float(num_cells)
        acceptance_rate = np.expand_dims(acceptance_rate, axis=0)
        return ber_rate, acceptance_rate

    @staticmethod
    def get_rates_given_counts(results: Dict, test_readings_count: int) -> Dict:
        """ Get rates given counts for all thresholds. """
        rate_results = {}
        
        for threshold_key, value in results.items():
            error_count = value["error_count"]
            valid_patterns_count = value["valid_patterns_count"]
            
            # Calculate BER rate
            safe_total = np.where(valid_patterns_count == 0, 1, 
                                test_readings_count * valid_patterns_count.astype(np.float64))
            ber_rate = np.divide(error_count, safe_total)
            ber_rate[valid_patterns_count == 0] = 0
            
            rate_results[threshold_key] = {"ber_rate": ber_rate}
        
        return rate_results

    def initialize(self) -> Dict:
        """
        Load or compute BER based on the class parameters.
        """
        if self.use_cache:
            # Check if all requested thresholds are cached
            missing_thresholds = [t for t in self.threshold_values_list
                                  if not self.cache_manager.check_threshold_in_cache(self.chip_id, t, self.num_enroll_readings)]
            
            if missing_thresholds:
                print(f"Computing missing thresholds for chip {self.chip_id} (K={self.num_enroll_readings}): {missing_thresholds}")
                # Temporarily update threshold list to only process missing ones
                self.threshold_values_list = missing_thresholds
                self.compute_and_save_global_ber()
            else:
                print(f"All thresholds cached for chip {self.chip_id} (K={self.num_enroll_readings}).")
            
            # Always load from cache after potential computation
            results = self.cache_manager.load_cache(self.chip_id, self.num_enroll_readings)
        else:
            # Compute without reading/writing cache
            print("Computing results without using cache...")
            results = self.compute_and_save_global_ber()
        
        heldout_readings_count = self.total_reads - self.num_enroll_readings
        rate_results = self.get_rates_given_counts(results, heldout_readings_count)
        return rate_results


if __name__ == "__main__":
    all_files = get_files()
    chip_ids = list(all_files.keys())
    # chip_ids = ['L45']
    
    # Example threshold values (delta or D values)
    # example_threshold_values = [0.499] #[0.49, 0.499, 0.4999, 0.5-10**(-9)]
    example_threshold_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49, 0.499]
    example_num_enroll_readings = 500  # K or N values
    
    all_readouts: List[ReadoutList] = [read_readouts(all_files[chip_id])
                                     for chip_id in chip_ids]
    
    for readouts_val in all_readouts:
        print(f"Processing chip {readouts_val.chip_id}")
        
        # # Option 1: Iterative BER computation (original behavior)
        # print("\n" + "="*60)
        # print("OPTION 1: ITERATIVE BER COMPUTATION")
        # print("="*60)
        # ber_processor_iterative = GlobalBERProcessor(
        #     readouts_val, example_threshold_values, example_num_enroll_readings, 
        #     process_all_ranges=False, iterative_ber=True
        # )
        # ber_processor_iterative.compute_and_save_global_ber()
        
        # Option 2: Single BER computation (faster)
        print("\n" + "="*60)
        print("OPTION 2: SINGLE BER COMPUTATION (FASTER)")
        print("="*60)
        ber_processor_single = GlobalBERProcessor(
            readouts_val, example_threshold_values, example_num_enroll_readings,
            process_all_ranges=True, iterative_ber=True, use_cache=True
        )
        _ = ber_processor_single.initialize()
        
    # Average over chips
    # per_chip = []
    # for cid in chip_ids:
    #     print(f"\nProcessing chip: {cid}")
    #     readouts = read_readouts(all_files[cid])
    #     gp = GlobalBERProcessor(readouts, example_threshold_values, example_num_enroll_readings,
    #                             process_all_ranges=False, iterative_ber=False, use_cache=False)
    #     rates = gp.initialize()  # {th: {"ber_rate": (ranges, T)}}
    #     per_chip.append(rates)

    # avg_over_chips = {}
    # for th in example_threshold_values:
    #     mats = [c[th]["ber_rate"] for c in per_chip]  # list of (ranges, T)
    #     # Average over ranges and chips
    #     avg_over_chips[th] = float(np.mean([m.mean() for m in mats]))
    # print(avg_over_chips)
