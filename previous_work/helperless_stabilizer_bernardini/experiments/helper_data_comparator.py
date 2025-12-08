""" Helper Data Comparator for helperless stabilizer Bernardini approach.

This module compares helper data (bit masks) across multiple enrollments using the same chip.
Unlike the nvm_free_tmvs approach, this works with bit matrices directly and doesn't require
multiprocessing since bit operations are computationally lighter.
"""
import time
from typing import List, Tuple, Dict
import numpy as np
from common.data_reading_utils import get_files, read_readouts, ReadoutList
from ..evaluate_ber import build_bit_matrix, mask_from_threshold, majority_from_matrix
from .analysis_utils import get_enrollment_ranges
from .comparator_cache_manager import ComparatorCacheManager


class HelperDataComparator:
    """ Comparator between the helper data (bit masks) across multiple enrollments. """
    
    def __init__(self, readouts: ReadoutList, threshold_values_list: List[float], 
                 num_enroll_readings: int,
                 reference_K: int | None = None, reference_delta: float | None = None,
                 test_N_list: List[int] | None = None, test_D_list: List[float] | None = None,
                 use_cache: bool = True):
        """ Initialize the HelperDataComparator. """
        self.readouts = readouts
        self.chip_id = readouts.chip_id
        self.threshold_values_list = threshold_values_list
        self.num_enroll_readings = num_enroll_readings
        self.use_cache = use_cache
        # Optional split of reference vs test parameters (used for non-equal ranges analysis)
        self.reference_K = int(num_enroll_readings if reference_K is None else reference_K)
        self.reference_delta = float(reference_delta) if reference_delta is not None else None
        self.test_N_list = list(test_N_list) if test_N_list is not None else None
        self.test_D_list = list(test_D_list) if test_D_list is not None else None

        self.cache_manager = ComparatorCacheManager()
        
        # Validate constraint during initialization
        self._validate_constraints()
        
        # Build bit matrix once for efficiency
        self.bit_matrix = build_bit_matrix(readouts)
        self.total_reads, self.num_cells = self.bit_matrix.shape

    def _validate_constraints(self):
        """
        Validate hard constraints for the HelperDataComparator.
        
        Raises:
            ValueError: If any constraint is violated
        """
        # Constraint: When using non-equal ranges analysis, all values in test_D_list must be > reference_delta
        if self.test_D_list is not None and self.reference_delta is not None:
            for test_D in self.test_D_list:
                if test_D <= self.reference_delta:
                    raise ValueError(f"Constraint violation: test_D ({test_D}) must be > reference_delta ({self.reference_delta}). "
                                    f"All values in test_D_list must be greater than reference_delta.")

    @staticmethod
    def calculate_error_count(mask_comparison: np.ndarray) -> np.ndarray:
        """ Calculate the error count between masks. """
        # mask_comparison shape: (num_cells,) - single comparison result
        # Return total errors (sum over all cells)
        return np.sum(mask_comparison)

    @staticmethod
    def calculate_accepted_cells_count(accept_mask: np.ndarray) -> int:
        """ Calculate the number of accepted cells in the mask. """
        return np.sum(accept_mask)

    @staticmethod
    def calculate_key_bits_count(majority: np.ndarray, accept_mask: np.ndarray) -> Tuple[int, int]:
        """ Calculate the key bits count (zeros and ones) for accepted cells. """
        accepted_majority = majority[accept_mask]
        zero_count = np.sum(accepted_majority == 0)
        one_count = np.sum(accepted_majority == 1)
        return zero_count, one_count

    @staticmethod
    def compute_masks_incrementally(bit_matrix: np.ndarray, K: int, threshold: float) -> np.ndarray:
        """
        Compute acceptance masks incrementally for 1, 2, ..., K readings.
        
        Args:
            bit_matrix: (total_readings, num_cells) bit matrix
            K: Maximum number of enrollment readings
            threshold: Threshold value (delta or D) for mask generation
            
        Returns:
            masks: (K, num_cells) boolean array where masks[i] is the mask using readings 0 to i
        """
        K = int(K)
        num_cells = bit_matrix.shape[1]
        masks = np.zeros((K, num_cells), dtype=bool)
        
        # Compute cumulative means efficiently
        cumulative_sum = np.cumsum(bit_matrix[:K], axis=0).astype(np.float64)
        
        for i in range(K):
            # Compute p_hat for first i+1 readings (0 to i inclusive)
            p_hat = cumulative_sum[i] / (i + 1)
            
            # Acceptance mask: accept if |p_hat - 0.5| >= threshold
            masks[i] = np.abs(p_hat - 0.5) >= np.float64(threshold)
        
        return masks

    @staticmethod
    def series_rates_from_counts(error_counts: np.ndarray, compared_cells: np.ndarray) -> np.ndarray:
        """Compute enrollment-only BER series: errors / compared_cells with safe division.

        Args:
            error_counts: array of length N with error counts per iteration
            compared_cells: array of length N with number of compared cells per iteration

        Returns:
            Array of length N with BER per iteration (zeros where compared_cells == 0)
        """
        err = np.asarray(error_counts, dtype=np.float64)
        cmp_cells = np.asarray(compared_cells)
        safe = np.where(cmp_cells == 0, 1, cmp_cells).astype(np.float64)
        rates = np.divide(err, safe)
        rates[cmp_cells == 0] = 0.0
        return rates

    def process_enroll_range(self, threshold: float, reference_mask: np.ndarray,
                              start_enroll_idx: int, end_enroll_idx: int) -> Tuple[np.ndarray, int, int, int]:
        """
        Process a single enrollment range and compare with reference.
        
        Args:
            threshold: Threshold value (delta or D) for mask generation
            reference_mask: Reference acceptance mask
            start_enroll_idx: inclusive start index of the enrollment readings to use
            end_enroll_idx: exclusive end index of the enrollment readings to use
            
        Returns:
            Tuple of (error_count, accepted_cells_count, zero_count, one_count)
        """
        # Slice bit matrix for the indicated range and compute with its own N
        sub_matrix = self.bit_matrix[start_enroll_idx:end_enroll_idx, :]
        N_sub = max(0, int(end_enroll_idx - start_enroll_idx))
        if N_sub <= 0:
            return np.zeros(self.num_enroll_readings, dtype=np.uint32), 0, 0, 0

        enroll_mask = mask_from_threshold(sub_matrix, N_sub, threshold)
        enroll_majority = majority_from_matrix(sub_matrix, N_sub)
        
        # Compare masks (XOR operation)
        mask_comparison = reference_mask ^ enroll_mask
        
        # Calculate error count (total errors between masks)
        error_count = self.calculate_error_count(mask_comparison)
        
        # Calculate counts for this enrollment
        accepted_cells_count = self.calculate_accepted_cells_count(enroll_mask)
        zero_count, one_count = self.calculate_key_bits_count(enroll_majority, enroll_mask)
        
        return error_count, accepted_cells_count, zero_count, one_count

    # def compute_enrollment_ber_series(self, reference_mask: np.ndarray,
    #                                   N: int, D: float) -> tuple[np.ndarray, np.ndarray]:
    #     """
    #     Compute enrollment-only BER series comparing the reference at K to test iterations 1..N
    #     using explicit (N, D). Error rate is calculated by XORing reference and test masks.
        
    #     Args:
    #         reference_mask: Reference acceptance mask computed from first K readings
    #         N: Number of test iterations
    #         D: Threshold for test enrollment ranges
            
    #     Returns:
    #         Tuple of (error_counts, ber_rates) for each test iteration
    #     """
    #     error_counts = np.zeros(int(N), dtype=np.uint32)
    #     ber_rates = np.zeros(int(N), dtype=np.float64)

    #     for i in range(1, int(N) + 1):
    #         test_mask_i = mask_from_threshold(self.bit_matrix, i, D)

    #         # Calculate error rate by XORing the masks and dividing by mask length
    #         mask_xor = reference_mask ^ test_mask_i
    #         error_count = int(np.sum(mask_xor))
    #         mask_length = len(reference_mask)
            
    #         error_counts[i - 1] = error_count
    #         ber_rates[i - 1] = float(error_count) / float(mask_length)

    #     return error_counts, ber_rates

    def compute_enrollment_ber_series_non_equal_ranges(self, reference_mask: np.ndarray,
                                                      K: int, N: int, D: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute enrollment-only BER series for non-equal ranges analysis.
        Uses enrollment ranges starting from K+1, K+1+N, K+1+2N, etc.
        Each range uses N readings and doesn't overlap with the reference enrollment (first K readings).
        Error rate is calculated by XORing reference and test masks.
        
        Args:
            reference_mask: Reference acceptance mask computed from first K readings
            K: Number of readings used for reference (first K readings)
            N: Number of readings per test enrollment range
            D: Threshold for test enrollment ranges
            
        Returns:
            Tuple of (error_counts, ber_rates) for each test range
        """
        # Calculate how many test ranges we can fit in the remaining readings
        remaining_readings = self.total_reads - K
        max_test_ranges = remaining_readings // N
        
        if max_test_ranges == 0:
            print(f"Warning: Not enough readings for test ranges. Need at least {N} readings after K={K}")
            return np.array([]), np.array([])
        
        error_counts = np.zeros(max_test_ranges, dtype=np.uint32)
        ber_rates = np.zeros(max_test_ranges, dtype=np.float64)
        
        print(f"Computing {max_test_ranges} test ranges, each using {N} readings")
        
        for range_idx in range(max_test_ranges):
            # Calculate the starting index for this test range
            start_idx = K + range_idx * N
            end_idx = start_idx + N
            
            # print(f"Test range {range_idx + 1}: readings {start_idx} to {end_idx-1}")
            
            # Extract the bit matrix for this test range
            test_bit_matrix = self.bit_matrix[start_idx:end_idx, :]
            
            # Compute mask for this test range
            test_mask = mask_from_threshold(test_bit_matrix, N, D)
            
            # Calculate error rate by XORing the masks and dividing by mask length
            mask_xor = reference_mask ^ test_mask
            error_count = int(np.sum(mask_xor))
            mask_length = len(reference_mask)
            
            error_counts[range_idx] = error_count
            ber_rates[range_idx] = float(error_count) / float(mask_length)
        
        return error_counts, ber_rates

    def equal_ranges_analysis(self):
        """
        Perform equal ranges analysis using a single num_enroll_readings and multiple thresholds.
        This is called when reference_K, reference_delta, test_N_list, and test_D_list are not provided.
        """
        results_dict: Dict[float, Dict[str, np.ndarray]] = {}
        
        # Original approach: equal-sized enrollment ranges (D=K case)
        enroll_ranges = get_enrollment_ranges(self.num_enroll_readings)
        
        # Initialize result arrays
        num_thresholds = len(self.threshold_values_list)
        num_ranges = len(enroll_ranges)
        
        error_count = np.zeros((num_thresholds, num_ranges, self.num_enroll_readings))
        accepted_cells_count = np.zeros((num_thresholds, num_ranges))
        zero_key_bits_count = np.zeros((num_thresholds, num_ranges))
        one_key_bits_count = np.zeros((num_thresholds, num_ranges))

        for i, threshold in enumerate(self.threshold_values_list):
            # Check if this threshold exists in the cache (only if using cache)
            if self.use_cache and self.cache_manager.check_threshold_in_cache(self.chip_id, threshold, 
                                                          self.num_enroll_readings):
                print(f"Skipping computation for threshold {threshold}, already in cache.")
                continue

            print(f"\nProcessing threshold {i}: {threshold}")
            start_time = time.time()

            # Compute reference data for first enrollment range
            reference_mask = mask_from_threshold(self.bit_matrix, self.num_enroll_readings, threshold)
            reference_majority = majority_from_matrix(self.bit_matrix, self.num_enroll_readings)
            
            # Store reference results
            accepted_cells_count[i, 0] = self.calculate_accepted_cells_count(reference_mask)
            zero_count, one_count = self.calculate_key_bits_count(reference_majority, reference_mask)
            zero_key_bits_count[i, 0] = zero_count
            one_key_bits_count[i, 0] = one_count

            # Process other enrollment ranges
            for range_idx, (start_enroll_idx, end_enroll_idx) in enumerate(enroll_ranges[1:], 1):
                print(f"Processing enrollment range {range_idx}: {start_enroll_idx} to {end_enroll_idx}")
                
                # For simplicity, we use the same threshold for all ranges
                # In practice, you might want to use different thresholds per range
                error_count_val, accepted_count, zero_count, one_count = self.process_enroll_range(
                        threshold, reference_mask, int(start_enroll_idx), int(end_enroll_idx)
                    )
                
                error_count[i, range_idx, :] = error_count_val
                accepted_cells_count[i, range_idx] = accepted_count
                zero_key_bits_count[i, range_idx] = zero_count
                one_key_bits_count[i, range_idx] = one_count

            # Store or cache results
            if self.use_cache:
                self.cache_manager.save_equal_ranges(
                    chip_id=self.chip_id,
                    K=self.num_enroll_readings,
                    delta=threshold,
                    D=threshold,
                    error_count=error_count[i],
                    accepted_cells_count=accepted_cells_count[i],
                    zero_key_bits_count=zero_key_bits_count[i],
                    one_key_bits_count=one_key_bits_count[i]
                )
            else:
                results_dict[float(threshold)] = {
                    "error_count": error_count[i],
                    "accepted_cells_count": accepted_cells_count[i],
                    "zero_key_bits_count": zero_key_bits_count[i],
                    "one_key_bits_count": one_key_bits_count[i]
                }

            end_time = time.time()
            print(f"Time taken for threshold {i}: {end_time - start_time}")
            print("***********\n")

        # Print summary results regardless of cache setting
        print("\n=== SUMMARY RESULTS (Equal Ranges Analysis) ===")
        if not results_dict:
            print("No results to display (all thresholds were cached)")
        else:
            for threshold, data in results_dict.items():
                error_count = data["error_count"]
                accepted_cells_count = data["accepted_cells_count"]
                
                # Calculate averages over ranges
                avg_error_count = np.mean(error_count)
                avg_accepted_cells = np.mean(accepted_cells_count)
                
                print(f"Threshold {threshold}:")
                print(f"  Average BER: {avg_error_count / self.num_cells:.6e}")
                print(f"  Average Acceptance Rate: {avg_accepted_cells / self.num_cells:.4f}")
                print()
        
        # If not using cache, return in-memory results
        if not self.use_cache:
            return results_dict
        
        # When using cache, return empty dict since results are saved to cache
        return {}

    def non_equal_ranges_analysis(self):
        """
        Perform non-equal ranges analysis using reference (K, delta) and test ranges (1000-K)/N.
        This is called when reference_K, reference_delta, test_N_list, and test_D_list are provided.
        The reference mask is derived using K and delta, and tested using different enrollment ranges
        that don't overlap with the reference enrollment.
        """
        results_dict: Dict[float, Dict[str, np.ndarray]] = {}
        
        # Determine reference parameters
        K = int(self.reference_K)
        # If reference_delta not provided, fall back to first threshold in the list
        if self.reference_delta is None:
            if not self.threshold_values_list:
                raise ValueError("threshold_values_list must be provided when reference_delta is None")
            delta = float(self.threshold_values_list[0])
        else:
            delta = float(self.reference_delta)

        print(f"\nReference: K={K}, delta={delta}")
        print(f"Total readings available: {self.total_reads}")

        # Compute reference data once using the first K readouts
        reference_mask = mask_from_threshold(self.bit_matrix, K, delta)
        reference_majority = majority_from_matrix(self.bit_matrix, K)

        accepted_cells_ref = self.calculate_accepted_cells_count(reference_mask)
        zero_ref, one_ref = self.calculate_key_bits_count(reference_majority, reference_mask)

        print(f"Reference mask: {accepted_cells_ref} accepted cells")

        # Determine tested parameter sets
        N_list = self.test_N_list if self.test_N_list is not None else [self.num_enroll_readings]
        D_list = self.test_D_list if self.test_D_list is not None else list(self.threshold_values_list)

        for N in N_list:
            for D in D_list:
                print(f"\nProcessing tested parameters: N={int(N)}, D={float(D)}")
                start_time = time.time()

                # Compute enrollment-only BER series vs K-th reference for non-equal ranges
                err_series, rate_series = self.compute_enrollment_ber_series_non_equal_ranges(
                    reference_mask, K, int(N), float(D)
                )

                if len(err_series) == 0:
                    print(f"Skipping (N={N}, D={D}) - insufficient readings")
                    continue

                # Store or cache results grouped per (K, delta), keyed by (N, D)
                if self.use_cache:
                    self.cache_manager.save_keqne_series(
                        chip_id=self.chip_id,
                        K=K,
                        delta=delta,
                        N=int(N),
                        D=float(D),
                        accepted_cells_ref=accepted_cells_ref,
                        zero_key_bits_ref=zero_ref,
                        one_key_bits_ref=one_ref,
                        enrollment_ber_error_counts=err_series,
                        enrollment_ber_rates=rate_series
                    )
                else:
                    # For non-cache mode, store results with key based on D
                    results_dict[float(D)] = {
                        "error_count": err_series,
                        "accepted_cells_count": np.array([accepted_cells_ref]),
                        "zero_key_bits_count": np.array([zero_ref]),
                        "one_key_bits_count": np.array([one_ref])
                    }
                
                print(f"Error counts: {err_series}")
                # print(f"BER rates: {rate_series}")

                # Calculate and print average BER across all test ranges
                end_time = time.time()
                print(f"Time taken for (N={N}, D={D}): {end_time - start_time:.2f}s")
                print("***********")

        # Print summary results regardless of cache setting
        print("\n=== SUMMARY RESULTS (Non-Equal Ranges Analysis) ===")
        if not results_dict:
            print("No results to display (all thresholds were cached)")
        else:
            for threshold_key, data in results_dict.items():
                error_count = data["error_count"]
                accepted_cells_count = data["accepted_cells_count"]
                
                # Calculate averages over ranges
                avg_error_count = np.mean(error_count)
                avg_accepted_cells = np.mean(accepted_cells_count)
                
                print(f"Threshold {threshold_key}:")
                print(f"  Average BER: {avg_error_count / self.num_cells:.6e}")
                print(f"  Average Acceptance Rate: {avg_accepted_cells / self.num_cells:.4f}")
                print()
        
        # If not using cache, return in-memory results
        if not self.use_cache:
            return results_dict
        
        # When using cache, return empty dict since results are saved to cache
        return {}

    def incremental_enrollment_ber_analysis(self, K: int, delta: float, D: float):
        """
        Perform incremental enrollment BER analysis for non-overlapping ranges.
        
        Similar to extract_key_and_helper_data_incrementally in nvm_free_tmvs, but for Bernardini:
        - Reference range: first K readings with threshold delta
        - Test ranges: non-overlapping ranges of size K starting after K, each with threshold D
        - Compare masks incrementally at iterations 1, 2, ..., K
        
        Args:
            K: Number of readings per range (reference and test ranges have same size K)
            delta: Threshold for reference mask
            D: Threshold for test range masks
            
        Returns:
            Dict with error_count and discarded_patterns_count
        """
        print(f"\nIncremental Enrollment BER Analysis: K={K}, delta={delta}, D={D}")
        print(f"Total readings available: {self.total_reads}")
        
        # Check cache
        if self.use_cache:
            cached_data = self.cache_manager.load_incremental_enrollment_ber(
                self.chip_id, K, delta, D
            )
            if cached_data is not None:
                print(f"Loading from cache for K={K}, delta={delta}, D={D}")
                return cached_data
        
        # Calculate how many test ranges we can fit
        remaining_readings = self.total_reads - K
        max_test_ranges = remaining_readings // K
        
        if max_test_ranges == 0:
            print(f"Warning: Not enough readings for test ranges. Need at least {K} readings after K={K}")
            return {
                "error_count": np.zeros((0, K), dtype=np.uint32),
                "discarded_patterns_count": np.zeros((0, K), dtype=np.uint32)
            }
        
        print(f"Computing {max_test_ranges} test ranges, each using {K} readings")
        
        # Compute reference masks incrementally (for 1, 2, ..., K readings)
        print("Computing reference masks incrementally...")
        reference_masks = self.compute_masks_incrementally(
            self.bit_matrix[:K], K, delta
        )  # Shape: (K, num_cells)
        
        # Initialize result arrays
        # error_count: (num_test_ranges, K) - one error count per iteration per test range
        # discarded_patterns_count: (num_test_ranges, K) - discarded cells per iteration per test range
        error_count = np.zeros((max_test_ranges, K), dtype=np.uint32)
        discarded_patterns_count = np.zeros((max_test_ranges, K), dtype=np.uint32)
        
        # Process each test range
        for range_idx in range(max_test_ranges):
            start_idx = K + range_idx * K
            end_idx = start_idx + K
            
            if end_idx > self.total_reads:
                # Truncate if we don't have enough readings
                max_test_ranges = range_idx
                error_count = error_count[:max_test_ranges]
                discarded_patterns_count = discarded_patterns_count[:max_test_ranges]
                break
            
            print(f"Processing test range {range_idx + 1}: readings {start_idx} to {end_idx-1}")
            
            # Extract bit matrix for this test range
            test_bit_matrix = self.bit_matrix[start_idx:end_idx]
            
            # Compute test masks incrementally (for 1, 2, ..., K readings within this range)
            test_masks = self.compute_masks_incrementally(
                test_bit_matrix, K, D
            )  # Shape: (K, num_cells)
            
            # Compare incrementally at each iteration
            for i in range(K):
                # Compare reference mask[i] with test mask[i]
                mask_xor = reference_masks[i] ^ test_masks[i]  # Shape: (num_cells,)
                error_count[range_idx, i] = np.sum(mask_xor).astype(np.uint32)
                
                # Discarded patterns count: cells not accepted in the reference mask
                # This represents cells that would be discarded during enrollment
                rejected_in_ref = ~reference_masks[i]  # Cells not accepted in reference
                discarded_patterns_count[range_idx, i] = np.sum(rejected_in_ref).astype(np.uint32)
        
        results = {
            "error_count": error_count,
            "discarded_patterns_count": discarded_patterns_count
        }
        
        # Save to cache
        if self.use_cache:
            self.cache_manager.save_incremental_enrollment_ber(
                self.chip_id, K, delta, D,
                error_count, discarded_patterns_count
            )
            print(f"Saved to cache for K={K}, delta={delta}, D={D}")
        
        return results

    @staticmethod
    def get_rates_given_counts_single_threshold(results: Tuple, num_cells: int) -> np.ndarray:
        """ Get rates given counts for a single threshold. """
        error_count, accepted_cells_count, zero_key_bits_count, one_key_bits_count = results
        
        # Handle both scalar and array inputs
        if np.isscalar(error_count):
            # Scalar case - return a single array
            error_rate = error_count / num_cells
            acceptance_rate = accepted_cells_count / num_cells
            
            # Calculate extraction rate (fraction of accepted cells that contribute to key)
            safe_total = max(accepted_cells_count, 1) if accepted_cells_count == 0 else accepted_cells_count
            extraction_rate = (zero_key_bits_count + one_key_bits_count) / safe_total
            if accepted_cells_count == 0:
                extraction_rate = 0
            
            zero_rate = zero_key_bits_count / num_cells
            one_rate = one_key_bits_count / num_cells
            
            return np.array([error_rate, acceptance_rate, extraction_rate, zero_rate, one_rate])
        else:
            # Array case - return arrays for each rate
            error_rate = error_count / num_cells
            acceptance_rate = accepted_cells_count / num_cells
            
            # Calculate extraction rate (fraction of accepted cells that contribute to key)
            safe_total = np.where(accepted_cells_count == 0, 1, accepted_cells_count)
            extraction_rate = (zero_key_bits_count + one_key_bits_count) / safe_total
            extraction_rate[accepted_cells_count == 0] = 0
            
            zero_rate = zero_key_bits_count / num_cells
            one_rate = one_key_bits_count / num_cells
            
            return np.array([error_rate, acceptance_rate, extraction_rate, zero_rate, one_rate])

    @staticmethod
    def get_rates_given_counts(results: Dict, num_cells: int) -> Dict:
        """ Get rates given counts for all thresholds, averaging over enrollment ranges if present. """
        # Guard against None or empty
        if results is None or (isinstance(results, dict) and len(results) == 0):
            return {}
        rate_results = {}

        for threshold_key, value in results.items():
            error_count = value["error_count"]
            accepted_cells_count = value["accepted_cells_count"]
            zero_key_bits_count = value["zero_key_bits_count"]
            one_key_bits_count = value["one_key_bits_count"]

            # Per-range rates
            error_rate = error_count / num_cells
            acceptance_rate = accepted_cells_count / num_cells
            safe_total = np.where(accepted_cells_count == 0, 1, accepted_cells_count)
            extraction_rate = (zero_key_bits_count + one_key_bits_count) / safe_total
            if isinstance(extraction_rate, np.ndarray):
                extraction_rate = extraction_rate.astype(np.float64)
                extraction_rate[accepted_cells_count == 0] = 0
            else:
                extraction_rate = 0 if accepted_cells_count == 0 else extraction_rate
            zero_rate = zero_key_bits_count / num_cells
            one_rate = one_key_bits_count / num_cells

            # Average across ranges when arrays have a range dimension
            def avg_over_ranges(x):
                if isinstance(x, np.ndarray):
                    return float(np.mean(x)) if x.ndim >= 1 else float(x)
                return float(x)

            error_rate_avg = avg_over_ranges(error_rate)
            acceptance_rate_avg = avg_over_ranges(acceptance_rate)
            extraction_rate_avg = avg_over_ranges(extraction_rate)
            zero_rate_avg = avg_over_ranges(zero_rate)
            one_rate_avg = avg_over_ranges(one_rate)

            rate_results[threshold_key] = {
                "error_rate": error_rate_avg,
                "acceptance_rate": acceptance_rate_avg,
                "extraction_rate": extraction_rate_avg,
                "zero_rate": zero_rate_avg,
                "one_rate": one_rate_avg
            }

        return rate_results

    @staticmethod
    def get_rates_given_counts_single_threshold_incremental(
        error_count_ranges: np.ndarray,
        discarded_patterns_count_ranges: np.ndarray,
        num_cells: int,
    ) -> Dict[str, np.ndarray]:
        """
        Compute incremental rates averaged over ranges for a single threshold.

        Args:
            error_count_ranges: (num_ranges, K) error counts per iteration (1..K) per range
            discarded_patterns_count_ranges: (num_ranges, K) discarded (rejected) cells in reference per iteration
            num_cells: total number of cells (mask length)

        Returns:
            Dict with arrays of length K:
                - error_rate: average over ranges of (error_count / num_cells)
                - acceptance_rate: average over ranges of (accepted_cells / num_cells),
                  where accepted_cells = num_cells - discarded_in_reference
        """
        if error_count_ranges is None or discarded_patterns_count_ranges is None:
            return {"error_rate": np.array([]), "acceptance_rate": np.array([])}

        err = np.asarray(error_count_ranges, dtype=np.float64)
        disc_ref = np.asarray(discarded_patterns_count_ranges, dtype=np.float64)

        if err.ndim != 2 or disc_ref.ndim != 2 or err.shape != disc_ref.shape:
            return {"error_rate": np.array([]), "acceptance_rate": np.array([])}

        # Average over ranges (axis=0) -> length K
        err_avg = np.mean(err, axis=0)
        disc_avg = np.mean(disc_ref, axis=0)

        error_rate = err_avg / float(num_cells)
        accepted_cells = float(num_cells) - disc_avg
        accepted_cells = np.clip(accepted_cells, 0.0, float(num_cells))
        acceptance_rate = accepted_cells / float(num_cells)

        return {"error_rate": error_rate, "acceptance_rate": acceptance_rate}

    def _transform_cached_keqne_data(self, cached: Dict) -> Dict:
        """Transform cached KEQNE data into results dict format."""
        results = {}
        for (N_val, D_val), entry in cached.items():
            err_series = np.asarray(entry.get("enrollment_ber_error_counts", []))
            if err_series.size == 0:
                continue
                
            acc_ref = float(entry.get("accepted_cells_ref", 0.0))
            zero_ref = float(entry.get("zero_key_bits_ref", 0.0))
            one_ref = float(entry.get("one_key_bits_ref", 0.0))
            
            results[float(D_val)] = {
                "error_count": err_series,
                "accepted_cells_count": np.full_like(err_series, acc_ref, dtype=np.float64),
                "zero_key_bits_count": np.full_like(err_series, zero_ref, dtype=np.float64),
                "one_key_bits_count": np.full_like(err_series, one_ref, dtype=np.float64),
            }
        return results

    def initialize(self) -> Dict:
        """
        Load or compute helper data based on the class parameters.
        Determines whether to use equal ranges or non-equal ranges analysis based on provided parameters.
        """
        # Determine which analysis to use based on provided parameters
        use_non_equal_ranges = (self.reference_K is not None and 
                               self.reference_delta is not None and 
                               self.test_N_list is not None and 
                               self.test_D_list is not None)
        
        if self.use_cache:
            # Try to load from cache first
            if use_non_equal_ranges:
                K = int(self.reference_K)
                delta = float(self.reference_delta)
                cached = self.cache_manager.load_all(self.chip_id, K, delta, use_equal_ranges=False)
                if cached:
                    results = self._transform_cached_keqne_data(cached)
                else:
                    print("Cache not found. Computing results...")
                    results = self.non_equal_ranges_analysis()
            else:
                results = self.cache_manager.load_cache(self.chip_id, self.num_enroll_readings)
                if results is None:
                    print("Cache not found. Computing results...")
                    results = self.equal_ranges_analysis()
        else:
            # Compute without reading/writing cache
            print("Computing results without using cache...")
            if use_non_equal_ranges:
                print("Using non-equal ranges analysis...")
                results = self.non_equal_ranges_analysis()
            else:
                print("Using equal ranges analysis...")
                results = self.equal_ranges_analysis()
        
        # Ensure results is a dict to avoid NoneType downstream
        if results is None:
            results = {}
        rate_results = self.get_rates_given_counts(results, self.num_cells)
        return rate_results


if __name__ == "__main__":
    all_files = get_files()
    chip_ids = list(all_files.keys())
    # chip_ids = ['L45']
    
    # Example threshold values (delta or D values)
    threshold_values_list = [0.3]
    # threshold_values_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49]
    num_enroll_readings = 500  # K or N values

    # Old demo (equal/non-equal ranges) - commented out
    # # Aggregators for averages over chips
    # chips_agg: Dict[float, Dict[str, List[float]]] = {}

    # for chip_id in chip_ids:
    #     print(f"\nProcessing chip {chip_id}")
    #     readouts_val = read_readouts(all_files[chip_id])

    #     helper_data_comparator = HelperDataComparator(
    #         readouts_val, threshold_values_list, num_enroll_readings,
    #         reference_K=500, reference_delta=0.499,
    #         test_N_list=[500], test_D_list=[0.4991],
    #         use_cache=True
    #     )

    #     rate_results = helper_data_comparator.initialize()
    #     if not isinstance(rate_results, dict) or not rate_results:
    #         continue

    #     for thr, rates in rate_results.items():
    #         chips_agg.setdefault(float(thr), {"error_rate": [], "acceptance_rate": []})
    #         # rates are already averaged over ranges
    #         chips_agg[float(thr)]["error_rate"].append(float(rates["error_rate"]))
    #         chips_agg[float(thr)]["acceptance_rate"].append(float(rates["acceptance_rate"]))

    # # Print averages over chips
    # if chips_agg:
    #     print("\n=== AVERAGE OVER CHIPS ===")
    #     for thr in sorted(chips_agg.keys()):
    #         err_list = chips_agg[thr]["error_rate"]
    #         acc_list = chips_agg[thr]["acceptance_rate"]
    #         if len(err_list) == 0:
    #             continue
    #         avg_err = float(np.mean(err_list))
    #         avg_acc = float(np.mean(acc_list)) if len(acc_list) > 0 else 0.0
    #         print(f"Threshold {thr}:")
    #         print(f"  Average BER (over chips): {avg_err:.6e}")
    #         print(f"  Average Acceptance Rate (over chips): {avg_acc:.4f}")

    # print(f"\nCompleted processing {len(chip_ids)} chips.")

    # New: generate and cache incremental enrollment BER for all chips
    K = 500
    delta = 0.499
    D = 0.4991
    for chip_id in chip_ids:
        print(f"Processing chip {chip_id}")
        readouts_val = read_readouts(all_files[chip_id])
        helper_data_comparator = HelperDataComparator(
            readouts_val, threshold_values_list=[delta], num_enroll_readings=K,
            reference_K=K, reference_delta=delta,
            test_N_list=[K], test_D_list=[D],
            use_cache=True
        )
        res = helper_data_comparator.incremental_enrollment_ber_analysis(K=K, delta=delta, D=D)
        print("Saved cache:", chip_id,
              "error_count shape:", res["error_count"].shape,
              "discarded shape:", res["discarded_patterns_count"].shape)