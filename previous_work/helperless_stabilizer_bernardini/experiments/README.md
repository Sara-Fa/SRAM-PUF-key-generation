# Helperless Stabilizer Bernardini Experiments

This directory contains the experimental framework for analyzing the helperless stabilizer Bernardini approach, similar to the `nvm_free_tmvs/experiments` but adapted for the simpler bit-matrix based approach.

## Overview

The helperless stabilizer Bernardini approach (p2) is simpler than the nvm_free_tmvs approach (p1) because:

- **No Hamming distances**: Works directly with bit matrices
- **No codebooks**: Uses bit masks instead of addresses and codewords
- **No multiprocessing**: Bit operations are computationally lighter
- **Simpler helper data**: Bit masks instead of complex helper data structures

## Key Differences from nvm_free_tmvs/experiments

| Aspect | nvm_free_tmvs (p1) | helperless_stabilizer_bernardini (p2) |
|--------|-------------------|--------------------------------------|
| **Data Structure** | Hamming distances, codebooks | Bit matrices |
| **Helper Data** | Addresses and codewords | Bit masks |
| **Processing** | Multiprocessing required | Single-threaded (sufficient) |
| **Parameters** | Complex threshold calculations | Simple threshold values (delta/D) |
| **Enrollment** | Code length dependent | Direct bit matrix operations |

## Components

### 1. HelperDataComparator (`helper_data_comparator.py`)

Compares helper data (bit masks) across multiple enrollments using the same chip.

**Key Features:**
- Works with bit matrices directly
- Compares masks using XOR operations
- Calculates error rates, acceptance rates, and extraction rates
- No multiprocessing needed

**Parameters:**
- `threshold_values_list`: List of threshold values (delta or D)
- `num_enroll_readings`: Number of enrollment readings (K or N)
- `enroll_ranges`: Enrollment ranges for testing

### 2. GlobalBERProcessor (`global_ber_processor.py`)

Processes Bit Error Rate (BER) data for different enrollments.

**Key Features:**
- Computes BER on heldout reads
- Uses bit matrices for efficient processing
- Calculates error counts and valid pattern counts
- Single-threaded processing

**Parameters:**
- `threshold_values_list`: List of threshold values (delta or D)
- `num_enroll_readings`: Number of enrollment readings (K or N)
- `enroll_ranges`: Enrollment ranges for testing

### 3. SpecializedHelperDataComparator (`specialized_helper_data_comparator.py`)

Handles the specific case mentioned in the user requirements:
- **Reference enrollment**: Fixed parameters (K, delta)
- **Test enrollments**: Variable parameters (N, D) from lists

**Key Features:**
- Reference range with fixed (K, delta) parameters
- Test ranges with variable (N, D) parameters
- Matrix comparison across all (N, D) combinations
- Detailed statistics and reporting

### 4. Analysis Utilities (`analysis_utils.py`)

Provides utility functions for parameter generation:

- `get_enrollment_ranges()`: Generate enrollment ranges
- `get_enrollment_threshold_values()`: Generate threshold values
- `get_enrollment_readings_values()`: Generate enrollment reading values

### 5. Cache Managers (`comparator_cache_manager.py`)

Manages caching for both helper data and BER processing:

- `ComparatorCacheManager`: Caches helper data comparison results
- `BERCacheManager`: Caches BER processing results

## Usage

### Basic Usage

```python
from experiments.main import run_helper_data_comparison, run_global_ber_analysis

# Run helper data comparison
run_helper_data_comparison(
    chip_ids=['L45', 'M17', 'M39'],
    threshold_values=[0.1, 0.2, 0.3, 0.4],
    num_enroll_readings=100
)

# Run global BER analysis
run_global_ber_analysis(
    chip_ids=['L45', 'M17', 'M39'],
    threshold_values=[0.1, 0.2, 0.3, 0.4],
    num_enroll_readings=100
)
```

### Specialized Usage

```python
from experiments.specialized_helper_data_comparator import run_specialized_comparison

# Run specialized comparison with fixed reference and variable test parameters
run_specialized_comparison(
    chip_ids=['L45', 'M17'],
    reference_K=100,           # Fixed K for reference
    reference_delta=0.2,       # Fixed delta for reference
    test_N_list=[50, 100, 150, 200],  # Variable N values
    test_D_list=[0.1, 0.15, 0.2, 0.25, 0.3]  # Variable D values
)
```

### Direct Component Usage

```python
from experiments.helper_data_comparator import HelperDataComparator
from experiments.global_ber_processor import GlobalBERProcessor
from common.data_reading_utils import get_files, read_readouts

# Load data
all_files = get_files()
readouts = read_readouts(all_files['L45'])

# Initialize components
helper_comparator = HelperDataComparator(
    readouts, 
    threshold_values_list=[0.1, 0.2, 0.3], 
    num_enroll_readings=100
)

ber_processor = GlobalBERProcessor(
    readouts,
    threshold_values_list=[0.1, 0.2, 0.3],
    num_enroll_readings=100
)

# Run analysis
helper_comparator.compare_and_save_helper_data()
ber_processor.compute_and_save_global_ber()

# Get results
helper_results = helper_comparator.initialize()
ber_results = ber_processor.initialize()
```

## Parameters

### Threshold Values
- **delta**: Threshold for acceptance mask generation
- **D**: Alternative threshold parameter
- Range: Typically 0.05 to 0.49

### Enrollment Readings
- **K**: Number of enrollment reads for reference
- **N**: Number of enrollment reads for testing
- Range: Typically 10 to 200

### Enrollment Ranges
- Generated automatically based on available readings
- Each range represents a different enrollment period
- Used for cross-validation and robustness testing

## Output

### Helper Data Comparison Results
- `error_rate`: Rate of mask disagreements
- `acceptance_rate`: Rate of accepted cells
- `extraction_rate`: Rate of cells contributing to key
- `zero_rate`: Rate of zero bits
- `one_rate`: Rate of one bits

### BER Analysis Results
- `ber_rate`: Bit error rate on heldout reads
- `error_count`: Total error count
- `valid_patterns_count`: Number of valid patterns

### Specialized Comparison Results
- `disagreement_rate`: Rate of mask disagreements
- `reference_acceptance_rate`: Acceptance rate for reference
- `test_acceptance_rate`: Acceptance rate for test
- `overlap_rate`: Rate of overlapping accepted cells
- `union_rate`: Rate of union of accepted cells

## File Structure

```
experiments/
├── __init__.py
├── helper_data_comparator.py          # Main helper data comparator
├── global_ber_processor.py           # Main BER processor
├── specialized_helper_data_comparator.py  # Specialized comparator
├── analysis_utils.py                 # Utility functions
├── comparator_cache_manager.py       # Cache management
├── main.py                          # Main execution script
└── README.md                        # This file
```

## Dependencies

- `numpy`: Numerical computations
- `common.data_reading_utils`: Data loading utilities
- `common.data_constants`: Data constants
- `..evaluate_ber`: BER evaluation functions from parent directory

## Notes

1. **No Multiprocessing**: Unlike p1, this approach doesn't require multiprocessing due to simpler bit operations.

2. **Bit Matrix Operations**: All operations work directly with bit matrices for efficiency.

3. **Simplified Parameters**: Uses simple threshold values instead of complex margin coefficients.

4. **Cache Management**: Results are cached to avoid recomputation during development.

5. **Flexible Configuration**: Parameters can be easily adjusted for different experimental setups.

## Future Enhancements

- Add plotting capabilities for result visualization
- Implement additional statistical analysis functions
- Add support for different bit matrix formats
- Integrate with existing plotting infrastructure from nvm_free_tmvs
