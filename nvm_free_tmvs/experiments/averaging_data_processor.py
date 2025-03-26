""" A class to process and average data saved by incremental cache scripts"""
import os
import re
import time
import h5py
import numpy as np
from nvm_free_tmvs.utils.file_manager import enroll_comparator_dir
from nvm_free_tmvs.utils.file_manager import ber_comparator_dir
from nvm_free_tmvs.experiments.global_ber_processor import GlobalBERProcessor
from nvm_free_tmvs.experiments.helper_data_comparator import HelperDataComparator
import nvm_free_tmvs.analysis_constants as const

class AveragingDataProcessor:
    """
    A class to process and average data saved by incremental cache scripts
    over multiple chips (identified by chip_id) while considering matching parameters.
    """

    def __init__(self, base_directory, class_instance):
        """
        Initialize the processor with the directory where .h5 files are stored.
        """
        self.base_directory = base_directory
        self.class_instance = class_instance
        self.pattern = None
        if base_directory == enroll_comparator_dir:
            self.pattern = r"enrollment_comparator_chip_([A-Z0-9]+)_code_N(\d+)_Threshold_(\d+)_(\d+)_MaxEnrollReadings_(\d+)\.h5"
        elif base_directory == ber_comparator_dir:
            self.pattern = r"ber_comparator_chip([A-Z0-9]+)_N(\d+)_Threshold_(\d+)_(\d+)_MaxEnrollReadings_(\d+)\.h5"

    def get_output_filename(self, code_length, select_threshold, num_enroll_readings):
        """
        Generate the output filename based on the parameters.
        """
        output_file_name = (
            f"aggregated_code_N{code_length}"
            f"_Threshold_{select_threshold[0]}_{select_threshold[1]}"
            f"_MaxEnrollReadings_{num_enroll_readings}.h5"
        )
        output_file_path = self.base_directory / output_file_name
        return output_file_path

    def parse_filename(self, filename):
        """
        Extract parameters from the filename using a regex pattern.
        """
        match = re.match(self.pattern, filename)
        if match:
            chip_id = match.group(1)
            code_length = int(match.group(2))
            select_threshold = (int(match.group(3)), int(match.group(4)))
            num_enroll_readings = int(match.group(5))
            return chip_id, code_length, select_threshold, num_enroll_readings
        return None

    def get_all_cache_files(self):
        """
        Retrieve all .h5 files in the base directory and their extracted parameters.
        """
        files_with_params = {}
        for file in os.listdir(self.base_directory):
            if file.endswith(".h5"):
                parsed = self.parse_filename(file)
                if parsed:
                    chip_id, code_length, select_threshold, num_enroll_readings = parsed
                    if num_enroll_readings != const.MAX_ENROLLMENT_READINGS:
                        continue
                    key = (code_length, select_threshold, num_enroll_readings)
                    if key not in files_with_params:
                        files_with_params[key] = []
                    files_with_params[key].append(os.path.join(self.base_directory, file))
        return files_with_params

    def check_file_exists(self, code_length, select_threshold, num_enroll_readings):
        """
        Check if the file exists.
        """
        output_file_path = self.get_output_filename(code_length, select_threshold, num_enroll_readings)
        if output_file_path.exists():
            print(f"Skipping computation: {output_file_path} already exists.")
            return 1  # Skip processing if the file is already present
        return 0

    def aggregate_data(self):
        """
        Read and average data across different chips, grouped by matching parameters.
        """
        files_with_params = self.get_all_cache_files()
        aggregated_results = {}

        for key, file_list in files_with_params.items():
            code_length, select_threshold, num_enroll_readings = key
            if self.check_file_exists(code_length, select_threshold, num_enroll_readings):
                continue
            grouped_data = {}  # {group_name: [data arrays from different chips]}
            print("code length, select_threshold, num_enroll_readings:", code_length,
                  select_threshold, num_enroll_readings)

            for file_path in file_list:
                with h5py.File(file_path, "r") as hf:
                    for group_name,_ in hf.items():  # Iterate over thresholds (group names)
                        if group_name not in grouped_data:
                            grouped_data[group_name] = []
                        # shape of combined_data: (criteria, num_enroll_ranges, num_enroll_readings))
                        combined_data = hf[group_name]["combined_data"][:]

                        # Convert to rate results using the provided function
                        combined_data_rate = self.class_instance.get_rates_given_counts_single_threshold(
                            combined_data, code_length, select_threshold
                            ) # Shape: (criteria, num_enroll_ranges, num_enroll_readings)

                        # Initialize placeholders for results
                        reduced_data = {
                            "mean": np.empty_like(combined_data_rate[:, 0]),  
                            "min": np.empty_like(combined_data_rate[:, 0]),
                            "max": np.empty_like(combined_data_rate[:, 0]),
                        }

                        # Apply special handling when condition below is satisfied
                        if self.class_instance == HelperDataComparator:
                            # Special handling for combined_data_rate[0], excluding the first element along axis 1
                            # This criteria is the error_count where we exclude values for first 
                            # enrollment range because it's null value (reference for other ranges)
                            reduced_data["mean"][0] = np.mean(combined_data_rate[0, 1:], axis=0)
                            reduced_data["min"][0] = np.min(combined_data_rate[0, 1:], axis=0)
                            reduced_data["max"][0] = np.max(combined_data_rate[0, 1:], axis=0)
                        else:
                            # Compute normally for combined_data_rate[0]
                            reduced_data["mean"][0] = np.mean(combined_data_rate[0], axis=0)
                            reduced_data["min"][0] = np.min(combined_data_rate[0], axis=0)
                            reduced_data["max"][0] = np.max(combined_data_rate[0], axis=0)

                        # Normal computation for the rest of the array (if it exists)
                        if combined_data_rate.shape[0] > 1:
                            reduced_data["mean"][1:] = np.mean(combined_data_rate[1:], axis=1)
                            reduced_data["min"][1:] = np.min(combined_data_rate[1:], axis=1)
                            reduced_data["max"][1:] = np.max(combined_data_rate[1:], axis=1)

                            
                        # Aggregate over num_enroll_ranges (Axis 1) **per chip**
                        # not considering first element as zero
                        # reduced_data = {
                        #     # Shape: (criteria, num_enroll_readings)
                        #     "mean": np.mean(combined_data_rate, axis=1),
                        #     "min": np.min(combined_data_rate, axis=1),
                        #     "max": np.max(combined_data_rate, axis=1),
                        # }

                        grouped_data[group_name].append(reduced_data) # List of dicts per chip

            # Convert lists to numpy arrays (chips now in Axis 0)
            grouped_data_arrays = {
                group_name: {
                    "mean": np.array([chip_data["mean"] for chip_data in data_list]),
                    "min": np.array([chip_data["min"] for chip_data in data_list]),
                    "max": np.array([chip_data["max"] for chip_data in data_list]),
                }
                for group_name, data_list in grouped_data.items()
            }

            # Final aggregation over all chips (Axis 0)
            aggregated_stats = {
                group_name: {
                    "mean": np.mean(data["mean"], axis=0).astype(np.float64),
                    "min": np.min(data["min"], axis=0).astype(np.float64),
                    "max": np.max(data["max"], axis=0).astype(np.float64),
                }
                for group_name, data in grouped_data_arrays.items()
            }
            aggregated_results[key] = aggregated_stats

        return aggregated_results

    def save_aggregated_results(self, aggregated_results):
        """
        Save the averaged results to a new .h5 file for each parameter set.
        """
        for key, data in aggregated_results.items():
            code_length, select_threshold, num_enroll_readings = key
            output_file_path = self.get_output_filename(code_length, select_threshold, num_enroll_readings)

            with h5py.File(output_file_path, "w") as hf:
                for group_name, stats in data.items():
                    group = hf.create_group(group_name)
                    group.create_dataset("mean", data=stats["mean"], dtype=np.float64,
                                         compression="gzip", compression_opts=9)
                    group.create_dataset("min", data=stats["min"], dtype=np.float64,
                                         compression="gzip", compression_opts=9)
                    group.create_dataset("max", data=stats["max"], dtype=np.float64,
                                         compression="gzip", compression_opts=9)

                print(f"Saved aggregated results to {output_file_path}.")

    def process_and_save(self):
        """
        Perform the entire processing and saving workflow.
        """
        print("Starting data aggregation...")
        aggregated_results = self.aggregate_data()
        print("Data aggregation complete.")
        if aggregated_results:
            self.save_aggregated_results(aggregated_results)
            print("All results saved successfully.")


# Example usage:
if __name__ == "__main__":

    processor = AveragingDataProcessor(enroll_comparator_dir, HelperDataComparator)
    # processor = AveragingDataProcessor(ber_comparator_dir, GlobalBERProcessor)
    start_time = time.time()
    processor.process_and_save()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds.")
