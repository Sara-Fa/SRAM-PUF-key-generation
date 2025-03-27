""" Read aggregated .h5 files and prepare data for plotting. """
import h5py
import numpy as np
from nvm_free_tmvs.utils.file_manager import ber_comparator_dir
from nvm_free_tmvs.utils.file_manager import enroll_comparator_dir
from nvm_free_tmvs.utils.file_manager import read_codebook
from nvm_free_tmvs.utils.analysis_utils import get_shifted_selection_threshold
from nvm_free_tmvs.plotting.plotting_functions import Plotting


class AggregatedDataReader:
    """
    A class to read aggregated .h5 files and prepare data for plotting.
    """

    def __init__(self, code_length, select_threshold, num_enroll_readings, directory):
        """
        Initialize with parameters to locate the correct aggregated .h5 file.
        """
        self.code_length = code_length
        self.select_threshold = select_threshold
        self.num_enroll_readings = num_enroll_readings
        self.directory = directory
        self.file_path = self._find_aggregated_file()

    def _find_aggregated_file(self):
        """
        Find the appropriate aggregated .h5 file based on the naming pattern.
        """
        file_pattern = (f"aggregated_code_N{self.code_length}_Threshold_{self.select_threshold[0]}_"
                        f"{self.select_threshold[1]}_MaxEnrollReadings_{self.num_enroll_readings}.h5")

        file_path = self.directory / file_pattern
        if file_path.exists():
            return file_path
        raise FileNotFoundError(f"Aggregated file not found: {file_pattern}")

    def read_aggregated_data(self):
        """
        Read the aggregated .h5 file and extract data for plotting.
        
        Returns:
            x_axis: Num enroll readings
            y_axis: Enroll select threshold values
            z_axis: Dictionary of mean, min, max values
        """
        x_axis = None
        y_axis = []
        z_axis = {"mean": [], "min": [], "max": []}
        temp_list = []  # Temporary list to store tuples of (key, data)

        with h5py.File(self.file_path, "r") as hf:
            for group_name, data_list in hf.items():
                # Extract enroll_select_threshold from group_name
                parts = group_name.split("_")[1:]
                enroll_select_threshold = (float(parts[0] + '.' + parts[1]),
                                           float(parts[2] + '.' + parts[3]))

                # Read datasets (mean, min, max)
                mean_data = data_list["mean"][:]
                min_data = data_list["min"][:]
                max_data = data_list["max"][:]
                # print("dtype:", mean_data.dtype)

                # Extract num_enroll_readings from shape
                num_enroll_readings = mean_data.shape[1]  # (criteria, num_enroll_readings)
                assert num_enroll_readings == self.num_enroll_readings, "Mismatch in num_enroll_readings"

                if x_axis is None:
                    x_axis = 1 + np.arange(num_enroll_readings)  # Define x-axis values

                # Store extracted values for sorting
                temp_list.append((enroll_select_threshold, mean_data, min_data, max_data))


        # Sort based on y_axis (enroll_select_threshold)
        temp_list.sort(key=lambda item: item[0], reverse=True)  # Sort by (-a, a)

        # Unpack sorted data
        y_axis, mean_sorted, min_sorted, max_sorted = zip(*temp_list)

        # Convert to numpy arrays
        y_axis = np.array(y_axis)
        z_axis["mean"] = np.array(mean_sorted)
        z_axis["min"] = np.array(min_sorted)
        z_axis["max"] = np.array(max_sorted)

        return x_axis, y_axis, z_axis

    @staticmethod
    def plot_2d_results(range_enroll_readings, select_margin, ber_results,
                        selection_rate_results=None, select_threshold=None):
        """ Plot results. """
        # 2D line graphs for BER and validity rates
        Plotting.plot_2d_line_graphs(x=range_enroll_readings,
                                     y=ber_results, z=select_margin,
                                     xlabel='Number of Readings', ylabel='BER',
                                     title='BER vs Number of Readings for Different Thresholds',
                                     legend_label='Threshold')

        if selection_rate_results is not None:
            Plotting.plot_2d_line_graphs(x=range_enroll_readings,
                                        y=selection_rate_results, z=select_margin,
                                        xlabel='Number of Readings', ylabel='BER',
                                        title='Selection Rate vs Number of Readings for Different Thresholds',
                                        legend_label='Threshold')

    @staticmethod
    def plot_3d_results(range_enroll_readings, select_margin, ber_results,
                        xlabel, ylabel, zlabel,
                        selection_rate_results=None,
                     select_threshold=None):
        """ Plot results. """  

        # 3D surface plots for BER and validity rates
        Plotting.plot_3d_surface(x=range_enroll_readings,
                                 y=select_margin, z=ber_results,
                                 xlabel=xlabel, ylabel=ylabel,
                                 zlabel=zlabel,
                                 title='BER vs Thresholds and Number of Readings',
                                 select_threshold=select_threshold, log_scale=True)
        if selection_rate_results is not None:
            Plotting.plot_3d_surface(x=range_enroll_readings,
                                    y=select_margin, z=selection_rate_results,
                                    xlabel=xlabel, ylabel=ylabel,
                                    zlabel=zlabel,
                                    title='Selection Rate vs Thresholds and Number of Readings',
                                    select_threshold=select_threshold)

# Example usage:
if __name__ == "__main__":
    # Define the enroll_select_threshold value we want
    num_enroll_reading = 10
    dir_name = enroll_comparator_dir
    # dir_name = ber_comparator_dir

    all_parameters = [(7,1,6), (9,1,8), (11, 1, 10), (11, 2, 9), (13, 1, 12), (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]
    parameters = [(27, 3, 24)]

    coeff = [0,0]
    for code_len, coeff[0], coeff[1] in parameters:
        select_th = get_shifted_selection_threshold(code_len, coeff)
        print("\ncode_length=", code_len, ", threshold=",coeff,
              "shifted_th=", select_th)
        codebook_length = len(read_codebook(code_len,
                                 coeff[0],
                                 coeff[1]))
        print("codebook_length=", codebook_length)
        target_threshold = select_th
        # target_threshold = [-13,13]
    
        # code_len = 17
        # select_th = [1, 16]

        # Initialize the reader
        reader = AggregatedDataReader(code_len, coeff, num_enroll_reading, dir_name)
        x_axis_val, y_axis_val, z_axis_val = reader.read_aggregated_data()
        
        # Find the index where y_axis (enroll_select_threshold) matches (-3.0, 3.0)
        threshold_index = np.where((y_axis_val[:, 0] == target_threshold[0]) &
                            (y_axis_val[:, 1] == target_threshold[1]))[0]

        # print("x_axis:", x_axis_val)
        # print("y_axis:", y_axis_val)
        z_axis_mean = np.array(z_axis_val["mean"])
        z_axis_min = np.array(z_axis_val["min"])
        z_axis_max = np.array(z_axis_val["max"])
        
        # print("z_axis[\"mean\"] shape:", z_axis_mean.shape)
        # print("z_axis[\"mean\"] dtype:", z_axis_mean.dtype)
        # print("Results at threshold:", threshold_index)
        
        # BER results
        # print("Mean BER at select_th:", z_axis_mean[threshold_index,0,:])
        # print("Mean Selection rate at select_th:", 1 - z_axis_mean[:,1,:])
        # reader.plot_3d_results(x_axis_val, y_axis_val, z_axis_mean[:,0,:])
        
        # print the results for helper data comparison
        print("error_count z_axis[\"mean\"]:", z_axis_mean[threshold_index,0,:])
        print("discarded_patterns_count z_axis[\"mean\"]:", z_axis_mean[threshold_index,1,:])
        # # print("error_count z_axis[\"min\"]:", z_axis_min[threshold_index,0,:])
        # # print("error_count z_axis[\"max\"]:", z_axis_max[threshold_index,0,:])
        # print("Mean Extraction rate at select_th:", z_axis_mean[threshold_index,4,:])
        # reader.plot_3d_results(x_axis_val, y_axis_val, z_axis_mean[:,0,:], 1 - z_axis_mean[:,1,:],
        #                     select_th)
        reader.plot_2d_results(x_axis_val, y_axis_val[-10:], z_axis_mean[:,0,:][-10:],  None,
                            select_th)
        