"""A module to handle all plotting functionalities."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import FuncFormatter, ScalarFormatter
# from matplotlib.ticker import LogFormatterSciNotation, LogLocator
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker


class Plotting:
    """A class to handle all plotting functionalities."""

    @staticmethod
    def sample_indices(intervals_length, num_samples=20):
        indices = np.linspace(0, intervals_length - 1, num_samples, dtype=int)
        return indices.tolist()

    @staticmethod
    def plot_2d_line_graphs(x, y, z, xlabel, ylabel, title, legend_label):
        """Plots a 2D line graph."""
        _, ax = plt.subplots(figsize=(12, 6))
        for idx, z_values in enumerate(z):
            ax.plot(x, y[idx, :], label=f'{legend_label}={z_values}') #{z_values:.1f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True)
        plt.show()

    @staticmethod
    def plot_2d_line_graphs_with_intervals_on_xaxis_and_second_yaxis(x, y, z, xlabel, ylabel, title, legend_label,
                                                    second_y=None,second_ylabel=None):
        """Plots a 2D line graph."""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = None
        if second_y is not None:
            ax2 = ax1.twinx()  # Create a second y-axis
            ax2.set_ylabel(second_ylabel, fontsize=16)

        for idx, z_values in enumerate(z):
            # Plot the first measure on ax1
            ax1.plot(np.arange(len(x)), y[:, z_values-1], label=f'{legend_label}={z_values}') #{z_values:.1f}')

            if second_y is not None:
                # Plot the second measure on ax2
                ax2.plot(np.arange(len(x)), second_y[:, z_values-1], linestyle='dashed', label='Second Measure')

        x_labels = [f"[{val[0]:.1f}, {val[1]:.1f}]" for val in x]
        selected_indices = Plotting.sample_indices(len(x_labels), num_samples=15)

        ax1.set_xticks(selected_indices)  # Set tick positions
        ax1.set_xticklabels([x_labels[i] for i in selected_indices],
                            fontsize=10, rotation=-45)  # Set tick labels        
        fig.subplots_adjust(bottom=0.2)

        ax1.set_yscale("log")
        ax1.set_xlabel(xlabel, fontsize=16)
        ax1.set_ylabel(ylabel, fontsize=16)
        ax1.set_title(title, fontsize=16)
        ax1.legend(loc='upper right')

        if ax2:
            ax2.set_yscale("log")
            ax2.legend(loc='upper left')

        ax1.grid(True)
        plt.show()

    @staticmethod
    def plot_2d_line_graphs_with_second_yaxis(x, y, z, xlabel, ylabel, title, legend_label,
                                                    second_y=None,second_ylabel=None):
        """Plots a 2D line graph."""
        print("x=",x)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        color1 = 'blue'
        color2 = 'green'
        ax2 = None
        if second_y is not None:
            ax2 = ax1.twinx()  # Create a second y-axis
            ax2.set_ylabel(second_ylabel, rotation=0, fontsize=16, color=color2,  labelpad=20, weight='bold')

        for idx, z_values in enumerate(z):
            # Determine linestyle
            linestyle = 'dashed' if idx == 0 else 'solid'
            # Plot the first measure on ax1
            ax1.plot(np.arange(len(x)), y[:, z_values-1], linestyle=linestyle, color=color1,
                        label=f'{legend_label}={z_values}') #{z_values:.1f}')

            if second_y is not None:
                # Plot the second measure on ax2
                ax2.plot(np.arange(len(x)), second_y[:, z_values-1], linestyle=linestyle,
                            color=color2, label=f'{legend_label}={z_values}')

        selected_x_indices = [idx for idx, num in enumerate(x) if num.is_integer()]
        selected_x_values = [int(x[i]) for i in selected_x_indices]
        selected_x_labels = [f"{val}" for val in selected_x_values]  # Format labels
        ax1.set_xticks(selected_x_indices, labels=selected_x_labels)

        ax1.set_yscale("log")
        ax1.set_xlabel(xlabel, fontsize=16, weight='bold')
        ax1.set_ylabel(ylabel, rotation=0, fontsize=16, color=color1,  labelpad=20, weight='bold')
        # ax1.set_title(title, fontsize=16)
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.85), fontsize=14)

        if ax2:
            ax2.set_yscale("log")
            ax2.legend(loc='center left', fontsize=14)

        ax1.grid(True)
        # plt.tight_layout()
        plt.savefig("./nvm_free_tmvs/enroll_evaluation_vs_thresholds.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_colorbar(z_min, z_max):

        # Ensure proper log normalization
        norm = mcolors.LogNorm(vmin=z_min, vmax=z_max, clip=True)

        #  Use log-spaced ticks EXACTLY between z_min and z_max
        z_ticks = np.logspace(np.log10(z_min), np.log10(z_max), num=6)

        #  Create a figure and axis for the color bar
        fig, ax = plt.subplots(figsize=(1.5, 6))  # Adjust size if needed

        # Generate the color bar with corrected scaling
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), 
                            ax=ax, orientation="vertical")  # Correct placement

        # Apply the corrected tick positions
        cbar.set_ticks(z_ticks)
        cbar.set_ticklabels([f"$10^{{{int(np.log10(tick))}}}$" for tick in z_ticks])

        # Set colorbar label

        plt.show()

    
    
    @staticmethod
    def plot_3d_surface(x, y, z, xlabel, ylabel, zlabel, title,
                        select_threshold=None, log_scale=False):
        """Plots a 3D surface graph."""
        # X, Y = np.meshgrid(x, y)
        # Flip y-axis (reverse order)
        print("x axis:",x)
        
        z = np.array(z, dtype=np.float64)  # Ensure z is a float array
        z[np.isnan(z)] = np.nan  # Convert None to NaN
        X, Y = np.meshgrid(np.arange(len(x)), np.arange(len(y)))  # Use indices for meshgrid
        print("dtype:",z.dtype)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

        def log_tick_formatter(val, pos=None):
            """Format log scale ticks to display as 10^x notation."""
            return f"$10^{{{int(val)}}}$"

        # Convert invalid values to NaN
        print("min, max z_axis:", np.min(z), np.max(z))  # Ensure no unexpected values
        # z_safe = np.where(z > 0, np.log10(np.maximum(z, 1e-30)), np.nan)
        z_safe = np.where(z == 0, np.nan, z)
        # z_min, z_max = np.min(z[z > 0]), np.max(z)  # Avoid zero or negative values
        z_min, z_max = np.nanmin(z_safe), np.nanmax(z_safe)
        print("max value of z=",z_max)
        print("min value of z=",z_min)
        # # log_min, log_max = np.floor(np.log10(z_min)), np.ceil(np.log10(z_max))
        log_min, log_max = np.log10(z_min), np.log10(z_max)
        print("log_min=",log_min)
        print("log_max=",log_max)
        norm = LogNorm(vmin=z_min, vmax=z_max, clip=True)
        # # norm = LogNorm(vmin=log_min, vmax=log_max)
        
        # mappable = plt.cm.ScalarMappable()
        # mappable.set_array(z_safe)
        
        cmap = plt.cm.get_cmap('viridis')
        
        # surf = ax.plot_surface(X, Y, np.log10(z_safe), cmap='viridis', vmin=log_min, vmax=log_max, edgecolor='none')
        # surf = ax.plot_surface(X, Y, z_safe, cmap='viridis', norm=norm, edgecolor='none')
        surf = ax.plot_surface(X, Y, np.log10(z_safe), cmap='viridis', edgecolor='none')


        # this code line allows colorbar to appear, but use z_safe  and norm
        # fig.colorbar (surf, ax=ax)

        
        
        # cbar.set_ticklabels([f"$10^{{{int(np.log10(tick))}}}$" for tick in z_ticks])
        # Apply the same log scale formatting to the color bar
        # cbar.set_ticks(ax.get_zticks())
        # print("ax.get_zticks()=",ax.get_zticks())
        # print([log_tick_formatter(tick) for tick in ax.get_zticks()])
        # cbar.ax.set_yticklabels([log_tick_formatter(tick) for tick in ax.get_zticks()])
        # cbar.ax.set_yticklabels(mticker.FuncFormatter(log_tick_formatter))
        # cbar.set_label("Z Values (log scale)", fontsize=12, rotation=270, labelpad=15)


        # Highlight selected data in red
        if select_threshold is not None:
            # threshold_index = np.where((y[:, 0] == select_threshold[0]) &
            #                (y[:, 1] == select_threshold[1]))[0]
            threshold_index = np.where(y == select_threshold[1])[0]
            threshold_index_list = threshold_index.tolist()
            threshold_index_list.append(threshold_index+1)
            z_selected = z_safe[threshold_index]
            # Extract corresponding X and Y values
            Y_selected = Y[threshold_index]
            X_selected = X[threshold_index]  # Ensure correct shape
            # print("z_selected shape=",z_selected.shape)
            # print("Y[threshold_index] shape=",Y[threshold_index].shape)
            # print("X[threshold_index] shape=",X[threshold_index].shape)
            ax.scatter(X_selected, Y_selected,z_selected, color='red', s=50, label="Selected Points")
            # ax.plot_surface(X[threshold_index], Y[threshold_index], np.log10(z_selected), color='red', edgecolor='red')

        # ax.set_zscale('log')
        
        
        
         # Apply the custom tick formatter
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        
        # log_min, log_max = np.floor(np.log10(z_min)), np.ceil(np.log10(z_max))
        # print("log_min=",log_min)
        # print("log_max=",log_max)
        # z_ticks = np.logspace(log_min, log_max, num=int(log_max - log_min + 1))
        
        # ax.set_zticks(z_ticks)

        selected_y_indices = [idx for idx, num in enumerate(y) if num.is_integer()]
        selected_y_values = [int(y[i]) for i in selected_y_indices]
        print("selected_y_values=",selected_y_values)
        selected_y_labels = [f"{val}" for val in selected_y_values]  # Format labels
        ax.set_yticks(selected_y_indices, labels=selected_y_labels)
        ax.set_yticklabels(selected_y_labels, fontsize=8)


        # Show only every nth label to avoid overlapping
        # n = max(1, len(y_labels) // 5)  # Adjust step size dynamically
        # n = 5
        # selected_indices = np.arange(0, len(y_labels), step=n)
        
        ax.set_yticks(selected_y_indices, labels=selected_y_labels)  # Set tick positions
        # ax.set_yticklabels(selected_y_labels, fontsize=10,
        #                    rotation=0, verticalalignment='baseline', horizontalalignment='left')  # Set tick labels
        ax.invert_yaxis()  # Flip the y-axis

        x_labels = [f"{val}" for val in x]
        ax.set_xticks(np.arange(len(x)), labels=x_labels)  # Set x-axis ticks
        ax.set_xticklabels(x_labels, fontsize=8)  # Adjust fontsize as needed
        ax.set_xlabel(xlabel, fontsize=14, weight='bold')
        ax.set_ylabel(ylabel, fontsize=14,  weight='bold')
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(zlabel, fontsize=14,  weight='bold', labelpad=15)
        # ax.zaxis.label.set_rotation(0)  # Adjust the rotation angle (0 for horizontal, 90 for vertical)
        # ax.set_title(title, fontsize=16)


    
        plt.savefig("./nvm_free_tmvs/regeneration_3d_evaluation_vs_thresholds_and_readings.pdf",
                    dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_hd_values_histogram(sram_pattern_idx, hamming_distances, chip_id, threshold):
        """
        Plot Hamming distances for all readings for a specific sram_pattern and highlight top-codewords.
        """
        sram_pattern_hd = hamming_distances[sram_pattern_idx]
        # print("hamming distances at first element=",sram_pattern_hd)
        num_codewords, num_readings = sram_pattern_hd.shape

        plt.figure(figsize=(12, 6))
        avg_legend_added = False

        # Assign colors for better visualization
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, num_codewords))

        for codeword_idx, hd_values in enumerate(sram_pattern_hd):
            # Plot histogram-like bars for the current codeword
            x_positions = np.arange(num_readings) + codeword_idx * (num_readings + 1)
            avg_value = np.mean(hd_values)
            bar_color = 'red' if avg_value >= threshold[1] or avg_value <= threshold[0] else colors[codeword_idx]
            # bar_color = colors[codeword_idx]

            # Define label dynamically
            bar_label = r'Codeword $S_{}$'.format(codeword_idx+1)
            if np.any(bar_color == 'red'):
                bar_label += " (Selected)"
                
            plt.bar(
                x_positions, hd_values, label=bar_label,
                color=bar_color, alpha=0.7
            )

            # Draw short dashed line for the average value of each codeword
            if not avg_legend_added:
                plt.hlines(avg_value, x_positions[0], x_positions[-1], colors='blue', linewidth=1.4, label=r"Average $d^*_H$ per Codeword")
                avg_legend_added = True
            else:
                plt.hlines(avg_value, x_positions[0], x_positions[-1], colors='blue', linewidth=1.4)
        

            # Annotate the legend with the index of the readings
            for idx, value in enumerate(hd_values):
                vertical_position = 'bottom'
                if value < 0:
                    vertical_position ='top'
                plt.text(
                    x_positions[idx], value, f"{idx+1}",
                    ha='center', va=vertical_position, fontsize=10
                )

        # Add horizontal dashed lines for -a and a
        plt.axhline(threshold[1], color="gray", linewidth=1.5, linestyle="--", label="Selection Threshold")
        plt.axhline(threshold[0], color="gray", linewidth=1.5, linestyle="--")
        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")

        # Format plot
        # plt.title(f"Hamming Distances for sram_pattern {sram_pattern_idx} for Chip {chip_id}")
        plt.xlabel("Readout Index (grouped by Codewords)", fontsize=16, weight='bold')
        plt.ylabel(r'Modified Hamming Distance ($d^*_H$)', fontsize=16, weight='bold')
        # plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.xticks([])  # Hide x-tick labels for 
        plt.yticks(fontsize=12)
        plt.legend(#title=r"Codewords (Red: |Avg. $d^*_H$| $\geq$ Threshold)", 
                   loc='upper left',
                   fontsize=14, title_fontsize=12)
        plt.tight_layout()
        # plt.grid(True)
        plt.savefig("./nvm_free_tmvs/hamming_distance_histogram.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_2d_plot_with_horizontal_line (x, y, xlabel, ylabel, title, horizontal_line):
        """Plots a 2D line graph."""
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.plot(x, y)
        # ax.axhline(y=horizontal_line, color='r', linestyle='--', label="Target BER")
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        # ax.legend(loc='upper right')
        # ax.grid(True)
        # plt.show()

        _, ax = plt.subplots(figsize=(12, 6))
        # fig, ax1 = plt.subplots(figsize=(10, 5))
        ax.scatter(x, y, alpha=0.5)
        ax.plot(x, y, alpha=0.5)

        # Add a dashed horizontal line at y = 10^-6
        plt.axhline(horizontal_line , color='r', linestyle='--', label=r'Failure rate = $10^{-6}$')
        # label=r'$P_{\text{fail}}$ = $10^{-6}$'

        if not np.any(y == None):
            below_line = y < horizontal_line
            if len(below_line)>0:
                x_below = x[below_line]
                y_below = y[below_line]
                # Highlight points below the horizontal line
                plt.scatter(x_below, y_below, color='red', label='Points below $10^{-6}$')
                # Annotate the points below the horizontal line
                # for xi, yi in zip(x_below, y_below):
                #     plt.annotate(f'{yi:.2e}', (xi, yi), textcoords="offset points",
                #       xytext=(0, -13), ha='center', fontsize=9, color='red')
                #     # only first element
                #     break

        # y-axis format
        # plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        # plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # Set y-axis to scientific notation
        ax.set_yscale("log")  # Log scale to ensure proper display of 10^-4, 10^-5, etc.
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())

        # x-axis format
        # Format x labels
        x_labels = [f"{val:.1f} kB" for val in x]

        # nnotate x-axis values on the plot
        for xi, yi, label in zip(x, y, x_labels):
            if yi is not None:
                ax.text(70+ xi, yi*1.1 , label, ha='right', fontsize=16, color='black')

        # x-axis format

        # ax.set_xticks(x)
        # ax.set_xticklabels(x_labels, rotation=0)

        # Dynamic x-axis ticks generation**
        min_x = np.min(x)
        max_x = np.max(x)

        # Find the first multiple of 100 greater than min_x
        start = np.ceil(min_x / 100) * 100
        # Find the last multiple of 100 less than or equal to max_x
        end = np.floor(max_x / 100) * 100

        # Generate tick positions at multiples of 100
        xtick_positions = np.arange(start, end + 100, 100)

        # Set x-axis ticks and labels
        ax.set_xticks(xtick_positions)
        x_labels = [f"{int(val)}" for val in xtick_positions]  # Format as integers
        ax.set_xticklabels(x_labels, fontsize=16)

        ax.set_ylabel(ylabel, fontsize=16, weight='bold')
        ax.set_xlabel(xlabel, fontsize=16, labelpad=20, weight='bold')
        # ax.set_title(title)
        # plt.legend(loc=1, fontsize=16)
        ax.legend(loc='upper right', fontsize=16)
        plt.savefig("./nvm_free_tmvs/optimal_failure_rate_vs_memory.pdf", dpi=300, bbox_inches='tight')
        plt.show()
        
