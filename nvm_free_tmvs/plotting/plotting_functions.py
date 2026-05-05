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
import matplotlib.transforms as mtransforms


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

        linestyles = ['dashed', 'dotted', 'solid', 'dashdot']
        for idx, z_values in enumerate(z):
            linestyle = linestyles[idx % len(linestyles)]
            # Round label: 99->100, 499->500, 1->1
            label_val = round(z_values, -1) if z_values >= 5 else z_values
            # Plot the first measure on ax1 (with label for combined legend)
            ax1.plot(np.arange(len(x)), y[:, z_values-1], linestyle=linestyle, color=color1,
                        label=f'$N_{{\\mathrm{{res}}}}$ = {label_val}')

            if second_y is not None:
                # Plot the second measure on ax2 (no label — shared legend from ax1)
                ax2.plot(np.arange(len(x)), second_y[:, z_values-1], linestyle=linestyle,
                            color=color2)

        selected_x_indices = [idx for idx, num in enumerate(x) if num.is_integer()]
        selected_x_values = [int(x[i]) for i in selected_x_indices]
        selected_x_labels = [f"{val}" for val in selected_x_values]  # Format labels
        ax1.set_xticks(selected_x_indices, labels=selected_x_labels)

        ax1.set_yscale("log")
        ax1.set_xlabel(xlabel, fontsize=16, weight='bold')
        ax1.set_ylabel(ylabel, rotation=0, fontsize=16, color=color1, labelpad=20, weight='bold',
                       verticalalignment='top')
        ax1.legend(loc='lower center', fontsize=14)

        if ax2:
            ax2.set_yscale("log")

        ax1.grid(True)
        plt.savefig("./nvm_free_tmvs/figures/enroll_evaluation_vs_thresholds.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_2d_line_graphs_with_second_yaxis_and_overlay(
        x, y_main, y_overlay, z, xlabel, ylabel, title, legend_label,
        overlay_label="previous-work", second_y_main=None, second_y_overlay=None, second_ylabel=None
    ):
        """Plots a 2D line graph overlaying two datasets on the same axes, with optional second y-axis."""
        fig, ax1 = plt.subplots(figsize=(10, 5))
        color_main = 'blue'
        color_overlay = 'orange'
        color_second = 'green'

        ax2 = None
        if second_y_main is not None:
            ax2 = ax1.twinx()
            ax2.set_ylabel(second_ylabel, rotation=0, fontsize=16, color=color_second,  labelpad=20, weight='bold')

        for idx, z_values in enumerate(z):
            # Determine linestyle consistently
            if idx == 0:
                linestyle = 'dashed'
            elif idx == 1:
                linestyle = 'dotted'
            else:
                linestyle = 'solid'

            # Main dataset
            ax1.plot(np.arange(len(x)), y_main[:, z_values-1], linestyle=linestyle, color=color_main,
                        label=f'{legend_label}={z_values}')

            # Overlay dataset
            ax1.plot(np.arange(len(x)), y_overlay[:, z_values-1], linestyle=linestyle, color=color_overlay,
                        label=f'{legend_label}={z_values} ({overlay_label})')

            if ax2 is not None and second_y_main is not None:
                ax2.plot(np.arange(len(x)), second_y_main[:, z_values-1], linestyle=linestyle,
                            color=color_second, label=f'PSR')
                if second_y_overlay is not None:
                    ax2.plot(np.arange(len(x)), second_y_overlay[:, z_values-1], linestyle=linestyle,
                                color='lime', label=f'PSR ({overlay_label})')

        selected_x_indices = [idx for idx, num in enumerate(x) if float(num).is_integer()]
        selected_x_values = [int(x[i]) for i in selected_x_indices]
        selected_x_labels = [f"{val}" for val in selected_x_values]
        ax1.set_xticks(selected_x_indices, labels=selected_x_labels)

        ax1.set_yscale("log")
        ax1.set_xlabel(xlabel, fontsize=16, weight='bold')
        ax1.set_ylabel(ylabel, rotation=0, fontsize=16, color=color_main,  labelpad=20, weight='bold')
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.8), fontsize=14)

        if ax2:
            ax2.set_yscale("log")
            ax2.legend(loc='center left',bbox_to_anchor=(0, 0.35), fontsize=14)

        ax1.grid(True)
        try:
            plt.tight_layout()
        except Exception:
            pass
        plt.show()

    @staticmethod
    def plot_2d_overlay_with_bands(
        x,
        a_mean, a_min, a_max, a_label,
        b_mean, b_min, b_max, b_label,
        xlabel, ylabel,
        second_y_a_mean=None, second_y_a_min=None, second_y_a_max=None,
        second_y_b_mean=None, second_y_b_min=None, second_y_b_max=None,
        second_ylabel=None,
        ylabel_rotation=0,
        x_log=False,
        jump_tick_value=None,
        scaling_flag=False,
    ):
        """Plot two approaches (A and B) on the same primary axis with mean and min/max bands.
        Optionally plot two second-axis series (A and B) with bands.
        All series are 1D arrays aligned to x.
        """
        fig, ax1 = plt.subplots(figsize=(10, 5))
        # Prepare x positions
        x_arr = np.array(x)
        print("x_arr=",x_arr)
        print("jump_tick_value=",jump_tick_value)
        print("scaling_flag=",scaling_flag)
        if scaling_flag and jump_tick_value:
            x_plot = x_arr / float(jump_tick_value)
        else:
            x_plot = np.arange(len(x_arr))
        print("x_plot=",x_plot)
        # Colors: A (ODHD) = blue, B (Dark Bit) = red
        a_color = 'blue'
        b_color = 'red'
        sel_color = 'green'
        # Primary axis: scatter only (no fill bands)
        line_our_ber = ax1.scatter(x_plot, a_mean, color=a_color, s=4, label=f"{a_label}", zorder=3)
        line_dark_ber = ax1.scatter(x_plot, b_mean, color=b_color, s=4, label=f"{b_label}", zorder=3)

        ax1.set_yscale("log")
        # X-axis scaling and ticks
        if x_log:
            try:
                ax1.set_xscale('log')
            except Exception:
                pass
        ax1.set_xlabel(xlabel, fontsize=16, weight='bold')
        ax1.set_ylabel(ylabel, rotation=ylabel_rotation, fontsize=16, color='black', labelpad=20, weight='bold')
        # x tick selection

        # Set evenly spaced ticks based on final x value or provided jump
        max_x = int(x_arr[-1]) if x_arr.size > 0 else 0
        step = int(jump_tick_value) if jump_tick_value else (max(1, max_x // 10) if max_x > 0 else 1)
        tick_values = list(range(step, max_x + 1, step))
        if tick_values:
            if scaling_flag and jump_tick_value:
                tick_positions = [tv / float(jump_tick_value) for tv in tick_values]
                tick_labels = [f"{int(tv/float(jump_tick_value))}" for tv in tick_values]
            else:
                # positions are indices matching values in x
                tick_positions = [np.where(x_arr == tv)[0][0] if np.any(x_arr == tv) else tv for tv in tick_values]
                tick_labels = [f"{tv}" for tv in tick_values]
            ax1.set_xticks(tick_positions, labels=tick_labels)

        # Add scaling indicator at end of x-axis if scaled
        if scaling_flag and jump_tick_value:
            try:
                trans = mtransforms.blended_transform_factory(ax1.transAxes, ax1.get_xaxis_transform())
                ax1.text(1.01, 0, 'x100', transform=trans,
                         ha='left', va='center', fontsize=12)
            except Exception:
                pass

        ax2 = None
        line_our_sel = None
        line_dark_sel = None
        if second_y_a_mean is not None or second_y_b_mean is not None:
            ax2 = ax1.twinx()
            ax2.set_ylabel(second_ylabel or '', rotation=0, fontsize=16, color=sel_color, labelpad=20, weight='bold')
            # A second axis series (our selection): solid
            if second_y_a_mean is not None:
                line_our_sel, = ax2.plot(x_plot, second_y_a_mean, color=sel_color, linestyle='solid')
                if second_y_a_min is not None and second_y_a_max is not None:
                    ax2.fill_between(x_plot, second_y_a_min, second_y_a_max, color=sel_color, alpha=0.15)
            # B second axis series (dark bits selection): dashed
            if second_y_b_mean is not None:
                line_dark_sel, = ax2.plot(x_plot, second_y_b_mean, color=sel_color, linestyle='dashed')
                if second_y_b_min is not None and second_y_b_max is not None:
                    ax2.fill_between(x_plot, second_y_b_min, second_y_b_max, color=sel_color, alpha=0.08)
            # Display second axis values (PSR, discarded) as percentages
            try:
                ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            except Exception:
                pass

        # Single legend with combined blue+green entries for each approach (solid vs dashed)
        try:
            from matplotlib.legend_handler import HandlerTuple
            handles = []
            labels = []
            if line_our_sel is not None:
                handles.append((line_our_ber, line_our_sel))
                labels.append('ODHD')
            else:
                handles.append(line_our_ber)
                labels.append('ODHD')
            if line_dark_sel is not None:
                handles.append((line_dark_ber, line_dark_sel))
                labels.append(b_label)
            else:
                handles.append(line_dark_ber)
                labels.append(b_label)
            ax1.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='upper right', fontsize=14)
        except Exception:
            # Fallback to simple legend if HandlerTuple unavailable
            ax1.legend(loc='upper right', fontsize=14)
        if ax2:
            ax2.set_yscale("log")
        ax1.grid(True)
        try:
            plt.tight_layout()
        except Exception:
            pass
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


        # Colorbar with log-scale tick labels
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.15)
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))


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
        # Limit to ~10 ticks to avoid overlapping
        if len(selected_y_values) > 10:
            step = -(-len(selected_y_values) // 10)  # ceil division
            selected_y_indices = selected_y_indices[::step]
            selected_y_values = selected_y_values[::step]
        selected_y_labels = [f"{val}" for val in selected_y_values]
        ax.set_yticks(selected_y_indices)
        ax.set_yticklabels(selected_y_labels, fontsize=8)
        ax.invert_yaxis()  # Flip the y-axis

        # Show round-number ticks on x-axis, always including first and last
        x_min, x_max = int(x[0]), int(x[-1])
        x_range = x_max - x_min
        # Pick a round step that yields ~10 ticks
        raw_step = x_range / 10
        for nice in [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500]:
            if nice >= raw_step:
                x_step = nice
                break
        else:
            x_step = raw_step
        # Build tick values: first, then round multiples, then last
        tick_vals = [x_min]
        v = int(np.ceil(x_min / x_step)) * x_step
        while v < x_max:
            if v > x_min:
                tick_vals.append(v)
            v += x_step
        tick_vals.append(x_max)
        # Map tick values to indices in x
        x_arr = np.array(x)
        x_tick_indices = [int(np.argmin(np.abs(x_arr - v))) for v in tick_vals]
        x_tick_labels = [f"{v}" for v in tick_vals]
        ax.set_xticks(x_tick_indices)
        ax.set_xticklabels(x_tick_labels, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=14, weight='bold')
        ax.set_ylabel(ylabel, fontsize=14,  weight='bold')
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(zlabel, fontsize=14,  weight='bold', labelpad=15)
        # ax.zaxis.label.set_rotation(0)  # Adjust the rotation angle (0 for horizontal, 90 for vertical)
        # ax.set_title(title, fontsize=16)


    
        plt.subplots_adjust(bottom=0.15, left=0.05, right=0.85)
        plt.savefig("./nvm_free_tmvs/figures/regeneration_3d_evaluation_vs_thresholds_and_readings.pdf",
                    dpi=300, bbox_inches='tight', pad_inches=0.5)

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
            bar_label = r'Codeword $S^{}$'.format(codeword_idx+1)
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
        plt.savefig("./nvm_free_tmvs/figures/hamming_distance_histogram.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_2d_plots_with_horizontal_line (x_list, y_list, xlabel, ylabel, title, horizontal_line,
                                           nb_enroll_reading_list):
        """Plots multiple 2D line graphs based on the given x and y datasets."""

        _, ax = plt.subplots(figsize=(12, 6))
        
        points_below_label_added = False  # Ensure 'Points below $10^{-6}$' label is added only once
        flag = False
        plt.axhline(horizontal_line , color='r', linestyle='--', label=r'Failure rate = $10^{-6}$')

        for x, y, label_val in zip(x_list, y_list, nb_enroll_reading_list):
            
            below_line = y < horizontal_line
            # Highlight points below the horizontal line in red
            if not np.any(y == None):
                if np.any(below_line):
                    x_below = x[below_line]
                    y_below = y[below_line]
                    # Highlight points below the horizontal line
                    if not points_below_label_added:
                        plt.scatter(x_below, y_below, color='red', label='Points below $10^{-6}$',
                                    zorder=5)
                        points_below_label_added = True
                    else:
                        plt.scatter(x_below, y_below, color='red', zorder=5)

            ax.scatter(x, y, alpha=0.5, label=f'$N_{{\\mathrm{{res}}, \\max}}$ = {label_val}'
                       , zorder=4)
            ax.plot(x, y, alpha=0.5)

            # Annotate x-axis values on the plot
            for xi, yi in zip(x, y):
                if yi is not None:
                    if not flag:
                        ax.text(xi+50, yi * 1.1, f"{xi:.1f} kB", ha='right', fontsize=12, color='black')
                    else:
                        ax.text( xi-40, yi*1.1 , f"{xi:.1f} kB", ha='left', fontsize=12, color='black')
            flag = True    
            
        # Add a dashed horizontal line at y = 10^-6
        ax.set_yscale("log")
        plt.gca().yaxis.set_major_formatter(LogFormatterSciNotation())
        
        # Dynamic x-axis ticks generation
        all_x = np.concatenate(x_list)
        min_x, max_x = np.min(all_x), np.max(all_x)
        start = np.ceil(min_x / 100) * 100
        end = np.floor(max_x / 100) * 100
        xtick_positions = np.arange(start, end + 100, 100)
        plt.xticks(xtick_positions, [f"{int(val)}" for val in xtick_positions], fontsize=16)
        
        plt.xlabel(xlabel, fontsize=16, labelpad=20, weight='bold')
        plt.ylabel(ylabel, fontsize=16, weight='bold')
        plt.legend(loc='upper right', fontsize=16)
        # plt.title(title, fontsize=18, weight='bold')

        plt.savefig("./nvm_free_tmvs/figures/optimal_failure_rate_vs_memory.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_2d_overlay_with_scatter(
        x,
        a_mean, a_min, a_max, a_label,
        scatter_x, scatter_y, scatter_label,
        xlabel, ylabel,
        ylabel_rotation=0,
        x_log=False,
        jump_tick_value=None,
        scaling_flag=False,
        scatter_color='red',
        scatter_alpha=0.6,
        scatter_size=20,
    ):
        """Plot a line with bands (A) and overlay scatter points.
        
        Args:
            x: X-axis values for the line plot
            a_mean, a_min, a_max: Mean, min, max values for the line plot with bands
            a_label: Label for the line plot
            scatter_x, scatter_y: X and Y coordinates for scatter points
            scatter_label: Label for scatter points
            xlabel, ylabel: Axis labels
            ylabel_rotation: Rotation angle for y-axis label
            x_log: Whether to use log scale for x-axis
            jump_tick_value: Value for x-axis scaling
            scaling_flag: Whether to apply scaling to x-axis
            scatter_color: Color for scatter points
            scatter_alpha: Transparency for scatter points
            scatter_size: Size of scatter points
        """
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Prepare x positions for line plot
        x_arr = np.array(x)
        if scaling_flag and jump_tick_value:
            x_plot = x_arr / float(jump_tick_value)
        else:
            x_plot = np.arange(len(x_arr))
        
        # Plot line with bands
        line_main, = ax1.plot(x_plot, a_mean, color='blue', linestyle='solid', label=a_label, linewidth=2)
        ax1.fill_between(x_plot, a_min, a_max, color='blue', alpha=0.15)
        
        # Plot scatter points
        ax1.scatter(scatter_x, scatter_y, color=scatter_color, alpha=scatter_alpha, 
                   s=scatter_size, label=scatter_label)
        
        # Set scales
        ax1.set_yscale("log")
        if x_log:
            try:
                ax1.set_xscale('log')
            except Exception:
                pass
        
        # Labels and styling
        ax1.set_xlabel(xlabel, fontsize=16, weight='bold')
        # Ensure LaTeX y-label appears bold when mathtext is used (wrap with \mathbf{})
        if isinstance(ylabel, str) and ylabel.startswith('$') and ylabel.endswith('$') and len(ylabel) > 2:
            inner = ylabel[1:-1]
            ylabel_to_use = f'$\\mathbf{{{inner}}}$'
        else:
            ylabel_to_use = ylabel
        ax1.set_ylabel(ylabel_to_use, rotation=ylabel_rotation, fontsize=16, color='black', labelpad=20, weight='bold')
        
        # X-axis ticks
        max_x = int(x_arr[-1]) if x_arr.size > 0 else 0
        if jump_tick_value and scaling_flag:
            tick_positions = np.arange(0, max_x + jump_tick_value, jump_tick_value)
            tick_labels = [f"{int(pos)}" for pos in tick_positions]
            ax1.set_xticks(tick_positions / jump_tick_value)
            ax1.set_xticklabels(tick_labels, fontsize=14)
        else:
            ax1.tick_params(axis='x', labelsize=14)
        
        ax1.tick_params(axis='y', labelsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=14)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_2d_dual_scatter(
        scatter1_x, scatter1_y, scatter1_label,
        scatter2_x, scatter2_y, scatter2_label,
        xlabel, ylabel,
        ylabel_rotation=0,
        x_log=False,
        jump_tick_value=None,
        scaling_flag=False,
        scatter1_color='blue',
        scatter2_color='red',
        scatter_alpha=0.6,
        scatter_size=20,
    ):
        """Plot two scatter plots on the same figure with different x-axis ranges.
        
        Args:
            scatter1_x, scatter1_y: X and Y coordinates for first scatter plot
            scatter1_label: Label for first scatter plot
            scatter2_x, scatter2_y: X and Y coordinates for second scatter plot
            scatter2_label: Label for second scatter plot
            xlabel, ylabel: Axis labels
            ylabel_rotation: Rotation angle for y-axis label
            x_log: Whether to use log scale for x-axis
            jump_tick_value: Value for x-axis scaling
            scaling_flag: Whether to apply scaling to x-axis
            scatter1_color, scatter2_color: Colors for scatter points
            scatter_alpha: Transparency for scatter points
            scatter_size: Size of scatter points
        """
        fig, ax1 = plt.subplots(figsize=(8, 4))
        
        # Convert to numpy arrays and filter out invalid values
        scatter1_x_arr = np.array(scatter1_x, dtype=float)
        scatter1_y_arr = np.array(scatter1_y, dtype=float)
        scatter2_x_arr = np.array(scatter2_x, dtype=float)
        scatter2_y_arr = np.array(scatter2_y, dtype=float)
        
        # Filter out NaN and infinite values (for log scale, y must be > 0)
        valid1 = np.isfinite(scatter1_x_arr) & np.isfinite(scatter1_y_arr) & (scatter1_y_arr > 0) & (scatter1_x_arr > 0)
        valid2 = np.isfinite(scatter2_x_arr) & np.isfinite(scatter2_y_arr) & (scatter2_y_arr > 0) & (scatter2_x_arr > 0)
        
        scatter1_x_arr = scatter1_x_arr[valid1]
        scatter1_y_arr = scatter1_y_arr[valid1]
        scatter2_x_arr = scatter2_x_arr[valid2]
        scatter2_y_arr = scatter2_y_arr[valid2]
        
        # Optional x-axis scaling by jump_tick_value (e.g., show x in units of 100 with 'x100' indicator)
        if scaling_flag and jump_tick_value:
            scale = float(jump_tick_value)
            x1_plot = scatter1_x_arr / scale
            x2_plot = scatter2_x_arr / scale
        else:
            x1_plot = scatter1_x_arr
            x2_plot = scatter2_x_arr

        # Plot first scatter (ODHD)
        if x1_plot.size > 0:
            ax1.scatter(x1_plot, scatter1_y_arr, color=scatter1_color, alpha=scatter_alpha,
                        s=scatter_size, label=scatter1_label)
        # Plot second scatter (Bernardini)
        if x2_plot.size > 0:
            ax1.scatter(x2_plot, scatter2_y_arr, color=scatter2_color, alpha=scatter_alpha,
                        s=scatter_size, label=scatter2_label)
        
        # Set scales
        ax1.set_yscale("log")
        if x_log:
            try:
                ax1.set_xscale('log')
            except Exception:
                pass
        
        # Labels and styling
        ax1.set_xlabel(xlabel, fontsize=16, weight='bold')
        # Ensure LaTeX y-label appears bold when mathtext is used.
        # Prefer replacing the first \mathrm{...} with \mathbf{...} (so subscripts like _\mathrm{Reg} stay non-bold),
        # otherwise wrap the entire math expression in \mathbf{...}.
        ylabel_to_use = ylabel
        if isinstance(ylabel, str) and ylabel.startswith('$') and ylabel.endswith('$') and len(ylabel) > 2:
            inner = ylabel[1:-1]
            if '\\mathrm{' in inner:
                # replace only the first occurrence
                idx = inner.find('\\mathrm{')
                close_idx = inner.find('}', idx + len('\\mathrm{'))
                if close_idx != -1:
                    token = inner[idx + len('\\mathrm{'):close_idx]
                    inner_bold = inner[:idx] + f'\\mathbf{{{token}}}' + inner[close_idx+1:]
                    ylabel_to_use = f'$' + inner_bold + '$'
                else:
                    ylabel_to_use = f'$\\mathbf{{{inner}}}$'
            else:
                ylabel_to_use = f'$\\mathbf{{{inner}}}$'
        ax1.set_ylabel(ylabel_to_use, rotation=ylabel_rotation, fontsize=16, color='black', labelpad=20, weight='bold')
        
        # X-axis ticks consistent with overlay style (factor x100 when scaling_flag=True)
        max_x_orig = max(
            np.max(scatter1_x_arr) if scatter1_x_arr.size > 0 else 0,
            np.max(scatter2_x_arr) if scatter2_x_arr.size > 0 else 0,
        )
        if jump_tick_value:
            step = int(jump_tick_value)
            tick_values = list(range(step, int(max_x_orig) + 1, step))
            if scaling_flag:
                tick_positions = [tv / float(step) for tv in tick_values]
                tick_labels = [f"{int(tv/float(step))}" for tv in tick_values]
            else:
                tick_positions = tick_values
                tick_labels = [f"{tv}" for tv in tick_values]
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels(tick_labels, fontsize=14)
        else:
            ax1.tick_params(axis='x', labelsize=14)

        # Add scaling indicator text (e.g., 'x100')
        if scaling_flag and jump_tick_value:
            try:
                import matplotlib.transforms as mtransforms
                trans = mtransforms.blended_transform_factory(ax1.transAxes, ax1.get_xaxis_transform())
                ax1.text(1.01, 0, f"x{int(jump_tick_value)}", transform=trans,
                         ha='left', va='center', fontsize=12)
            except Exception:
                pass
        
        ax1.tick_params(axis='y', labelsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=14, loc='center right')
        
        plt.tight_layout()
        plt.savefig("./nvm_free_tmvs/figures/odhd_vs_dark_bit.pdf", dpi=300, bbox_inches='tight')

        plt.show()
        
