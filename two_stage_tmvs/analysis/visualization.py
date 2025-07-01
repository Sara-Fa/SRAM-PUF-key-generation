""" Visualization functions for Two-Stage TMVS analysis."""
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import pandas as pd
from two_stage_tmvs.analysis_constants import PLOT_DIR, MIN_TARGET_ERROR_PROB, MAX_CODE_LENGTH


def set_ieee_plot_style(use_latex=False, single_column=False):
    """Set global matplotlib style for IEEE-compatible plots."""

    figsize = (3.5, 2.8) if single_column else (7.2, 4.5)

    # Color-blind-friendly colors
    cbf_colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']

    plt.rcParams.update({
        # Font setup
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,                # Base text size
        'axes.titlesize': 12.5,         # Title size
        'axes.labelsize': 11.5,         # Axis label size
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 10.5,
        'figure.titlesize': 12.5,

        # Figure & output resolution
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': figsize,

        # Line styles
        'lines.linewidth': 1.4,
        'lines.markersize': 6,

        # Color cycle
        'axes.prop_cycle': plt.cycler(color=cbf_colors),

        # PDF font compatibility
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        })


def generate_analysis_plots(results, target_error=1e-9, simple_results=None):
    """Generate three key analysis plots"""
    print(f"\nStarting plot generation (target_error={target_error})...")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.DataFrame(results)
        simple_df = pd.DataFrame(simple_results) if simple_results is not None else None
        print(f"Processing {len(df)} data points...")

        # Plot 1
        print("Generating SRAM trajectory plot...")
        plot_sram_trajectory(df, target_error)

        # Plot 2
        print("Generating error-SRAM frontiers plot...")
        plot_error_sram_frontiers(df, target_error)

        # Plot 3
        print("Generating configuration heatmap...")
        plot_configuration_heatmap(df, target_error)

        # # Plot 4
        print("Generating simple vs. concatenated SRAM plot...")
        plot_sram_vs_pflip_simple_vs_concat(df, target_error, simple_df)

        print(f"All plots saved to: {PLOT_DIR.absolute()}")
    except Exception as e:
        print(f"Error during plot generation: {str(e)}")
        raise

def plot_sram_trajectory(df, target_error):
    """Plot minimum SRAM and helper data size configurations across p_flip values"""
    set_ieee_plot_style()
    try:
        fig, ax1 = plt.subplots()

        # Find minimal SRAM config for each p_flip meeting target
        optimal_configs = (df[df['error_prob'] <= target_error]
                         .groupby('p_flip', as_index=False)
                         .apply(lambda x: x.nsmallest(1, 'sram_size'))
                         .reset_index(drop=True))

        # Compute global min/max across both metrics
        sram = optimal_configs['sram_size']
        helper = optimal_configs['helper_data_size']
        y_min = min(sram.min(), helper.min())
        y_max = max(sram.max(), helper.max())

        # Define shared y-ticks (e.g., 6 evenly spaced ticks)
        num_ticks = 6
        yticks = np.linspace(y_min, y_max, num_ticks)

        # Plot SRAM size (primary y-axis)
        color = 'tab:blue'
        ax1.plot(optimal_configs['p_flip'], sram,
                'o-', color=color, lw=2, label='SRAM Size')
        ax1.set_xlabel(r'Bit Flipping Probability ($p_e$)')
        ax1.set_ylabel('SRAM Size (kiB)', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yticks(yticks)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.grid(True, alpha=0.3)

        # Create secondary y-axis for helper data
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.plot(optimal_configs['p_flip'], helper,
                's--', color=color, lw=1.5, markersize=6, label='Helper Data Size')
        ax2.set_ylabel('Helper Data Size (kiB)', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yticks(yticks)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # Annotate code configurations
        for _, row in optimal_configs.iterrows():
            label = f"{row['code1'][0]}/{row['code2'][0]}"
            ax1.annotate(label, (row['p_flip'], row['sram_size']),
                        textcoords="offset points", xytext=(0,5), ha='center')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # plt.title('Optimal Resource Configuration Trajectory\n'
        #         f'(Error Target ≤ {target_error:.1e})')
        fig.tight_layout()
        plt.savefig(PLOT_DIR / 'resource_trajectory.png', dpi=300)
    finally:
        plt.close()

def plot_error_sram_frontiers(df, target_error):
    """Plot Error-SRAM trade-off frontiers for key p_flip values"""
    set_ieee_plot_style()
    try:
        plt.figure()

        # Select representative p_flip values
        p_flip_values = np.linspace(df['p_flip'].min(), df['p_flip'].max(), 5)

        # Find all rows where p_flip ≈ value, allowing for small numerical tolerance.
        for p_flip in p_flip_values:
            subset = df[np.isclose(df['p_flip'], p_flip, atol=0.005)]
            if len(subset) == 0:
                continue

            # Find Pareto frontier
            pareto = []
            sorted_subset = subset.sort_values('sram_size')
            min_error = np.inf
            for _, row in sorted_subset.iterrows():
                if row['error_prob'] < min_error:
                    pareto.append(row)
                    min_error = row['error_prob']

            if not pareto:
                continue

            pareto_df = pd.DataFrame(pareto)
            pareto_df = pareto_df[pareto_df['error_prob'] >= MIN_TARGET_ERROR_PROB]
            plt.plot(np.log10(pareto_df['error_prob']), pareto_df['sram_size'],
                    'o-', lw=1.5, markersize=6, 
                    label=r'$p_e$'f'={p_flip:.3f}')
            below_target = pareto_df[pareto_df['error_prob'] <= target_error]
            if not below_target.empty:
                crossing = below_target.iloc[0]
                plt.plot(np.log10(crossing['error_prob']), crossing['sram_size'],
                        's', markersize=10, markerfacecolor='none',
                        markeredgewidth=2, markeredgecolor='k')

        mantissa, exponent = f"{target_error:.1e}".split("e")
        mantissa = float(mantissa)
        exponent = int(exponent)
        plt.axvline(np.log10(target_error), color='r', linestyle='--', 
                label=r'Target $P_\mathrm{error, TS}$ = 'f'${mantissa} \\times 10^{{{exponent}}}$')

        # Plot formatting
        plt.ylabel('SRAM Size (kiB)')  # Now on y-axis
        plt.xlabel(r'Error Probability ($P_\mathrm{error, TS}$)')
        ax = plt.gca()

        # Determine tick positions from min to target error
        all_errors = df['error_prob']
        exponents = np.arange(
            int(np.floor(np.log10(MIN_TARGET_ERROR_PROB))),
            int(np.ceil(np.log10(all_errors.max()))) + 1
        )
        tick_locs = np.log10(10.0 ** exponents)
        tick_labels = [rf'$10^{{{e}}}$' for e in exponents]

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        # plt.title('SRAM-Error Trade-off Frontiers')  # Updated title
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()  # Reverse x-axis direction
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'sram_error_frontiers.png', dpi=300)  # Updated filename
    finally:
        plt.close()

# remark: to have the full symmetric grid, we need to have all combinations of code1_len and code2_len in analysis_functions.py
def plot_configuration_heatmap(df, target_error, p_flip_value=0.05):
    """
    Heatmap of SRAM size for Code1 vs Code2 lengths at a given p_flip,
    showing SRAM size only for configurations meeting the error target.
    Invalid configurations are marked with white color.
    """
    set_ieee_plot_style()
    try:
        df = df[df['p_flip'] == p_flip_value].copy()
        if df.empty:
            print(f"No configurations found for p_flip = {p_flip_value}")
            return

        df['code1_len'] = df['code1'].apply(lambda x: x[0])
        df['code2_len'] = df['code2'].apply(lambda x: x[0])
        df['valid'] = df['error_prob'] <= target_error

        if MAX_CODE_LENGTH is not None:
            df = df[(df['code1_len'] <= MAX_CODE_LENGTH) & (df['code2_len'] <= MAX_CODE_LENGTH)]

        # Pivot table with SRAM size for valid configurations only
        pivot = df[df['valid']].pivot_table(
            index='code1_len',
            columns='code2_len',
            values='sram_size',
            aggfunc='min'
        )

        # Full grid to detect invalid configurations
        full_grid = df.pivot_table(
            index='code1_len',
            columns='code2_len',
            values='sram_size',
            aggfunc='min'
        )

        # Plot heatmap for valid only
        # plt.figure(figsize=(10, 8))
        plt.figure()
        ax = sns.heatmap(
            pivot,
            cmap='YlGnBu',
            linewidths=0.5,
            linecolor='gray',
            annot=True,
            fmt=".1f",
            cbar_kws={'label': 'SRAM Size (kiB)'},
            mask=pivot.isna()
        )
        ax.invert_yaxis()

        # Mark invalid configurations (✘)
        for i in range(full_grid.shape[0]):
            for j in range(full_grid.shape[1]):
                val = pivot.iloc[i, j] if i < pivot.shape[0] and j < pivot.shape[1] else np.nan
                if np.isnan(val) and not np.isnan(full_grid.iloc[i, j]):
                    ax.text(j + 0.5, i + 0.5, '', ha='center', va='center',
                            color='gray', fontsize=12)

        # Create the white cell legend item
        legend_elements = [Patch(facecolor='white', edgecolor='gray',
                                 label='Invalid Configuration')]

        # Place the legend on the right of the title area
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            bbox_to_anchor=(1.15, 1.08),  # Adjust 1.15 to move further right if needed
            frameon=False
        )

        # Reserve space on top
        plt.subplots_adjust(top=0.85)  # Adjust down if needed (e.g., 0.85)

        # plt.title(f"SRAM Size Heatmap at p_flip = {p_flip_value:.2f}\n(Valid if Error ≤ {target_error:.1e})")
        plt.xlabel(r'Code Length ($n_1$)')
        plt.ylabel(r'Code Length ($n_2$)')
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"sram_heatmap_pflip_{p_flip_value:.2f}.png", dpi=300)
    finally:
        plt.close()

def format_table_row(values, fmt=str):
    """ Format a row of values for display in a table."""
    return [fmt(v) if v is not None else "" for v in values]

def plot_sram_vs_pflip_simple_vs_concat(full_df, target_error, simple_df=None):
    """
    Plot SRAM size, Helper Data size, and Codebook size vs. p_flip 
        for both simple and concatenated codes.
    For simple codes: always show 1 point per p_flip. If no config meets the error target,
    show the one with minimum error in gray.
    """
    set_ieee_plot_style()
    try:
        # Filter full (concatenated) results
        full_filtered = full_df[full_df['error_prob'] <= target_error]
        if simple_df is None:
            simple_df = pd.DataFrame(columns=full_df.columns)

        # === Concatenated Code ===
        concat_grouped = (
            full_filtered.groupby('p_flip', as_index=False)
            .apply(lambda x: x.nsmallest(1, 'sram_size'))
            .reset_index(drop=True)
        )

        # === Simple Code ===
        simple_valid = simple_df[simple_df['error_prob'] <= target_error]
        simple_invalid = simple_df[simple_df['error_prob'] > target_error]
        all_simple_pflips = simple_df['p_flip'].unique()
        simple_points, fallback_points = [], []

        for p in sorted(all_simple_pflips):
            valid_at_p = simple_valid[simple_valid['p_flip'] == p]
            if not valid_at_p.empty:
                best = valid_at_p.nsmallest(1, 'sram_size').iloc[0]
                simple_points.append(best)
            else:
                fallback = simple_invalid[simple_invalid['p_flip'] == p]
                if not fallback.empty:
                    best_fallback = fallback.sort_values(['error_prob', 'sram_size']).iloc[0]
                    fallback_points.append(best_fallback)

        valid_df = pd.DataFrame(simple_points) if simple_points else pd.DataFrame()
        fallback_df = pd.DataFrame(fallback_points) if fallback_points else pd.DataFrame()

        # === Create Figure Layout ===
        fig = plt.figure(figsize=(7, 7))
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 0.6])  # last row for table

        ax_sram = fig.add_subplot(gs[0])
        ax_helper = fig.add_subplot(gs[1], sharex=ax_sram)
        ax_codebook = fig.add_subplot(gs[2], sharex=ax_sram)
        ax_table = fig.add_subplot(gs[3])

        # Hide axis of table
        ax_table.axis('off')

        # Plot concatenated codes
        ax_sram.plot(concat_grouped['p_flip'], concat_grouped['sram_size'],
                     's--', markersize=5, lw=1.5, label='TS-TMVS', color='tab:green')
        ax_helper.plot(concat_grouped['p_flip'], concat_grouped['helper_data_size'],
                       's--', markersize=5, lw=1.5, color='tab:green')
        ax_codebook.plot(concat_grouped['p_flip'], concat_grouped['codebook_size'],
                         's--', markersize=5, lw=1.5, color='tab:green')

        # Combine valid and fallback data for a continuous line
        combined_simple_df = pd.concat([valid_df, fallback_df]).sort_values('p_flip')

        # Plot continuous line through all simple points (including fallback)
        if not combined_simple_df.empty:
            print(f" SRAM size for simple codes: {combined_simple_df['sram_size']} kiB")
            ax_sram.plot(combined_simple_df['p_flip'], combined_simple_df['sram_size'],
                        'o-', markersize=5, lw=1.5, color='tab:blue', label='TMVS')
            ax_helper.plot(combined_simple_df['p_flip'], combined_simple_df['helper_data_size'],
                           'o-', markersize=5, lw=1.5, color='tab:blue')
            ax_codebook.plot(combined_simple_df['p_flip'], combined_simple_df['codebook_size'],
                             'o-', markersize=5, lw=1.5, color='tab:blue')

        if not fallback_df.empty:
            mantissa, exponent = f"{target_error:.1e}".split("e")
            mantissa = float(mantissa)
            exponent = int(exponent)
            # plt.title(rf'Optimal Resource Trajectory' f'\n(Error ≤ $10^{{{exponent}}}$)')
            ax_sram.plot(fallback_df['p_flip'], fallback_df['sram_size'],
                        'o', markersize=5, color='orange',
                        label=r'TMVS ($P_\mathrm{error}$ 'f'> ${mantissa} \\times 10^{{{exponent}}}$)')
            ax_helper.plot(fallback_df['p_flip'], fallback_df['helper_data_size'],
                           'o', markersize=5, color='orange')
            ax_codebook.plot(fallback_df['p_flip'], fallback_df['codebook_size'],
                             'o', markersize=5, color='orange')
    
            # Set more y-axis ticks
            ax_sram.yaxis.set_major_locator(MaxNLocator(nbins='auto', prune=None))
            ax_helper.yaxis.set_major_locator(MaxNLocator(nbins='auto', prune=None))
            ax_codebook.yaxis.set_major_locator(MaxNLocator(nbins='auto', prune=None))

        # Formatting
        ax_sram.set_ylabel('SRAM Size (kiB)')
        ax_helper.set_ylabel('Helper Data Size (kiB)')
        ax_codebook.set_ylabel('Codebook Size (kiB)')
        ax_codebook.set_xlabel(r'Bit Flipping Probability ($p_e$)')
        # ax_sram.set_title(r'Resource Usage vs. $p_\mathrm{flip}$'f'(Error ≤ {target_error:.1e})')
        for ax in [ax_sram, ax_helper, ax_codebook]:
            ax.grid(True, alpha=0.3)

        fig.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.03),  # just below the last subplot
            ncol=3,
            frameon=False
        )

        # === Build Table Data ===
        x_vals = sorted(full_df['p_flip'].unique())
        col_labels = [f"{p:.3f}" for p in x_vals]

        # Combine valid and fallback into one reference for simple
        combined_simple = pd.concat([valid_df, fallback_df])

        # Format scientific notation and retain numeric value
        def sci_notation_latex(x):
            mantissa, exponent = f"{x:.1e}".split("e")
            return float(x), rf"${float(mantissa)}\times 10^{{{int(exponent)}}}$"

        # Build row with optional formatter
        def build_row(df, label_fn, return_pair=False):
            row = []
            for p in x_vals:
                match = df[df['p_flip'] == p]
                if not match.empty:
                    val = label_fn(match.iloc[0])
                    row.append(val if return_pair else val[1])  # return formatted string if pair
                else:
                    row.append(("-", "-") if return_pair else "-")
            return row

        # Code labels
        row_simple = build_row(combined_simple, lambda r: (None, f"{r['code2'][0]}"))
        row_concat = build_row(concat_grouped, lambda r: (None, f"{r['code1'][0]}/{r['code2'][0]}"))

        # Error rows: keep numeric + formatted strings
        row_error_simple_vals = build_row(combined_simple, lambda r: sci_notation_latex(r['error_prob']), return_pair=True)
        row_error_concat_vals = build_row(concat_grouped, lambda r: sci_notation_latex(r['error_prob']), return_pair=True)

        # Separate values and formatted strings
        row_error_simple = [fmt for _, fmt in row_error_simple_vals]
        row_error_concat = [fmt for _, fmt in row_error_concat_vals]

        # === Table and labels ===
        table_data = [
            [r'$n$'] + row_simple,
            [r'$P_\mathrm{error}$'] + row_error_simple,
            [r'$n_1$/$n_2$'] + row_concat,
            [r'$P_{\mathrm{error, TS}}$ '] + row_error_concat,
        ]
        column_labels = [r'$p_e$'] + col_labels

        # === Cell coloring ===
        cell_colours = []
        for row_label, _, value_row in zip(
            ['Simple Code', r'$P_\mathrm{error}$ (Simple)',
             'Concat. Code', r'$P_\mathrm{error}$ (Concat.)'],
            [row_simple, row_error_simple, row_concat, row_error_concat],
            [row_simple, [val for val, _ in row_error_simple_vals], row_concat,
             [val for val, _ in row_error_concat_vals]]
        ):
            row_colors = ['white']
            for val in value_row:
                if "error" in row_label:
                    try:
                        if float(val) > target_error:
                            row_colors.append('orange')
                        else:
                            row_colors.append('white')
                    except ValueError:
                        row_colors.append('white')
                else:
                    row_colors.append('white')
            cell_colours.append(row_colors)

        table = ax_table.table(
                cellText=table_data,
                colLabels=column_labels,
                cellColours=cell_colours,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )

        table.auto_set_font_size(False)
        table.set_fontsize(8)

        # Add legend below the table with gray box and target error
        mantissa, exponent = f"{target_error:.1e}".split("e")
        mantissa = float(mantissa)
        exponent = int(exponent)
        legend_elements = [
            Patch(facecolor='orange', edgecolor='black',
                  label=r'$P_\mathrm{error}$ 'f'> ${mantissa} \\times 10^{{{exponent}}}$')
        ]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.03),
                frameon=False, ncol=1)

        fig.tight_layout()
        plt.savefig(PLOT_DIR / 'sram_helper_codebook_vs_pflip.png', dpi=300, bbox_inches='tight')
    finally:
        plt.close()
