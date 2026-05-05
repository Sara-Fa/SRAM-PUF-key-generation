"""Interactive plotting menu for NVM-free TMVS cached results.

Assumes BER/helper data caches already exist in `Enroll_comparator_data`
or `BER_comparator_data` (either computed or unzipped). Presents a list
of plot types and dispatches to the corresponding plotting functions.
"""
import questionary
import nvm_free_tmvs.plotting.plotting_configuration as config
from nvm_free_tmvs.utils.file_manager import ber_comparator_dir, enroll_comparator_dir


def main():
    """Prompt user for a plotting configuration and render the selection."""

    # Prompt user to select a plotting configuration
    choices = [
        "Calculate and plot failure rate vs memory trade",
        "Plot 3D enrollment/regeneration evaluation vs threshold and number of readings",
        "Plot 2D enrollment/regeneration evaluation vs threshold",
        "Plot 2D enrollment/regeneration evaluation vs number of readings",
        "Plot 2D enrollment evaluation vs number of readings (overlay dark-bits) [combined]",
        "Plot 2D enrollment evaluation vs number of readings (overlay dark-bits) [BER only]",
        "Plot 2D enrollment evaluation vs number of readings (overlay dark-bits) [Selection only]",
        "Plot 2D Bernardini evaluation vs number of readings (overlay Bernardini-dark-bits) [BER only]",
        "Plot 2D Bernardini evaluation vs number of readings (overlay Bernardini-dark-bits) [Selection only]",
        "Plot Hamming Distance histogram",
        "Plot BER vs n (trivial codebook, fixed threshold)",
        "Plot BER_Enr vs BER_Reg (per threshold)",
        "Plot KER_Enr/KER_Reg vs SRAM size (per threshold)"
    ]
    
    selected_option = questionary.select(
        "Select a plotting configuration:",
        choices=choices
    ).ask()

    if selected_option == "Calculate and plot failure rate vs memory trade":
        # all_parameters =  [(17, 1, 16)]
        all_parameters = [(7, 1, 6), (9, 1, 8), (11, 1, 10), (11, 2, 9), (13, 1, 12),
                          (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25),
                          (31, 5, 26), (33, 5, 28), (35, 6, 29), (37, 7, 30), (39, 8, 31),
                          (41, 6, 35), (45, 10, 35), (47, 8, 39)]
        nb_enroll_reading_list = [10, 100]
        config.calculate_and_plot_failure_rate_vs_memory(all_parameters, nb_enroll_reading_list)

    elif selected_option == "Plot 3D enrollment/regeneration evaluation vs threshold and number of readings":
        all_parameters = [(27, 3, 24)]
        trivial = True  # True for trivial, False for non-trivial, None for auto
        dir_choice = questionary.select(
            "Select directory for evaluation:",
            choices=["enroll_comparator_dir (Enrollment Evaluation)", "ber_comparator_dir (Regeneration Evaluation)"]
        ).ask()
        dir_name = enroll_comparator_dir if "enroll_comparator_dir" in dir_choice else ber_comparator_dir
        config.plot_3d_evaluation_vs_threshold_and_num_readings(all_parameters, dir_name, trivial=trivial)

    elif selected_option == "Plot 2D enrollment/regeneration evaluation vs threshold":
        # all_parameters = [(3, 1, 2), (27, 3, 24)]
        all_parameters = [(27, 3, 24)]
        target_num_readings = [1, 99, 499]
        trivial = False  # True for trivial, False for non-trivial, None for auto
        dir_choice = questionary.select(
            "Select directory for evaluation:",
            choices=["enroll_comparator_dir (Enrollment Evaluation)", "ber_comparator_dir (Regeneration Evaluation)"]
        ).ask()
        dir_name = enroll_comparator_dir if "enroll_comparator_dir" in dir_choice else ber_comparator_dir
        config.plot_2d_evaluation_vs_threshold(all_parameters, target_num_readings, dir_name, trivial=trivial)

    elif selected_option == "Plot 2D enrollment/regeneration evaluation vs number of readings":
        all_parameters = [(27, 3, 24)]
        target_num_readings = [1, 10]
        dir_choice = questionary.select(
            "Select directory for evaluation:",
            choices=["enroll_comparator_dir (Enrollment Evaluation)", "ber_comparator_dir (Regeneration Evaluation)"]
        ).ask()
        dir_name = enroll_comparator_dir if "enroll_comparator_dir" in dir_choice else ber_comparator_dir
        config.plot_2d_evaluation_vs_num_readings(all_parameters, dir_name)

    elif selected_option == "Plot 2D enrollment evaluation vs number of readings (overlay dark-bits) [combined]":
        all_parameters = [(27, 3, 24)]
        chip_id = 'M2'
        config.plot_2d_evaluation_vs_num_readings_overlay(all_parameters, chip_id, enroll_comparator_dir, overlay_label="Dark Bit")

    elif selected_option == "Plot 2D enrollment evaluation vs number of readings (overlay dark-bits) [BER only]":
        # --- Customize here ---
        all_parameters = [(3, 1, 2), (27, 3, 24)]  # (n, cb_low, cb_high)
        # th_high_star: float for same threshold on all, or list for per-parameter
        th_high_star = [0.0, 4.6]  # n=3 at TH*=0, n=27 at TH*=4.6
        config.plot_2d_evaluation_vs_num_readings_overlay_ber(
            all_parameters, enroll_comparator_dir,
            overlay_label="Dark Bit", th_high_star=th_high_star)

    elif selected_option == "Plot 2D enrollment evaluation vs number of readings (overlay dark-bits) [Selection only]":
        # --- Customize here (same config as BER only) ---
        all_parameters = [(3, 1, 2), (27, 3, 24)]
        th_high_star = [0.0, 4.6]  # per-parameter: n=3 at TH*=0, n=27 at TH*=4.6
        config.plot_2d_evaluation_vs_num_readings_overlay_psr(
            all_parameters, enroll_comparator_dir,
            overlay_label="Dark Bit", th_high_star=th_high_star)

    elif selected_option == "Plot 2D Bernardini evaluation vs number of readings (overlay Bernardini-dark-bits) [BER only]":
        # --- Customize here ---
        all_parameters = [(3, 1, 2), (27, 3, 24)]  # ODHD configs (n, cb_low, cb_high)
        trivial = True           # True for trivial codebook, False for non-trivial
        config.plot_2d_bernardini_iterative_vs_readings_overlay_ber(
            all_parameters, enroll_comparator_dir,
            K=500, reference_delta=0.499, test_D=0.4991,
            overlay_label="Dark Bit", trivial=trivial,
        )

    elif selected_option == "Plot 2D Bernardini evaluation vs number of readings (overlay Bernardini-dark-bits) [Selection only]":
        # --- Customize here ---
        all_parameters = [(3, 1, 2)]
        trivial = True
        config.plot_2d_bernardini_iterative_vs_readings_overlay_psr(
            all_parameters, enroll_comparator_dir,
            K=500, reference_delta=0.499, test_D=0.4991,
            enroll_select_threshold_override=None,
            overlay_label="Dark Bit", trivial=trivial,
        )

    elif selected_option == "Plot Hamming Distance histogram":
        parameters = [(7, 1, 6)]  # Add other cases if needed
        chip_id = 'M2'
        config.initialize_and_plot_hd_values_histogram(parameters, chip_id)

    elif selected_option == "Plot BER vs n (trivial codebook, fixed threshold)":
        all_parameters = [
            (3, 1, 2), (7, 1, 6), (9, 1, 8), (11, 1, 10), (11, 2, 9),
            (13, 1, 12), (13, 2, 11), (15, 1, 14), (17, 1, 16),
            (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),
            (35, 6, 29), (37, 7, 30), (39, 8, 31), (41, 6, 35),
            (45, 10, 35), (47, 8, 39)]
        target_th_high = 0.0  # no threshold (baseline)
        target_nr_read = [1, 9, 99, 499] # [10, 100, 500]  # one figure per NrRead
        dir_choice = questionary.select(
            "Select directory for evaluation:",
            choices=["enroll_comparator_dir (Enrollment Evaluation)", "ber_comparator_dir (Regeneration Evaluation)"]
        ).ask()
        dir_name = enroll_comparator_dir if "enroll_comparator_dir" in dir_choice else ber_comparator_dir
        config.plot_ber_vs_n_trivial(all_parameters, target_th_high, target_nr_read, dir_name)

    elif selected_option == "Plot BER_Enr vs BER_Reg (per threshold)":
        # --- Customize here ---
        all_parameters = [(27, 3, 24), (3, 1, 2)]
        trivial = True
        config.plot_ber_enr_vs_ber_reg(all_parameters, trivial=trivial,
                                        nr_read=499, min_selected=100, key_length=16,
                                        skip_integer_th=False)

    elif selected_option == "Plot KER_Enr/KER_Reg vs SRAM size (per threshold)":
        # --- Customize here ---
        all_parameters = [(3, 1, 2)] # [(27, 3, 24)]
        trivial = True
        config.plot_ker_vs_sram_size(all_parameters, trivial=trivial,
                                      nr_read=499, min_selected=100, key_length=16,
                                      skip_integer_th=False)

    else:
        print("No valid option selected.")

if __name__ == "__main__":
    main()
