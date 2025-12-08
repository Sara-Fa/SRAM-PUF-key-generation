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
        "Plot Hamming Distance histogram"
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
        dir_choice = questionary.select(
            "Select directory for evaluation:",
            choices=["enroll_comparator_dir (Enrollment Evaluation)", "ber_comparator_dir (Regeneration Evaluation)"]
        ).ask()
        dir_name = enroll_comparator_dir if "enroll_comparator_dir" in dir_choice else ber_comparator_dir
        config.plot_3d_evaluation_vs_threshold_and_num_readings(all_parameters, dir_name)

    elif selected_option == "Plot 2D enrollment/regeneration evaluation vs threshold":
        all_parameters = [(15, 1, 14)]
        # all_parameters = [(27, 3, 24)]
        target_num_readings = [1, 10, 100]
        dir_choice = questionary.select(
            "Select directory for evaluation:",
            choices=["enroll_comparator_dir (Enrollment Evaluation)", "ber_comparator_dir (Regeneration Evaluation)"]
        ).ask()
        dir_name = enroll_comparator_dir if "enroll_comparator_dir" in dir_choice else ber_comparator_dir
        # config.plot_2d_evaluation_vs_num_readings(all_parameters, dir_name)
        config.plot_2d_evaluation_vs_threshold(all_parameters, target_num_readings, dir_name)

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
        config.plot_2d_evaluation_vs_num_readings_overlay(all_parameters, chip_id, enroll_comparator_dir, overlay_label="Dark bits")

    elif selected_option == "Plot 2D enrollment evaluation vs number of readings (overlay dark-bits) [BER only]":
        all_parameters = [(27, 3, 24)]
        config.plot_2d_evaluation_vs_num_readings_overlay_ber(all_parameters, enroll_comparator_dir, overlay_label="Dark bits")

    elif selected_option == "Plot 2D enrollment evaluation vs number of readings (overlay dark-bits) [Selection only]":
        all_parameters = [(27, 3, 24)]
        config.plot_2d_evaluation_vs_num_readings_overlay_psr(all_parameters, enroll_comparator_dir, overlay_label="Dark bits")

    elif selected_option == "Plot 2D Bernardini evaluation vs number of readings (overlay Bernardini-dark-bits) [BER only]":
        all_parameters = [(27, 3, 24)]
        config.plot_2d_bernardini_iterative_vs_readings_overlay_ber(
            all_parameters,
            enroll_comparator_dir,
            K=500,
            reference_delta=0.499,
            test_D=0.4991,
            overlay_label="Dark Bit",
        )

    elif selected_option == "Plot 2D Bernardini evaluation vs number of readings (overlay Bernardini-dark-bits) [Selection only]":
        all_parameters = [(27, 3, 24)]
        config.plot_2d_bernardini_iterative_vs_readings_overlay_psr(
            all_parameters,
            enroll_comparator_dir,
            K=500,
            reference_delta=0.499,
            test_D=0.4991,
            enroll_select_threshold_override=None, #[-12.8, 12.8],
            overlay_label="Dark Bit",
        )

    elif selected_option == "Plot Hamming Distance histogram":
        parameters = [(7, 1, 6)]  # Add other cases if needed
        chip_id = 'M2'
        config.initialize_and_plot_hd_values_histogram(parameters, chip_id)

    else:
        print("No valid option selected.")

if __name__ == "__main__":
    main()
