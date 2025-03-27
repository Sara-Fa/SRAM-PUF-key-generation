""" Main file for running experiments and processing readouts. """
import questionary
import nvm_free_tmvs.plotting.plotting_configuration as config
from nvm_free_tmvs.utils.file_manager import ber_comparator_dir
from nvm_free_tmvs.utils.file_manager import enroll_comparator_dir

def main():
    """ Main function for running experiments and processing readouts. """

        # Prompt user to select a plotting configuration
    choices = [
        "Calculate and plot failure rate vs memory trade",
        "Plot 3D enrollment/regeneration evaluation vs threshold and number of readings",
        "Plot 2D enrollment/regeneration evaluation vs threshold",
        "Plot 2D enrollment/regeneration evaluation vs number of readings",
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
        config.calculate_and_plot_failure_rate_vs_memory(all_parameters)

    elif selected_option == "Plot 3D enrollment/regeneration evaluation vs threshold and number of readings":
        all_parameters = [(27, 3, 24)]
        dir_choice = questionary.select(
            "Select directory for evaluation:",
            choices=["enroll_comparator_dir (Enrollment Evaluation)", "ber_comparator_dir (Regeneration Evaluation)"]
        ).ask()
        dir_name = enroll_comparator_dir if "enroll_comparator_dir" in dir_choice else ber_comparator_dir
        config.plot_3d_evaluation_vs_threshold_and_num_readings(all_parameters, dir_name)

    elif selected_option == "Plot 2D enrollment/regeneration evaluation vs threshold":
        all_parameters = [(27, 3, 24)]
        target_num_readings = [1, 10]
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

    elif selected_option == "Plot Hamming Distance histogram":
        parameters = [(7, 1, 6)]  # Add other cases if needed
        chip_id = 'M2'
        config.initialize_and_plot_hd_values_histogram(parameters, chip_id)

    else:
        print("No valid option selected.")

    # calculate and plot failure rate vs memory trade
    # # all_parameters =  [(17, 1, 16)]
    # all_parameters = [(7,1,6), (9,1,8), (11, 1, 10), (11, 2, 9), (13, 1, 12), (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]
    # config.calculate_and_plot_failure_rate_vs_memory(all_parameters)

    # plot 3D enrollment/regeneration evaluation vs threshold and number of readings
    # all_parameters = [(27, 3, 24)]
    # # dir_name = enroll_comparator_dir
    # dir_name = ber_comparator_dir
    # config.plot_3d_evaluation_vs_threshold_and_num_readings(all_parameters, dir_name)

    # plot 2D enrollment/regeneration evaluation vs threshold and number of readings
    # all_parameters = [(27, 3, 24)] #   #
    # dir_name = enroll_comparator_dir # discarding rate is also ploted
    # # dir_name = ber_comparator_dir
    # target_num_readings = [1, 10]
    # config.plot_2d_evaluation_vs_threshold(all_parameters, target_num_readings, dir_name)
    # # plot_2d_evaluation_vs_num_readings(all_parameters, dir_name)

    # plot Hamming Distance histogram
    # parameters = [(7,1,6)] # # add the case 27 to the next list
    # chip_id='M2'
    # config.initialize_and_plot_hd_values_histogram(parameters, chip_id)
    

if __name__ == "__main__":
    main()
