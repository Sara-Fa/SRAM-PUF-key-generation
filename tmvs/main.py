"""
This is the main script that interacts with the user to choose the analysis
functions to execute from analysis.py and enter the required parameters.
"""
import logging
import questionary
from common.data_reading_utils import ReadoutList, get_files, read_readouts
from tmvs.utils import KeyInfoList, get_codebook_parameters, read_key_info
from tmvs.tmvs_algo import EnrollKey, RegenerateKey
from tmvs.analysis import BitErrorRate, SelectionRate, ErrorProbabilityValidation
from tmvs.analysis import SelectProbabilityValidation, TheoreticalMemoryTradeOff
from tmvs.analysis import FailureProbability, OptimalRequirementsSRAM

# Define a custom formatter
formatter = logging.Formatter(
    '[%(levelname)s][%(asctime)s] | %(message)s', datefmt='%H:%M:%S')

# Configure logging with the custom formatter and set level to INFO
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s][%(asctime)s] | %(message)s',
                    datefmt='%H:%M:%S')

analysis_suite = [EnrollKey, RegenerateKey, BitErrorRate, SelectionRate,
                  ErrorProbabilityValidation, SelectProbabilityValidation,
                  TheoreticalMemoryTradeOff, FailureProbability, OptimalRequirementsSRAM]

STOP_CHAR = 'x'

def ask_for_n_values():
    """
    Ask the user to input code length (n) values to be used for running the selected
    analysis functions
    """
    code_length_input_values = []
    # collect n values
    print(f'\nEnter positive odd code length values n (enter \'{STOP_CHAR}\' to stop)',
          'to enroll/regenerate keys or for theoretical analysis functions:')
    while True:
        user_input = input("Enter value of n: ")
        if user_input == STOP_CHAR:
            break
        try:
            integer_input = int(user_input)
            if integer_input > 0 and integer_input % 2 != 0:
                code_length_input_values.append(integer_input)
            else:
                print("Invalid input. Please enter a strictly positive odd integer:")
        except ValueError:
            print(f"Invalid input. Please enter an integer or '{STOP_CHAR}' to stop:")
    return code_length_input_values

def ask_for_margin_coefficients():
    """
    Ask the user to input sd coefficient values to be used for running the selected
    analysis functions
    """
    margin_coeff_input = []
    print('\nEnter error margin coefficients (mean, standard deviation)',
            f'(enter \'{STOP_CHAR}\' to stop)',
            'to enroll/regenerate keys or for theoretical analysis functions:')
    # collect (mean, standard deviation) inputs
    while True:
        user_input = input("Enter values of mean and sd coefficients separated by space: ")
        if user_input == STOP_CHAR:
            break
        try:
            float_str1, float_str2 = user_input.split()
            float_input = [float(float_str1), float(float_str2)]
            if float_input[0] >= 0 and float_input[1] >= 0:
                margin_coeff_input.append(float_input)
            else:
                print("Invalid input. Please enter positive values:")
        except ValueError:
            print('Invalid input. Please enter (mean, standard deviation) values or enter',
                f'\'{STOP_CHAR}\' to stop:')
    return margin_coeff_input


def ask_for_chips_ids():
    """
    Ask the user to input chips IDs to be used for running the selected
    analysis functions
    """
    all_files = get_files()
    chips_to_evaluate = questionary.checkbox(
    "\nSelect the chips IDs to evaluate",
    choices=list(all_files.keys())
    ).ask()
    return chips_to_evaluate

def main():
    """
    Entry point. Ask the user about which functions to run and their required parameters.
    """
    all_files = get_files()

    analyses_to_run = questionary.checkbox(
        "Select the function(s) to run",
        choices=[questionary.Choice(title=c.name, value=c)
                 for c in analysis_suite]
    ).ask()

    obtained_n_values = 0
    obtained_sd_coeff_values = 0
    obtained_chips_ids = 0

    code_length_input_values = []
    margin_coeff_input = []
    chips_to_evaluate = []

    common_elements = list({EnrollKey, RegenerateKey} & set(analyses_to_run))
    if common_elements:
        code_length_input_values = ask_for_n_values()
        margin_coeff_input = ask_for_margin_coefficients()
        chips_to_evaluate = ask_for_chips_ids()
        obtained_n_values = 1
        obtained_sd_coeff_values = 1
        obtained_chips_ids = 1
        all_readouts: list[ReadoutList] = [read_readouts(all_files[chip_id])
                                for chip_id in chips_to_evaluate]
        for analysis_class in common_elements:
            for coeff in margin_coeff_input:
                for n in code_length_input_values:
                    for readouts in all_readouts:
                        analysis_class.process(readouts=readouts, code_length=n,
                                                margin_coeff=coeff)

    common_elements = list({BitErrorRate, SelectionRate, ErrorProbabilityValidation,
                            SelectProbabilityValidation} & set(analyses_to_run))
    if common_elements:
        if not obtained_chips_ids:
            chips_to_evaluate = ask_for_chips_ids()
            obtained_chips_ids = 1
        enroll_parameters, regenerate_parameters = get_codebook_parameters(chips_to_evaluate)
        parameters_to_evaluate = questionary.checkbox(
            "\nSelect the parameters (n, TH_low, TH_high) to calculate BER, selection rate or"
            +" validate them",
            choices=[questionary.Choice(title=str(c), value=c)
                    for c in list(enroll_parameters.keys())]
        ).ask()

        all_key_info: list[KeyInfoList] = [read_key_info(enroll_parameters[parameter],
                                                        regenerate_parameters[parameter])
                                        for parameter in parameters_to_evaluate ]
        for analysis_class in common_elements:
            a = analysis_class(all_key_info=all_key_info)
            a.run()

    if TheoreticalMemoryTradeOff in analyses_to_run:
        if not obtained_n_values:
            code_length_input_values = ask_for_n_values()
            obtained_n_values = 1
        if not obtained_sd_coeff_values:
            margin_coeff_input = ask_for_margin_coefficients()
            obtained_sd_coeff_values = 1
        for coeff in margin_coeff_input:
            for n in code_length_input_values:
                TheoreticalMemoryTradeOff.process(n=n, margin_coeff=coeff)

    if FailureProbability in analyses_to_run:
        if not obtained_sd_coeff_values:
            margin_coeff_input = ask_for_margin_coefficients()
            obtained_sd_coeff_values = 1
        for coeff in margin_coeff_input:
            FailureProbability.process(margin_coeff=coeff)

    if OptimalRequirementsSRAM in analyses_to_run:
        OptimalRequirementsSRAM.process()

if __name__ == "__main__":
    main()
