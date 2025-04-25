"""
This script provides the analysis tools of TMVS, including
calculating the experimental bit error rate and selection rate,
comparing theoretical results with experimental data, and
other theoretical analytical metrics.
It also includes plotting capabilities.
"""
import logging
from abc import ABC, abstractmethod
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from adjustText import adjust_text
import tmvs.analysis_constants as const
import common.data_constants as data_const
from tmvs.utils import KeyInfoList, KeyInfo
from tmvs.formulas import theoretical_selection_probability, theoretical_error_probability
from tmvs.formulas import theoretical_required_sram_size, theoretical_required_helper_data_size
from tmvs.formulas import required_codebook_size, key_failure_probability
from common.formulas import calculate_threshold

class Analysis(ABC):
    """
    Abstract base class for all types of analysis.
    """
    name = None

    def __init__(self, all_key_info: list[KeyInfoList]):
        self.all_key_info = all_key_info

    @abstractmethod
    def run(self):
        pass


class SingleCodebookAnalysis(Analysis):
    """
    Class for analyses that only require data from a single pair of
    codebook parameters (n, [TH_low,TH_high]) (for all the chips).

    The main difference with ManyCodebookAnalysis is that the process
    method only receives the readouts from a single pair of parameters.
    """
    @staticmethod
    @abstractmethod
    def process(all_key_info: list[KeyInfoList]):
        pass

    def run(self):
        """
        An analysis is run for each pair of parameters.
        """
        for key_info in self.all_key_info:
            self.process(key_info)


class ManyCodebookAnalysis(Analysis):
    """
    Class for analyses that require data from all pairs of
    codebook parameters (n, [TH_low,TH_high]).

    The main difference with SingleCodebookAnalysis is that the process
    method receives all the key info from all parameters.
    """
    @staticmethod
    @abstractmethod
    def process(all_key_info: list[KeyInfoList]):
        pass

    def run(self):
        """
        An inter-chip analysis is run once for all pairs of parameters.
        """
        self.process(self.all_key_info)


class BitErrorRate(SingleCodebookAnalysis):
    """
    Class to calculate, print and plot the Bit Error Rate (BER) based on SCuM chip readings
    and using the enrollment and regeneration key info
    """
    name = "Bit Error Rate -- experimental (data+plot)"

    @staticmethod
    def calculate_experimental_ber_per_chip(chip_key_info: KeyInfo):
        """
        Calculate and return minimum, maximum and average BER using key info
        for a single chip over multiple key regeneration
        """
        error_count_per_key = chip_key_info.error_counts
        key_length = len(chip_key_info.reference_key)
        regeneration_count = len(error_count_per_key)

        avg_ber = float(sum(error_count_per_key)) / (regeneration_count * key_length)
        min_ber = float(min(error_count_per_key)) / key_length
        max_ber = float(max(error_count_per_key)) / key_length

        return avg_ber, min_ber, max_ber

    @staticmethod
    def calculate_experimental_ber_all_chips(key_info: KeyInfoList):
        """
        Calculate and return minimum, maximum and average BER using key info
        over multiple chips and key regeneration
        """
        # Get global ber over all chips
        chips_min_ber = np.inf
        chips_max_ber = -np.inf
        chips_avg_ber = 0
        ber_values = []

        for chip_key_info in key_info:
            avg_ber, min_ber, max_ber = BitErrorRate.calculate_experimental_ber_per_chip(
                chip_key_info=chip_key_info)
            chips_avg_ber += avg_ber
            chips_min_ber = min(chips_min_ber, min_ber)
            chips_max_ber = max(chips_max_ber, max_ber)
            ber_values.append(avg_ber)

        chips_avg_ber /= len(key_info)
        return chips_avg_ber, chips_min_ber, chips_max_ber, ber_values


    @staticmethod
    def process(key_info: KeyInfoList):
        """
        Print and plot BER values for different chips and fixed codebook parameters
        """
        n = key_info.n
        select_threshold = [key_info.thresh_low, key_info.thresh_high]
        logging.info('Bit Error Rate with n=%d and [TH_low,TH_high]=[%d,%d]', n,
                     select_threshold[0], select_threshold[1])

        # theoretical BER
        theor_ber = theoretical_error_probability(n, select_threshold, data_const.P_FLIP)
        theor_failure = key_failure_probability(theor_ber)

        chips_min_ber = np.inf
        chips_max_ber = -np.inf
        chips_avg_ber = 0

        for chip_key_info in key_info:

            avg_ber, min_ber, max_ber = BitErrorRate.calculate_experimental_ber_per_chip(
                chip_key_info=chip_key_info)

            # Get global ber over all chips
            chips_avg_ber += avg_ber
            chips_min_ber = min(chips_min_ber, min_ber)
            chips_max_ber = max(chips_max_ber, max_ber)

            # Divide by amount of bits (key length) to get BER
            plot_round_nb = const.BER_PLOT_RESOLUTION[(n,select_threshold[0],select_threshold[1])]
            ber = np.round(np.divide(chip_key_info.error_counts,
                                     len(chip_key_info.reference_key)), plot_round_nb)
            logging.info('Chip %s:',chip_key_info.chip_id)
            logging.info('\tAverage BER (bit error rate): %.6e', avg_ber)
            logging.info('\tMinimum BER (bit error rate): %.6e', min_ber)
            logging.info('\tMaximum BER (bit error rate): %.6e\n', max_ber)

            unique, counts = np.unique(ber, return_counts=True)
            plt.plot(unique, counts, label=chip_key_info.chip_id)
            plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            plt.title(r'Frequency of bit error rates for $n$='+f'{n}'
                      +r' and [$\mathrm{TH}_{\mathrm{low}}$,$\mathrm{TH}_{\mathrm{high}}$]='
                      +f'[{select_threshold[0]},{select_threshold[1]}]')
            plt.xlabel('BER')
            plt.ylabel('Frequency')
            plt.legend()

        chips_avg_ber /= len(key_info)
        failure = key_failure_probability(chips_avg_ber)
        logging.info('Over all Chips:')
        logging.info('\tAverage BER (bit error rate): %.6e', chips_avg_ber)
        logging.info('\tAverage Failure Rate (128 bits key): %.6e', failure)
        logging.info('\tMinimum BER (bit error rate): %.6e', chips_min_ber)
        logging.info('\tMaximum BER (bit error rate): %.6e', chips_max_ber)
        logging.info('\tTheoretical average BER: %.6e', theor_ber)
        logging.info('\tTheoretical Failure Rate (128 bits key): %.6e\n\n', theor_failure)

        plt.scatter(chips_avg_ber, 0, color='black', marker='x', label='Exp. avg BER')
        plt.scatter(theor_ber, 0, color='red', marker='x', label='Theor. avg BER')
        plt.legend()
        # Set the x-axis formatter to ScalarFormatter for scientific notation
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.gca().ticklabel_format(style='sci', scilimits=(0,0))
        plt.show()


class SelectionRate(SingleCodebookAnalysis):
    """
    Class to calculate and print the Selection Rate based on SCuM chip readings
    and using the enrollment key info
    """
    name = "Selection Rate -- experimental (data)"

    @staticmethod
    def calculate_experimental_pselect_per_chip(chip_key_info: KeyInfo):
        """
        Calculate and return the selection rate using key info for a single chip
        """
        key_length =len(chip_key_info.reference_key)
        non_select_count = chip_key_info.skiped_bits_count

        # Probability of selection
        select_rate = float(key_length) / (key_length + non_select_count)

        return select_rate

    @staticmethod
    def calculate_experimental_pselect_all_chips(key_info: KeyInfoList):
        """
        Calculate and return minimum, maximum, average and values of selection rate using
        key enrollment info over multiple chips
        """
        chips_min_select_rate = np.inf
        chips_max_select_rate = -np.inf
        chips_avg_select_rate = 0
        chips_select_rate_values = []

        for chip_key_info in key_info:

            select_rate = SelectionRate.calculate_experimental_pselect_per_chip(
                chip_key_info=chip_key_info)
            chips_avg_select_rate += select_rate
            chips_min_select_rate = min(chips_min_select_rate, select_rate)
            chips_max_select_rate = max(chips_max_select_rate, select_rate)
            chips_select_rate_values.append(select_rate)

        chips_avg_select_rate /= len(key_info)
        return (chips_avg_select_rate, chips_min_select_rate,
                chips_max_select_rate, chips_select_rate_values)

    @staticmethod
    def process(key_info: KeyInfoList):
        """
        Print selection rate values for different chips and fixed codebook parameters
        """
        n = key_info.n
        select_threshold = [key_info.thresh_low, key_info.thresh_high]
        logging.info('Selection Rate with n=%d and [TH_low,TH_high]=[%d,%d]', n,
                     select_threshold[0], select_threshold[1])

        # theoretical Pselect
        theor_pselect = theoretical_selection_probability(
            n, select_threshold, key_info.codebook_size)

        chips_min_select_rate = np.inf
        chips_max_select_rate = -np.inf
        chips_avg_select_rate = 0
        chips_sram_size_128_key_bits = 0

        for chip_key_info in key_info:

            select_rate = SelectionRate.calculate_experimental_pselect_per_chip(
                chip_key_info=chip_key_info)
            sram_size_128_key_bits = (
                chip_key_info.key_bits_addr[const.KEY_LENGTH-1] + n) / (8 * 1024)

            # Average probability of selection
            chips_avg_select_rate += select_rate
            chips_min_select_rate = min(chips_min_select_rate, select_rate)
            chips_max_select_rate = max(chips_max_select_rate, select_rate)
            chips_sram_size_128_key_bits += chip_key_info.key_bits_addr[const.KEY_LENGTH-1] + n

            logging.info('Chip %s:', chip_key_info.chip_id)
            logging.info('Required SRAM for the first 128 key bits: %.2fkB', sram_size_128_key_bits)
            logging.info('Total number of key bits (among 55kB): %d',
                         len(chip_key_info.reference_key))
            logging.info('Selection rate: %.7f \n', select_rate)

        chips_avg_select_rate /= len(key_info)
        chips_sram_size_128_key_bits /= (8 * 1024 * len(key_info))
        logging.info('Over all Chips:')
        logging.info('\tAverage Selection Rate: %.5f', chips_avg_select_rate)
        logging.info('\tMinimum Selection Rate: %.5f', chips_min_select_rate)
        logging.info('\tMaximum Selection Rate: %.5f', chips_max_select_rate)
        logging.info(
            '\tAverage required SRAM for the first 128 key bits: %.4fkB',
            chips_sram_size_128_key_bits)
        logging.info('\tTheoretical selection probability: %.7f \n\n', theor_pselect)


class ErrorProbabilityValidation(ManyCodebookAnalysis):
    """
    Class to validate theoretical error probability formula
    using experimental BER results
    """
    name = "Error Probability Validation -- experimental (plot)"

    @staticmethod
    def process(all_key_info: list[KeyInfoList]):
        """
        Print and plot BER values vs error probability for different chips
        and codebook parameters
        """
        experimental_avg_ber = []
        experimental_ber_values = []
        theoretical_ber = []
        parameters = []
        avg_key_length = 0
        total_enrolled_key_chips_values = []
        total_enrolled_key_chips = 0

        for key_info in all_key_info:

            n = key_info.n
            select_threshold = [key_info.thresh_low, key_info.thresh_high]
            parameters.append((n,select_threshold))

            ber,_,_,ber_values = BitErrorRate.calculate_experimental_ber_all_chips(
                key_info=key_info)
            experimental_avg_ber.append(ber)
            theoretical_ber.append(
                theoretical_error_probability(n, select_threshold, data_const.P_FLIP))
            experimental_ber_values.append(ber_values)

            total_enrolled_key_chips = 0
            for key_info_value in key_info:
                total_enrolled_key_chips += len(key_info_value.reference_key)
            avg_key_length = total_enrolled_key_chips / len(key_info)
            total_enrolled_key_chips_values.append(total_enrolled_key_chips)
            logging.info('Error Probability Validation with n=%d and [TH_low,TH_high]=[%d,%d]',
                         n, select_threshold[0], select_threshold[1])
            logging.info('\t Avg experimental BER (bit error rate): %.6e', ber)
            logging.info('\t Theoretical Error Probability: %.6e', theoretical_ber[-1])
            logging.info(
                '\t Relative deviation: %f%%', 100*(theoretical_ber[-1]-ber)/theoretical_ber[-1])
            logging.info('\t Average number of key bits (among 55kB): %d',
                         int(np.ceil(avg_key_length)))
            logging.info('\t Total Key bits number: %d \n\n', total_enrolled_key_chips)

        # Plotting
        _, ax1 = plt.subplots(figsize=(6, 3))
        bp = ax1.boxplot(experimental_ber_values, patch_artist=True,
                         positions=range(len(parameters)))
        # Overlay theoretical values
        for i, (_, theoretical_ber_value) in enumerate(zip(parameters, theoretical_ber)):
            ax1.scatter(i, theoretical_ber_value, color='red', marker='^', zorder=3,
                        label=r'Theoretical $P_{\mathrm{error}}$' if i == 0 else "")
        ax1.set_xticks(range(len(parameters)))
        ax1.set_xticklabels([f"({x}, {y})" for x, y in parameters])
        plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
        ax1.set_yscale('log')
        ax1.set_ylabel('Probability', fontsize=14)
        ax1.set_xlabel(r'($n$,[$\mathrm{TH}_{\mathrm{low}}$,$\mathrm{TH}_{\mathrm{high}}$])',fontsize=14)
        ax1.set_title('Theoretical vs Empirical Error Probability Results')
        # Highlighting outliers
        for flier in bp['fliers']:
            flier.set(marker='o', color='blue', alpha=0.5)
        # Outlier handle (white circle with black edge)
        outlier_handle = plt.Line2D([], [], color='white', marker='o', markersize=10,
                                    linestyle='none', markeredgecolor='black')
        # Theoretical points handle (red triangle)
        theoretical_handle = plt.Line2D([], [], color='red', marker='^', markersize=10,
                                        linestyle='none')
        # Create a second y-axis sharing the same x-axis
        ax2 = ax1.twinx()
        ax2.plot(range(len(parameters)), total_enrolled_key_chips_values, color='black',
                 label='# Tested key bits')
        # Customize the y-axis ticks and labels
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # Customize the second y-axis
        ax2.set_ylabel(r'# Tested key bits',fontsize=14)
        handles1, labels1 = [outlier_handle, theoretical_handle], [
            'Outliers', r'Theoretical $P_{\mathrm{error}}$']
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', frameon=True)
        plt.show()


class SelectProbabilityValidation(ManyCodebookAnalysis):
    """
    Class to validate theoretical selection probability formula
    using experimental selection rate results
    """
    name = "Select Probability Validation -- experimental (plot)"

    @staticmethod
    def process(all_key_info: list[KeyInfoList]):
        """
        Plot selection rate values vs selection probability for
        different chips and codebook parameters
        """
        experimental_pselect = []
        experimental_pselect_chips = []
        theoretical_pselect = []
        parameters = []
        codebook_size_values = []

        for key_info in all_key_info:

            n = key_info.n
            select_threshold = [key_info.thresh_low, key_info.thresh_high]

            codebook_size = key_info.codebook_size
            parameters.append((n,select_threshold))
            codebook_size_values.append(codebook_size)

            pselect,_,_,pselect_values = SelectionRate.calculate_experimental_pselect_all_chips(
                key_info=key_info)
            experimental_pselect.append(pselect)
            experimental_pselect_chips.append(pselect_values)
            theoretical_pselect.append(theoretical_selection_probability(n, select_threshold,
                                                                         codebook_size))
        # Plotting
        _, ax1 = plt.subplots(figsize=(6, 3))
        bp = ax1.boxplot(experimental_pselect_chips, patch_artist=True,
                         positions=range(len(parameters)))
        # Overlay theoretical values
        for i, (_, theoretical_pselect_value) in enumerate(zip(parameters, theoretical_pselect)):
            ax1.scatter(i, theoretical_pselect_value, color='red', marker='^', zorder=3,
                        label=r'Theoretical $P_{\mathrm{error}}$' if i == 0 else "")
        ax1.set_xticks(range(len(parameters)))
        ax1.set_xticklabels([f"({x}, {y})" for x, y in parameters])
        plt.xticks(rotation=45)  # Rotate x-ax1is labels by 45 degrees
        ax1.set_ylabel('Probability', fontsize=14)
        ax1.set_yscale('log')
        ax1.set_xlabel(r'($n$,[$\mathrm{TH}_{\mathrm{low}}$,$\mathrm{TH}_{\mathrm{high}}$])',fontsize=14)
        ax1.set_title('Theoretical vs Empirical Selection Probability Results')
        # Highlighting outliers
        for flier in bp['fliers']:
            flier.set(marker='o', color='blue', alpha=0.5)
        # Outlier handle (white circle with black edge)
        outlier_handle = plt.Line2D([], [], color='white', marker='o', markersize=10,
                                    linestyle='none', markeredgecolor='black')
        # Theoretical points handle (red triangle)
        theoretical_handle = plt.Line2D([], [], color='red', marker='^', markersize=10,
                                        linestyle='none')
        # Create a second y-axis sharing the same x-axis
        ax2 = ax1.twinx()
        # Plot the second dataset on the new y-axis
        ax2.plot(range(len(parameters)), codebook_size_values, color='green',  label=r'$M$')
        # Customize the second y-axis
        ax2.set_ylabel(r'# Codewords ($M$)',fontsize=14)
        handles1, labels1 = [outlier_handle, theoretical_handle], [
            'Outliers', r'Theoretical $P_{\mathrm{select}}$']
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', frameon=True)
        plt.show()


class TheoreticalMemoryTradeOff:
    """
    Class to show theoretically the trade-off between
    different memory requirements
    """
    name = "Theoretical memory trade-off -- theoretical (plot)"

    @staticmethod
    def process(n: int, margin_coeff: list[float]):
        """
        Calculate and plot the trade-off between required memory size of
        SRAM, helper data and codebook for codebook parameters (n, margin_coeff)
        """
        codebook_size_array = const.CODEBOOK_SIZES_TO_TEST
        select_threshold, _ = calculate_threshold (n, margin_coeff)
        convergence_threshold = const.CONVERGENCE_THRESHOLD

        sram_size = []
        helper_data_memory_size = []
        codebook_memory_size = []
        min_select_proba = np.inf

        for size in codebook_size_array:
            sram_size.append(theoretical_required_sram_size(n, select_threshold, size))
            codebook_memory_size.append(required_codebook_size(n, size))
            helper_data_memory_size.append(
                theoretical_required_helper_data_size(n, select_threshold, size))
            min_select_proba = min(min_select_proba,
                                   theoretical_selection_probability (n, select_threshold, size))
        logging.info('Theoretical memory trade-off with n=%d and [TH_low,TH_high]=[%d,%d]', n,
                     select_threshold[0], select_threshold[1])

        plt.plot(codebook_size_array, sram_size, label='SRAM source bits')
        plt.plot(codebook_size_array, helper_data_memory_size, label='Helper Data')
        plt.plot(codebook_size_array, codebook_memory_size, label='Codebook')

        # Analyze the data to find the convergence point
        codebook_size_diff = codebook_size_array[1] - codebook_size_array[0]
        diff = np.abs(np.diff(sram_size)) / codebook_size_diff
        # First point where difference is below the threshold
        convergence_indices = np.where(diff < convergence_threshold)[0]
        # Check if convergence is found
        if convergence_indices.size > 0:
            convergence_index = convergence_indices[0]
            convergence_x = codebook_size_array[convergence_index]
            convergence_y = sram_size[convergence_index]
            convergence_codebook = codebook_memory_size[convergence_index]

            logging.info('Convergence occures at codebook size = %d with:', convergence_x)
            logging.info('\t sram size = %.2fkB', convergence_y)
            logging.info('\t codewords size = %.2fkB \n\n', convergence_codebook)

            # Annotate the convergence point on the plot
            plt.axvline(convergence_x, color='r', linestyle='--', label='Convergence')
            plt.scatter([convergence_x], [convergence_y], color='red')
        else:
            logging.info('Convergence not found within the given threshold.')

        plt.title(
            r'Required Memory Size ($n = %d$, $\mathrm{TH}_{\mathrm{low}} = %d$, $\mathrm{TH}_{\mathrm{high}} = %d$)'
                  % (n, select_threshold[0], select_threshold[1]))
        # Label the axes
        plt.xlabel(r'# Codewords ($M$)',fontsize=14)
        plt.ylabel('Memory size (kB)',fontsize=14)
        plt.legend(loc='right')
        plt.show()


class FailureProbability:
    """
    Class to show theoretically the failure probability
    across different code length values
    """
    name = "Failure Probability -- theoretical (plot+table)"

    @staticmethod
    def calculate_failure_probability_vs_code_length (
        margin_coeff: list[float], code_length_array: list[int]):
        """
        Calculate the key failure probability across different
        code length values
        """
        p_error = []
        p_failure = []
        threshold_values = []
        for n in code_length_array:
            threshold, _ =  calculate_threshold (n, margin_coeff)
            if threshold[0] == -1:
                threshold_values.append(np.nan)
                p_error.append(np.nan)
                p_failure.append(np.nan)
            else:
                threshold_values.append(threshold)
                p_error_value = theoretical_error_probability (n, threshold,
                                                               data_const.P_FLIP)
                p_error.append(p_error_value)
                p_failure.append(key_failure_probability(p_error_value))
        return threshold_values, p_error, p_failure

    @staticmethod
    def process(margin_coeff: list[float]):
        """
        Plot failure probability across different code length values
        given the standard deviation coefficient scalar value
        """

        threshold_values, p_error, p_failure = (
            FailureProbability.calculate_failure_probability_vs_code_length(
                margin_coeff, const.CODE_LENGTH_TO_TEST_FAILURE))

        # plot
        plt.plot(const.CODE_LENGTH_TO_TEST_FAILURE, p_failure, alpha=0.5)
        y_line  = 1e-6
        below_line = np.array(p_failure) < y_line
        if len(below_line)>0:
            x_below = const.CODE_LENGTH_TO_TEST_FAILURE[below_line]
            y_below = np.array(p_failure)[below_line]
            # Highlight points below the horizontal line
            plt.scatter(x_below, y_below, color='red', label='Points below $10^{-6}$')
            # Annotate the points below the horizontal line
            for xi, yi in zip(x_below, y_below):
                plt.annotate(f'{yi:.2e}', (xi, yi), textcoords="offset points",
                            xytext=(0, -13), ha='center', fontsize=9, color='red')
                # only first element
                break
        # Add a dashed horizontal line at y = 10^-6
        plt.axhline(y_line , color='r', linestyle='--', label=r'$P_{\mathrm{fail}}$ = $10^{-6}$')
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # x-axis format
        plt.xticks(ticks=const.CODE_LENGTH_TO_TEST_FAILURE,
                   labels=const.CODE_LENGTH_TO_TEST_FAILURE)
        plt.ylabel(r'$P_{\mathrm{fail}}$', fontsize=14)
        plt.xlabel(r'Codeword length ($n$)', fontsize=14)
        plt.title(r'Failure probability vs $n$ with $x_{\sigma}$ ='+f'{margin_coeff[1]}')
        plt.legend(loc=1, fontsize=14)
        plt.show()

          # assign data to table
        data = [
                ["Error probability"]+ [f"{value:.2e}" for value in p_error],
                ["Failure probability"]+ [f"{value:.2e}" for value in p_failure],
                ["thresholds"]+ threshold_values,
                ]
        # create header
        headers = ["Code Size"]+ list(const.CODE_LENGTH_TO_TEST_FAILURE)
        print(tabulate(data, headers, tablefmt="pretty"))

class OptimalRequirementsSRAM:
    """
    Class to show theoretically the optimal SRAM memory requirements
    across different codebook parameters and sizes
    """
    name = "Optimal SRAM requirements -- theoretical (plot)"

    @staticmethod
    def find_feasible_codebooks():
        """
        Find the feasible codebooks parameters across a wide range of code length (n)
        and sd coefficient (x_sigma) values, satisfying failure probability < 10^{-6}
        and constraints on threshold values (equations (15)-(16) in the journal paper)
        """
        feasible_cases = {}
        for sd_coefficient in const.SD_COEFFICIENT_TO_TEST_OPTIMALITY:
            threshold_values, _, p_failure = (
                FailureProbability.calculate_failure_probability_vs_code_length(
                    [1,sd_coefficient], const.CODE_LENGTH_TO_TEST_OPTIMALITY))
            feasible_case_index = [i for i, x in enumerate(p_failure) if x != np.nan
                                   and 10**-7 < x < 10**-6 and threshold_values[i][0] >= 0
                                   # equations (15-16)
                                   and (2*threshold_values[i][0]+1
                                        <= threshold_values[i][1] - threshold_values[i][0] -1)]
            if feasible_case_index:
                feasible_cases[sd_coefficient] = [[ const.CODE_LENGTH_TO_TEST_OPTIMALITY[i],
                                                  threshold_values[i], p_failure[i]]
                                                 for i in feasible_case_index]
        return feasible_cases

    @staticmethod
    def calculate_optimal_sram_requirements():
        """
        Find among the feasible codebook parameters, the codebooks that
        requiring minimum SRAM memory size across different codebook size values
        Return the memory size of SRAM (optimal), helper data and codebook,
        and other metrics such as the corresponding failure probability and threshold
        """
        codebook_size_array = const.CODEBOOK_SIZES_TO_TEST
        minimum_tuple = [()] * len(codebook_size_array)
        minimum_sram_size = [np.inf] * len(codebook_size_array)
        minimum_helper_data_size = [np.inf] * len(codebook_size_array)
        minimum_codebook_memory_size = [np.inf] * len(codebook_size_array)

        feasible_codebooks = OptimalRequirementsSRAM.find_feasible_codebooks()
        for sd_coeff, feasible_codebooks_info in feasible_codebooks.items():
            for [n, select_threshold, p_failure] in feasible_codebooks_info:
                sram_size = []
                helper_data_size = []
                codebook_memory_size = []
                for i, codebook_size in enumerate(codebook_size_array):
                    sram_size.append(theoretical_required_sram_size(
                        n, select_threshold, codebook_size))
                    helper_data_size.append(theoretical_required_helper_data_size(
                        n, select_threshold, codebook_size))
                    codebook_memory_size.append(required_codebook_size(n, codebook_size))
                    if sram_size[-1] < minimum_sram_size[i]:
                        minimum_sram_size[i] = sram_size[-1]
                        minimum_tuple[i] = (sd_coeff, select_threshold, n, p_failure)
                        minimum_helper_data_size[i] = helper_data_size[-1]
                        minimum_codebook_memory_size[i] = codebook_memory_size[-1]
        return (minimum_tuple, minimum_sram_size, minimum_helper_data_size,
                minimum_codebook_memory_size)

    @staticmethod
    def process():
        """
        Plot the trade-off between required memory size of SRAM (optimal across different
        codebooks), helper data and codebook, showing the optimal codebook parameters for
        different codebook size values
        """
        codebook_size_array = const.CODEBOOK_SIZES_TO_TEST
        (minimum_tuple, minimum_sram_size, minimum_helper_data_size, minimum_codebook_memory_size
         ) = OptimalRequirementsSRAM.calculate_optimal_sram_requirements()

        #plot
        plt.plot(codebook_size_array, minimum_sram_size, color='#ff7f0e', label='SRAM source bits')
        plt.plot(codebook_size_array, minimum_codebook_memory_size,color='#2ca02c',label='Codebook')
        plt.plot(codebook_size_array, minimum_helper_data_size, color='#1f77b4',label='Helper data')
        plt.title(r'Optimal SRAM requirements vs. $M$',fontsize=14)
        # Annotate the intervals
        intervals = [0]  # Starting index of each interval
        current_param = minimum_tuple[0]
        for i in range(1, len(minimum_tuple)):
            if minimum_tuple[i] != current_param:
                intervals.append(i)
                current_param = minimum_tuple[i]
        intervals.append(len(minimum_tuple))  # End of the last interval
        # Store annotation texts and positions
        texts = []
        positions = []
        # Define colors for the vertical lines and annotations
        colors = ['black', 'gray', 'darkolivegreen', 'midnightblue']
        # Add annotations
        for i in range(len(intervals)-1):
            start = intervals[i]
            color = colors[i % len(colors)]
            # Ensure indices are within valid range
            if (0 <= start < len(minimum_tuple) and 0 <= start < len(codebook_size_array)
                and 0 <= start < len(minimum_sram_size)):
                annotation_text = (fr'$n=${minimum_tuple[start][2]}'+'\n'
                                   +fr' $x_\sigma=${minimum_tuple[start][0]}'+'\n'
                                   +r' $P_{\mathrm{fail}}=$'+f'{minimum_tuple[start][3]:.2e}')
                plt.axvline(x=codebook_size_array[start], color=color, linestyle='--', linewidth=1)
                texts.append(plt.text(codebook_size_array[start], 3.2 + 1, annotation_text,
                                      fontsize=10, ha='right', va='center',color=color))
                positions.append((codebook_size_array[start], 3.2 + 1))
        # Adjust text to avoid overlapping
        adjust_text(texts, force_text=0.5, expand_text=1.5, only_move={'points':'xy', 'texts':'xy'})

        # convergence
        # Analyze the data to find the convergence point
        rate_codebook_size = codebook_size_array[1] - codebook_size_array[0]
        diff = np.abs(np.diff(minimum_sram_size)) / rate_codebook_size
        convergence_indices = np.where(diff < 0.0001)[0]
        # Check if convergence is found
        if convergence_indices.size > 0:
            convergence_index = convergence_indices[0]
            convergence_x = codebook_size_array[convergence_index]
            convergence_y = minimum_sram_size[convergence_index]
            logging.info('Optimal SRAM requirements')
            logging.info('Convergence occures at codebook size =%d, where:', convergence_x)
            logging.info('\tsram size = %.3fkB', convergence_y)
            logging.info('\thelper data size = %.3fkB', minimum_helper_data_size[convergence_index])
            logging.info('\tcodebook memory size = %.3fkB',
                         minimum_codebook_memory_size[convergence_index])

            # Annotate the convergence point on the plot
            plt.axvline(convergence_x, color='r',  label='Convergence')
            plt.scatter([convergence_x], [convergence_y], color='red')
        else:
            print("Convergence not found within the given threshold.")
        plt.legend(loc=(0.65,0.25))
        # Label the axes
        plt.xlabel(r'# Codewords ($M$)',fontsize=14)
        plt.ylabel('Memory size (kB)',fontsize=14)
        plt.grid(True)
        plt.show()
