"""
NOTE: This implementation slightly deviates from the dark bit implementation where
we need to set a threshold on the reliability of cells. Here the threshold is
maximum = 0.5. An accurate implementation is the one of Bernardini et al. in the
project implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from common.data_reading_utils import ReadoutList, get_files, read_readouts


class GenerateBitsMask:
    """
    Class to generate the stable cells masks (for every SRAM readout).
    """
    name = "Generate Stable Cells Masks -- experimental"


    @staticmethod
    def compute_metrics_for_chip(chip_id: str, nb_enroll_readings: int | None = None):
        """
        Read readouts for a chip, optionally limit readings, and compute metrics.
        Compute BER and discarded fraction per readout using the dark bits approach.

        Returns:
            x_axis: numpy array of readout indices starting at 1 (excluding the reference)
            ber_values: numpy array of BER per readout
            discarded_bits_values: numpy array of discarded fraction per readout
        """
        files = get_files()
        if chip_id not in files:
            raise ValueError(f"Chip id '{chip_id}' not found in readouts directory")
        readouts: ReadoutList = read_readouts(files[chip_id])
        if nb_enroll_readings is not None:
            limit = min(len(readouts), nb_enroll_readings + 1)
            readouts = ReadoutList(readouts.chip_id, readouts[:limit])
        
        ber_values = []
        discarded_bits_values = []

        reference_readout = readouts[0].data.astype(np.int8)
        sram_size = reference_readout.size
        bit_mask = np.zeros(sram_size, dtype=np.int8)

        for i in range(1, len(readouts)):
            new_readout = readouts[i].data.astype(np.int8)
            flipped_bits = reference_readout ^ new_readout
            masked_flipped_bits = flipped_bits & (1 - bit_mask)
            valid_count = np.sum(1 - bit_mask)
            ber = np.nan if valid_count == 0 else (np.sum(masked_flipped_bits) / valid_count)
            bit_mask = flipped_bits | bit_mask
            discarded_bits_count = np.sum(bit_mask)
            ber_values.append(ber)
            discarded_bits_values.append(discarded_bits_count / sram_size)

        x_axis = np.arange(1, len(readouts))
        return x_axis, np.array(ber_values, dtype=float), np.array(discarded_bits_values, dtype=float)

    @staticmethod
    def compute_aggregate_metrics_over_chips(nb_enroll_readings: int | None = None,
                                             chip_ids: list[str] | None = None):
        """Compute aggregate Dark Bits metrics (BER and discarded fraction) across chips.

        Args:
            nb_enroll_readings: If provided, limit to the first N readings after the reference
                (i.e., x-axis will be 1..N). If None, uses the minimum length across chips.
            chip_ids: Optional list of chip IDs to include. If None, includes all available chips.

        Returns:
            x_axis: numpy array of length L (1..L)
            results: dict with keys 'ber' and 'discarded', each mapping to a dict
                    {'mean': (L,), 'min': (L,), 'max': (L,)}
        """
        files = get_files()
        selected_chips = chip_ids if chip_ids is not None else list(files.keys())

        # Collect per-chip series
        ber_series_list = []
        discarded_series_list = []
        lengths = []
        for chip_id in selected_chips:
            if chip_id not in files:
                continue
            # Delegate reading and optional limiting to the chip-aware function
            x_axis, ber_vals, discarded_vals = GenerateBitsMask.compute_metrics_for_chip(
                chip_id, nb_enroll_readings
            )
            ber_series_list.append(ber_vals)
            discarded_series_list.append(discarded_vals)
            lengths.append(len(x_axis))

        if not ber_series_list:
            return np.array([]), {
                'ber': {'mean': np.array([]), 'min': np.array([]), 'max': np.array([])},
                'discarded': {'mean': np.array([]), 'min': np.array([]), 'max': np.array([])}
            }

        # Align lengths across chips
        L = min(lengths)
        if nb_enroll_readings is not None:
            L = min(L, nb_enroll_readings)
        ber_mat = np.vstack([s[:L] for s in ber_series_list])  # shape: (num_chips, L)
        discarded_mat = np.vstack([s[:L] for s in discarded_series_list])

        # Aggregate across chips (axis 0)
        ber_mean = np.mean(ber_mat, axis=0)
        ber_min = np.min(ber_mat, axis=0)
        ber_max = np.max(ber_mat, axis=0)
        disc_mean = np.mean(discarded_mat, axis=0)
        disc_min = np.min(discarded_mat, axis=0)
        disc_max = np.max(discarded_mat, axis=0)

        x_axis = np.arange(1, L + 1)
        results = {
            'ber': {'mean': ber_mean, 'min': ber_min, 'max': ber_max},
            'discarded': {'mean': disc_mean, 'min': disc_min, 'max': disc_max}
        }
        return x_axis, results

    @staticmethod
    def plot_results(x_axis: np.ndarray, results: dict, title: str):
        """Plot results for either a single chip or aggregated across chips.
        results schema:
          - Per-chip: {'ber': np.ndarray(L,), 'discarded': np.ndarray(L,)}
          - Aggregate: {'ber': {'mean': (L,), 'min': (L,), 'max': (L,)},
                        'discarded': {'mean': (L,), 'min': (L,), 'max': (L,)}}
        """
        ber = results['ber']
        discarded = results['discarded']

        _, ax1 = plt.subplots(figsize=(7, 4))
        # BER primary axis
        if isinstance(ber, dict):
            ax1.plot(x_axis, ber['mean'], 'g-', label='BER (mean)')
            ax1.fill_between(x_axis, ber['min'], ber['max'], color='g', alpha=0.2, label='BER (min/max)')
        else:
            ax1.plot(x_axis, ber, 'g-', label='BER')
        ax1.set_xlabel('Number of power-off-power-on cycles')
        ax1.set_ylabel('BER', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax1.set_yscale('log')

        # Discarded secondary axis
        ax2 = ax1.twinx()
        if isinstance(discarded, dict):
            ax2.plot(x_axis, discarded['mean'], 'b--', label='Discarded (mean)')
            ax2.fill_between(x_axis, discarded['min'], discarded['max'], color='b', alpha=0.2, label='Discarded (min/max)')
        else:
            ax2.plot(x_axis, discarded, 'b--', label='Discarded')
        ax2.set_ylabel('Disacarded bits (%)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

        plt.title(title)
        plt.show()

    # removed: use plot_results for both single-chip and aggregate

    @staticmethod
    def process(chip_id: str | None = None, nb_enroll_readings: int | None = None):
        """Compute and plot Dark Bits metrics.
        - If chip_id provided: load and plot that chip (optionally limited).
        - Else: aggregate across all chips and plot mean with min/max.
        """
        if chip_id is not None:
            # Use the chip-aware metric computation which handles reading and optional limiting
            x_axis, ber_values, discarded_bits_values = GenerateBitsMask.compute_metrics_for_chip(
                chip_id, nb_enroll_readings
            )
            results = {'ber': ber_values, 'discarded': discarded_bits_values}
            GenerateBitsMask.plot_results(x_axis, results, title=f'Dark Bit Approach Performance for chip {chip_id}')
            return

        x_axis, results = GenerateBitsMask.compute_aggregate_metrics_over_chips(nb_enroll_readings=nb_enroll_readings)
        GenerateBitsMask.plot_results(x_axis, results, title='Dark Bit Approach Performance (Aggregate across chips)')

if __name__ == "__main__":

    all_files = get_files()
    all_readouts: ReadoutList = read_readouts(all_files['M2'])
    GenerateBitsMask.process(nb_enroll_readings=100)
    # all_readouts: list[ReadoutList] = [read_readouts(all_files['M49'])]
    # for readouts in all_readouts:
    #     GenerateBitsMask.process(readouts=readouts)