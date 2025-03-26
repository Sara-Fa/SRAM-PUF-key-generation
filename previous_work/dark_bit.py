"""
TODO
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
    def process(readouts: ReadoutList):
        """
        TODO
        Get the SRAM readings for a single chip, and use the readings (except the first
        one) to regenerate key bits using the helper data generated previously.
        Save the regenerated key bits and bit error counts in regenerated_key_dir directory. 
        """

        ber_values = []
        discarded_bits_values = []

        reference_readout = readouts[0].data.astype(np.int8)

        sram_size = reference_readout.size

        bit_mask = np.zeros(sram_size, dtype=np.int8)

        # logging.info('info for chip %s:', readouts.chip_id)

        for i in range(1,len(readouts)): # skip first readout

            new_readout = readouts[i].data.astype(np.int8)

            flipped_bits = reference_readout ^ new_readout
            # print(np.sum(flipped_bits))

            masked_flipped_bits = flipped_bits & (1-bit_mask)

            ber = np.sum(masked_flipped_bits) / np.sum(1-bit_mask)
            

            bit_mask = flipped_bits | bit_mask
            discarded_bits_count  = np.sum(bit_mask)

            ber_values.append(ber)

            # if ber < 5*10**(-4):
                # print( np.sum(masked_flipped_bits))
                # indices = np.where(masked_flipped_bits == 1)[0]
                # print(indices)

            discarded_bits_values.append(discarded_bits_count / sram_size)

            # print(f'ber after readout #{i}: {ber*100} %')
            # print(f'discarded bits after readout #{i}: {100 * discarded_bits_count / sram_size} %')
            # logging.info('ber after readout #%d: %f %', i, ber*100)
            # logging.info('discarded bits after readout #%d: %f %', i, 100 * discarded_bits_count / sram_size)

        # Plotting
        _, ax1 = plt.subplots(figsize=(7, 4))
        # _, ax1 = plt.subplots(figsize=(6, 3))

        # Plot the first line with the first y-axis
        ax1.plot(range(1,len(readouts)), ber_values, 'g-',
                 label='BER')  # 'g-' means green solid line
        ax1.set_xlabel('Number of power-off-power-on cycles')
        ax1.set_ylabel('BER', color='g')
        ax1.tick_params(axis='y', labelcolor='g')

        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # Apply log scale to the first y-axis
        ax1.set_yscale('log')

        # Create a second y-axis sharing the same x-axis
        ax2 = ax1.twinx()

        # Plot the second line with the second y-axis
        ax2.plot(range(1,len(readouts)), discarded_bits_values, 'b--',
                 label='Disacarded bits')  # 'b--' means blue dashed line
        ax2.set_ylabel('Disacarded bits (%)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        # Format the second y-axis as percentage
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

        # Optionally add a title
        plt.title(f'Dark Bit Approach Performance for chip {readouts.chip_id}')

        # Show the plot
        plt.show()



if __name__ == "__main__":

    all_files = get_files()
    all_readouts: list[ReadoutList] = [read_readouts(all_files['M49'])]
    for readouts in all_readouts:
        GenerateBitsMask.process(readouts=readouts)

