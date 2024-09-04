"""
This script contains the core classes for the enrollment and regeneration phases of TMVS.
"""
import logging
import pathlib
import pickle
import numpy as np
import constants as const
from utils import ReadoutList, hamming_distance, read_codebook, read_enrollment_info
from utils import enrolled_key_dir, regenerated_key_dir
from formulas import calculate_threshold


class EnrollKey:
    """
    Class to execute the TMVS enrollment phase.
    """
    name = "Enroll Key -- experimental (save key + helper data)"

    @staticmethod
    def process(readouts: ReadoutList, code_length:int, margin_coeff: list[float]):
        """
        Get the SRAM readings for a single chip, and use the first reading to
        generate the reference key using TMVS algorithm for given:
        code length, and sd coefficient.
        Save the key, helper data, and nb. of discarded SRAM patterns
        in enrolled_key_dir directory. 
        """
        sram_size = readouts[0].data.size
        readout_0 = readouts[0].data.astype(np.int8)
        n = code_length

        select_threshold, _ = calculate_threshold (n, margin_coeff)
        logging.info('n=%d and [TH_low,TH_high]=[%d,%d]', n,
                     select_threshold[0], select_threshold[1])

        # Read codebook list
        codebook = read_codebook(n, select_threshold[0], select_threshold[1])
        codebook_size = len(codebook)
        logging.info('codebook size = %d', codebook_size)

        reference_key = []
        key_bits_address = []
        assigned_code_words = []

        sram_idx = np.int32(0)
        non_select_count = np.int32(0)
        while sram_idx < sram_size - n + 1:
            # sram pattern
            sram_pattern = readout_0[sram_idx : sram_idx + n]

            # testing different code words
            for code_word_idx in range(codebook_size):

                # code word value
                codeword = codebook[code_word_idx]
                codeword = np.array([int(bit) for bit in bin(codeword)[2:].zfill(n)])

                # hamming distance between sram_pattern and codeword
                hd = hamming_distance(codeword,sram_pattern)

                if select_threshold[0] < hd < select_threshold[1]:
                    if code_word_idx == codebook_size - 1:
                        sram_idx += 1
                        non_select_count += 1
                        break
                    continue

                if hd <= select_threshold[0]:
                    reference_key.append(np.int32(0))
                else:
                    reference_key.append(np.int32(1))

                # helper data
                key_bits_address.append(sram_idx)
                assigned_code_words.append(np.int32(code_word_idx))

                sram_idx += n
                break

        # save key + helper data
        filename = (f'reference_key_and_helper_data_{readouts.chip_id}_N{n}_TH_low_'
                    +f'{select_threshold[0]}_TH_high_{select_threshold[1]}.npz')
        np.savez(pathlib.Path(enrolled_key_dir / filename), key=reference_key,
                 address=key_bits_address, code_words=assigned_code_words,
                 non_select_count = [non_select_count])
        logging.info(
            'Saved key and helper data info for chip %s to %s', readouts.chip_id, filename)


class RegenerateKey:
    """
    Class to execute the TMVS regeneration phase.
    """
    name = "Regenerate Key -- experimental (save key)"

    @staticmethod
    def process(readouts: ReadoutList, code_length:int, margin_coeff: list[float]):
        """
        Get the SRAM readings for a single chip, and use the readings (except the first
        one) to regenerate key bits using the helper data generated previously.
        Save the regenerated key bits and bit error counts in regenerated_key_dir directory. 
        """

        n = code_length
        select_threshold, _ = calculate_threshold (n, margin_coeff)
        logging.info('n=%d and [TH_low,TH_high]=[%d,%d]', n,
                     select_threshold[0], select_threshold[1])

        # Read codebook list
        codebook = read_codebook(n, select_threshold[0], select_threshold[1])
        codebook_size = len(codebook)
        logging.info('codebook size = %d', codebook_size)

        regenerated_keys = []

        # read reference key and helper data
        filename = (f'reference_key_and_helper_data_{readouts.chip_id}_N{n}_TH_low_'
                    +f'{select_threshold[0]}_TH_high_{select_threshold[1]}.npz')
        data = read_enrollment_info(filename)

        reference_key = data['key']
        key_bits_address = data['address']
        assigned_code_words = data['code_words']

        avg_ber = 0
        error_count = []

        for i in range(1,len(readouts)): # skip first readout

            readout = readouts[i].data.astype(np.int8)
            wrong_key_bit_count = np.int32(0)
            key = []

            for j, reference_key_bit in enumerate(reference_key):

                # SRAM pattern
                sram_address = key_bits_address[j]
                sram_pattern = readout[sram_address : sram_address + n]

                # code word
                code_word_idx = assigned_code_words[j]
                codeword = codebook[code_word_idx]
                codeword = np.array([int(bit) for bit in bin(codeword)[2:].zfill(n)])

                # hamming distance between sram_pattern and codeword
                hd = hamming_distance(codeword,sram_pattern)
                wrong_key_bit = 0

                if hd <= np.floor(n * const.P_SRAM):
                    key.append(np.int32(0))
                    wrong_key_bit = reference_key_bit
                else:
                    key.append(np.int32(1))
                    wrong_key_bit = 1 - reference_key_bit

                wrong_key_bit_count += wrong_key_bit

            regenerated_keys.append(key)
            error_count.append(wrong_key_bit_count)

        # save regenerated keys
        filename = (f'regenerated_keys_{readouts.chip_id}_N{n}_TH_low_'
                    +f'{select_threshold[0]}_TH_high_{select_threshold[1]}.pkl')
        pickle.dump([regenerated_keys, error_count],
                    open(pathlib.Path(regenerated_key_dir / filename), 'wb'))
        logging.info('Saved regenerated keys and ber info for chip %s to %s',
                     readouts.chip_id, filename)

        avg_ber = sum(error_count) / (len(reference_key) * (len(readouts) - 1) )
        logging.info('average BER = %f\n', avg_ber)
   