"""
This script includes utility classes and functions for data handling, including
SRAM readings from different SCuM chips, codebooks, enrolled and regenerated key bits.
"""
# The implementation of get_files() and read_results(files: list[pathlib.Path]) functions
# is copied from https://github.com/bkorecic/scum-sram-evaluation/blob/main/analysis/utils.py
import pathlib
import pickle
import logging
import json
import re
import os
from typing import List
import numpy as np
import constants as const

# directory to store sram readouts data
readouts_dir = pathlib.Path(__file__).parent.absolute() / 'data' /'SRAM_readouts'

# directory to store enrolled key bits + helper data using the first SRAM readout
enrolled_key_dir = pathlib.Path(__file__).parent.absolute() / 'enrolled_keys'
enrolled_key_dir.mkdir(exist_ok=True)

# directory to store regenerated key bits from other SRAM readouts
regenerated_key_dir = pathlib.Path(__file__).parent.absolute() / 'regenerated_keys'
regenerated_key_dir.mkdir(exist_ok=True)

# filepath to store generated codebooks
codebooks_dir = pathlib.Path(__file__).parent.absolute() / 'data' / 'codebooks.json'

class Readout:
    """
    Class to represent a single readout from a chip
    """

    def __init__(self,
                 start_timestamp: float,
                 end_timestamp: float,
                 data: bytes):
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.data = np.unpackbits(np.frombuffer(data, dtype=np.uint8))


class ReadoutList (list):
    """
    Class to represent a list of readouts from a chip. Has the same interface
    as a python list but adds a "chip_id" attribute
    """

    def __init__(self, chip_id: str, *args):
        super().__init__(*args)
        self.chip_id = chip_id


class KeyInfo:
    """
    Class to represent key bits info from a single chip with codebook
    parameters (n,[TH_low,TH_high])
    """
    def __init__(self,
                 chip_id: str,
                 reference_key: List[int],
                 key_bits_addr: List[int],
                 error_counts:  List[int],
                 code_words: List[int],
                 skiped_bits_count:  List[int]):
        self.chip_id = chip_id
        self.reference_key = reference_key
        self.key_bits_addr = key_bits_addr
        self.error_counts = error_counts
        self.code_words = code_words
        self.skiped_bits_count = skiped_bits_count


class KeyInfoList (list):
    """
    Class to represent a list of key bits info for multiple chips
    with codebook parameters (n,[TH_low,TH_high]). 
    Has the same interface as a python list but adds n, [TH_low,TH_high] attributes
    """

    def __init__(self, n: int, thresh_low: int, thresh_high: int, *args):
        super().__init__(*args)
        self.n = n
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high

        # Read codebook with parameters (n,[TH_low,TH_high])
        codebook = read_codebook(n, thresh_low, thresh_high)
        self.codebook = codebook
        self.codebook_size = len(codebook)


def hamming_distance(bit_arr1: np.ndarray, bit_arr2: np.ndarray) -> int:
    """
    Calculate the Hamming distance between two bit arrays
    """
    return np.sum(bit_arr1 ^ bit_arr2)


def get_files() -> dict:
    """
    Gets the files in the sibling data/SRAM_readouts directory.
    Puts them into a dictionary where the key is the chip ID
    """
    sram_readings = {}
    for f in readouts_dir.iterdir():
        chip_id = f.parts[-1].split('-')[0]
        if chip_id in sram_readings:
            sram_readings[chip_id].append(f)
        else:
            sram_readings[chip_id] = [f]
    return sram_readings


def read_readouts(files: list[pathlib.Path]) -> ReadoutList:
    """
    Read, unpickle, merge, trim and return a list of readout files

    files -- list of paths to readout files
    """
    files.sort()  # Sort to read chronologically
    chip_id = files[0].parts[-1].split('-')[0]
    readouts = ReadoutList(chip_id)
    for fp in files:
        with open(fp, 'rb') as f:
            try:
                while len(readouts) < const.READINGS_TO_ANALYZE:
                    data = pickle.load(f)
                    readouts.append(Readout(*data))
            except EOFError:
                pass
    if len(readouts) != const.READINGS_TO_ANALYZE:
        logging.warning('Expected at least %d readings for chip %s, got %d.',
                        const.READINGS_TO_ANALYZE, chip_id, len(readouts))
    return readouts


def extract_filename_info(file_path: pathlib.Path):
    """
    Extract chip ID, code length (n), hreshold values from filename
    given the file path
    """
    filename_pattern = re.compile(
        r"(?:reference_key_and_helper_data|regenerated_keys)"+
        r"_(\w+)_N(\d+)_TH_low_(\d+)_TH_high_(\d+)")
    match = filename_pattern.match(os.path.basename(file_path))
    chip_id = match.group(1)
    n = int(match.group(2))
    thresh_low = int(match.group(3))
    thresh_high = int(match.group(4))
    return chip_id, n, thresh_low, thresh_high


def get_codebook_parameters(chips_to_evaluate: list[str]) -> tuple[dict, dict]:
    """
    Get the files in the sibling enrolled_key_dir and regenerated_key_dir directories
    belonging chips_to_evaluate.
    Put them in two dictionaries where the key of every item is the codebook
    parameters (n,[TH_low,TH_high])
    
    chips_to_evaluate -- list of chip_id of SCuM chips 
    """
    parameters_enroll = {}
    parameters_regenerate = {}
    for f in enrolled_key_dir.iterdir():
        if f.is_file():
            chip_id, n, thresh_low, thresh_high = extract_filename_info(f)
            if chip_id in chips_to_evaluate:
                if (n,thresh_low,thresh_high) in parameters_enroll:
                    parameters_enroll[(n,thresh_low,thresh_high)].append(f)
                else:
                    parameters_enroll[(n,thresh_low,thresh_high)] = [f]

    all_keys = list(parameters_enroll.keys())
    all_keys.sort()
    # sorted dictionary
    sorted_parameters_enroll = {i: parameters_enroll[i] for i in all_keys}

    for f in regenerated_key_dir.iterdir():
        if f.is_file():
            chip_id, n, thresh_low, thresh_high = extract_filename_info(f)
            if chip_id in chips_to_evaluate:
                if (n,thresh_low,thresh_high) in parameters_regenerate:
                    parameters_regenerate[(n,thresh_low,thresh_high)].append(f)
                else:
                    parameters_regenerate[(n,thresh_low,thresh_high)] = [f]

    all_keys = list(parameters_regenerate.keys())
    all_keys.sort()
    # sorted dictionary
    sorted_parameters_regenerate = {i: parameters_regenerate[i] for i in all_keys}

    return sorted_parameters_enroll, sorted_parameters_regenerate


def read_key_info(files_enroll: list[pathlib.Path],
                  files_regenerate: list[pathlib.Path]) -> KeyInfoList:
    """
    Read, unpickle, merge, trim and return a list of key bits info for different chips
    with codebook parameters (n, [TH_low,TH_high])
    files_enroll -- list of paths to files storing enrolled key bits and helper data info
                    for different chips with codebook parameters (n, [TH_low,TH_high])
    files_regenerate -- list of paths to files storing regenerated key bits info
                        for different chips with codebook parameters (n, [TH_low,TH_high])
    """
    files_enroll.sort()  # Sort to read chronologically
    files_regenerate.sort()
    _, n, thresh_low, thresh_high = extract_filename_info(files_enroll[0])
    extracted_info = KeyInfoList(n, thresh_low, thresh_high)
    for fp_enroll, fp_reg in zip(files_enroll, files_regenerate):
        chip_id, _, _, _ = extract_filename_info(fp_enroll)
        try:
            enrollement_data = np.load(fp_enroll)
        except EOFError:
            pass
        # regenrated keys
        with open(fp_reg, 'rb') as f:
            try:
                _, bit_error_count = pickle.load(f)
            except EOFError:
                pass

        extracted_info.append(KeyInfo(chip_id, enrollement_data['key'],
                                      enrollement_data['address'],
                                      bit_error_count,
                                      enrollement_data['code_words'],
                                      enrollement_data['non_select_count'][0]))
    return extracted_info


def read_codebook(n:int, thresh_low:int, thresh_high:int) -> List[int]:
    """
    Read and return the codebook with parameters (n, [TH_low,TH_high])
    generated and stored in file_path
    """
    try:
        with open(pathlib.Path(codebooks_dir), 'r', encoding='utf-8') as f:
            codebooks = json.load(f)
        return codebooks[f'CODE_N{n}_TH_low_{thresh_low}_TH_high_{thresh_high}']
    except FileNotFoundError:
        print(f"Error: File not found at {codebooks_dir}")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in file {codebooks_dir}")
    except KeyError:
        print(f"Error: Key 'CODE_N{n}_TH_low_{thresh_low}_TH_high_{thresh_high}'",
              "not found in the codebooks")


def read_enrollment_info(filename:str):
    """
    Read and return the data stored during enrollment in filename
    """
    try:
        # load file if found
        data = np.load(pathlib.Path(enrolled_key_dir / filename))
        return data
    except FileNotFoundError:
        logging.info('File not found: %s', filename)
        return  # Skip to the next iteration
    except (OSError, IOError) as e:
        logging.info('File read error for %s: %s', filename, e)
        return  # Skip to the next iteration in case of file-related issues
    except ValueError as e:
        logging.info('Value error while processing %s: %s', filename, e)
        return  # Skip to the next iteration in case of value parsing issues
