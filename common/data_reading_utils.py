"""
This script includes utility classes and functions for data handling, including
SRAM readings from different SCuM chips and codebooks.
"""
# The implementation of get_files() and read_results(files: list[pathlib.Path]) functions
# is copied from https://github.com/bkorecic/scum-sram-evaluation/blob/main/analysis/utils.py
import pathlib
import pickle
import logging
import json
from typing import List
import numpy as np
import common.data_constants as data_const

# directory to store sram readouts data
readouts_dir = pathlib.Path(__file__).parent.parent.absolute() / 'data' /'SRAM_readouts'

# filepath to store generated codebooks
codebooks_dir = pathlib.Path(__file__).parent.parent.absolute() / 'data' / 'codebooks.json'


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
                while len(readouts) < data_const.READINGS_TO_ANALYZE:
                    data = pickle.load(f)
                    readouts.append(Readout(*data))
            except EOFError:
                pass
    if len(readouts) != data_const.READINGS_TO_ANALYZE:
        logging.warning('Expected at least %d readings for chip %s, got %d.',
                        data_const.READINGS_TO_ANALYZE, chip_id, len(readouts))
    return readouts

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
