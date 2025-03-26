"""
Base class for the nvm_free_tmvs_algo module.
"""
from abc import ABC, abstractmethod
from nvm_free_tmvs.core.hamming_processor import HammingProcessor

class BaseAnalysis(ABC): # change name to Analysis
    """ Base class for the nvm_free_tmvs_algo module. """
    def __init__(self, hamming_processor: HammingProcessor,
                 data_start_idx=None, num_enroll_readings=None,
                 incremental_computation: bool = False):
        self.hamming_processor = hamming_processor
        self.code_length = hamming_processor.code_length
        self.select_threshold = hamming_processor.select_threshold
        self.data_start_idx = data_start_idx
        self.num_enroll_readings = num_enroll_readings
        self.incremental_computation = incremental_computation
        # self.dataset = processor.dataset
        # self.codebook = processor.codebook


    @abstractmethod
    def execute(self, enroll_select_threshold,
                enroll_hamming_distances=None,
                boolean_hamming_distances=None):
        """
        Main logic for the specific analysis.
        Subclasses must implement this method, optionally accepting additional inputs.
        """
