""" Module for chunking data from a list of readouts. """
import multiprocessing
import threading
import numpy as np
from nvm_free_tmvs.utils.file_manager import ReadoutList
from nvm_free_tmvs.utils.analysis_utils import get_optimal_data_type


class ChunkDataProcessor:
    """
    Class for processing chunks of data from a list of readouts.
    """
    def __init__(self, chunk_len: int, readouts: ReadoutList, active_multithreading: bool):
        """
        Initialize the ChunkDataProcessor.
        """
        self.chunk_len = chunk_len
        self.readouts = readouts
        # Determine the optimal data type for the chunks
        self.chunk_dtype = get_optimal_data_type(2**chunk_len)
        self.active_multithreading = active_multithreading

    def chunk_bits(self, data: bytes):
        """
        Chunk bytes into integers of size `self.chunk_len` bits using the optimal data type.
        """
        # Convert bytes to a NumPy array of uint8
        data = np.frombuffer(data, dtype=np.uint8)

        # Total bits in the input data
        total_bits = len(data) * 8

        # Calculate the number of bits to keep (discard the remainder)
        bits_to_keep = total_bits - (total_bits % self.chunk_len)

        # Convert uint8 data into a flat array of bits
        bit_array = np.unpackbits(data, bitorder='big')

        # Trim the bit array to the largest multiple of `self.chunk_len`
        bit_array = bit_array[:bits_to_keep]

        # Reshape the bit array into rows of `self.chunk_len` bits
        bit_array = bit_array.reshape(-1, self.chunk_len)

        # Convert each row of bits into integers using NumPy's dot product
        powers = 1 << np.arange(self.chunk_len - 1, -1, -1, dtype=np.uint64)  # Powers of 2
        chunked_data = np.dot(bit_array, powers).astype(self.chunk_dtype)  # Convert to target dtype
        return chunked_data

    def _chunk_readouts(self, index: int, result_dict: dict, range_start: int, range_end: int):
        """
        Chunk readouts from `range_start` to `range_end` and store the result in `result_dict`.
        """
        result_dict[index] = np.array([self.chunk_bits(r.data)
                                       for r in self.readouts[range_start:range_end]])
        
    def chunk_readouts(self):
        """
        Chunk readouts into integers of size `self.chunk_len` bits using the optimal data type.
        Returns chunked_data of shape (num_readouts, num_sram_patterns)
        """
        if self.active_multithreading:
            num_cores = multiprocessing.cpu_count()
            range_width = len(self.readouts)//num_cores
            remainder = len(self.readouts) % num_cores  # Remaining readouts after division
            range_threads = []
            parallel_results = {}
            start = 0
            index = 0
            while start < len(self.readouts):
                end = start + range_width + (1 if remainder > 0 else 0)
                end = min(end, len(self.readouts))
                remainder = max(remainder - 1, 0)  # Deduct from remainder for extra readouts
                range_thread = threading.Thread(
                    target=self._chunk_readouts,
                    args=(index, parallel_results, start, end),
                    )
                range_threads.append(range_thread)
                range_thread.start()
                start = end
                index += 1

            for thread in range_threads:
                thread.join()

            chunked_data = []
            # for _,value in enumerate(parallel_results): # Iterate over all threads' results
            #     chunked_data.append(value)
            for i in range(num_cores):
                chunked_data.append(parallel_results[i])

            return np.concatenate(chunked_data, axis=0)

        result_dict = {}
        self._chunk_readouts(0, result_dict, 0, len(self.readouts))
        return result_dict[0]
