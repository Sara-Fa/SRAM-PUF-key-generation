""" Module for processing Hamming distances between SRAM patterns and codebook elements. """
# import time
# from common.formulas import calculate_threshold
import os
import gc
from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory
from typing import List
import numpy as np
import common.data_constants as data_const
# from nvm_free_tmvs.core.cache_manager import CacheManager
from nvm_free_tmvs.core.chunk_data_processor import ChunkDataProcessor
from nvm_free_tmvs.utils.file_manager import read_codebook, read_readouts
from nvm_free_tmvs.utils.file_manager import ReadoutList, get_files
import nvm_free_tmvs.analysis_constants as const


class HammingProcessor:
    """ Class for processing Hamming distances between SRAM patterns and codebook elements. """
    def __init__(self, code_length: int, readouts: ReadoutList, select_threshold: List[float],
                 active_multithreading: bool): # replace select_threshold by margin_ceoff
        """ Initialize the HammingProcessor. """
        self.readouts = readouts
        self.chip_id = readouts.chip_id
        self.code_length = code_length
        self.select_threshold = select_threshold
        # self.margin_coeff = margin_coeff
        # self.select_threshold, _ = calculate_threshold (code_length, margin_coeff)
        # self.cache_manager = CacheManager()
        self.chunk_data_processor = ChunkDataProcessor(code_length, readouts, active_multithreading)
        self.active_multithreading = active_multithreading


    @staticmethod
    def create_lookup_table():
        """
        Generate a lookup table for Hamming distances between two bit values.
        """
        max_value = 2 ** 16
        return np.array([ int(i).bit_count() for i in range(max_value)], dtype=const.HD_DATA_TYPE)

    def get_codebook(self, codewords_indices=None):
        """ Get the codebook based on the specified indices. """
        if codewords_indices is None:
            return read_codebook(self.code_length, self.select_threshold[0],
                                    self.select_threshold[1])
        return np.array(read_codebook(self.code_length, self.select_threshold[0],
                                    self.select_threshold[1]))[codewords_indices]

    @staticmethod
    def _worker_compute_hamming_distances(codebook_chunk, chunked_data, lookup_table_name,
            shared_mem_name, offset, total_codebook_size, floor_half):
        """
        Worker function to compute Hamming distances for a subset of the codebook.
        """

        try:
            # (codebook_chunk, chunked_data, lookup_table_name,
            # shared_mem_name, offset, total_codebook_size, floor_half) = tasks
            # Reconnect to shared memory
            shared_hamming_mem = SharedMemory(name=shared_mem_name)
            shared_lookup_mem = SharedMemory(name=lookup_table_name)

            # Recreate lookup table
            lookup_table = np.ndarray((2 ** 16,), dtype=np.int8, buffer=shared_lookup_mem.buf).astype(chunked_data.dtype)
            # Recreate numpy arrays
            shared_result = np.ndarray(
                # (total_codebook_size, chunked_data.shape[0], chunked_data.shape[1]),
                (chunked_data.shape[1], total_codebook_size, chunked_data.shape[0]),
                dtype=const.HD_DATA_TYPE,
                buffer=shared_hamming_mem.buf
            )#[offset: offset + len(codebook_chunk)]

            dtype_num_chunks = int(np.ceil(
                chunked_data.dtype.itemsize / 2)) # Number of 16-bit chunks

             # Temporary buffer to hold results for current codebook chunk
            temp_result = np.zeros((chunked_data.shape[0], chunked_data.shape[1]), dtype=np.int8)


            # Compute Hamming distances for each element in the codebook chunk
            for i, cb_val in enumerate(codebook_chunk):
                xor_result = np.bitwise_xor(cb_val, chunked_data)
                # shared_result[i] = lookup_table[xor_result] # + look_table

                if dtype_num_chunks > 1:
                    # Initialize partial results array
                    partial_results = np.zeros_like(xor_result, dtype=chunked_data.dtype)

                    # Split xor_result into chunks and calculate partial results
                    for chunk_idx in range(dtype_num_chunks):
                        # Extract 16-bit chunks by shifting and masking
                        chunk = np.bitwise_and(np.right_shift(xor_result, chunk_idx * 16), 0xFFFF)
                        # Accumulate lookup table results
                        partial_results += lookup_table[chunk]

                    # Assign the results for the current codebook chunk
                    temp_result = partial_results
                else:
                    # Compute results directly using the lookup table
                    temp_result = lookup_table[xor_result]

                # Apply modification rules
                # Resulting values are in the ranges [-floor_half-1, -1] or [1, floor_half]
                temp_result = np.where(temp_result <= floor_half,
                                       temp_result - floor_half - 1,
                                       # following step ~ temp_result - (floor_half + 1) + 1
                                       temp_result - floor_half
                                       )

                # Write the results into shared_result with the new layout
                for j in range(chunked_data.shape[1]):  # Iterate over SRAM patterns
                    shared_result[j, offset + i, :] = temp_result[:, j]
        except (MemoryError, ValueError, OSError) as e:
            print(f"Error in worker process: {e}")

        finally:
            shared_hamming_mem.close()
            shared_lookup_mem.close()

    def compute_hamming_distances(self, data_start_idx, num_readings,
                                  chunked_data=None, codewords_indices=None):
        """ Compute Hamming distances between SRAM patterns and codebook elements. """
        # print("Chunking data.")
        # put this step outside this function
        if chunked_data is None:
            chunked_data = self.chunk_data_processor.chunk_readouts()
        codebook = self.get_codebook(codewords_indices)
        print(f"codebook length: {len(codebook)}")

        # Create a lookup table for bit counts
        max_value = 2 ** 16 # 2 ** dtype_bits
        bit_count_lookup = self.create_lookup_table()

        floor_half = int(self.code_length * data_const.P_SRAM)
        # print(f"Floor half: {floor_half}")

        # n_chunks, chunk_size = chunked_data.shape
        if 'SLURM_CPUS_ON_NODE' in os.environ:
            num_cores = int(os.environ['SLURM_CPUS_ON_NODE'])
        else:
            num_cores = min(cpu_count() - 1, 4)  # Adjust cores as needed

        total_codebook_size = len(codebook)
        # Split the codebook into manageable chunks for parallel processing
        # if total_codebook_size > const.MAX_CODEWORDS_PER_CHUNK:
        #     codebook_chunks = np.array_split(codebook, num_cores)
        # else:
        #     codebook_chunks = [codebook]
        codebook_chunks = []
        for start in range(0, len(codebook), const.MAX_CODEWORDS_PER_CHUNK):
            end = min(start + const.MAX_CODEWORDS_PER_CHUNK, len(codebook))
            codebook_chunks.append(codebook[start:end])
        # print(f"Processing {len(codebook_chunks)} codebook chunks.")

        # Determine the max chunk size for chunked_data to fit memory constraints
        ########## !!!!!!!!!!!! try value 250 !!!!!!!!!!!! ##########
        data_chunks = [
            chunked_data[i : min(i + const.MAX_NUM_CHUNKS, data_start_idx + num_readings)]
            for i in range(data_start_idx, data_start_idx + num_readings,
                           const.MAX_NUM_CHUNKS)
            ]

        # print(f"Processing {len(data_chunks)} data chunks sequentially.")

        hamming_results = []

        try:
            # Allocate shared memory for the lookup table
            shared_lookup_mem = SharedMemory(create=True, size=bit_count_lookup.nbytes)

            # Write the lookup table to shared memory
            np.ndarray((max_value,), dtype=const.HD_DATA_TYPE,
                       buffer=shared_lookup_mem.buf)[:] = bit_count_lookup

            for data_idx, data_chunk in enumerate(data_chunks):
                print(f"Starting data chunk {data_idx + 1}/{len(data_chunks)}.")
                print(f"Data chunk shape: {data_chunk.shape}")
                # Calculate memory size for this chunk
                chunk_hamming_mem_size = (total_codebook_size * data_chunk.shape[0]
                                          * data_chunk.shape[1])
                shared_hamming_mem = SharedMemory(create=True, size=chunk_hamming_mem_size)

                try:
                    # Prepare tasks for multiprocessing
                    tasks = []
                    offset = 0  # Start of the shared memory
                    for codebook_chunk in codebook_chunks:
                        tasks.append((
                            codebook_chunk,         # Current chunk of codebook
                            data_chunk,             # Current chunk of chunked_data
                            shared_lookup_mem.name,    # Shared memory for lookup table
                            shared_hamming_mem.name,   # Shared memory for results
                            offset,
                            total_codebook_size,
                            floor_half
                        ))
                        offset += len(codebook_chunk)

                    if self.active_multithreading and len(tasks) > 1:
                        # Process tasks in parallel
                        print(f"Starting parallel processing with {len(tasks)} tasks.")
                        with Pool(processes=num_cores) as pool:
                            # pool.starmap(self._worker_compute_hamming_distances, tasks)
                            pool.starmap(self._worker_compute_hamming_distances, tasks)
                            # pool.close()      # Prevents new tasks
                            # pool.join()       # Ensures proper cleanup
                    else:
                        # Run in single-threaded mode
                        for task in tasks:
                            self._worker_compute_hamming_distances(*task)
                    # print("Reached point 1")
                    # Retrieve the results from shared memory for this chunk
                    hamming_chunk = np.ndarray(
                        # (total_codebook_size, data_chunk.shape[0], data_chunk.shape[1]),
                        (data_chunk.shape[1], total_codebook_size, data_chunk.shape[0]),
                        dtype=const.HD_DATA_TYPE,
                        buffer=shared_hamming_mem.buf,
                    ).copy()  # Copy to avoid issues when the shared memory is released
                    hamming_results.append(hamming_chunk)
                    del hamming_chunk
                    # # Save incrementally
                    # start_idx = data_idx * const.MAX_NUM_CHUNKS
                    # end_idx = start_idx + data_chunk.shape[0]
                    # h5_file["hamming_distances"][:, start_idx:end_idx, :] = hamming_chunk

                finally:
                    shared_hamming_mem.close()
                    shared_hamming_mem.unlink()
                    gc.collect()

        finally:
            # Release shared memory for the lookup table
            shared_lookup_mem.close()
            shared_lookup_mem.unlink()
            # h5_file.close()
        print("Hamming distances computation completed.")
        # Combine results from all chunks
        hamming_distances = np.concatenate(hamming_results, axis=2) #axis=1)
        return hamming_distances
        # print("Hamming distances computation and caching completed.")
        # return self.cache_manager.load_cache(self.code_length, self.select_threshold, self.readouts.chip_id)

# Example usage:
if __name__ == "__main__":
    # Define the enroll_select_threshold value we want
    all_files = get_files()
    parameters = [(15, 1, 14)] #, (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]
    # parameters = [(7,1,6), (9,1,8), (11, 1, 10), (11, 2, 9), (13, 1, 12), (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]
    all_readouts: list[ReadoutList] = [read_readouts(all_files['L45'])]
    # all_readouts: list[ReadoutList] = [read_readouts(all_files[chip_id])
    #                                    for chip_id in all_files.keys()]
    coeff = [0,0]
    for n, coeff[0], coeff[1] in parameters:
        print("\n\nn =",n,"sigma=",coeff[1])
        for readouts_val in all_readouts:
            print("---------------------------------")
            # dir_name = enroll_comparator_dir
            active_multi = True
            hamming_processor = HammingProcessor(n, readouts_val,
                                                 coeff, active_multi)
            hamming_distances_res = hamming_processor.compute_hamming_distances(
                    0, 1000, None)
            print("hamming distance result:", hamming_distances_res)
            
