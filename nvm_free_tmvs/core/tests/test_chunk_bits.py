import pathlib
import numpy as np
from common.data_reading_utils import ReadoutList, get_files, read_readouts


from nvm_free_tmvs.core.chunk_data_processor import ChunkDataProcessor

# directory to store sram readouts data
readouts_dir = pathlib.Path(__file__).parent.parent.parent.absolute() / 'data' /'SRAM_readouts'

# filepath to store generated codebooks
codebooks_dir = pathlib.Path(__file__).parent.parent.parent.absolute() / 'data' / 'codebooks.json'


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
        # self.data = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        self.data = np.frombuffer(data, dtype=np.uint8)
        # self.data = data

def hamming_distance(bit_arr1: np.ndarray, bit_arr2: np.ndarray) -> int:
    """
    Calculate the Hamming distance between two bit arrays
    """
    return np.sum(bit_arr1 ^ bit_arr2)


def assign_optimal_data_type (bits_num: int):
    """TODO_summary_

    Args:
        bits_num (int): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
# Assign an optimal data type for the SRAM data based on the bits number
    if bits_num <= 8:
        return np.uint8
    if bits_num <= 16:
        return np.uint16
    if bits_num <= 32:
        return np.uint32
    return np.uint64

def optimized_chunk_bits(bits_num: int, data: bytes):
    """
    Chunk bytes into integers of size `bits_num` bits using the optimal data type.

    Args:
        data (bytes): Input data in bytes.

    Returns:
        numpy.ndarray: Array of integers with the appropriate data type.
    """

    # Convert bytes to a NumPy array of uint8
    data = np.frombuffer(data, dtype=np.uint8)

    # Determine the optimal data type for the chunks
    sram_dtype = assign_optimal_data_type(bits_num)
    dtype_bits = np.dtype(sram_dtype).itemsize * 8  # Bits in the target dtype

    # Total bits in the input data
    total_bits = len(data) * 8

    # Calculate the number of bits to keep (discard the remainder)
    bits_to_keep = total_bits - (total_bits % bits_num)

    # Convert uint8 data into a flat array of bits
    bit_array = np.unpackbits(data, bitorder='big')

    # Trim the bit array to the largest multiple of `bits_num`
    bit_array = bit_array[:bits_to_keep]

    # Reshape the bit array into rows of `bits_num` bits
    bit_array = bit_array.reshape(-1, bits_num)

    # Convert each row of bits into integers using NumPy's dot product
    powers = 2 ** np.arange(bits_num - 1, -1, -1, dtype=np.uint64)  # Powers of 2
    chunked_data = np.dot(bit_array, powers).astype(sram_dtype)  # Convert to target dtype
    

    return chunked_data


def non_optimized_chunk_bits(bits_num: int, data: bytes):
    """
    TODO
    Chunk bytes into given number of bits "bits_num"
    """ 
    # Convert bytes to numpy data with type uint8
    data = np.frombuffer(data, dtype=np.uint8)

    # The numpy data type of the elements in the reference SRAM data
    element_dtype = data.dtype.type

    # Calculate the number of bits in the numpy type used for reference SRAM data
    reference_chunk_len = data.dtype.itemsize * 8

    # Assign an optimal data type for the SRAM data based on the code length
    sram_dtype = assign_optimal_data_type(bits_num)

    # Initialize the current SRAM pattern with a zero of the assigned data type
    sram_pattern = sram_dtype(0)

    # Carry tracks how many bits are already added from a partial chunk
    carry = 0

    # Initialize a list to hold the newly formatted SRAM data
    sram_data = []

    # Index to track position within the `data` 
    i = 0

    # Length of the `data` array
    data_len = len(data)

    # Iterate through the bits in the data
    while i < data_len:

        # Determine how many bits are needed to complete a chunk
        bits_to_collect = bits_num - carry
        if bits_to_collect < 0:
            raise Exception("Sorry, chunk_bits does not handle such bits_num values") 

        # Calculate how many full chunks of `reference_chunk_len` bits can fit
        entire_chunks_num = bits_to_collect // reference_chunk_len # quotient
 
        # Calculate the remaining bits after the full chunks
        partial_chunks_num = bits_to_collect % reference_chunk_len # remainder

        # If there aren't enough bits remaining in the data, break the loop
        if i + entire_chunks_num > data_len:
            break
            
        # Shift the current pattern to make space for the new chunks
        sram_pattern = sram_pattern << sram_dtype(reference_chunk_len * entire_chunks_num)
            
        # Add the bits from the full chunks to the SRAM pattern
        sram_pattern |= sram_dtype(int.from_bytes(data[i:i+entire_chunks_num]))

        # Move the index forward by the number of full chunks added
        i += entire_chunks_num

        # Handle the remaining partial chunk, if applicable
        if i >= data_len:
            if not partial_chunks_num:
                # Append the completed SRAM pattern to the output list
                sram_data.append(sram_dtype(sram_pattern))
            break

        # Create a mask to extract the relevant bits for the partial chunk
        # Equivalent to: 2^reference_chunk_len-1-(2^(reference_chunk_len-partial_chunks_num)-1)
        mask = (1<<reference_chunk_len) - (1<<(reference_chunk_len-partial_chunks_num))

        # Shift the current pattern to make space for the partial chunk
        sram_pattern = sram_pattern << sram_dtype(partial_chunks_num)

        # Extract the relevant bits from the current position in the data
        shifted_data = sram_dtype((mask & data[i]) >> (reference_chunk_len-partial_chunks_num))

        # Add the partial chunk to the current SRAM pattern
        sram_pattern |= shifted_data

        # Append the completed SRAM pattern to the output list
        sram_data.append(sram_dtype(sram_pattern))

        # Prepare a new SRAM pattern for the next iteration, removing used bits
        sram_pattern = (~element_dtype(mask)) & data[i]

        # Update the carry to reflect the leftover bits from the partial chunk
        carry = reference_chunk_len - partial_chunks_num

        i += 1

        # If the carry completes a chunk, add the pattern
        if carry == bits_num:
            sram_data.append(sram_dtype(sram_pattern))
            carry = 0
            sram_pattern = 0

    # Convert the list of SRAM data into a numpy array with the assigned data type
    # return np.array(sram_data, dtype=sram_dtype)
    return sram_data

def chunk_readouts_traditionally (bits_num: int, data: bytes):
    data = np.unpackbits(data)
    chunked_data = []
    num_sram_patterns = len(data) // bits_num
    for sram_pattern_idx in range(num_sram_patterns):
        start = sram_pattern_idx * bits_num
        end = start + bits_num
        sram_pattern = data[start:end]
        # convert bits to integers
        # Reshape the array into chunks of size n
        reshaped = sram_pattern.reshape(-1, bits_num)
        # Convert each chunk to an integer
        integers = np.array([int("".join(map(str, chunk)), 2) for chunk in reshaped])[0]
        chunked_data.append(integers)
    return chunked_data

def test_chunk_readouts (bits_num: int, readouts: ReadoutList, data_processor: ChunkDataProcessor):
    """_summary_
    TODO
    Args:
        readouts (ReadoutList): _description_

    Returns:
        _type_: _description_
    """
    # chunked_bits_1 = optimized_chunk_bits(bits_num, readouts[0].data)
    chunked_bits_1 = data_processor.chunk_bits(readouts[0].data)
    # chunked_bits_1 = non_optimized_chunk_bits(bits_num, readouts[0].data)
    print("first sram pattern (optimized): ",chunked_bits_1[0])
    chunked_bits_2 = chunk_readouts_traditionally(bits_num, readouts[0].data)
    print("first sram pattern (traditional): ",chunked_bits_2[0])

    # print(len(chunked_bits_1))
    # print(len(chunked_bits_2))

    # Set a flag to True to assume lists are identical initially
    flag = True
    count = 0
    # Check if lengths of both lists are equal
    if len(chunked_bits_1) != len(chunked_bits_2):
        print(len(chunked_bits_1))
        print(len(chunked_bits_2))
        # If lengths differ, set flag to False
        flag = False
    else:
        # Iterate over each element in lists
        for i,_ in enumerate(chunked_bits_1):
            # Compare corresponding elements of both lists
            if chunked_bits_1[i] != chunked_bits_2[i]:

                # If any elements differ, set flag to False
                flag = False
                count += 1
                # print(flag,"at position: ",i)
                # print(chunked_bits_1[i])
                # print(chunked_bits_2[i])

                # Exit the loop since we found a difference

    print(flag, "count=",count)
        
    return 

if __name__ == "__main__":
    all_files = get_files()
    # n_values = [11,11,13,,]
    
    # n = 11
    # coeff = [1,3.5]
    # parameters = [(7,1,6)]
    parameters = [(7,1,6), (9,1,8), (11, 1, 10), (11, 2, 9), (13, 1, 12), (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]
    # parameters = [(7,1,6), (11, 1, 10), (11, 2, 9),(13, 1, 12), (13, 2, 11), (15, 1, 14), (17, 1, 16), (27, 3, 24), (29, 4, 25), (31, 5, 26), (33, 5, 28),(35, 6, 29), (37, 7, 30),(39, 8, 31),(41, 6, 35),(45, 10, 35),(47, 8, 39)]

    # print("n =",n,"sigma=",coeff[1])

    # all_readouts: list[ReadoutList] = [read_readouts(all_files[chip_id])
    #                                    for chip_id in all_files.keys()]

    all_readouts: list[ReadoutList] = [read_readouts(all_files['M22'])]
    # data = np.array([r.data for r in all_readouts[0]])
    # # data = np.array([transform_to_integers(r.data, n) for r in all_readouts[0]])
    # print(data)
    # print(f"Shape of data: {data.shape}")
    # print(f"Data type: {data.dtype}")
    # print(f"Size in memory (bytes): {data.nbytes}")

    # average_ber = 0
    coeff = [0,0]
    for n, coeff[0], coeff[1] in parameters:
        print("n =",n,"sigma=",coeff[1])
        for readouts in all_readouts:
            # print("\nChip ",readouts.chip_id)
            data_processor = ChunkDataProcessor(n, readouts, active_multithreading=True)
            # start_time = time.time()
            # chunked_data = data_processor.chunk_readouts()
            # end_time = time.time()
            test_chunk_readouts(n, readouts, data_processor)