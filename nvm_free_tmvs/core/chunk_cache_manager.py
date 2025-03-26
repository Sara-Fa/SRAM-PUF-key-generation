""" Module for saving and loading the output of chunk_readouts function. """
import numpy as np
from nvm_free_tmvs.utils.file_manager import chunked_readouts_dir


class ChunkedDataManager:
    """
    Class for saving and loading the output of chunk_readouts function.
    """

    def __init__(self):
        """
        Initialize the ChunkedDataManager and ensure the directory exists.
        """
        chunked_readouts_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    @staticmethod
    def _generate_filename(chip_id: str, chunk_length: int) -> str:
        """
        Generate a filename based on chip ID and chunk length.

        :param chip_id: ID of the chip.
        :param chunk_length: Length of the chunk.
        :return: Generated filename.
        """
        return f"chip_ID_{chip_id}_chunklen_{chunk_length}"

    def save_chunked_data(self, chip_id: str, chunk_length: int, chunked_data: np.ndarray):
        """
        Save chunked data to a compressed .npz file. Checks if the file already exists.

        :param chip_id: Chip ID.
        :param chunk_length: Length of the chunk.
        :param chunked_data: A 2D numpy array representing the chunked data.
        """
        filename = self._generate_filename(chip_id, chunk_length)
        file_path = chunked_readouts_dir / f"{filename}.npz"
         # Check if file exists
        if file_path.exists():
            print(f"Warning: File {file_path} already exists. Skipping save.")
        else:
            np.savez_compressed(file_path, chunked_data=chunked_data)
            print(f"Chunked data saved to {file_path}")

    def load_chunked_data(self, chip_id: str, chunk_length: int) -> np.ndarray:
        """
        Load chunked data from a compressed .npz file.

        :param chip_id: Chip ID.
        :param chunk_length: Length of the chunk.
        :return: A 2D numpy array representing the chunked data.
        """
        filename = self._generate_filename(chip_id, chunk_length)
        file_path = chunked_readouts_dir / f"{filename}.npz"
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with np.load(file_path) as data:
            chunked_data = data["chunked_data"]
        print(f"Chunked data loaded from {file_path}")
        return chunked_data
