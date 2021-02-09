""" Numpy tools """
from typing import Union

import numpy as np


def unpackbits(array: np.ndarray, nof_bits: int) -> np.ndarray:
    """
    Function found here:
    https://stackoverflow.com/questions/18296035/how-to-extract-the-bits-of-larger-numeric-numpy-data-types
    Args:
        array (np.ndarray): Array to unpack
        nof_bits (int): Number of bits to unpack

    Returns:
        np.ndarray: Unpacked array
    """
    xshape = list(array.shape)
    array = array.reshape([-1, 1])
    mask = 2 ** np.arange(nof_bits).reshape([1, nof_bits])
    return (array & mask).astype(bool).astype(np.uint8).reshape(xshape + [nof_bits])


def read_bit_array(bit_mask: np.ndarray, bit_id: Union[list, int]) -> Union[np.ndarray, list]:
    """
    Read bit arrays as a succession of binary masks (sort of read a slice of the bit mask, slice number bit_id)

    Args:
        bit_mask (np.ndarray): Bit array to read
        bit_id (int): Bit ID of the slice to be read
          Example: read the bit 0 of the mask as a cloud mask (Theia)

    Returns:
        Union[np.ndarray, list]: Binary mask or list of binary masks if a list of bit_id is given
    """
    # Get the number of bits
    nof_bits = 8 * bit_mask.dtype.itemsize

    # Read cloud mask as bits
    mask = unpackbits(bit_mask, nof_bits)

    # Only keep the bit number bit_id and reshape the vector
    if isinstance(bit_id, list):
        bit_arr = [mask[..., id] for id in bit_id]
    else:
        bit_arr = mask[..., bit_id]

    return bit_arr
