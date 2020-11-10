""" Script testing the np_utils """
import os
import numpy as np
from sertit_utils.core import np_utils


def test_bit():
    """ Test bit arrays """
    np_ones = np.ones((1, 2, 2), dtype=np.uint16)
    ones = np_utils.read_bit_array(np_ones, bit_id=0)
    zeros = np_utils.read_bit_array(np_ones, bit_id=list(np.arange(1, 15)))
    assert (np_ones == ones).all()
    for arr in zeros:
        assert (np_ones == 1 + arr).all()
