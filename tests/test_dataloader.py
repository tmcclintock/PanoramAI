import PanoramAI as PA
import numpy as np
import numpy.testing as npt
import pytest

#Only run this test locally
def LOCAL_ONLY_test_DataLoader():
    data = PA.load_data()
    M = 32
    N = 256
    Npics = 10000
    npt.assert_equal(data.shape, (Npics, M, N, 3))

#if __name__ == "__main__":
#    local_test_DataLoader()
