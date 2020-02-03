import PanoramAI as PA
import numpy as np
import numpy.testing as npt
import pytest

def test_GENERICorama():
    #Smoke test
    data = np.random.randn(10, 4, 8, 3)
    o = PA.GENERICorama(data)

def test_save_model():
    data = np.random.randn(10, 4, 8, 3)
    o = PA.GENERICorama(data)
    o.save_model(0, 0, 0, 0)
    import os
    res = os.system("rm -rf saved_models")
    npt.assert_equal(res, 0)
    
if __name__ == "__main__":
    test_GENERICorama()
