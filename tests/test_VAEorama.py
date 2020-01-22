import PanoramAI as PA
import numpy as np
import numpy.testing as npt
import pytest

def test_VAEorama():
    #Smoke test
    V = PA.VAEorama()
