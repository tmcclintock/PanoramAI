import PanoramAI as PA
import numpy as np
import numpy.testing as npt
import pytest

def test_VAEorama():
    #Smoke test
    V = PA.VAEorama()

def test_training():
    M, N = 16, 128
    data = np.random.randn(100, M, N, 3)
    V = PA.VAEorama(
        M, N, train_dataset=data,
        test_dataset=data, BATCH_SIZE=25)
    V.train(10)

if __name__ == "__main__":
    test_training()
