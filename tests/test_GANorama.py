import PanoramAI as PA
import numpy as np
import numpy.testing as npt
import pytest

def test_GANorama():
    #Smoke test
    M, N = 16, 128
    LD = 100
    BS = 25
    data = np.random.randn(128, M, N, 3)

    G = PA.GANorama(data)

def test_attributes():
    M, N = 16, 128
    LD = 100
    BS = 25
    data = np.random.randn(128, M, N, 3)
    G = PA.GANorama(data, latent_dim = LD, BATCH_SIZE = BS)
    npt.assert_equal(G.dimensions, [M, N])
    npt.assert_equal(G.latent_dim, LD)
    npt.assert_equal(G.BATCH_SIZE, BS)

def test_training():
    M, N = 16, 128
    LD = 100
    BS = 25
    data = np.random.randn(1000, M, N, 3)
    G = PA.GANorama(data, latent_dim = LD, BATCH_SIZE = BS)
    G.train(1)

    #Test the loss directly
    for x in G.train_dataset:
        DL = G.discriminator_loss(x, x)
        GL = G.generator_loss(x)
        npt.assert_equal(DL.dtype, np.float64)
        npt.assert_equal(GL.dtype, np.float64)
        break
        
def test_sample():
    M, N = 16, 128
    LD = 100
    BS = 25
    data = np.random.randn(128, M, N, 3)
    G = PA.GANorama(data, latent_dim = LD, BATCH_SIZE = BS)

    examples = G.generate_samples(1)[0]
    npt.assert_equal(examples.shape, (M, N, 3))


if __name__ == "__main__":
    #test_GANorama()
    #test_attributes()
    test_training()
    test_sample()
