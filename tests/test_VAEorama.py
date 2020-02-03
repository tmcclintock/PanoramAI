import PanoramAI as PA
import numpy as np
import numpy.testing as npt
import pytest

def test_VAEorama_smoketest():
    M, N = 8, 8
    data = np.random.randn(128, M, N, 3)
    V = PA.VAEorama(data)

def test_attributes():
    M, N = 16, 128
    LD = 100
    BS = 25
    data = np.random.randn(128, M, N, 3)
    V = PA.VAEorama(data, latent_dim = LD, BATCH_SIZE = BS)
    npt.assert_equal(V.dimensions, [M, N])
    npt.assert_equal(V.latent_dim, LD)
    npt.assert_equal(V.BATCH_SIZE, BS)

def test_training():
    M, N = 16, 128
    LD = 100
    BS = 25
    data = np.random.randn(128, M, N, 3)
    V = PA.VAEorama(data, latent_dim = LD, BATCH_SIZE = BS)
    V.train(1)

def test_sample():
    M, N = 16, 128
    LD = 100
    BS = 25
    data = np.random.randn(128, M, N, 3)
    V = PA.VAEorama(data, latent_dim = LD, BATCH_SIZE = BS)

    examples = V.generate_samples(1)[0]
    assert examples.shape == (M, N, 3)

def test_encode_decode():
    #Test shapes of the encode->decode process
    M, N = 16, 128
    LD = 100
    BS = 25
    data = np.random.randn(128, M, N, 3)
    V = PA.VAEorama(data, latent_dim = LD, BATCH_SIZE = BS)

    for x in V.train_dataset:
        mean, logvar = V.model.encode(x)
        npt.assert_equal(
            mean.get_shape().as_list(), (V.BATCH_SIZE, V.latent_dim,))
        npt.assert_equal(
            logvar.get_shape().as_list(), (V.BATCH_SIZE, V.latent_dim,))
        z = V.model.reparameterize(mean, logvar)
        npt.assert_equal(z.shape, mean.shape)

        xprime = V.model.decode(z)
        npt.assert_equal(x.shape, xprime.shape)
        break

def test_save_and_load():
    import os
    M, N = 16, 128
    LD = 100
    BS = 25
    data = np.random.randn(128, M, N, 3)
    V = PA.VAEorama(data, latent_dim = LD, BATCH_SIZE = BS)
    V.save_model_weights("temp/")
    V.load_model_weights("temp/")
    res = os.system("rm -rf temp")
    npt.assert_equal(res, 0)
    

if __name__ == "__main__":
    #test_VAEorama_smoketest()
    #test_attributes()
    test_training()
