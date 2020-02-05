from .generic import GENERICorama

import numpy as np
import time, os

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.models import Model, Sequential

class CondtionalVAEorama(GENERICorama):
    """Convolutional VAE conditional on an input image.
    """
    def __init__(self, obs, dataset,
                 BATCH_SIZE = 64, test_size = 0.25,
                 latent_dim = 100):
        super().__init__(dataset, BATCH_SIZE, test_size, latent_dim)

        obs = np.asarray(obs)
        assert len(obs.shape) == 4
        assert obs.shape[-1] == 3 #3 channels
        self.obs_dimensions = obs.shape[1:3]
        self.obs_dimensions[0] % 4 == 0
        self.obs_dimensions[1] % 4 == 0

        self.obs_dimensions[0] >= 8 == 0
        self.obs_dimensions[1] >= 8 == 0

    def create_model(self):
        pass
