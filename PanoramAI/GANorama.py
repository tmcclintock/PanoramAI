from .generic import GENERICorama

import numpy as np
import time

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.models import Model, Sequential

class GANorama(GENERICorama):
    """(Deep convolutional) Generative adversarial network
    used to learn and genreate panoramic images.
    """
    def __init__(self, dataset,
                 BATCH_SIZE = 64, test_size = 0.25,
                 latent_dim = 100):
        super().__init__(dataset, BATCH_SIZE, test_size, latent_dim)

    def reset_optimizer(self):
        """Reset the optimizers attached to the generator and descriminator.

        """
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        return


    def generate_samples(self, n_samples):
        return self.generator(tf.random.normal(shape = [n_samples,
                                                        self.latent_dim]))

    def train(self, epochs):
        raise Exception("Not implemented yet")

    def create_model(self):
        M, N = self.dimensions
        latent_dim = self.latent_dim

        #Make the generator
        generator = tf.keras.Sequential()
        generator.add(tfkl.Dense(M * N * 4, use_bias=False, input_shape=(latent_dim,)))
        generator.add(tfkl.BatchNormalization())
        generator.add(tfkl.LeakyReLU())
        
        generator.add(tfkl.Reshape((M//4, N//4, 64)))
        generator.add(tfkl.Conv2DTranspose(32, (5, 5), strides=(1, 1),
                                           padding='same', use_bias=False))
        generator.add(tfkl.BatchNormalization())
        generator.add(tfkl.LeakyReLU())
        
        generator.add(tfkl.Conv2DTranspose(16, (5, 5), strides=(2, 2),
                                           padding='same', use_bias=False))
        generator.add(tfkl.BatchNormalization())
        generator.add(tfkl.LeakyReLU())
        
        generator.add(tfkl.Conv2DTranspose(3, (5, 5), strides=(2, 2),
                                           padding='same', use_bias=False,
                                           activation='sigmoid'))
        self.generator = generator

        #Make the discriminator
        discriminator = tf.keras.Sequential()
        discriminator.add(tfkl.Conv2D(64, (5, 5), strides=(2, 2), padding='valid',
                                input_shape=[M, N, 3]))
        discriminator.add(tfkl.LeakyReLU())
        discriminator.add(tfkl.Dropout(0.3))
        
        discriminator.add(tfkl.Conv2D(128, (5, 5), strides=(2, 2), padding='valid'))
        discriminator.add(tfkl.LeakyReLU())
        discriminator.add(tfkl.Dropout(0.3))
        
        discriminator.add(tfkl.Flatten())
        discriminator.add(tfkl.Dense(1))
        self.discriminator = discriminator

        #Design the losses
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        def discriminator_loss(real_output, fake_output):
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss

        def generator_loss(fake_output):
            return cross_entropy(tf.ones_like(fake_output), fake_output)
