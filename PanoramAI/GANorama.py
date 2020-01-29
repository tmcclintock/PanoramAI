from .generic import GENERICorama

import numpy as np
import time, os

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.models import Model, Sequential

#To save space
_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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
        return self.generator(
            tf.random.normal(shape = [n_samples, self.latent_dim]))

    def discriminator_loss(self, real_output, fake_output):
        real_loss = _cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = _cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def generator_loss(self, fake_output):
        return _cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def _train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss

    def train(self, epochs, steps_for_update = None, make_checkpoints = False):
        if not steps_for_update:
            steps_for_update = max(1, epochs // 10)

        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)
        
        for epoch in range(1, epochs+1):
            start = time.time()
            for train_x in self.train_dataset:
                gen_loss, disc_loss = self._train_step(train_x)

            # Save the model every 15 epochs
            if epoch % steps_for_update == 0:
                if make_checkpoints:
                    checkpoint.save(file_prefix = checkpoint_prefix)
            print(f"Time for epoch {epoch+1} "
                  f" is {time.time() - start:.4f} sec -- "
                  f"Gen loss: {gen_loss:.4f}   Disc loss: {disc_loss:.4f} per batch")
            #.format(epoch + 1, time.time()-start))
        return

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
