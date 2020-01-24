import numpy as np
import time
import tensorflow as tf

class VAEorama(object):
    def __init__(self, M = 16, N = 128,
                 latent_dimension = 200,
                 optimizer = None,
                 n_samples_to_generate = 16,
                 train_dataset = None,
                 test_dataset = None):
        assert M % 4 == 0
        assert N % 4 == 0

        self.M, self.N = M, N
        self.latent_dimension = latent_dimension
        self.optimizer = optimizer
        self.create_CVAE(M, N, latent_dimension)
        self.CVAE = _CVAE(M, N, latent_dimension)
        if not optimizer:
            self.reset_optimizer()
        self.TOTAL_EPOCHS = 0

        if train_dataset is not None:
            self.set_train_dataset(train_dataset)
        if test_dataset is not None:
            self.set_test_dataset(test_dataset)
            
        self._generate_random_vector(n_samples_to_generate)

    def set_train_dataset(self, train_dataset):
        assert self.M == len(train_dataset[0])
        assert self.N == len(train_dataset[0][0])
        assert 3 == len(train_dataset[0][0][0])
        self.train_dataset = train_dataset
        return

    def set_test_dataset(self, test_dataset):
        assert self.M == len(test_dataset[0])
        assert self.N == len(test_dataset[0][0])
        assert 3 == len(test_dataset[0][0][0])
        self.test_dataset = test_dataset
        return

    def _generate_random_vector(self, n_samples):
        self.n_samples_to_generate = n_samples
        self.random_vector_for_generation = tf.random.normal(
            shape=[n_samples, self.latent_dimension])
        return

    def generate_samples(self, n_samples):
        if n_samples > self.n_samples_to_generate:
            print("Regenerating sample vector.")
            self._generate_random_vector(n_samples)
        return self.CVAE.sample(self.random_vector_for_generation)

    def reset_optimizer(self, opt = tf.keras.optimizers.Adam):
        self.optimizer = opt(1e-4)
        return

    def create_CVAE(self, M, N, latent_dimension):
        self.CVAE = _CVAE(M, N, latent_dimension)
        return

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    @tf.function
    def compute_loss(self, x):
        mean, logvar = self.CVAE.encode(x)
        z = self.CVAE.reparameterize(mean, logvar)
        x_predicted = self.CVAE.decode(z)
        MSE = tf.losses.MSE(x, x_predicted)
        logpx_z = -tf.reduce_sum(MSE)
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def compute_apply_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
            gradients = tape.gradient(loss, self.CVAE.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.CVAE.trainable_variables))

    def train(self, epochs, steps_for_update = None):
        if not steps_for_update:
            steps_for_update == epochs // 10
        update_length = epochs//10

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            for train_x in [self.train_dataset]:
                self.compute_apply_gradients(train_x)
    
            if epoch % update_length == 0:
                end_time = time.time()
                loss = tf.keras.metrics.Mean()
                for test_x in self.test_dataset:
                    loss(self.compute_loss(tf.expand_dims(test_x, 0)))
                elbo = -loss.result()
        
                print(f'Epoch: {epoch}, Test set ELBO: {elbo:.4f}, '
                      f'time elapsed for current epoch batch {end_time - start_time:.4f}')
                start_time = time.time()
        self.TOTAL_EPOCHS += epochs
        print(f"Total epochs: {self.TOTAL_EPOCHS}")
        return

class _CVAE(tf.keras.Model):
    """A convolutional variational autoencoder used to
    create panoramic images.

    Note: we assume there are 3 input (RGB) channels.

    """
    def __init__(self, M, N, latent_dimension):
        super(_CVAE, self).__init__()        
        self.input_dimensions = [M, N]
        self.latent_dimension = latent_dimension
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(M, N, 3)), #(bs, M, N, 3)
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2),
                    activation='relu', padding="valid"), #(bs, M/2, N/2, 32)
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), #(bs, M/4, N/4, 64)
                    activation='relu', padding="valid"),
                tf.keras.layers.Flatten(), #(bs, (M/4) * (N/4) * 64)
                #predicting mean and logvar
                tf.keras.layers.Dense(latent_dimension + latent_dimension), # (bs, D * D)
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dimension,)), #(bs, D)
                tf.keras.layers.Dense(units= M * N * 4, activation=tf.nn.relu), #M * N * 4 = (M/4)*(N/4)*64
                tf.keras.layers.Reshape(target_shape=(M//4, N//4, 64)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=(2, 2),
                    padding="SAME", activation='relu'), #(bs, M/2, N/2, 64)
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=(2, 2),
                    padding="SAME", activation='relu'), #(bs, M, N, 32)
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=3, strides=(1, 1), padding="SAME",
                    activation='sigmoid'), #(bs, M, N, 3)
            ]
        )
        
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dimension))
        return self.decode(eps)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), 
                                num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        return self.generative_net(z)