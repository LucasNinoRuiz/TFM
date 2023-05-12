import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.losses import MeanSquaredError
from keras.optimizers import Adam


class VAE(tf.keras.Model):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config
        self.encoder, self.z_mean, self.z_log_var = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = self.build_vae()

    def build_encoder(self):
        """
        Construye la arquitectura del codificador del VAE.
        """
        # Definir las dimensiones de entrada, las dimensiones del espacio latente y las dimensiones intermedias
        input_dim = self.config["input_dim"]
        latent_dim = self.config["latent_dim"]
        intermediate_dim = self.config["intermediate_dim"]

        # Construir el codificador
        encoder_input = Input(shape=(input_dim,), name="encoder_input")
        x = Dense(intermediate_dim, activation="relu", name="encoder_hidden")(encoder_input)

        # Definir las capas para la media y la log-varianza del espacio latente
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)

        # Definir una función Lambda para muestrear puntos en el espacio latente
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

        return Model(encoder_input, [z_mean, z_log_var, z], name="encoder"), z_mean, z_log_var

    def build_decoder(self):
        """
        Construye la arquitectura del decodificador del VAE.
        """
        # Definir las dimensiones del espacio latente y las dimensiones intermedias
        latent_dim = self.config["latent_dim"]
        intermediate_dim = self.config["intermediate_dim"]
        output_dim = self.config["output_dim"]

        # Construir el decodificador
        decoder_input = Input(shape=(latent_dim,), name="decoder_input")
        x = Dense(intermediate_dim, activation="relu", name="decoder_hidden")(decoder_input)
        decoder_output = Dense(output_dim, activation="sigmoid", name="decoder_output")(x)

        return Model(decoder_input, decoder_output, name="decoder")

    def encode(self, inputs):
        z_mean, z_log_var, _ = self.encoder(inputs)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.config["latent_dim"]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, inputs):
        return self.decoder(inputs)

    def build_vae(self):
        """
        Construye el modelo VAE completo a partir del codificador y el decodificador.
        """
        inputs = Input(shape=(self.config["input_dim"],))
        z_mean, z_log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)
        return Model(inputs, outputs, name="vae")

    def vae_loss(self, x, x_decoded):
        """
        Calcula la pérdida de reconstrucción y la divergencia KL para el VAE.
        """
        reconstruction_loss = MeanSquaredError()(x, x_decoded)
        kl_loss = -0.5 * tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=-1)
        return reconstruction_loss + kl_loss

    def train(self, train_data, epochs=100, batch_size=32):
        """
        Entrena el modelo VAE con los datos de entrenamiento.
        """
        optimizer = Adam(learning_rate=self.config["learning_rate"])
        self.vae.compile(optimizer=optimizer, loss=self.vae_loss)

        history = self.vae.fit(
            train_data,
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=self.config["validation_split"],
        )
        return history

    def generate_sample(self, num_samples=1):
        """
        Genera una muestra a partir del espacio latente utilizando el decodificador.
        """
        z_sample = tf.random.normal(shape=(num_samples, self.config["latent_dim"]))
        generated_sample = self.decode(z_sample)
        return generated_sample.numpy()
    
    def generate_image():
        pass
