import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.losses import MeanSquaredError
from keras.optimizers import Adam

class GAN:
    def __init__(self, config):
        self.config = config
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        """
        Construye el generador del GAN.
        """
        generator = tf.keras.Sequential(name="generator")

        # Aquí se agregan las capas del generador
        generator.add(layers.Dense(self.config["generator_intermediate_dim"], input_shape=(self.config["latent_dim"],)))
        generator.add(layers.LeakyReLU(alpha=self.config["leaky_relu_alpha"]))
        
        generator.add(layers.Dense(self.config["generator_output_dim"]))
        generator.add(layers.Activation("tanh"))

        return generator

    def build_discriminator(self):
        """
        Construye el discriminador del GAN.
        """
        discriminator = tf.keras.Sequential(name="discriminator")

        # Aquí se agregan las capas del discriminador
        discriminator.add(layers.Dense(self.config["discriminator_intermediate_dim"], input_shape=(self.config["input_dim"],)))
        discriminator.add(layers.LeakyReLU(alpha=self.config["leaky_relu_alpha"]))
        discriminator.add(layers.Dropout(self.config["dropout_rate"]))
        
        discriminator.add(layers.Dense(1))
        discriminator.add(layers.Activation("sigmoid"))

        return discriminator

    def build_gan(self):
        """
        Construye el modelo GAN combinando el generador y el discriminador.
        """
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # El generador toma el ruido como entrada y genera muestras
        noise = Input(shape=(self.latent_dim,))
        sample = self.generator(noise)

        # Para el modelo combinado, solo entrenaremos el generador
        self.discriminator.trainable = False

        # El discriminador toma las muestras generadas como entrada y determina su validez
        validity = self.discriminator(sample)

        # El modelo combinado toma el ruido como entrada y devuelve la validez de las muestras generadas
        self.gan = Model(noise, validity)

    def compile_models(self):
        """
        Compila los modelos: el generador, el discriminador y el GAN.
        """
        # Compilar el discriminador
        discriminator_optimizer = Adam(lr=self.learning_rate, beta_1=self.beta_1)
        self.discriminator.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer, metrics=["accuracy"])

        # Compilar el modelo GAN
        gan_optimizer = Adam(lr=self.learning_rate, beta_1=self.beta_1)
        self.gan.compile(loss="binary_crossentropy", optimizer=gan_optimizer)

    def train_step(self, real_data):
        """
        Realiza un paso de entrenamiento tanto para el generador como para el discriminador.
        """
        batch_size = real_data.shape[0]

        # Generar ruido aleatorio como entrada para el generador
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

        # Generar muestras falsas
        fake_data = self.generator.predict(noise)

        # Entrenar el discriminador
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = self.discriminator.train_on_batch(fake_data, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Entrenar el generador
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        g_loss = self.gan.train_on_batch(noise, real_labels)

        return d_loss, g_loss

    def train(self, train_data, epochs, batch_size=128, sample_interval=50):
        """
        Entrena el modelo GAN utilizando los datos de entrenamiento proporcionados.
        """
        train_data = np.array(train_data)
        num_batches = train_data.shape[0] // batch_size

        for epoch in range(epochs):
            for batch in range(num_batches):
                real_data = train_data[batch * batch_size : (batch + 1) * batch_size]
                d_loss, g_loss = self.train_step(real_data)

            # Imprimir las métricas de pérdida y guardar muestras generadas periódicamente
            print(f"Epoch {epoch + 1}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")

            if (epoch + 1) % sample_interval == 0:
                self.generate_sample()