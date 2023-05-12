import json
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm
from models.gan.standard_gan.gan_model import GAN
from utils.data_loader import load_data
from utils.data_preprocessing import clean_data, normalize_data, split_data

class GANTrainer:
    def __init__(self, config):
        # Inicializa el entrenador GAN con la configuración proporcionada.
        # Args: config (dict): Configuración para el entrenador y el modelo GAN.
        self.config = config

    def prepare_data(self, data_file, features):
        # Cargar datos
        raw_data = load_data(data_file, features)

        # Preprocesar datos
        cleaned_data = clean_data(raw_data)
        normalized_data = normalize_data(cleaned_data)
        self.train_data, self.val_data = split_data(normalized_data)

    def build_model(self):
        # Construye el modelo GAN utilizando la configuración proporcionada.
        self.model = GAN(self.config)
        
        # También podría hacerse así para proporcionar una ruta:
        # self.model = GAN(config_file=self.config["gan_config_file"])

    def save_model(self, models_dir):
        """
        Guarda el modelo GAN (Generador y Discriminador) en un directorio.
        """
        os.makedirs(models_dir, exist_ok=True)

        self.model.generator.save(os.path.join(models_dir, 'generator.h5'))
        self.model.discriminator.save(os.path.join(models_dir, 'discriminator.h5'))

        with open(os.path.join(models_dir, 'config.json'), 'w') as f:
            json.dump(self.model.config, f)

    def load_model(self, models_dir):
        """
        Carga el modelo GAN (Generador y Discriminador) desde un directorio.
        """
        generator_path = os.path.join(models_dir, 'generator.h5')
        discriminator_path = os.path.join(models_dir, 'discriminator.h5')

        if not os.path.exists(generator_path) or not os.path.exists(discriminator_path):
            raise ValueError("Los archivos de modelo no se encuentran en el directorio proporcionado.")

        self.model.generator = load_model(generator_path)
        self.model.discriminator = load_model(discriminator_path)

        with open(os.path.join(models_dir, 'config.json'), 'r') as f:
            self.model.config = json.load(f)

    def train(self, epochs, batch_size):
        # Entrenar el modelo GAN
        history = self.model.train(self.train_data, epochs, batch_size)
        return history

    def calculate_fid(self, act1, act2):
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def evaluate(self, num_samples=10000):
        # Crear un modelo InceptionV3 para extraer características
        inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

        # Redimensionar y preprocesar imágenes de validación para InceptionV3
        val_data_resized = tf.image.resize(self.val_data, (299, 299))
        val_data_preprocessed = preprocess_input(val_data_resized)

        # Generar imágenes sintéticas utilizando el generador GAN
        generated_images = self.model.generate_sample(num_samples)

        # Redimensionar y preprocesar imágenes generadas para InceptionV3
        generated_images_resized = tf.image.resize(generated_images, (299, 299))
        generated_images_preprocessed = preprocess_input(generated_images_resized)

        # Extraer características de las imágenes de validación y las imágenes generadas
        act_val = inception_model.predict(val_data_preprocessed)
        act_gen = inception_model.predict(generated_images_preprocessed)

        # Calcular la métrica FID entre las imágenes de validación y las imágenes generadas
        fid = self.calculate_fid(act_val, act_gen)
        return fid

    def save_results(self, image_save_path, text_save_path, num_samples=100, results=None):
        """
        Guarda imágenes generadas y métricas de evaluación (si se proporcionan) en archivos.

        Args:
        gan: el modelo GAN entrenado.
        image_save_path: la ruta donde se guardarán las imágenes generadas.
        text_save_path: la ruta donde se guardará el archivo de texto con los resultados de la evaluación.
        num_samples: el número de imágenes generadas que se guardarán (default: 100).
        results: un diccionario que contiene los resultados de la evaluación (default: None).
        """

        # Generar imágenes utilizando el modelo GAN
        generated_images = self.model.generate_sample(num_samples)

        # Crear la carpeta para guardar las imágenes si no existe
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)

        # Guardar imágenes generadas como archivos .png
        for i in range(num_samples):
            image = (generated_images[i] * 255).astype(np.uint8)
            image_pil = Image.fromarray(image)
            image_pil.save(os.path.join(image_save_path, f"generated_image_{i}.png"))

        # Guardar resultados de evaluación en un archivo de texto si se proporcionan
        if results is not None:
            with open(text_save_path, "w") as f:
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")