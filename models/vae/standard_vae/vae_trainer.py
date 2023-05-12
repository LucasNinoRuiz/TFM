import os
from utils import data_preprocessing
from utils.data_preprocessing import clean_data, normalize_data, split_data
from utils.data_loader import load_data
from models.vae.standard_vae.vae_model import VAE

class VAETrainer:
    def __init__(self, config):
        """ 
        Inicializa el entrenador VAE con la configuración proporcionada.
        """
        # Args: config (dict): Configuración para el entrenador y el modelo VAE.
        self.config = config

    def prepare_data(self, raw_data):
    # def prepare_data(self, data_file, features):
        # Cargar datos
        # raw_data = load_data(data_file, features)

        # Preprocesar datos
        cleaned_data = clean_data(raw_data)
        normalized_data = normalize_data(cleaned_data)
        self.train_data, self.val_data = split_data(normalized_data)

    '''def prepare_data(self, data):
        """
        Prepara los datos para el entrenamiento y la validación del VAE.

        Args:
            data (pd.DataFrame): Conjunto de datos sin procesar.
        """
        cleaned_data = clean_data(data)
        normalized_data, self.scaler = normalize_data(cleaned_data)
        self.train_data, self.val_data = split_data(normalized_data)'''

    def build_model(self):
        # Construye el modelo VAE utilizando la configuración proporcionada.
        self.model = VAE(self.config)

    def save_model(self, save_dir):
        """
        Guarda el modelo VAE en el directorio especificado.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_weights(os.path.join(save_dir, 'vae_weights.h5'))

    def load_model(self, load_dir):
        """
        Carga el modelo VAE desde el directorio especificado.
        """
        self.model.load_weights(os.path.join(load_dir, 'vae_weights.h5'))
    
    def train(self, epochs, batch_size):
        """
        Entrena el modelo VAE con los datos de entrenamiento.
        """
        history = self.model.train(self.train_data, epochs, batch_size)

        # Return: history (tf.keras.callbacks.History): Objeto de historial de entrenamiento.
        return history

    def evaluate(self):
        """
        Evalúa el modelo VAE utilizando los datos de validación.
        """

        # Esta evaluación es más sencilla que en el GAN, 
        # ya que está basada en la pérdida de reconstrucción.
        val_loss = self.model.evaluate(self.val_data, self.val_data)
        print(f"Validation loss: {val_loss}")
