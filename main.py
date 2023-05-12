import argparse
from utils.data_loader import load_data
from models.vae.standard_vae.vae_model import VAE
from models.vae.standard_vae.vae_trainer import VAETrainer
from models.gan.standard_gan.gan_model import GAN
from models.gan.standard_gan.gan_trainer import GANTrainer
from utils.visualization import visualize_results
from utils.data_preprocessing import clean_data, normalize_data, split_data

def main(args):
    # Cargar datos
    raw_data = load_data(args.data_file)

    # Construir y entrenar modelos (también se realiza el preprocesado de datos)
    if args.model_type == "VAE":
        model = VAE(args.config_file)
        trainer = VAETrainer(model)
        trainer.prepare_data(raw_data)
        trainer.train(args.epochs, args.batch_size)
        trainer.evaluate()

        # Guardar el modelo
        # trainer.save_model(args.models_folder)

        # Cargar el modelo
        # trainer.load_model(args.models_folder)
    elif args.model_type == "GAN":
        model = GAN(args.config_file)
        trainer = GANTrainer(model)
        trainer.prepare_data(raw_data)
        trainer.train(args.epochs, args.batch_size)
        trainer.evaluate()

        # Guardar el modelo
        # trainer.save_model(args.models_folder)

        # Cargar el modelo
        # trainer.load_model(args.models_folder)
    else:
        raise ValueError("Modelo no soportado")

    # Visualizar resultados
    visualize_results(model, args.results_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proyecto de fin de máster en Big Data y Data Science")
    parser.add_argument("--data_file", default="data/raw_data/materials_data.json", help="Ruta del archivo de datos")
    parser.add_argument("--model_type", default="VAE", choices=["VAE", "GAN"], help="Tipo de modelo generativo (VAE o GAN)")

    args = parser.parse_args()

    # Seleccionar archivo de configuración y carpeta de resultados según el tipo de modelo
    if args.model_type == "VAE":
        default_config_file = "config/vae_config.json"
        default_results_folder = "results/vae"
    elif args.model_type == "GAN":
        default_config_file = "config/gan_config.json"
        default_results_folder = "results/gan"

    parser.add_argument("--config_file", default=default_config_file, help="Ruta del archivo de configuración del modelo")
    parser.add_argument("--results_folder", default=default_results_folder, help="Carpeta para guardar los resultados")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas para entrenar el modelo")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño del lote para el entrenamiento")
    parser.add_argument("--models_folder", default="models", help="Carpeta para guardar y cargar los modelos")

    args = parser.parse_known_args()[0]
    main(args)
