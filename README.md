###### OUT-OF-DATE ######

# Proyecto de fin de máster en Big Data y Data Science

# Autor Lucas Niño Ruiz

Este proyecto tiene como objetivo generar imágenes sintéticas utilizando dos tipos de modelos generativos: Variational Autoencoders (VAE) y Generative Adversarial Networks (GAN).

## Estructura del proyecto

El proyecto se organiza en las siguientes carpetas y archivos:

- `app/`: Carpeta opcional que contiene una interfaz de usuario básica para visualizar los resultados.
- `config/`: Archivos de configuración para los modelos VAE y GAN.
- `data/`: Datos en bruto y preprocesados utilizados para entrenar y evaluar los modelos.
- `logs/`: Registros de entrenamiento para los modelos VAE y GAN.
- `models/`: Implementaciones de modelos VAE y GAN.
- `results/`: Resultados de entrenamiento, imágenes generadas y gráficas de pérdida.
- `utils/`: Funciones auxiliares para cargar datos, preprocesarlos y visualizar los resultados.
- `main.py`: Script principal que ejecuta el proyecto y entrena los modelos.

## Requisitos

Para instalar las dependencias necesarias, ejecuta el siguiente comando en tu entorno virtual:

`pip install -r requirements.txt`


## Uso

Para entrenar y evaluar un modelo, ejecuta `main.py` con los siguientes argumentos:

- `--data_file`: Ruta del archivo de datos (por defecto: `processed_data/data_normalized.csv`)
- `--model_type`: Tipo de modelo generativo a utilizar (`VAE` o `GAN`, por defecto: `VAE`)
- `--config_file`: Ruta del archivo de configuración del modelo (por defecto: depende del tipo de modelo)
- `--results_folder`: Carpeta para guardar los resultados (por defecto: depende del tipo de modelo)

Ejemplo de uso:

`python main.py --model_type=VAE`

Esto entrenará y evaluará un modelo VAE con la configuración y datos predeterminados.

## Contribuciones

Este proyecto es un trabajo en solitario para un proyecto de fin de máster. La única retroalimentación ha sido por parte de Caroline König, tutora del alumno.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para obtener más información.
