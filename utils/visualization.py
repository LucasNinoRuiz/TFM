import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(model, results_folder, n=5):
    """
    Visualiza los resultados del modelo generativo, incluyendo imágenes generadas y gráficas de pérdida.
    Guarda las imágenes y gráficas en la carpeta de resultados especificada.

    Args:
    model: Modelo generativo entrenado (VAE o GAN).
    results_folder (str): Ruta de la carpeta donde se guardarán los resultados.
    n (int): Número de imágenes generadas a visualizar en una fila.
    """

    # Generar imágenes
    # Modificar este método
    generated_images = model.generate_images(n * n)

    # Guardar imágenes generadas
    images_path = os.path.join(results_folder, 'generated_images.png')
    save_generated_images(generated_images, images_path, n)

    # Guardar gráficas de pérdida
    loss_plot_path = os.path.join(results_folder, 'loss_plot.png')
    plot_loss(model, loss_plot_path)

def save_generated_images(generated_images, images_path, n):
    """
    Guarda las imágenes generadas en un archivo de imagen.

    Args:
    generated_images (np.array): Array de imágenes generadas.
    images_path (str): Ruta del archivo donde se guardarán las imágenes generadas.
    n (int): Número de imágenes generadas a visualizar en una fila.
    """

    generated_images = (generated_images * 255).astype(np.uint8)
    img_size = generated_images.shape[1]
    grid = np.zeros((n * img_size, n * img_size, 3), dtype=np.uint8)

    for i in range(n):
        for j in range(n):
            grid[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size] = generated_images[i * n + j]

    plt.imsave(images_path, grid)

def plot_loss(model, loss_plot_path):
    """
    Genera una gráfica de pérdida a partir de los registros de entrenamiento del modelo.

    Args:
    model: Modelo generativo entrenado (VAE o GAN).
    loss_plot_path (str): Ruta del archivo donde se guardará la gráfica de pérdida.
    """

    # Obtener historial de pérdida
    loss_history = model.get_loss_history()

    # Crear gráficas de pérdida
    plt.figure()
    plt.plot(loss_history['loss'], label='Loss')
    if 'val_loss' in loss_history:
        plt.plot(loss_history['val_loss'], label='Validation Loss')

    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Guardar gráficas de pérdida
    plt.savefig(loss_plot_path)
    plt.close()
