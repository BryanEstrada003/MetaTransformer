import cv2
import os
from LIME import LIME

# Rutas de las carpetas de entrada y salida
input_dir = "/media/vision/Almacenamiento e/BANANA DEFORM/RGB/ORIGINAL/CAT1"
output_dir = "/media/vision/Almacenamiento e/BANANA DEFORM/RGB/BANANA_ILLUM/CAT1"

# Crear la carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Lista de valores de gamma para ajustar la luminosidad
gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Procesar cada imagen en la carpeta de entrada
for filename in os.listdir(input_dir):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # Filtrar por tipos de imagen válidos
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        # Procesar la imagen con cada valor de gamma
        for gamma in gamma_values:
            # Crear una instancia de LIME con el valor actual de gamma
            lime = LIME(img, alpha=1, gamma=gamma, rho=2)
            lime.optimizeIllumMap()
            enhanced_img = lime.enhance()

            # Generar el nombre de archivo de salida
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_LIME_0_{int(gamma * 10)}{ext}"
            output_path = os.path.join(output_dir, output_filename)

            # Guardar la imagen mejorada
            cv2.imwrite(output_path, enhanced_img)
            print(f"Imagen guardada: {output_path}")

print("Procesamiento completado.")