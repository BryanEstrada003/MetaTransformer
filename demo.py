# demo.py - Versión corregida
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data2Seq"))

from Data2Seq import Data2Seq

import torch 
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

image_tokenizer = Data2Seq(modality='image', dim=768)

# Opción 1: Todo el vector tabular como un solo token
tabular_tokenizer_single = Data2Seq(
    modality='tabular', 
    dim=768,
    num_features=3,
    embed_type='single'  # Necesitamos agregar este parámetro
)

# Opción 2: Cada feature como token separado (RECOMENDADO para tu caso)
tabular_tokenizer = Data2Seq(
    modality='tabular', 
    dim=768,
    num_features=3,
    embed_type='separate'  # Cada sensor es un token separado
)

def load_and_preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, image

def prepare_tabular_data(sensor_readings):
    """
    Prepara los datos tabulares para 3 sensores
    
    Args:
        sensor_readings: Lista de 3 valores de sensores
        
    Returns:
        Tensor de forma (1, 3) donde cada columna es un sensor
    """
    # Convertir a tensor: [batch_size, num_features]
    # Cada columna es una característica (sensor)
    return torch.tensor([sensor_readings], dtype=torch.float32)

def visualize_tokens(img_tokens, tabular_tokens, encoded_features):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    img_tokens_np = img_tokens[0].detach().cpu().numpy()
    tabular_tokens_np = tabular_tokens[0].detach().cpu().numpy()
    encoded_np = encoded_features[0].detach().cpu().numpy()
    
    # Visualización de tokens de imagen
    img_activation = img_tokens_np.mean(axis=1)
    axes[0, 0].imshow(img_activation.reshape(14, 14), cmap='viridis')
    axes[0, 0].set_title('Mapa de activación promedio de tokens de imagen')
    
    # Visualización de embeddings tabulares
    # tabular_tokens_np tiene forma (4, 768) donde:
    # token 0: CLS
    # token 1-3: Sensores
    for i in range(1, 4):  # Saltar CLS token
        axes[0, 1].plot(tabular_tokens_np[i], label=f'Sensor {i}')
    
    axes[0, 1].set_title('Embeddings de sensores')
    axes[0, 1].set_xlabel('Dimensión')
    axes[0, 1].set_ylabel('Valor')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Visualización de características codificadas
    encoded_avg = encoded_np.mean(axis=0)
    axes[0, 2].imshow(encoded_avg.reshape(1, -1), aspect='auto', cmap='hot', 
                     extent=[0, 768, 0, 1])
    axes[0, 2].set_title('Características codificadas promedio')
    axes[0, 2].set_xlabel('Dimensión')
    axes[0, 2].set_yticks([])
    
    # Histogramas
    axes[1, 0].hist(img_tokens_np.flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 0].set_title('Distribución - Tokens de imagen')
    
    axes[1, 1].hist(tabular_tokens_np.flatten(), bins=50, alpha=0.7, color='green')
    axes[1, 1].set_title('Distribución - Tokens tabulares')
    
    axes[1, 2].hist(encoded_np.flatten(), bins=50, alpha=0.7, color='red')
    axes[1, 2].set_title('Distribución - Características codificadas')
    
    plt.tight_layout()
    plt.savefig('visualizacion_tokens.png')
    plt.show()

def main():
    # Cargar imagen
    image_path = "./IMG_4646.JPG"
    
    try:
        img_data, original_image = load_and_preprocess_image(image_path)
        print(f"Imagen cargada exitosamente: {image_path}")
        print(f"Forma del tensor de imagen: {img_data.shape}")
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        img_data = torch.randn(1, 3, 224, 224)
        original_image = None
        print("Usando datos aleatorios como fallback")

    # Preparar datos tabulares (3 sensores)
    sensor_readings = [6.348, 0.411, 0.455]
    tabular_data = prepare_tabular_data(sensor_readings)
    print(f"\nDatos tabulares (3 sensores): {sensor_readings}")
    print(f"Forma de datos tabulares: {tabular_data.shape}")
    
    # Obtener tokens
    img_tokens = image_tokenizer(img_data)
    tabular_tokens = tabular_tokenizer(tabular_data)
    
    print(f"\nForma de tokens de imagen: {img_tokens.shape}")
    print(f"Forma de tokens tabulares: {tabular_tokens.shape}")
    
    # Concatenar tokens
    features = torch.concat([img_tokens, tabular_tokens], dim=1)
    print(f"Total de tokens concatenados: {features.shape[1]}")
    
    # Cargar encoder
    try:
        ckpt = torch.load("Meta-Transformer_base_patch16_encoder.pth", map_location='cpu')
        encoder = nn.Sequential(*[
            Block(
                dim=768,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(12)])
        encoder.load_state_dict(ckpt, strict=True)
        
        encoder.eval()
        
        with torch.no_grad():
            encoded_features = encoder(features)
        
        print(f"\nCaracterísticas codificadas: {encoded_features.shape}")
        
        # Mostrar información de los tokens
        print(f"\nTokens tabulares desglose:")
        print(f"  Total tokens: {tabular_tokens.shape[1]}")
        print(f"  - CLS token: posición 0")
        for i in range(3):
            sensor_token = tabular_tokens[0, i+1, :]
            print(f"  - Sensor {i+1}: posición {i+1}, "
                  f"min={sensor_token.min():.4f}, max={sensor_token.max():.4f}, "
                  f"mean={sensor_token.mean():.4f}")
        
        # Calcular similitudes
        img_avg = img_tokens.mean(dim=1, keepdim=True)
        tabular_avg = tabular_tokens[:, 1:, :].mean(dim=1, keepdim=True)  # Excluir CLS
        
        similarity = F.cosine_similarity(img_avg, tabular_avg, dim=-1)
        print(f"\nSimilitud coseno imagen-sensores: {similarity.item():.4f}")
        
        # Visualizar
        if original_image is not None:
            visualize_tokens(img_tokens, tabular_tokens, encoded_features)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()