# train_multimodal.py
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import argparse
from sklearn.utils import class_weight
warnings.filterwarnings('ignore')

# Agregar Data2Seq al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "Data2Seq"))

from Data2Seq import Data2Seq
from timm.models.vision_transformer import Block

class ConfigLoader:
    """Cargador de configuración desde JSON"""
    
    @staticmethod
    def load_config(config_path):
        """Carga la configuración desde un archivo JSON"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Convertir paths a objetos Path
        paths = config.get('paths', {})
        for key, value in paths.items():
            if isinstance(value, str):
                paths[key] = Path(value)
        
        # Convertir paths en data
        data = config.get('data', {})
        if 'data_files' in data:
            for key, value in data['data_files'].items():
                if isinstance(value, str):
                    data['data_files'][key] = Path(value)
        
        return config

class MultimodalDataset(Dataset):
    """Dataset para datos multimodales de xxx (imagen + tabular)"""
    
    def __init__(self, image_dir, column_tabular, class_mapping, transform=None, mode='train'):
        """
        Args:
            image_dir: Directorio base de imágenes
            column_tabular: DataFrame con datos
            class_mapping: Diccionario de mapeo de clases
            transform: Transformaciones para imágenes
            mode: 'train', 'val' o 'test'
        """
        self.image_dir = Path(image_dir)
        self.column_tabular = column_tabular
        self.class_mapping = class_mapping
        self.transform = transform
        self.mode = mode
        
        # Recopilar todos los datos
        self.samples = []
        
        # Procesar cada clase
        for class_name, class_idx in class_mapping.items():
            class_dir = self.image_dir / class_name
            
            if not class_dir.exists():
                print(f"Advertencia: Directorio no encontrado: {class_dir}")
                continue
            
            # Obtener archivos de imagen
            image_files = sorted(list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.jpg')))
            
            # Filtrar datos de tabular para esta clase
            class_column_tabular = self.column_tabular[self.column_tabular['class_idx'] == class_idx]
            
            # Emparejar imágenes con datos de tabular por índice
            for i, img_path in enumerate(image_files):
                if i < len(class_column_tabular):
                    tabular_row = class_column_tabular.iloc[i]
                    self.samples.append({
                        'image_path': str(img_path),
                        'tabular_values': tabular_row[self.column_tabular.columns[:3]].values.astype(np.float32),
                        'label': class_idx
                    })
        
        print(f"Cargados {len(self.samples)} muestras para modo {mode}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Cargar imagen
        image = Image.open(sample['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Obtener valores de tabular
        tabular_values = torch.tensor(sample['tabular_values'], dtype=torch.float32)
        
        # Etiqueta
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return image, tabular_values, label

class MultimodalModel(nn.Module):
    """Modelo multimodal para clasificación"""
    
    def __init__(self, config):
        super().__init__()
        
        model_config = config['model']
        training_config = config['training']
        paths_config = config['paths']
        
        self.dim = model_config['dim']
        num_classes = model_config['num_classes']
        
        # Tokenizadores
        self.image_tokenizer = Data2Seq(modality='image', dim=self.dim)
        self.tabular_tokenizer = Data2Seq(
            modality='tabular',
            dim=self.dim,
            num_features=3,  # 3 columns
            embed_type='separate'
        )
        
        # Encoder (Meta-Transformer)
        encoder_weights_path = paths_config['encoder_weights']
        try:
            if os.path.exists(encoder_weights_path):
                ckpt = torch.load(encoder_weights_path, map_location='cpu')
                self.encoder = nn.Sequential(*[
                    Block(
                        dim=self.dim,
                        num_heads=12,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        norm_layer=nn.LayerNorm,
                        act_layer=nn.GELU
                    )
                    for _ in range(12)
                ])
                self.encoder.load_state_dict(ckpt, strict=True)
                
                # Congelar encoder si se solicita
                if training_config['freeze_encoder']:
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                    print("Encoder congelado")
                else:
                    print("Encoder entrenable")
                    
            else:
                print(f"Advertencia: Archivo de pesos no encontrado: {encoder_weights_path}")
                print("Inicializando encoder desde cero")
                self.encoder = nn.Sequential(*[
                    Block(
                        dim=self.dim,
                        num_heads=12,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        norm_layer=nn.LayerNorm,
                        act_layer=nn.GELU
                    )
                    for _ in range(12)
                ])
                
        except Exception as e:
            print(f"Error cargando encoder: {e}")
            print("Inicializando encoder desde cero")
            self.encoder = nn.Sequential(*[
                Block(
                    dim=self.dim,
                    num_heads=12,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU
                )
                for _ in range(12)
            ])
        
        # Head de clasificación
        hidden_dim = model_config['classifier_hidden_dim']
        dropout = model_config['classifier_dropout']
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Estadísticas del modelo
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parámetros: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        print(f"Porcentaje entrenable: {100*trainable_params/total_params:.2f}%")
    
    def forward(self, images, column_tabular):
        # Tokenizar imágenes (batch_size, 197, dim)
        img_tokens = self.image_tokenizer(images)
        
        # Tokenizar datos tabulares (batch_size, 4, dim)
        tabular_tokens = self.tabular_tokenizer(column_tabular)
        
        # Concatenar tokens (excluyendo CLS token de tabular)
        features = torch.cat([img_tokens, tabular_tokens[:, 1:, :]], dim=1)
        
        # Pasar por encoder
        encoded = self.encoder(features)
        
        # Usar CLS token para clasificación (posición 0)
        cls_token = encoded[:, 0, :]
        
        # Clasificación
        logits = self.classifier(cls_token)
        
        return logits

class TabularDataLoader:
    """Cargador de datos"""

    @staticmethod
    def load_full_tabular_by_class(config):
        """
        Carga los datos tabulares completos de cada clase.
        Retorna: dict {class_idx: DataFrame con columnas tabulares}
        """
        data_config = config['data']
        column_tabular = config['column_tabular']
        
        typeA_excel = data_config['data_files']['typeA']
        typeB_excel = data_config['data_files']['typeB']
        
        class_data = {}
        
        # Clases 0-4
        for i, sheet in enumerate(data_config['sheets']['typeA_sheets']):
            try:
                df = pd.read_excel(typeA_excel, sheet_name=sheet)
                df = df[column_tabular].copy()
                class_data[i] = df
            except Exception as e:
                print(f"Error cargando hoja {sheet}: {e}")
        
        # Clases 5-9
        for i, sheet in enumerate(data_config['sheets']['typeB_sheets']):
            try:
                df = pd.read_excel(typeB_excel, sheet_name=sheet)
                df = df[column_tabular].copy()
                class_data[i + 5] = df
            except Exception as e:
                print(f"Error cargando hoja {sheet}: {e}")
        
        print("Datos tabulares cargados por clase:")
        for idx, df in class_data.items():
            print(f"  Clase {idx}: {len(df)} filas")
        return class_data

def count_images_per_class_split(base_dir, image_dirs, class_mapping):
    """Cuenta el número real de imágenes por clase en cada split."""
    counts = {'train': {}, 'val': {}, 'test': {}}
    for split_name, dir_name in image_dirs.items():
        split_dir = base_dir / dir_name
        for class_name, class_idx in class_mapping.items():
            class_dir = split_dir / class_name
            if class_dir.exists():
                # Busca .JPG y .jpg (puedes añadir más formatos)
                images = list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.jpg'))
                counts[split_name][class_idx] = len(images)
            else:
                counts[split_name][class_idx] = 0
    return counts

def create_datasets(config, class_tabular_data):
    """
    Crea los datasets train/val/test alineando los datos tabulares
    con la cantidad real de imágenes por clase.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    
    base_dir = config['paths']['base_dir']
    image_dirs = config['data']['image_dirs']
    class_mapping = config['classes']
    
    # 1. Contar imágenes reales
    print("\nContando imágenes reales por split y clase...")
    image_counts = count_images_per_class_split(base_dir, image_dirs, class_mapping)
    
    # Mostrar conteos
    for split_name in ['train', 'val', 'test']:
        print(f"\n{split_name.upper()}:")
        for class_idx in sorted(image_counts[split_name].keys()):
            cnt = image_counts[split_name][class_idx]
            print(f"  Clase {class_idx}: {cnt} imágenes")
    
    # 2. Construir DataFrames tabulares para cada split
    train_list, val_list, test_list = [], [], []
    
    for class_idx, full_df in class_tabular_data.items():
        total_imgs = (image_counts['train'].get(class_idx, 0) +
                      image_counts['val'].get(class_idx, 0) +
                      image_counts['test'].get(class_idx, 0))
        
        if total_imgs == 0:
            print(f"  Clase {class_idx}: sin imágenes, se omite.")
            continue
        
        # Tomar solo las primeras 'total_imgs' filas (respetando el orden)
        available = len(full_df)
        if total_imgs > available:
            print(f"  Advertencia: Clase {class_idx} tiene {total_imgs} imágenes pero solo {available} filas tabulares. Se usarán {available}.")
            total_imgs = available
        
        class_subset = full_df.iloc[:total_imgs].copy()
        class_subset['class_idx'] = class_idx
        
        # Dividir según los conteos reales
        train_cnt = image_counts['train'].get(class_idx, 0)
        val_cnt   = image_counts['val'].get(class_idx, 0)
        test_cnt  = image_counts['test'].get(class_idx, 0)
        
        start = 0
        if train_cnt > 0:
            train_list.append(class_subset.iloc[start:start+train_cnt])
            start += train_cnt
        if val_cnt > 0:
            val_list.append(class_subset.iloc[start:start+val_cnt])
            start += val_cnt
        if test_cnt > 0:
            test_list.append(class_subset.iloc[start:start+test_cnt])
            # start += test_cnt
    
    # Concatenar por split
    train_tabular = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    val_tabular   = pd.concat(val_list, ignore_index=True)   if val_list   else pd.DataFrame()
    test_tabular  = pd.concat(test_list, ignore_index=True)  if test_list  else pd.DataFrame()
    
    print(f"\nDatos tabulares asignados por conteo real:")
    print(f"  Train: {len(train_tabular)} muestras")
    print(f"  Val:   {len(val_tabular)} muestras")
    print(f"  Test:  {len(test_tabular)} muestras")
    
    # 3. Crear datasets (tu clase MultimodalDataset no necesita cambios)
    train_dataset = MultimodalDataset(
        base_dir / image_dirs['train'], train_tabular, class_mapping, transform_train, mode='train'
    )
    val_dataset = MultimodalDataset(
        base_dir / image_dirs['val'], val_tabular, class_mapping, transform, mode='val'
    )
    test_dataset = MultimodalDataset(
        base_dir / image_dirs['test'], test_tabular, class_mapping, transform, mode='test'
    )
    
    return train_dataset, val_dataset, test_dataset

def calculate_class_weights(train_loader, num_classes=10):
    """Calcula pesos de clase a partir de un DataLoader"""
    class_counts = torch.zeros(num_classes)
    
    for _, _, labels in train_loader:  # labels es tensor de batch
        for label in labels:
            class_counts[label] += 1
    
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    
    # Evitar infinitos si alguna clase tiene 0
    class_weights = torch.nan_to_num(class_weights, nan=1.0)
    
    return class_weights

class Trainer:
    """Entrenador para el modelo multimodal"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
    def train_epoch(self, model, dataloader, criterion, optimizer):
        """Entrenar una época"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Entrenando')
        for images, tabular, labels in pbar:
            images, tabular, labels = images.to(self.device), tabular.to(self.device), labels.to(self.device)           
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, tabular)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Estadísticas
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, model, dataloader, criterion):
        """Validación"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, tabular, labels in tqdm(dataloader, desc='Validando'):
                images, tabular, labels = images.to(self.device), tabular.to(self.device), labels.to(self.device)
                
                outputs = model(images, tabular)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def test(self, model, dataloader):
        """Evaluación en test"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, tabular, labels in tqdm(dataloader, desc='Testeando'):
                images, tabular, labels = images.to(self.device), tabular.to(self.device), labels.to(self.device)
                
                outputs = model(images, tabular)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calcular accuracy
        correct = sum(np.array(all_preds) == np.array(all_labels))
        accuracy = 100. * correct / len(all_labels)
        
        return accuracy, all_preds, all_labels

def main(config_path):
    """Función principal"""
    
    # Cargar configuración
    print(f"Cargando configuración desde: {config_path}")
    config = ConfigLoader.load_config(config_path)
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Crear directorio de salida
    output_dir = config['paths']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos de tabular
    print("\nCargando datos tabulares completos...")
    class_tabular_data = TabularDataLoader.load_full_tabular_by_class(config)

    # Crear datasets
    print("\nCreando datasets con alineación por conteo real...")
    train_dataset, val_dataset, test_dataset = create_datasets(config, class_tabular_data)
    
    # Crear dataloaders
    training_config = config['training']
    batch_size = training_config['batch_size']
    num_workers = training_config['num_workers']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers)
    
    # Crear modelo
    print("\nCreando modelo...")
    model = MultimodalModel(config)
    model = model.to(device)

    class_weights = calculate_class_weights(train_loader)

    print(f"Pesos de clase calculados: {class_weights}")
    
    # Configurar entrenamiento
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Solo entrenar parámetros que requieren gradiente
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=training_config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_config['num_epochs']
    )
    
    # Entrenador
    trainer = Trainer(config, device)
    
    # Entrenamiento
    print("\nIniciando entrenamiento...")
    best_val_acc = 0
    
    for epoch in range(training_config['num_epochs']):
        print(f"\nÉpoca {epoch+1}/{training_config['num_epochs']}")
        
        # Entrenar
        train_loss, train_acc = trainer.train_epoch(model, train_loader, criterion, optimizer)
        
        # Validar
        val_loss, val_acc = trainer.validate(model, val_loader, criterion)
        
        # Actualizar scheduler
        scheduler.step()
        
        # Guardar historial
        trainer.history['train_loss'].append(train_loss)
        trainer.history['train_acc'].append(train_acc)
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            config_clean = json.loads(json.dumps(config, default=str))
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': trainer.history,
                'config': config_clean
            }
            
            model_path = output_dir / 'best_multimodal_model.pth'
            torch.save(checkpoint, str(model_path))
    
    # Evaluar en test con mejor modelo
    print("\nCargando mejor modelo para evaluación en test...")
    model_path = output_dir / 'best_multimodal_model.pth'
    checkpoint = torch.load(str(model_path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_acc, test_preds, test_labels = trainer.test(model, test_loader)
    print(f"Accuracy en test: {test_acc:.2f}%")
    
    # Guardar resultados
    save_results(config, output_dir, trainer, test_acc, test_preds, test_labels)
    
    return model, trainer.history

def save_results(config, output_dir, trainer, test_acc, test_preds, test_labels):
    """Guardar resultados y visualizaciones"""
    
    # Guardar predicciones
    predictions_df = pd.DataFrame({
        'true_label': test_labels,
        'predicted_label': test_preds
    })
    predictions_path = output_dir / 'test_predictions.csv'
    predictions_df.to_csv(str(predictions_path), index=False)
    print(f"Predicciones guardadas en {predictions_path}")
    
    # Guardar historial como JSON
    history_path = output_dir / 'training_history.json'
    with open(str(history_path), 'w') as f:
        json.dump(trainer.history, f, indent=4)
    
    # Guardar configuración usada
    config_path = output_dir / 'training_config.json'
    with open(str(config_path), 'w') as f:
        # Convertir Path objects a strings para JSON
        config_serializable = json.loads(json.dumps(config, default=str))
        json.dump(config_serializable, f, indent=4)
    
    # Visualizar historial de entrenamiento
    plot_training_history(trainer.history, output_dir)
    
    # Guardar métricas finales
    metrics = {
        'best_val_accuracy': max(trainer.history['val_acc']),
        'final_test_accuracy': test_acc,
        'final_train_accuracy': trainer.history['train_acc'][-1],
        'final_val_accuracy': trainer.history['val_acc'][-1],
        'num_epochs': len(trainer.history['train_acc']),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    metrics_path = output_dir / 'training_metrics.json'
    with open(str(metrics_path), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Resultados guardados en {output_dir}")

def plot_training_history(history, output_dir):
    """Graficar historial de entrenamiento"""
    plt.figure(figsize=(12, 4))
    
    # Gráfico de loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss durante entrenamiento')
    plt.grid(True, alpha=0.3)
    
    # Gráfico de accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Época')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy durante entrenamiento')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_path = output_dir / 'training_history.png'
    plt.savefig(str(history_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de entrenamiento guardado en {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar modelo multimodal para clasificación de xxx')
    parser.add_argument('--config', type=str, required=True, 
                       help='Ruta al archivo de configuración JSON')
    parser.add_argument('--create-example', action='store_true',
                       help='Crear un archivo de configuración de ejemplo')
    
    args = parser.parse_args()
    
    if args.create_example:
        ConfigLoader.create_example_config()
    else:
        if not os.path.exists(args.config):
            print(f"Error: Archivo de configuración no encontrado: {args.config}")
            print("Usa --create-example para crear un archivo de configuración de ejemplo")
            sys.exit(1)
        
        main(args.config)