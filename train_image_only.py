import os
import sys
import json
from torch.utils.data import Dataset  # Añadir esta línea
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings('ignore')

# Agregar Data2Seq al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "Data2Seq"))

from Data2Seq import Data2Seq
from timm.models.vision_transformer import Block

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
            nn.Dropout(dropout),
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
    
    def forward(self, images):
        # Tokenizar imágenes (batch_size, 197, dim)
        img_tokens = self.image_tokenizer(images)        
        
        # Concatenar tokens (excluyendo CLS token de tabular)
        features = torch.cat([img_tokens], dim=1)
        
        # Pasar por encoder
        encoded = self.encoder(features)
        
        # Usar CLS token para clasificación (posición 0)
        cls_token = encoded[:, 0, :]
        
        # Clasificación
        logits = self.classifier(cls_token)
        
        return logits

class FilteredImageFolder(Dataset):
    """Dataset que filtra ImageFolder por clases permitidas"""
    
    def __init__(self, root, allowed_classes, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.allowed_classes = allowed_classes
        
        # Cargar ImageFolder completo
        self.full_dataset = ImageFolder(str(self.root), transform=transform)
        
        # Crear mapping de índices originales a nuevos índices
        self.original_to_new = {}
        self.new_class_names = []
        
        # Filtrar solo las clases permitidas
        for class_name, class_idx in self.full_dataset.class_to_idx.items():

            if class_name in list(allowed_classes.keys()):
                class_idx_new = allowed_classes[class_name]
                self.original_to_new[class_idx] = class_idx_new
                self.new_class_names.append(class_name)
                print(f"Clase permitida encontrada: {class_name} (original idx: {class_idx}, nuevo idx: {self.original_to_new[class_idx]})")

                # ordena la nueva lista de clases según los nuevos índices
                self.new_class_names.sort(key=lambda x: allowed_classes[x])
        
        if not self.original_to_new:
            raise ValueError(f"No se encontraron clases permitidas en {root}. "
                           f"Clases disponibles: {list(self.full_dataset.class_to_idx.keys())}")
        
        # Filtrar muestras
        self.samples = []
        self.targets = []
        
        for img_path, original_idx in self.full_dataset.samples:
            if original_idx in self.original_to_new:
                new_idx = self.original_to_new[original_idx]
                self.samples.append((img_path, new_idx))
                self.targets.append(new_idx)
        
        print(f"  Clases encontradas en {root.name}: {[self.full_dataset.classes[i] for i in self.original_to_new.keys()]}")
        print(f"  Total muestras: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.full_dataset.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, target
    
    @property
    def classes(self):
        return self.new_class_names
    
    @property
    def class_to_idx(self):
        return {name: i for i, name in enumerate(self.new_class_names)}

class ConfigLoader:
    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        paths = config.get('paths', {})
        for k, v in paths.items():
            if isinstance(v, str):
                paths[k] = Path(v)
        data = config.get('data', {})
        if 'data_files' in data:
            for k, v in data['data_files'].items():
                if isinstance(v, str):
                    data['data_files'][k] = Path(v)
        return config

def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc='Train')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Val'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return total_loss / len(loader), 100. * correct / total

def test(model, loader, device):
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Test'):
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.numpy())
    acc = 100. * np.mean(np.array(preds_list) == np.array(labels_list))
    return acc, preds_list, labels_list

def save_results(config, output_dir, history, test_acc, test_preds, test_labels, class_names):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar nombres de clases
    with open(output_dir / 'class_names.json', 'w') as f:
        json.dump(class_names, f, indent=4)
    
    # predicciones
    pd.DataFrame({
        'true_label': test_labels, 
        'predicted_label': test_preds,
        'true_class': [class_names[l] for l in test_labels],
        'predicted_class': [class_names[p] for p in test_preds]
    }).to_csv(output_dir / 'test_predictions.csv', index=False)# historial

    with open(output_dir / 'history_image.json', 'w') as f:
        json.dump(history, f)

    # Guardar configuración usada
    config_path = output_dir / 'training_config.json'
    with open(str(config_path), 'w') as f:
        # Convertir Path objects a strings para JSON
        config_serializable = json.loads(json.dumps(config, default=str))
        json.dump(config_serializable, f, indent=4)

    # métricas
    metrics = {
        'best_val_accuracy': max(history['val_acc']),
        'test_acc': test_acc,
        'final_train_accuracy': history['train_acc'][-1],
        'final_val_accuracy': history['val_acc'][-1],
        'num_epochs': len(history['train_acc']),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open(output_dir / 'metrics_image.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    # gráficas
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.savefig(output_dir / 'training_history_image.png')
    plt.close()
    print(f"Resultados imagen guardados en {output_dir}")

def main(config_path):
    config = ConfigLoader.load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    base_dir = config['paths']['base_dir']
    image_dirs = config['data']['image_dirs']
    output_uni = config['paths']['output_uni'] / 'image_only'
    output_uni.mkdir(parents=True, exist_ok=True)
    
    # Datasets
    # Obtener clases permitidas del metadata
    allowed_classes = config['classes']
    print(f"Clases permitidas según metadata: {allowed_classes}")

    # Crear datasets filtrados
    print("\nFiltrando datasets por clases del metadata...")
    train_dataset = FilteredImageFolder(
        base_dir / image_dirs['train'], 
        allowed_classes, 
        transform=get_transforms('train')
    )
    val_dataset = FilteredImageFolder(
        base_dir / image_dirs['val'], 
        allowed_classes, 
        transform=get_transforms('val')
    )
    test_dataset = FilteredImageFolder(
        base_dir / image_dirs['test'], 
        allowed_classes, 
        transform=get_transforms('val')
    )

    class_names = train_dataset.classes
    print(f"\nClases finales para entrenamiento: {class_names}")
    print(f"Número de clases: {len(class_names)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=config['training']['num_workers'])
    val_loader   = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                              shuffle=False, num_workers=config['training']['num_workers'])
    test_loader  = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                              shuffle=False, num_workers=config['training']['num_workers'])
    
    # Modelo
    model = MultimodalModel(config)
    model = model.to(device)
    
    # Pesos de clase (opcional)
    from sklearn.utils.class_weight import compute_class_weight
    # Obtener labels del dataset filtrado
    labels = train_dataset.targets
    unique_classes = np.unique(labels)
    print(f"Clases únicas en train: {unique_classes}")

    # Calcular pesos de clase
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Pesos de clase calculados: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'])
    
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    best_val_acc = 0
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_uni / 'best_model_image.pth')
    
    # Evaluación test
    model.load_state_dict(torch.load(output_uni / 'best_model_image.pth'))
    test_acc, preds, labels = test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    save_results(config, output_uni, history, test_acc, preds, labels, class_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)