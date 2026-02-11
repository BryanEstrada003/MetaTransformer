import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------- Importar Data2Seq y bloques transformer ----------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Data2Seq"))
from Data2Seq import Data2Seq
from timm.models.vision_transformer import Block

# ---------- Reutilizar funciones del script multimodal ----------
from train_multimodal import ConfigLoader, TabularDataLoader, count_images_per_class_split

# ------------------------------------------------------------
#   MODELO UNIMODAL PARA TABULAR (MISMA ARQUITECTURA)
# ------------------------------------------------------------
class TabularOnlyModel(nn.Module):
    """Modelo transformer solo para datos tabulares (mismo encoder que multimodal)."""
    
    def __init__(self, config):
        super().__init__()
        
        model_config = config['model']
        training_config = config['training']
        paths_config = config['paths']
        
        self.dim = model_config['dim']
        num_classes = model_config['num_classes']
        
        # Tokenizador solo para tabular
        self.tabular_tokenizer = Data2Seq(
            modality='tabular',
            dim=self.dim,
            num_features=3,          # 3 columnas: MQ135, MQ4, MQ136
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
                    print("Encoder congelado (tabular)")
                else:
                    print("Encoder entrenable (tabular)")
            else:
                print(f"Advertencia: Pesos no encontrados: {encoder_weights_path}")
                print("Inicializando encoder desde cero (tabular)")
                self.encoder = nn.Sequential(*[
                    Block(dim=self.dim, num_heads=12, mlp_ratio=4., qkv_bias=True,
                          norm_layer=nn.LayerNorm, act_layer=nn.GELU)
                    for _ in range(12)
                ])
        except Exception as e:
            print(f"Error cargando encoder: {e}")
            print("Inicializando encoder desde cero (tabular)")
            self.encoder = nn.Sequential(*[
                Block(dim=self.dim, num_heads=12, mlp_ratio=4., qkv_bias=True,
                      norm_layer=nn.LayerNorm, act_layer=nn.GELU)
                for _ in range(12)
            ])
        
        # Cabeza de clasificación (idéntica a multimodal)
        hidden_dim = model_config['classifier_hidden_dim']
        dropout = model_config['classifier_dropout']
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Estadísticas
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parámetros (tabular): {total_params:,}")
        print(f"Parámetros entrenables (tabular): {trainable_params:,}")
        print(f"Porcentaje entrenable: {100*trainable_params/total_params:.2f}%")
    
    def forward(self, tabular):
        # Tokenizar datos tabulares → (batch_size, 4, dim)  [CLS + 3 features]
        tabular_tokens = self.tabular_tokenizer(tabular)
        
        # Pasar por encoder
        encoded = self.encoder(tabular_tokens)
        
        # Usar CLS token (posición 0) para clasificación
        cls_token = encoded[:, 0, :]
        
        # Clasificación
        logits = self.classifier(cls_token)
        return logits

# ------------------------------------------------------------
#   DATASET (sin normalización, el tokenizador se encarga)
# ------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, df, feature_cols, label_col='class_idx'):
        self.features = df[feature_cols].values.astype(np.float32)  # valores crudos
        self.labels = df[label_col].values.astype(np.int64)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

# ------------------------------------------------------------
#   CREAR DATAFRAMES CON LA MISMA PARTICIÓN QUE MULTIMODAL
# ------------------------------------------------------------
def create_tabular_datasets(config, class_tabular_data):
    """Devuelve DataFrames train/val/test con el mismo número de muestras
       y el mismo orden que en el entrenamiento multimodal."""
    base_dir = config['paths']['base_dir']
    image_dirs = config['data']['image_dirs']
    class_mapping = config['classes']
    
    image_counts = count_images_per_class_split(base_dir, image_dirs, class_mapping)
    
    train_list, val_list, test_list = [], [], []
    for class_idx, full_df in class_tabular_data.items():
        total_imgs = sum(image_counts[split].get(class_idx, 0) for split in ['train','val','test'])
        if total_imgs == 0:
            continue
        available = len(full_df)
        if total_imgs > available:
            print(f"Clase {class_idx}: {total_imgs} imágenes pero {available} filas. Usando {available}.")
            total_imgs = available
        class_subset = full_df.iloc[:total_imgs].copy()
        class_subset['class_idx'] = class_idx
        
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
    
    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    val_df   = pd.concat(val_list, ignore_index=True)   if val_list   else pd.DataFrame()
    test_df  = pd.concat(test_list, ignore_index=True)  if test_list  else pd.DataFrame()
    
    return train_df, val_df, test_df

# ------------------------------------------------------------
#   ENTRENAMIENTO / VALIDACIÓN / TEST
# ------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc='Train')
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = out.max(1)
        total += y.size(0)
        correct += preds.eq(y).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return total_loss/len(loader), 100.*correct/total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Val'):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            _, preds = out.max(1)
            total += y.size(0)
            correct += preds.eq(y).sum().item()
    return total_loss/len(loader), 100.*correct/total

def test(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            _, p = out.max(1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.numpy())
    acc = 100. * np.mean(np.array(preds) == np.array(labels))
    return acc, preds, labels

def save_results(output_dir, history, test_acc, test_preds, test_labels):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicciones
    pd.DataFrame({'true': test_labels, 'pred': test_preds}).to_csv(
        output_dir / 'test_predictions_tabular.csv', index=False)
    
    # Historial
    with open(output_dir / 'history_tabular.json', 'w') as f:
        json.dump(history, f)
    
    # Métricas
    metrics = {
        'best_val_acc': max(history['val_acc']),
        'test_acc': test_acc,
        'epochs': len(history['train_acc'])
    }
    with open(output_dir / 'metrics_tabular.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Gráficas
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.savefig(output_dir / 'training_history_tabular.png')
    plt.close()
    print(f"Resultados tabulares guardados en {output_dir}")

# ------------------------------------------------------------
#   MAIN
# ------------------------------------------------------------
def main(config_path):
    # 1. Cargar configuración
    config = ConfigLoader.load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 2. Directorio de salida (compatible con train_image_only.py)
    if 'output_uni' in config['paths']:
        base_output = config['paths']['output_uni']
    else:
        base_output = config['paths']['output_dir']  # fallback
    output_dir = base_output / 'tabular_only'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Cargar datos tabulares completos
    print("\nCargando datos tabulares...")
    class_tabular_data = TabularDataLoader.load_full_tabular_by_class(config)
    
    # 4. Crear DataFrames con la misma partición que el multimodal
    print("\nCreando particiones train/val/test (mismo orden que multimodal)...")
    train_df, val_df, test_df = create_tabular_datasets(config, class_tabular_data)
    
    # 5. Datasets y DataLoaders (SIN normalización, valores crudos)
    feature_cols = config['column_tabular']
    train_dataset = TabularDataset(train_df, feature_cols)
    val_dataset   = TabularDataset(val_df, feature_cols)
    test_dataset  = TabularDataset(test_df, feature_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # 6. Modelo transformer solo tabular
    print("\nCreando modelo transformer solo tabular...")
    model = TabularOnlyModel(config)
    model = model.to(device)
    
    # 7. Pesos de clase (balanceados)
    from sklearn.utils.class_weight import compute_class_weight
    labels = train_df['class_idx'].values
    classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 8. Optimizador y scheduler (solo parámetros entrenables)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['num_epochs'])
    
    # 9. Entrenamiento
    print("\nIniciando entrenamiento...")
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    best_val_acc = 0
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nÉpoca {epoch+1}/{config['training']['num_epochs']}")
        
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
            torch.save(model.state_dict(), output_dir / 'best_model_tabular.pth')
    
    # 10. Evaluación en test
    print("\nEvaluando en test...")
    model.load_state_dict(torch.load(output_dir / 'best_model_tabular.pth'))
    test_acc, preds, labels = test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # 11. Guardar resultados
    save_results(output_dir, history, test_acc, preds, labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)