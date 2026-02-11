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
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------- reutilizar funciones del script multimodal ----------
from train_multimodal import ConfigLoader, TabularDataLoader, count_images_per_class_split

class TabularDataset(Dataset):
    def __init__(self, df, feature_cols, label_col='class_idx'):
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = df[label_col].values.astype(np.int64)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def create_tabular_datasets(config, class_tabular_data):
    """Igual que create_datasets pero devuelve DataFrames en lugar de datasets multimodales."""
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

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0,0,0
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
    total_loss, correct, total = 0,0,0
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

def save_results(output_uni, history, test_acc, test_preds, test_labels, scaler=None):
    output_uni.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'true': test_labels, 'pred': test_preds}).to_csv(output_uni / 'test_predictions_tabular.csv', index=False)
    with open(output_uni / 'history_tabular.json', 'w') as f:
        json.dump(history, f)
    metrics = {'best_val_acc': max(history['val_acc']), 'test_acc': test_acc}
    with open(output_uni / 'metrics_tabular.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    if scaler:
        joblib.dump(scaler, output_uni / 'scaler_tabular.pkl')
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.legend()
    plt.subplot(1,2,2); plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val'); plt.legend()
    plt.savefig(output_uni / 'training_history_tabular.png')
    plt.close()
    print(f"Resultados tabulares guardados en {output_uni}")

def main(config_path):
    config = ConfigLoader.load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_uni = config['paths']['output_uni'] / 'tabular_only'
    
    # Cargar datos tabulares completos
    class_tabular_data = TabularDataLoader.load_full_tabular_by_class(config)
    
    # Crear DataFrames para cada split (misma partición que en el modelo multimodal)
    train_df, val_df, test_df = create_tabular_datasets(config, class_tabular_data)
    
    # Normalizar características
    feature_cols = config['column_tabular']
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
    
    # Datasets
    train_dataset = TabularDataset(train_df, feature_cols)
    val_dataset   = TabularDataset(val_df, feature_cols)
    test_dataset  = TabularDataset(test_df, feature_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Modelo
    input_dim = len(feature_cols)
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=config['model']['classifier_hidden_dim'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['classifier_dropout']
    ).to(device)
    
    # Pesos de clase
    from sklearn.utils.class_weight import compute_class_weight
    labels = train_df['class_idx'].values
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
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
            torch.save(model.state_dict(), output_uni / 'best_model_tabular.pth')
    
    model.load_state_dict(torch.load(output_uni / 'best_model_tabular.pth'))
    test_acc, preds, labels = test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    save_results(output_uni, history, test_acc, preds, labels, scaler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)