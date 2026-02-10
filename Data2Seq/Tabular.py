# Tabular.py - Versión corregida
import torch
import torch.nn as nn

class TabularEmbed(nn.Module):
    """
    Simple tabular embedding for quick implementation
    Treats all features as continuous and projects them
    """
    def __init__(self, num_features=None, embed_dim=768, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # Linear projection for each feature
        self.projection = nn.Linear(num_features, embed_dim)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_features, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_features)
        """
        batch_size = x.shape[0]
        
        # Project each feature
        embeddings = self.projection(x)  # (batch_size, embed_dim)
        # Necesitamos agregar dimensión para que sea (batch_size, num_features, embed_dim)
        embeddings = embeddings.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Si queremos tratar cada feature como token separado, necesitamos un enfoque diferente
        # En su lugar, tratamos todo el vector tabular como un solo token
        # CLS token ya es (1, 1, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        
        # Concatenar CLS token con embeddings
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)  # (batch_size, 2, embed_dim)
        
        # Agregar posición para CLS y para el token tabular
        # self.pos_embedding es (1, num_features, embed_dim) pero solo necesitamos 2 posiciones
        # Tomamos las primeras 2 posiciones
        pos_embed = self.pos_embedding[:, :embeddings.shape[1]-1, :]  # (1, 1, embed_dim)
        pos_embed = torch.cat([torch.zeros_like(cls_tokens), pos_embed], dim=1)  # (1, 2, embed_dim)
        
        embeddings = embeddings + pos_embed
        
        # Apply layer norm and dropout
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


# Versión alternativa: Cada feature como token separado
class TabularEmbedSeparateTokens(nn.Module):
    """
    Tabular embedding where each feature is a separate token
    """
    def __init__(self, num_features=None, embed_dim=768, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # Project each feature separately (1D conv over features)
        self.projection = nn.Linear(1, embed_dim)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_features + 1, embed_dim))  # +1 for CLS
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_features)
        """
        batch_size = x.shape[0]
        
        # Reshape para tratar cada feature como entrada separada
        x = x.unsqueeze(-1)  # (batch_size, num_features, 1)
        
        # Proyectar cada feature
        embeddings = self.projection(x)  # (batch_size, num_features, embed_dim)
        
        # Agregar CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)  # (batch_size, num_features + 1, embed_dim)
        
        # Agregar embeddings posicionales
        embeddings = embeddings + self.pos_embedding[:, :embeddings.shape[1], :]
        
        # Aplicar normalización y dropout
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings