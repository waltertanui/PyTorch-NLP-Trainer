# -*- coding: utf-8 -*-
"""
    @Author : [Your Name]
    @E-mail : [Your Email]
    @Date   : 2025-03-01
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, num_classes, num_embeddings, context_size, embedding_dim=512, num_heads=8, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, context_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Apply transformer encoder
        x = self.transformer(x)  # [batch_size, seq_len, embedding_dim]
        
        # Global pooling (mean)
        x = x.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Classification
        x = self.fc(x)  # [batch_size, num_classes]
        return x