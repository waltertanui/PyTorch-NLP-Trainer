# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-01-17 17:46:38
"""

import os
import torch
import torch.nn as nn
from torchvision import models
from core.models import textcnn, textcnnv2
from core.models import ngram
from core.models import lstm


def get_models(net_type, num_classes, num_embeddings, embedding_dim=128, context_size=None, is_train=True,
               pretrained=True, **kwargs):
    """
    :param net_type: 模型名称
    :param num_classes: 输出维度(类别数num_classes)
    :param num_embeddings: size of the dictionary of embeddings,词典的大小(vocab_size)
    :param embedding_dim:  the size of each embedding vector，词向量特征长度
    :param context_size: 句子长度
    :param is_train: 是否训练模式
    :param pretrained:
    :param kwargs:
    :return:
    """
    if net_type.lower() == "NGram".lower():
        model = ngram.NGram(num_classes=num_classes, context_size=context_size, num_embeddings=num_embeddings,
                            embedding_dim=embedding_dim)
    elif net_type.lower() == "TextCNN".lower() or net_type.lower() == "TextCNNv1".lower():
        model = textcnn.TextCNN(num_classes=num_classes, num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    elif net_type.lower() == "TextCNNv2".lower():
        model = textcnnv2.TextCNNv2(num_classes=num_classes, num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    elif net_type.lower() == "LSTM".lower():
        model = lstm.LSTMNet(num_classes=num_classes, num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                             bidirectional=False)
    elif net_type.lower() == "BiLSTM".lower():
        model = lstm.LSTMNet(num_classes=num_classes, num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                             bidirectional=True)
    elif net_type.lower() == "Transformer".lower():
        # Extract transformer-specific parameters from kwargs
        num_heads = kwargs.get('num_heads', 8)
        num_layers = kwargs.get('num_layers', 6)
        dropout = kwargs.get('dropout', 0.1)
        
        from core.models.transformer import TransformerModel  # Import here to avoid circular imports
        model = TransformerModel(
            num_classes=num_classes, 
            num_embeddings=num_embeddings, 
            context_size=context_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise Exception("Error: net_type:{}".format(net_type))
    
    model = model.train() if is_train else model.eval()
    return model


if __name__ == "__main__":
    from core.utils import torch_tools

    device = "cuda:0"
    batch_size = 4
    num_classes = 2  # 输出类别
    context_size = 128  # 句子长度（字词个数）
    num_embeddings = 1024  # 词典的大小(vocab_size)
    embedding_dim = 128  # 词向量特征长度

    x = torch.ones(size=(batch_size, context_size)).long().to(device)
    net_type = 'TextCNNv2'
    model = get_models(net_type, num_classes, num_embeddings, embedding_dim=embedding_dim, context_size=context_size)
    model = model.to(device)
    model.eval()
    out = model(x)
    print("x.shape:{}".format(x.shape))
    print("out.shape:{}".format(out.shape))
    torch_tools.nni_summary_model(model, inputs=x, plot=True)
