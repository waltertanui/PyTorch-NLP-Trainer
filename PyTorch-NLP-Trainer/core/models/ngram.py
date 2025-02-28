# -*-coding: utf-8 -*-
"""
    @Author : 390737991
    @E-mail : 390737991@163.com
    @Date   : 2022-11-01 17:54:33
    @Brief  :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NGram(nn.Module):
    """N-Gram模型m"""

    def __init__(self, num_classes, context_size, num_embeddings, embedding_dim=128, embeddings_pretrained=None):
        """
        :param num_classes: 输出维度(类别数num_classes)
        :param context_size: 句子长度
        :param num_embeddings: size of the dictionary of embeddings,词典的大小(vocab_size)
        :param embedding_dim:  the size of each embedding vector，词向量特征长度
        :param embeddings_pretrained: embeddings pretrained参数，默认None
        :return:
        """
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        super(NGram, self).__init__()
        # embedding层
        if self.num_embeddings > 0:
            # embedding之后的shape: torch.Size([200, 8, 300])
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            if embeddings_pretrained is not None:
                self.embedding = self.embedding.from_pretrained(embeddings_pretrained, freeze=False)
        self.classify = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim * context_size, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        if self.num_embeddings > 0:
            x = self.embedding(x)  # 得到词嵌入
        x = x.view(x.size(0), -1)  # 将两个词向量拼在一起
        out = self.classify(x)
        return out


if __name__ == "__main__":
    batch_size = 2
    num_classes = 100
    num_embeddings = num_classes  # 预测的类别数目和单词数目一样
    context_size = 8  # 句子长度，即最大依赖的单词数目
    input = torch.ones(batch_size, context_size).long().cuda()
    model = NGram(num_classes, context_size, num_embeddings=num_embeddings, embedding_dim=64).cuda()
    print(model)
    out = model(input)
    print(out)
    print("input", input.shape)
    print("out  ", out.shape)
