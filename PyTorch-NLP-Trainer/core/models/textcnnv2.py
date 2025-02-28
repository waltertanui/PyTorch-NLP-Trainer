# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-11-01 17:54:33
    @Brief  :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])  # shape: (batch_size, channel, 1)


class TextCNNv2(nn.Module):
    def __init__(self, num_classes, num_embeddings=-1, embedding_dim=128, strides=[1, 2, 2], kernel_sizes=[3, 4, 5, 6],
                 num_channels=[256, 256, 256, 256], embeddings_pretrained=None):
        """
        :param num_classes: 输出维度(类别数num_classes)
        :param num_embeddings: size of the dictionary of embeddings,词典的大小(vocab_size),
                               当num_embeddings<0,模型会去除embedding层
        :param embedding_dim:  the size of each embedding vector，词向量特征长度
        :param kernel_sizes: CNN层卷积核大小
        :param num_channels: CNN层卷积核通道数
        :param embeddings_pretrained: embeddings pretrained参数，默认None
        :return:
        """
        super(TextCNNv2, self).__init__()
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        # embedding层
        if self.num_embeddings > 0:
            # embedding之后的shape: torch.Size([200, 8, 300])
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            if embeddings_pretrained is not None:
                self.embedding = self.embedding.from_pretrained(embeddings_pretrained, freeze=False)
        # 卷积层
        in_features = []
        self.cnn_layers = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            stride = strides[0]
            in_channels = embedding_dim
            out_channels = c * stride
            cnn = [nn.Conv1d(in_channels=in_channels,
                             out_channels=out_channels,
                             stride=stride,
                             kernel_size=k),
                   nn.BatchNorm1d(out_channels),
                   nn.ReLU(inplace=True)]
            for stride in strides[1:]:
                in_channels = out_channels
                out_channels = in_channels * stride
                cnn += [nn.Conv1d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  stride=stride,
                                  kernel_size=k),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=True)]
            in_features.append(out_channels)
            self.cnn_layers.append(nn.Sequential(*cnn))

        # 最大池化层
        self.pool = GlobalMaxPool1d()
        # 输出层
        self.classify = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(sum(in_features), self.num_classes)
        )
        # 初始化模型参数
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        """
        :param input:  (batch_size, context_size, embedding_size(in_channels))
        :return:
        """
        if self.num_embeddings > 0:
            # 得到词嵌入(b,context_size)-->(b,context_size,embedding_dim)
            input = self.embedding(input)
            # (batch_size, context_size, channel)->(batch_size, channel, context_size)
        input = input.permute(0, 2, 1)
        y = []
        for layer in self.cnn_layers:
            x = layer(input)  # torch.Size([4, 6, 11])->torch.Size([4, 4, 10])
            x = self.pool(x).squeeze(-1)
            y.append(x)
        y = torch.cat(y, dim=1)
        out = self.classify(y)
        return out


if __name__ == "__main__":
    device = "cuda:0"
    batch_size = 4
    num_classes = 2  # 输出类别
    context_size = 128  # 句子长度（字词个数）
    num_embeddings = 1024  # 词典的大小(vocab_size)
    embedding_dim = 128  # 词向量特征长度

    input = torch.ones(size=(batch_size, context_size)).long().to(device)
    model = TextCNNv2(num_classes=num_classes,
                      num_embeddings=num_embeddings,
                      embedding_dim=embedding_dim,
                      strides=[1, 2, 2],
                      # kernel_sizes=kernel_sizes,
                      # num_channels=num_channels,
                      )
    model = model.to(device)
    model.eval()
    output = model(input)
    print("-----" * 10)
    print(model)
    print("-----" * 10)
    print(" input.shape:{}".format(input.shape))
    print("output.shape:{}".format(output.shape))
    print("-----" * 10)
