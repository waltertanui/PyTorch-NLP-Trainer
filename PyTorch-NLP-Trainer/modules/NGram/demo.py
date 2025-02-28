# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-11-02 08:58:38
    @Brief  :
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


class NGram(nn.Module):
    """N-Gram模型m"""

    def __init__(self, vocab_size, context_size, n_dim):
        super(NGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_dim)
        self.classify = nn.Sequential(
            nn.Linear(context_size * n_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, vocab_size)
        )

    def forward(self, x):
        emb = self.embedding(x)  # 得到词嵌入
        emb = emb.view(emb.size(0), -1)  # 将两个词向量拼在一起
        out = self.classify(emb)
        return out


class TextDataset(object):
    def __init__(self, context_size):
        # 我们使用莎士比亚的诗
        test_sentence = """When forty winters shall besiege thy brow,
        And dig deep trenches in thy beauty's field,
        Thy youth's proud livery so gazed on now,
        Will be a totter'd weed of small worth held:
        Then being asked, where all thy beauty lies,
        Where all the treasure of thy lusty days;
        To say, within thine own deep sunken eyes,
        Were an all-eating shame, and thriftless praise.
        How much more praise deserv'd thy beauty's use,
        If thou couldst answer 'This fair child of mine Shall sum my count, and make my old excuse,'
        Proving his beauty by succession thine!
        This were to be new made when thou art old,
        And see thy blood warm when thou feel'st it cold."""
        test_sentence = test_sentence.split()
        # 接着我们建立训练集，便利整个语料库，将单词三个分组，前面两个作为输入，最后一个作为预测的结果。
        self.dataset = []
        for i in range(len(test_sentence) - context_size):
            inputs = test_sentence[i:(i + context_size)]
            target = test_sentence[i + context_size]
            self.dataset.append([inputs, target])
        # 建立每个词与数字的编码，据此构建词嵌入
        vocb = set(test_sentence)  # 使用 set 将重复的元素去掉
        self.word_to_idx = {word: i for i, word in enumerate(vocb)}
        self.idx_to_word = {self.word_to_idx[word]: word for word in self.word_to_idx}
        self.vocab_size = len(self.word_to_idx)

    def __getitem__(self, index):
        """
        :param index:
        :return: embedding,label id
        """
        input, label = self.dataset[index]
        input = [self.word_to_idx[n] for n in input]
        label = self.word_to_idx[label]
        input = np.asarray(input, dtype=np.int32)
        label = np.asarray(label, dtype=np.int32)
        input = torch.from_numpy(input).long()
        label = torch.from_numpy(label).long()
        return input, label

    def __len__(self):
        self.num_sample = len(self.dataset)
        return self.num_sample


class Trainer(object):
    def __init__(self, CONTEXT_SIZE=2, EMBEDDING_DIM=10):
        """
        :param CONTEXT_SIZE: 依据的单词数
        :param EMBEDDING_DIM: 词向量的维度
        """
        self.batch_size = 4
        self.num_workers = 0
        self.epochs = 100
        self.dataset = self.build_train_loader()
        self.word_to_idx = self.dataset.dataset.word_to_idx
        self.idx_to_word = self.dataset.dataset.idx_to_word
        self.model = NGram(vocab_size=self.dataset.dataset.vocab_size, context_size=CONTEXT_SIZE, n_dim=EMBEDDING_DIM)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

    def build_train_loader(self, ) -> DataLoader:
        """build_train_loader"""
        dataset = TextDataset(context_size=2)

        loader = DataLoader(dataset=dataset, batch_size=self.batch_size,
                            shuffle=True, num_workers=self.num_workers)
        return loader

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for inputs, labels in self.dataset:  # 使用前 100 个作为训练集
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            train_loss += loss.item()
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if (epoch + 1) % 20 == 0:
            print('epoch: {}, Loss: {:.6f}'.format(epoch + 1, train_loss / len(self.dataset)))

    def predict(self, words, labels):
        self.model.eval()
        # 测试一下结果
        inputs = []
        for i in range(len(words)):
            input = [[self.word_to_idx[i] for i in words[i]]]
            inputs.append(input)
        inputs = np.asarray(inputs, dtype=np.int32)
        inputs = torch.from_numpy(inputs).long()
        outputs = self.model(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        pred_score, pred_index = torch.max(outputs, dim=1)
        for i in range(len(pred_index)):
            label = self.idx_to_word[int(pred_index[i])]
            score = pred_score[i]
            true_label = labels[i]
            print("words:{},true_label:{},pred_label:{},pred_score:{}".format(words[i], true_label, label, score))

    def run(self):
        """开始运行"""
        for epoch in range(self.epochs):
            self.train(epoch)
        words = [('so', 'gazed'), ('my', 'old')]
        labels = ["on", "excuse"]
        self.predict(words, labels)


if __name__ == "__main__":
    train = Trainer()
    train.run()
