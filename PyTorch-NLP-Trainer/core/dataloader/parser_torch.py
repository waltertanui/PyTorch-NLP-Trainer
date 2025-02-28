# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @Date   : 2019-9-20 13:18:34
# --------------------------------------------------------
"""

import os
import math
import numpy as np
import jieba
from torchtext import data
from torchtext.vocab import Vectors


# 分词
def tokenizer(text):
    return [word for word in jieba.lcut(text) if word not in stop_words]


# 去停用词
def get_stop_words(file):
    file_object = open(file, encoding='utf-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words


if __name__ == '__main__':
    stop_file = "/home/dm/nasdata/release/CSDN/PyTorch-NLP-Trainer/data/stopwords.txt"
    stop_words = get_stop_words(stop_file)  # 加载停用词表
    text = data.Field(sequential=True,
                      lower=True,
                      tokenize=tokenizer,
                      stop_words=stop_words)
    label = data.Field(sequential=False)
    train, val = data.TabularDataset.splits(
        path='data/',
        skip_header=True,
        train='train.tsv',
        validation='validation.tsv',
        format='tsv',
        fields=[('index', None), ('label', label), ('text', text)],
    )
    print(train)
