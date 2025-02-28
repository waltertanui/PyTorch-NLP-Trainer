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
import random
import torch
import copy
import json  # Add this import
from tqdm import tqdm
from torch.utils.data import Dataset
from pybaseutils import file_utils
from core.dataloader import balanced_classes, data_resample, augment
from core.utils import nlp_utils, jieba_utils

user_file = "data/text/user_dict.txt"
if os.path.exists(user_file):
    print("加载用户词汇:{}".format(user_file))
    jieba_utils.load_userdict(user_file)


class TextDataset(Dataset):
    @staticmethod
    def read_json_with_utf8(vocab_file):
        """Read JSON file with UTF-8 encoding"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        return json_data

    def __init__(self, data_root, vocab_file="", context_size=2, is_train=False, resample=False, shuffle=True):
        """
        :param data_root: [text_dir]->list or `path/to/text_dir`->str
        :param vocab_file: 字典文件
        :param context_size: 句子长度
        :param class_name: 类别
        :param is_train: 是否是训练模式
        :param resample: 是否使用重采样
        :param shuffle: 是否打乱顺序
        """
        if not vocab_file: vocab_file = os.path.join("./vocabulary.json")
        self.pad_token = '<pad>'
        self.context_size = context_size
        self.is_train = is_train
        self.resample = resample
        self.shuffle = shuffle
        self.vocab_file = vocab_file
        file_list = self.get_text_files(data_root, shuffle=False)
        sentences, vocabulary = self.get_vocabulary(file_list, self.context_size,
                                                    padding=self.pad_token, shuffle=self.shuffle)
        # 接着我们建立训练集，遍历整个语料库，将单词三个分组，前面context_size个作为输入，最后一个作为预测的结果。
        self.item_list = self.get_item_list(sentences, context_size=self.context_size,
                                            padding=self.pad_token, shuffle=self.shuffle)
        # self.item_list = self.item_list * 10
        if self.resample:
            self.data_resample = data_resample.DataResample(self.item_list,
                                                            label_index=1,
                                                            balance="mean",
                                                            shuffle=shuffle,
                                                            disp=False)
            self.item_list = self.data_resample.update(True)
            class_count = self.data_resample.class_count  # resample前，每个类别的分布
            balance_nums = self.data_resample.balance_nums  # resample后，每个类别的分布

        self.classes_weights = self.get_classes_weights(label_index=1)
        # 建立每个词与数字的编码，据此构建词嵌入
        if self.is_train:
            print("save vocabulary file          : {}".format(vocab_file))
            self.vocabulary = list(vocabulary.keys())  # 使用set将重复的元素去掉
            json_data = {"vocabulary": self.vocabulary, }
            file_utils.write_json_path(vocab_file, json_data=json_data)
        else:
            assert os.path.exists(vocab_file), Exception("Error: vocabulary file not exists: {}".format(vocab_file))
            print("load vocabulary file          : {}".format(vocab_file))
            json_data = self.read_json_with_utf8(vocab_file)  # Changed this line
            self.vocabulary = json_data["vocabulary"]
        self.word_to_idx = {word: i for i, word in enumerate(self.vocabulary)}
        self.idx_to_word = {self.word_to_idx[word]: word for word in self.word_to_idx}
        self.num_classes = len(self.word_to_idx)
        self.num_embeddings = len(self.word_to_idx)
        self.class_name = list(self.word_to_idx.keys())

        print("embedding_size(num_classes)   : {}".format(self.num_classes))
        print("embedding shape               : ({},{})".format(self.num_classes, context_size))
        print("context_size                  : {}".format(context_size))
        print("dataset nums                  : {}".format(len(self.item_list)))

    @staticmethod
    def get_vocabulary(file_list, context_size, padding="<pad>", shuffle=False):
        """
        获得词典
        :param file_list:
        :param context_size:
        :return: padding
        """
        sentences = []
        vocabulary = {}
        for file in tqdm(file_list):
            sentence = nlp_utils.read_data(file)
            sentence = nlp_utils.get_sentences_splitword(sentence, context_size=context_size + 1, only_chinese=True,
                                                         padding=None, min_nums=2)
            sentence_ = []
            for line in sentence:
                line_ = []
                for word in line:
                    if not word: continue
                    line_.append(word)
                    try:
                        vocabulary[word] += 1
                    except Exception as e:
                        vocabulary[word] = 1
                sentence_.append(line_)
            sentences += sentence_
        if shuffle:
            random.seed(100)
            random.shuffle(sentences)
        vocabulary = {n: vocabulary[n] for n in sorted(vocabulary.keys())}  # 排序
        vocabulary[padding] = 1
        return sentences, vocabulary

    @staticmethod
    def get_text_files(data_root, shuffle=True):
        """
        get text list and classes
        :param data_root:
        :param shuffle:
        :return:
        """
        if isinstance(data_root, str): data_root = [data_root]
        file_list = []
        for i, text_dir in enumerate(data_root):
            print("loading text from:{}".format(text_dir))
            if not os.path.exists(text_dir):
                raise Exception("text_dir:{}".format(text_dir))
            files = file_utils.get_files_list(text_dir, prefix="", postfix=["*.txt"])
            if files: file_list += files
        if shuffle:
            random.seed(100)
            random.shuffle(file_list)
        return file_list

    def get_item_list(self, sentences, context_size, stride=1, padding="<pad>", shuffle=False):
        """
        构建数据集
        :param sentences: 语料数据[list],一句话一个列表
        :param context_size: 句子最大长度
        :param stride: 步长，默认1
        :param padding: 不足context_size，进行填充
        :return:
        """
        item_list = []
        for content in sentences:
            pad_size = context_size + 1 - len(content)
            if pad_size > 0:
                content = [padding] * pad_size + content
            for i in range(0, len(content) - context_size, stride):
                inputs = content[i:(i + context_size)]
                target = content[i + context_size]
                item_list.append((inputs, target))
        if shuffle:
            random.seed(100)
            random.shuffle(item_list)
        return item_list

    def __getitem__(self, index):
        """
        :param index:
        :return: embedding,label id
        """
        text, label = copy.deepcopy(self.item_list[index])  # fix a bug: 深拷贝，避免修改原始数据
        if self.is_train:
            text, label = augment.random_text_mask(text, label, len_range=(0, 2), token=self.pad_token)
        text = self.map_word_to_idx(text, unknown=self.pad_token)
        label = self.word_to_idx[label]
        if len(text) > 0:
            text = np.asarray(text, dtype=np.int32)
            label = np.asarray(label, dtype=np.int32)
            text = torch.from_numpy(text).long()
            label = torch.from_numpy(label).long()
        else:
            index = int(random.uniform(0, len(self)))
            return self.__getitem__(index)
        return text, label

    def map_word_to_idx(self, text, unknown="<pad>"):
        """
        将文本映射ID
        :param text: 输入任意文本
        :param unknown: 如果ID不存在，则映射unknown
        :return:
        """
        for i in range(len(text)):
            try:
                text[i] = self.word_to_idx[text[i]]
            except Exception as e:
                text[i] = self.word_to_idx[unknown]
        return text

    def __len__(self):
        if self.resample:
            self.item_list = self.data_resample.update(True)
        self.num_sample = len(self.item_list)
        return self.num_sample

    def get_classes_weights(self, label_index=1):
        """
        :param label_index:
        :return:
        """
        labels_list = []
        for item in self.item_list:
            label = item[label_index]
            labels_list.append(label)
        # weight = balanced_classes.create_sample_weight_torch(labels_list)
        weight = balanced_classes.create_class_sample_weight_custom(labels_list,
                                                                    balanced="auto",
                                                                    weight_type="sample_weight")
        return weight


if __name__ == '__main__':
    data_root = "PyTorch-NLP-Trainer/data/text/data"
    dataset = TextDataset(data_root, context_size=5, is_train=True, resample=False, shuffle=False)
    for index in range(len(dataset)):
        index = 1
        text, label = dataset.__getitem__(index)
        print(len(text), text, label)
