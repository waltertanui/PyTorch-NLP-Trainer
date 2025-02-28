# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-04-13 15:12:08
"""

from core.dataloader import parser_textDataset


def load_dataset(data_type,
                 filename,
                 vocab_file,
                 context_size,
                 class_name=None,
                 resample=False,
                 shuffle=False,
                 check=False,
                 phase="train",
                 **kwargs):
    """
    :param data_type: 加载数据DataLoader方法
    :param filename: 数据文件
    :param vocab_file:  字典文件(会根据训练数据集自动生成)
    :param context_size: 句子长度
    :param class_name: 类别
    :param resample: 是否重采样
    :param shuffle: 是否打乱顺序
    :param check: 是否对数据进行检测
    :param phase: 数据处理模式(train,test,val)
    :param kwargs:
    :return:
    """
    is_train = phase == "train"
    if data_type.lower() == "text_dataset":  # Changed from "textdata" to "text_dataset"
        dataset = parser_textDataset.TextDataset(filename,
                                                 vocab_file=vocab_file,
                                                 context_size=context_size,
                                                 is_train=is_train,
                                                 resample=resample,
                                                 shuffle=shuffle)
    else:
        raise Exception("Error:data_type:{}".format(data_type))
    return dataset
