# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @Date   : 2019-9-20 13:18:34
# --------------------------------------------------------
"""
import math
import random
from typing import List


def random_text_crop(text: List, label, context_size, token="<pad>", p=0.5):
    """
    句⼦中的每个词，以概率p随机截取
    :param text:
    :param label:
    :param context_size:
    :param token:
    :param p:
    :return:
    """
    context_size = int(context_size)
    nums = len(text)
    pad = context_size - nums
    if pad > 0 and token:
        text = [token] * pad + text
    if random.random() < p and pad < 0:
        start = random.randint(0, nums - context_size)
        text = text[start:start + context_size]
    elif len(text) > context_size:
        text = text[0:context_size]
    return text, label


def random_text_mask(text: List, label, len_range=(0, 2), token="<pad>", p=0.5):
    """
    句⼦中的每个词，以概率p替换成token
    :param text:
    :param label:
    :param len_range:
    :param p:
    :return:
    """
    if random.random() < p and len(text) > 2 * len_range[1]:
        nums = math.ceil(random.uniform(len_range[0], len_range[1]))
        for i in range(nums):
            index = int(random.uniform(0, len(text) - 1))
            text[index] = token
    return text, label


def random_text_delete(text: List, label, len_min, p=0.5):
    """
    句⼦中的每个词，以概率p随机删除
    :param text:
    :param label:
    :param len_min: 句子最小长度，低于该值，不会删除
    :param p:
    :return:
    """
    if random.random() < p and len(text) > len_min:
        nums = int(random.uniform(0, len(text) - len_min))
        for i in range(nums):
            index = int(random.uniform(0, len(text)))
            del text[index]
    return text, label


def random_text_insert(text: List, label, len_range=(0, 2), token="<pad>", p=0.5):
    """
    句⼦中的每个词，以概率p随机插入
    :param text:
    :param label:
    :param len_range: 
    :param p:
    :return:
    """
    if random.random() < p and len(text) > 2 * len_range[1]:
        nums = math.ceil(random.uniform(len_range[0], len_range[1]))
        for i in range(nums):
            index = int(random.uniform(0, len(text) - 1))
            text.insert(index, token)
    return text, label


if __name__ == '__main__':
    label = 1
    context_size = 10
    pad_token = "<pad>"
    p = 10
    for i in range(10):
        text = "我是一名中国人，我爱中国，我的家乡在广东"
        text = "_".join(text).split("_")
        len_range = (0, context_size // 4)
        # text, label = random_text_crop(text, label, 1.8 * context_size, token=None, p=0.8)
        # text, label = random_text_delete(text, label, len_min=1.5 * context_size)
        text, label = random_text_insert(text, label, len_range=len_range, token=pad_token)
        # text, label = random_text_mask(text, label, len_range=len_range, token=pad_token)
        # text, label = random_text_crop(text, label, context_size, token=pad_token, p=0.8)
        print(text, len(text))
