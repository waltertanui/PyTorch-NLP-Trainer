# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : segment.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2017-05-11 17:51:53
"""

import jieba
import os
import io
import math
import re
from pybaseutils import file_utils


# jieba.enable_parallel()

def get_common_stop_words():
    """
    常用的需要ignore的字词
    :return:
    """
    # stop_words=['\n','',' ','  ','\n\n']
    stop_words = ['\n', '', ' ', '\n\n', '　', '\t', '\"', '']
    stop_words += ["“", "”"]
    stop_words += [",", "，", ":", "：", ".", "。", "?", "？", "！", "!", "；", ";"]
    return stop_words


def load_userdict(file):
    if file: jieba.load_userdict(file)


def padding_words(words, context_size, pad_token="<pad>"):
    """
    :param words:
    :param label:
    :param context_size:
    :param pad_token:
    :return:
    """
    context_size = int(context_size)
    pad = context_size - len(words)
    if pad > 0 and pad_token:
        words = [pad_token] * pad + words
    if context_size > 0:
        words = words[0:context_size]
    return words


def cut_words(content):
    """
    按字词word进行分割
    :param content: str
    :return:
    """
    words = list(jieba.cut(content))
    return words


def delete_words(lines_list, stop_words=[]):
    """

    :param lines_list:
    :param stop_words:
    :return:
    """
    sentence_segment = []
    for word in lines_list:
        if word not in stop_words:
            sentence_segment.append(word)
    return sentence_segment


def cut_content_word(content, stop_words=[], only_chinese=False):
    """
    :param content:
    :param stop_words:
    :param only_chinese:
    :return:
    """
    words = cut_words(content)
    if only_chinese: words = [get_string_chinese(w) for w in words]
    words = delete_words(words, stop_words)
    return words


def cut_content_char(content, stop_words=[], only_chinese=True):
    """
    new = re.sub('([^\u4e00-\u9fa5])', '', old) # 字符串删掉除汉字以外的所有字符
    new = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', old) # 字符串删掉除汉字和数字以外的其他字符
    按字符char进行分割
    :param content:
    :return:
    """
    words = clean_str(seperate_line(content))
    words = words.split(' ')
    if only_chinese: words = [get_string_chinese(w) for w in words]
    words = delete_words(words, stop_words)
    return words


def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string.strip()


def seperate_line(line):
    return ''.join([word + ' ' for word in line])


def combine_files_content(file_list, out_file):
    """
    合并输出一个文件
    :param file_list:
    :param out_file:
    :return:
    """
    f2 = open(out_file, 'wb')
    for i, file in enumerate(file_list):
        with io.open(file, encoding='utf8') as file:
            lines = file.readlines()
            lines = ''.join(lines)
        result = ' '.join(lines)
        result += '\n'
        result = result.encode('utf-8')
        f2.write(result)
    f2.close()


def is_chinese(uchar):
    """
    判断一个字符uchar是否为汉字
    :param uchar:一个字符，如"我"
    :return: True or False
    """
    for ch in uchar:
        if '\u4e00' <= ch <= '\u9fff': return True
    return False


def get_string_chinese(string, repl=""):
    """
    https://zhuanlan.zhihu.com/p/407918235
    获得字符串中所有汉字，其他字符删除
    new = re.sub('([^\u4e00-\u9fa5])', '', old) # 字符串删掉除汉字以外的所有字符
    :param string:
    :param repl:
    :return:
    """
    new = re.sub('([^\u4e00-\u9fa5])', repl, string)  # 字符串删掉除汉字以外的所有字符
    return new


def get_string_chinese_number(string, repl=""):
    """
    获得字符串中所有汉字和数字，其他字符删除，PS小数点也会被删除
    new = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', old) # 字符串删掉除汉字和数字以外的其他字符
    :param string:
    :param repl:
    :return:
    """
    new = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', repl, string)  # 字符串删掉除汉字和数字以外的其他字符
    return new


def match_string_chinese_number(string):
    new = re.match(r"[a-zA-z]", string)
    new = new.group() if new else new
    return new


def remove_string_special_characters(string, repl=""):
    """
    string = "你3.39好@、/、小，*、明&，在 %%%么100（）"
    去除所有特殊字符
    :param string:
    :param repl:
    :return:
    """
    # new = re.sub(r"[^\w]", repl, string)  # 删除特殊字符，数字除外
    new = re.sub('[0-9’!"#$%&\'()（）*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', repl, string)  # 删除特殊字符，数字也删除
    return new


if __name__ == '__main__':
    # 多线程分词
    # jieba.enable_parallel()
    # 加载自定义词典
    # user_path = '../data/user_dict.txt'
    # jieba.load_userdict(user_path)

    stop_words = get_common_stop_words()

    # file_dir='../data/source2'
    file_dir = '/home/ubuntu/project/tfTest/THUCNews/THUCNews'

    segment_out_dir = '../../modules/word2vec/data/cutwords'
    files_list = file_utils.get_files_list(file_dir, postfix='*.txt')