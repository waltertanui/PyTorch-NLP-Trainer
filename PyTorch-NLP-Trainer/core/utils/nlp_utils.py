# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-11-08 19:41:37
    @Brief  :
"""
import os
import io
import re
import math
from tqdm import tqdm
from core.utils import jieba_utils
from pybaseutils import file_utils


def read_text_file(filename):
    """读取文本数据"""
    text = read_data(filename)
    text = '\n'.join(text)
    return text


def read_data(filename):
    """
    读取txt数据函数
    :param filename:文件名
    :param convertNum :是否将list中的string转为int/float类型的数字
    :return: txt的数据列表
    Python中有三个去除头尾字符、空白符的函数，它们依次为:
    strip： 用来去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    lstrip：用来去除开头字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    注意：这些函数都只会删除头和尾的字符，中间的不会删除。
    """
    with open(filename, mode="r", encoding='utf-8') as f:
        content_list = f.readlines()
    content_list = [content.strip() for content in content_list if content.strip()]
    return content_list


def get_sentences_splitword(sentences: list, context_size, use_word=True, cutting=False, min_nums=1, only_chinese=False,
                            split='[|,|，| |:|：|.|。|?]|？|！|!|；|;|]', stop_words=["“", "”", "《", "》"], padding="<pad>"):
    """
    :param sentence:
    :param context_size:
    :param use_word:
    :param cutting:
    :param min_nums: 句子最新字词数目
    :param split   :分割符 re.split('[分隔符1|分隔符1|分隔符1]', content)
    :param padding:
    :return:
    """
    sentences1 = []
    for line in sentences:  # 分割字词
        line = re.split(split, line)
        sentences1 += line
    stop_words = stop_words + split.split("|")  #
    sentences2 = get_sentences_cutword(sentences1, context_size, use_word=use_word, cutting=cutting, min_nums=min_nums,
                                       stop_words=stop_words, only_chinese=only_chinese, padding=padding)
    return sentences2


def get_sentences_cutword(sentences: list, context_size, use_word=True, cutting=False, min_nums=1,
                          stop_words=jieba_utils.get_common_stop_words(), only_chinese=False, padding="<pad>"):
    """
    :param sentence:
    :param context_size:
    :param use_word: True对句子进行分词;否则按照字符进行分割
    :param cutting: 是否裁剪掉超过context_size的部分内容
    :param min_nums: 句子最新字词数目
    :param split   :分割符 re.split('[分隔符1|分隔符1|分隔符1]', content)
    :param only_chinese: True只保留中文内容
    :param padding:
    :return:
    """
    if use_word:
        sentences = [jieba_utils.cut_content_word(line, stop_words, only_chinese) for line in sentences if line]
    else:
        sentences = [jieba_utils.cut_content_char(line, stop_words, only_chinese) for line in sentences if line]
    sentences1 = []
    for content in sentences:  # 分割字词
        content = [c for c in content if c]
        if len(content) < min_nums: continue
        content = [w.lower() for w in content]  # 全部转为小写
        pad_size = context_size - len(content)
        if padding and pad_size > 0:
            content = [padding] * pad_size + content
        if cutting and pad_size < 0:
            content = content[-context_size:]
        sentences1.append(content)
    return sentences1


def get_file_sentences_cutword(file_list, stop_words=[], use_word=True, only_chinese=False):
    """
    字词分割
    :param file_list:
    :param stop_words:
    :param segment_type: word or char，选择分割类型，按照字符char，还是字词word分割
    :return:
    """
    sentences = []
    for i, file in enumerate(file_list):
        text = read_text_file(file)
        if use_word:
            sentence = jieba_utils.cut_content_word(text, stop_words, only_chinese)
        else:
            sentence = jieba_utils.cut_content_char(text, stop_words, only_chinese)
        sentences.append(sentence)
    return sentences


def get_files_sentences_cutword(file_dir, word_out, stop_words=[], block_size=10000):
    """
    批量分割文件字词，并将batchSize的文件合并一个文件
    :param file_dir: *.txt文件目录,支持多个文件
    :param word_out: jieba字词分割文件输出的目录
    :param stop_words: 停用词,用于ignore的字词
    :param block_size: 块合并的数量，默认10000个文件合并为一个jieba字词文件
    :return:
    """
    file_utils.create_dir(word_out)  # 创建输出的目录
    file_list = file_utils.get_files_list(file_dir, postfix=['*.txt'])  # 读取所有*.txt文件
    nums = len(file_list)
    batch_nums = int(math.ceil(1.0 * nums / block_size))
    print("have total files:{},batch_nums：{}".format(nums, batch_nums))
    cutword_files = []
    for i in tqdm(range(batch_nums)):
        out_file = os.path.join(word_out, 'cutwords_{:0=4d}.txt'.format(i))
        start = i * block_size
        end = min((i + 1) * block_size, nums)
        batch_file = file_list[start:end]
        content_list = get_file_sentences_cutword(batch_file, stop_words)
        # content_list=padding_sentences(content_list, padding_token='<PAD>', padding_sentence_length=15)
        file_utils.write_data(out_file, content_list, split=" ", mode='w')
        cutword_files.append(out_file)
        print("cut words files:{}".format(out_file))
    return cutword_files
