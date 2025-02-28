# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : 390737991@qq.com
    @Date   : 2022-09-26 14:50:34
    @Brief  :
"""
import os
import re
import sys
import argparse
import time
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from core.models import build_models
from core.dataloader import parser_textDataset
from pybaseutils import file_utils, config_utils, numpy_utils
from core.utils import nlp_utils, jieba_utils
import jieba

user_file = "data/text/user_dict.txt"
if os.path.exists(user_file):
    print("加载用户词汇:{}".format(user_file))
    jieba_utils.load_userdict(user_file)

class Predictor(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.device == 'cuda' else "cpu")
        self.vocabulary = file_utils.read_json_data(self.cfg.vocab_file)["vocabulary"]
        self.word_to_idx = {word: i for i, word in enumerate(self.vocabulary)}
        self.idx_to_word = {self.word_to_idx[word]: word for word in self.word_to_idx}
        self.model = self.build_model().to(self.device)
        self.word_vectors = self.word_to_idx
        print("load vocabulary file          : {}".format(self.cfg.vocab_file))
        self.context_size = 8  # 确保配置中有 context_size 参数

    def build_model(self):
        model = build_models.get_models(net_type=self.cfg.net_type,
                                        num_classes=len(self.word_to_idx),
                                        num_embeddings=len(self.word_to_idx),
                                        context_size=self.cfg.context_size,
                                        embedding_dim=128,
                                        is_train=False)
        state_dict = torch.load(self.cfg.model_file, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def pre_process(self, inputs):
        if not isinstance(inputs, list): inputs = [inputs]
        input_tensors = []
        for input_ in inputs:
            if isinstance(input_, str):
                input_ = [input_]  # 将单个字符包装成列表
            input_tensors.append([self.word_to_idx.get(word, 0) for word in input_])
        input_tensors = np.asarray(input_tensors, dtype=np.int32)
        input_tensors = torch.from_numpy(input_tensors).long().to(self.device)
        return input_tensors

    def post_process(self, output, topk=5):
        prob_scores = self.softmax(output.cpu().numpy(), axis=1)
        pred_score, pred_index = numpy_utils.get_topK(prob_scores, k=topk)
        return [[self.idx_to_word[idx] for idx in idxs] for idxs in pred_index], pred_score

    @staticmethod
    def softmax(x, axis=1):
        row_max = x.max(axis=axis, keepdims=True)
        x = x - row_max
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        return x_exp / x_sum

    def forward(self, input_tensor):
        with torch.no_grad():
            return self.model(input_tensor)


    def inference(self, inputs):
        """
        :param inputs:
        :return:pred_index: predict label index
                pred_score: predict label score
        """
        # 图像预处理
        input_tensor = self.pre_process(inputs)
        output = self.forward(input_tensor)
        # 模型输出后处理
        pred_index, pred_score = self.post_process(output.cpu())
        pred_index = self.label2class_name(pred_index)
        return pred_index, pred_score

    def label2class_name(self, pred_index):
        if isinstance(pred_index, np.ndarray): pred_index = pred_index.tolist()
        for i in range(len(pred_index)):
            pred_index[i] = [self.idx_to_word[w] for w in pred_index[i]]
        return pred_index

    def predict(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]  # 如果输入是单个字符串，将其转换为字符串列表
        input_tensor = self.pre_process(inputs)
        output = self.forward(input_tensor)
        pred_index, pred_score = self.post_process(output)
        return pred_index, pred_score

    def tokenize_and_pad(self, sentence):
        # 使用jieba分词
        words = jieba.lcut(sentence)
        # 编码分词
        indexed_words = [self.vocab.get(word, self.vocab.get('<UNK>')) for word in words]
        # 填充序列
        padded_words = pad_sequence([torch.tensor(indexed_words, dtype=torch.long)],
                                   padding_value=self.vocab.get('<PAD>'),
                                   batch_first=True)
        return padded_words

    def predict_next_word(self, word_seq):
        if not word_seq:
            print("No input for prediction")
            return None

        word_seq_encoded = torch.tensor([self.word_to_idx.get(wd, 0) for wd in word_seq], dtype=torch.long).unsqueeze(
            0).to(self.device)
        with torch.no_grad():
            prediction = self.model(word_seq_encoded)
        predicted_index = torch.argmax(prediction, dim=1).item()
        predicted_word = self.idx_to_word.get(predicted_index, None)
        print(f"Predicted word: {predicted_word}")  # 调试信息
        return predicted_word

    '''
    def predict_unknown_words(self, sentence, placeholder='□'):
        words = jieba.lcut(sentence)
        predicted_text = ''
        unknown_words = []
        for i, word in enumerate(words):
            if word == placeholder:
                unknown_words.append(word)
            else:
                if unknown_words:
                    if i == 0:  # 文本开始处的未知词
                        context = ['<START>'] + words[:min(i + self.context_size, len(words))]
                    elif i + 1 == len(words):  # 文本结束处的未知词
                        context = words[max(0, i - self.context_size):i]
                    else:
                        context = words[max(0, i - self.context_size):min(i + self.context_size, len(words))]

                    if context:
                        predicted_word = self.predict_next_word(context)
                        if predicted_word is not None:
                            predicted_text += predicted_word + ' '
                            print(f"Predicted word: {predicted_word}")
                            unknown_words = []
                        else:
                            print("No prediction made")
                            predicted_text += ' '.join(unknown_words) + ' '
                            unknown_words = []
                    else:
                        print("No context available for prediction")
                        predicted_text += ' '.join(unknown_words) + ' '
                        unknown_words = []
                predicted_text += word + ' '
        if unknown_words:
            if len(words) == len(unknown_words):  # 整个句子都是未知词
                context = ['<START>'] * self.context_size
            else:
                context = words[max(0, len(words) - self.context_size):len(words)]
            predicted_word = self.predict_next_word(context)
            if predicted_word is not None:
                predicted_text += predicted_word
                print(f"Predicted word: {predicted_word}")
            else:
                print("No prediction made for the last unknown words")
                predicted_text += ' '.join(unknown_words)
        return predicted_text.strip()
        '''


    def predict_unknown_words(self, sentence, placeholder='□'):
        words = jieba.lcut(sentence)
        predicted_text = ''
        unknown_words_count = sum(1 for word in words if word == placeholder)
        if unknown_words_count == 0:
            return sentence

        i = 0
        while i < len(words):
            if words[i] == placeholder:
                context = []
                if i > 0:
                    context = words[max(0, i - self.context_size):i]
                if i + 1 < len(words) and unknown_words_count <= 5:
                    context += words[i + 1:min(i + self.context_size + 1, len(words))]

                if context:
                    predicted_word = self.predict_next_word(context)
                    if predicted_word is not None:
                        predicted_text += predicted_word + ' '
                        i += 1
                    else:
                        predicted_text += placeholder + ' '
                        i += 1
                else:
                    predicted_text += placeholder + ' '
                    i += 1
            else:
                predicted_text += words[i] + ' '
                i += 1
        return predicted_text.strip()


    def get_context(self, words, unknown_words):
        # 获取上下文
        start_idx = max(0, len(words) - self.context_size)
        end_idx = min(len(words), len(unknown_words) + self.context_size)
        context = words[start_idx:end_idx]
        print(f"Context for prediction: {context}")  # 调试信息
        return context if context else None

    def save_predicted_text(self, text, output_file_path):
        predicted_text = self.predict_unknown_words(text)
        if predicted_text:  # 确保 predicted_text 非空
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(predicted_text)
        else:
            print("预测文本为空，不执行写入操作。")

    def load_word_vectors(self):
        # 假设 word_vectors 是一个已经加载的词汇向量
        self.word_vectors = np.random.rand(len(self.word_to_idx), self.embedding_dim)  # 随机初始化词向量
        self.word_vectors['□'] = np.zeros(self.embedding_dim)  # 未知字的向量



    def get_result(self, texts, pred_index, pred_score):
        print("==" * 30)
        res = []
        for i in range(len(texts)):
            if i < len(pred_index) and i < len(pred_score):
                info = "【预测结果】:{}({})".format(texts[i], pred_index[i][0])
                print(info)
                for k in range(len(pred_index[i])):
                    top = "{}\t{}\t{}\tscore:{:3.3f}".format(" " * len(info), k + 1, pred_index[i][k], pred_score[i][k])
                    print(top)
            else:
                print("没有找到预测结果")
            time.sleep(1)
            print("--" * 20)
        res = ["{}({})".format(texts[i], pred_index[i][0]) for i in range(len(texts)) if i < len(pred_index)]
        print("【返回结果】:{}".format(",".join(res)))
        print("--" * 20)
        return res
def get_parser():
    model_file = "work_space/BiLSTM_CELoss_20241016160136/model/best_model_082_0.5405.pth"
    config_file = os.path.join(os.path.dirname(os.path.dirname(model_file)), "config.yaml")
    vocab_file = os.path.join(os.path.dirname(os.path.dirname(model_file)), "vocabulary.json")
    parser = argparse.ArgumentParser(description="Inference Argument")
    parser.add_argument("-c", "--config_file", help="configs file", default=config_file, type=str)
    parser.add_argument("-m", "--model_file", help="model_file", default=model_file, type=str)
    parser.add_argument("-v", "--vocab_file", help="vocab_file", default=vocab_file, type=str)
    parser.add_argument("--device", help="cuda device id", default="cuda:0", type=str)
    parser.add_argument("--input", help="text", type=str)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cfg = config_utils.parser_config(args, cfg_updata=False)
    predictor = Predictor(cfg)

    test_file_path = 'test.txt'
    output_file_path = 'predicted_test.txt'

    with open(test_file_path, 'r', encoding='utf-8') as file:
        test_text = file.read().strip()  # 读取文本内容
    print(f"Original text: {test_text}")  # 输出原始文本

    predictor.save_predicted_text(test_text, output_file_path)  # 进行预测并保存结果

    with open(output_file_path, 'r', encoding='utf-8') as file:
        predicted_text = file.read()  # 读取预测结果
    print("Predicted text from file:", predicted_text)  # 输出预测结果

