import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    """nn.LayerNorm"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        output = (input - mu) / sigma
        return output * self.weight.expand_as(output) + self.bias.expand_as(output)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        self.final_linear = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hidden_size // self.num_heads)
        attn_weights = self.softmax(scores)
        context = torch.matmul(attn_weights, value).transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return self.final_linear(context)


class CRF(nn.Module):
    def __init__(self, tagset_size):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        # 转移矩阵参数
        self.transition_matrix = nn.Parameter(torch.randn(tagset_size, tagset_size))

    def forward(self, emissions, tags, mask):
        # emissions: batch_size x seq_len x tagset_size
        # tags: batch_size x seq_len
        # mask: batch_size x seq_len
        batch_size, seq_len, _ = emissions.size()

        # 计算转移分数
        trans_score = torch.matmul(tags.unsqueeze(2), self.transition_matrix).squeeze(2)

        # 计算路径分数
        path_score = emissions + trans_score

        # 应用mask
        path_score = path_score * mask.unsqueeze(2).float()

        # 计算所有路径的分数
        all_path_score = torch.sum(path_score, dim=1)

        # 计算标签序列的分数
        tag_seq_score = torch.sum(path_score.gather(dim=2, index=tags.unsqueeze(2)).squeeze(2), dim=1)

        # 返回所有路径的分数和标签序列的分数
        return all_path_score, tag_seq_score

class LSTMNet(nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_dim=64, hidden_size=128, num_layers=2,
                 batch_first=True, bidirectional=True, dropout=0, use_norm=True, embeddings_pretrained=None, num_heads=4, use_crf =False):
        super(LSTMNet, self).__init__()
        self.use_norm = use_norm
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.use_crf = use_crf
        if self.num_embeddings > 0:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            if embeddings_pretrained is not None:
                self.embedding = self.embedding.from_pretrained(embeddings_pretrained, freeze=False)

        if self.use_crf:
            self.crf = CRF(num_classes)
        num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=batch_first,
                            bidirectional=bidirectional, dropout=dropout)
        if self.use_norm:
            self.layer_norm = LayerNorm(hidden_size * num_directions)

        # 添加多头注意力层
        self.attention = MultiHeadAttention(hidden_size * num_directions, num_heads)

        # 输出层，确保输出尺寸与目标标签相匹配
        self.classify = nn.Linear(hidden_size * num_directions, num_classes)

    def forward(self, x,tags=None, mask=None):
        if self.num_embeddings > 0:
            x = self.embedding(x)
        x, _ = self.lstm(x)
        if self.use_norm:
            x = self.layer_norm(x)

        # 应用多头注意力机制
        x = self.attention(x, x, x)
        if self.use_crf:
            all_path_score, tag_seq_score = self.crf(x, tags, mask)
            return tag_seq_score
        else:
            # 使用最后一个时间步的输出进行分类
            x = self.classify(x[:, -1, :])
            return x

if __name__ == "__main__":
    batch_size = 2
    num_embeddings = 128
    context_size = 8
    num_classes = 24339  # 假设有24339个类别
    input = torch.ones(batch_size, context_size).long().cuda()
    model = LSTMNet(num_classes, num_embeddings, embedding_dim=64, num_heads=4,use_crf=True).cuda()
    # 创建一个mask，其中1表示有效位置，0表示填充位置
    mask = torch.ones(batch_size, context_size).bool().cuda()

    # 如果使用CRF，还需要传递标签
    tags = torch.randint(0, num_classes, (batch_size, context_size)).cuda()

    out = model(input, tags=tags, mask=mask)
    print(out)
    print(model)
    print("input", input.shape)
    print("out  ", out.shape)