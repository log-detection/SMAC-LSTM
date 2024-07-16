import json
import os
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# 根目录
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))  # 当前文件所在目录的绝对路径

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, "data")  # 数据目录
DATA_PATH = os.path.join(DATA_DIR, "BGL_2k.csv")  # 数据路径
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")  # 词典路径
FINALLY_DATA_PATH = os.path.join(DATA_DIR, "finally_data2.csv")  # 最终数据路径

# 标签
LABELS = ["normal", "anomalous"]  # 标签


class Vocabulary(object):
    # 词典类
    def __init__(self):
        self.token2idx = {"<K>": 0, "<unk>": 1}  # token2idx <K>表示填充，<unk>表示未知词
        self.idx2token = {0: "<K>", 1: "<unk>"}  # idx2token <K>表示填充，<unk>表示未知词
        self.idx = 2  # 词典中词的个数

    def add_token(self, token):
        # 如果token不在词典中，就添加到词典中
        if token not in self.token2idx:  # token2idx是一个字典 key是token value是idx
            self.token2idx[token] = self.idx  # token2idx 添加idx
            self.idx2token[self.idx] = token  # idx2token 添加token
            self.idx += 1  # 词的个数加1

    def add_sequence(self, sequence):
        # 将sequence中的词添加到词典中
        for token in sequence.split():  # sequence是一个字符串 用空格分割
            self.add_token(token)  # 将sequence中的词添加到词典中

    def seq2vec(self, sequence, fix_length=32):
        # 将sequence转换为向量
        idxs = [
            self.token2idx.get(token, self.token2idx["<unk>"]) for token in sequence.split()
        ]  # sequence是一个字符串 用空格分割
        if len(idxs) >= fix_length:  # 如果idxs的长度大于fix_length
            idxs = idxs[:fix_length]  # 截取前fix_length个
        else:  # 如果idxs的长度小于fix_length
            idxs.extend([self.token2idx["<K>"]] * (fix_length - len(idxs)))  # 将<K>填充到idxs中
        return idxs

    def save_dict(self, dict_path):
        # 保存词典
        with open(dict_path, "w", encoding="utf-8") as fw:  # 打开文件
            json.dump(self.token2idx, fw, ensure_ascii=False)  # 保存词典

    def load_dict(self, dict_path):
        # 加载词典
        with open(dict_path, "r", encoding="utf-8") as fr:  # 打开文件
            self.token2idx = json.load(fp=fr)  # 加载词典 token2idx
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}  # idx2token
        self.idx = len(self.token2idx)  # 词典中词的个数

    def __call__(self, word):
        return self.token2idx.get(word, self.token2idx["<unk>"])  # 如果word在词典中 返回idx 否则返回<unk>的idx

    def __len__(self):
        return self.idx  # 返回词典中词的个数


def build_vocab(text_list):
    vocabulary_obj = Vocabulary()  # 创建词典对象
    for text in tqdm(text_list, desc="Build vocab", file=sys.stdout):  # 遍历文本
        vocabulary_obj.add_sequence(text)  # 将文本中的词添加到词典中
    vocabulary_obj.save_dict(VOCAB_PATH)  # 保存词典
    return vocabulary_obj  # 返回词典对象


if __name__ == "__main__":
    ############### 构建词典 转换向量 ################
    data = pd.read_csv(DATA_PATH)  # 读取数据

    data = data[["GroupTemplate", "Level"]]  # 只保留GroupTemplate和Level
    data = data.dropna(axis=0, how="any")  # 删除空行
    data = data[data["Level"].isin(LABELS)]  # 只保留LABELS中的类别

    data["GroupTemplate"] = data["GroupTemplate"].str.lower()  # 转换为小写
    data["GroupTemplate"] = data["GroupTemplate"].apply(
        lambda x: re.sub(r'([!@#$%^&*()_+={}\[\]:;<>,.?~\\/\|"])', r" \1 ", x)
    )  # 将特殊字符前后加空格
    data["GroupTemplate"] = data["GroupTemplate"].apply(lambda x: re.sub(r"\s{2,}", " ", x))  # 将多个空格替换为一个空格

    print(data["Level"].value_counts())  # 统计每个类别的数量

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Level"])  # 切分训练集和测试集
    vocabulary_obj = build_vocab(train_data["GroupTemplate"].tolist())  # 创建词典对象 并且将训练集中的词添加到词典中
    data["GroupTemplate"] = data["GroupTemplate"].apply(lambda x: vocabulary_obj.seq2vec(x))  # 将文本转换为向量
    data["Level"] = data["Level"].apply(lambda x: LABELS.index(x))  # 顺序编码

    data.to_csv(FINALLY_DATA_PATH, index=False)  # 保存数据
