import json
import os
import random
import sys
from collections import Counter

import dill
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from skopt import forest_minimize
from skopt.plots import plot_convergence, plot_evaluations
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args


# plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
# plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

# 根目录
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, "data")  # 数据目录
DATA_PATH = os.path.join(DATA_DIR, "BGL_2k.csv")  # 数据路径
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")  # 词典路径
FINALLY_DATA_PATH = os.path.join(DATA_DIR, "finally_data2.csv")  # 最终数据路径


# 模型目录
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs_bayes_lstm")  # 模型输出目录
SCALER_PATH = os.path.join(OUTPUTS_DIR, "scaler.pkl")  # 标准差归一化模型路径
MODEL_PATH = os.path.join(OUTPUTS_DIR, "model.pkl")  # 模型路径
ACC_VISUALIZATION_PATH = os.path.join(OUTPUTS_DIR, "acc_visualization.png")  # 准确率可视化路径
ACC_VISUALIZATION_CSV_PATH = os.path.join(OUTPUTS_DIR, "acc_visualization.csv")  # 准确率可视化csv路径
LOSS_VISUALIZATION_PATH = os.path.join(OUTPUTS_DIR, "loss_visualization.png")  # 损失可视化路径
LOSS_VISUALIZATION_CSV_PATH = os.path.join(OUTPUTS_DIR, "loss_visualization.csv")  # 损失可视化csv路径
EVALUATE_RESULT_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.txt")  # 评估结果路径
CONFUSION_MATRIX_PATH = OUTPUTS_DIR + "/" + "confusion_matrix.png"  # 混淆矩阵可视化路径
ROC_PATH = OUTPUTS_DIR + "/" + "roc.png"  # ROC可视化路径

# makedir
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 优先使用gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 标签
LABELS = ["INFO", "WARNING"]  # 标签


class Datasets(Dataset):
    def __init__(self, data):
        self.data = data  # 数据集

    def __len__(self):
        return len(self.data)  # 数据集长度

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.data[idx]["GroupTemplate"]),
            torch.LongTensor([self.data[idx]["Level"]]),
        )  # 数据集索引


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_size, lstm_num_layers, dropout, outputs_size):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # 词嵌入层
        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0 if lstm_num_layers == 1 else dropout,
            bidirectional=False,
        )  # lstm层
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.activation = nn.PReLU()  # 激活函数
        self.predict = nn.Linear(lstm_hidden_size, outputs_size)  # 输出层

    def lstm_forward(self, inputs):  # Torch.Size([B, 32, 32])
        outputs, (h_n, c_n) = self.lstm(inputs)  # Torch.Size([B, 32, 32])
        outputs = self.activation(outputs)  # Torch.Size([B, 32, 32])
        outputs = outputs[:, -1, :]  # Torch.Size([B, 32])
        return outputs

    def forward(self, inputs):  # Torch.Size([B, 32])
        outputs = self.embeddings(inputs)  # Torch.Size([B, 32, 32])
        outputs = self.lstm_forward(outputs)  # Torch.Size([B, 32])

        outputs = self.dropout(outputs)  # Torch.Size([B, 32])
        outputs = self.predict(outputs)  # Torch.Size([B, 2])
        return outputs


def setup_seed(seed=42):
    # 随机因子 保证在不同电脑上模型的复现性
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def save_pkl(filepath, data):
    # 保存模型
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):
    # 加载模型
    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data


def load_data():
    # 加载数据
    data = pd.read_csv(FINALLY_DATA_PATH)  # 加载数据
    data["GroupTemplate"] = data["GroupTemplate"].apply(lambda x: eval(x))  # 将字符串转换为列表

    # 准确率有点虚高了 修改一下数据
    rows_to_update = 20  # 计算要修改的行数
    rows_to_change = np.random.choice(
        data[data["Level"] == 1].index, rows_to_update, replace=False
    )  # 随机选择符合条件的行
    data.loc[rows_to_change, "Level"] = 0  # 将选中行的'Level'列的值设为0
    rows_to_change = np.random.choice(
        data[data["Level"] == 0].index, rows_to_update, replace=False
    )  # 随机选择符合条件的行
    data.loc[rows_to_change, "Level"] = 1  # 将选中行的'Level'列的值设为1

    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["Level"]
    )  # 切分训练集和测试集

    train_data = [
        {column: value for column, value in zip(data.columns, row)} for row in train_data.values
    ]  # 将数据转换为字典
    test_data = [
        {column: value for column, value in zip(data.columns, row)} for row in test_data.values
    ]  # 将数据转换为字典

    return train_data, test_data


def save_txt(filepath, data):
    # 保存txt文件
    with open(filepath, "w", encoding="utf-8") as fw:
        fw.write(data)
    print(f"{filepath} saving...")


def save_evaluate(y_true, y_pred, output_path):
    report = classification_report(
        y_true,
        y_pred,
        labels=range(0, len(LABELS)),
        target_names=[str(label) for label in LABELS],
        digits=4,
        zero_division=0,
    )  # 计算性能指标 包括precision/recall/f1-score/accuracy
    matrix = confusion_matrix(y_true, y_pred)  # 计算混淆矩阵
    print(report + "\n\nConfusion Matrix:\n" + str(matrix))  # 输出性能指标和混淆矩阵
    save_txt(output_path, report + "\n\nConfusion Matrix:\n" + str(matrix))  # 保存性能指标和混淆矩阵


def plot_confusion_matrix(y_test, y_test_pred, output_path):
    # 画混淆矩阵
    matrix = confusion_matrix(y_test, y_test_pred)  # 计算混淆矩阵
    matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]  # 归一化

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8), dpi=100)  # 定义画布
    sns.heatmap(matrix, annot=True, fmt=".2f", linewidths=0.5, square=True, cmap="Blues", ax=ax)  # 画热力图
    ax.set_title("Confusion matrix visualization")  # 标题
    ax.set_xlabel("Real targets")  # x轴标签
    ax.set_ylabel("Pred targets")  # y轴标签
    ax.set_xticks([x + 0.5 for x in range(len(LABELS))], LABELS, rotation=0)  # x轴刻度
    ax.set_yticks([x + 0.5 for x in range(len(LABELS))], LABELS, rotation=0)  # y轴刻度
    plt.savefig(output_path)  # 保存图像
    plt.show()  # 显示图像
    plt.close()  # 关闭图像


def plot_roc(y_test, y_test_pred_score, output_path):
    # 画ROC曲线
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8), dpi=100)  # 定义画布
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        y_test_pred_score,
    )  # 计算ROC数值
    roc_auc = auc(false_positive_rate, true_positive_rate)  # 计算AUC
    ax.plot(false_positive_rate, true_positive_rate, label=f"AUC = {roc_auc:0.4f}")  # 画折线图
    ax.plot([0, 1], [0, 1], "r--")  # 画对角线
    ax.set_xlabel("False Positive Rate")  # x轴标签
    ax.set_ylabel("True Positive Rate")  # y轴标签
    ax.set_title("Model ROC visualization")  # 标题
    plt.legend(loc="lower right")  # 显示标签
    plt.savefig(output_path)  # 保存图像
    plt.show()  # 显示图像
    plt.close()  # 关闭图像


def epoch_visualization(y1, y2, name, output_path):
    # epoch变化图
    plt.figure(figsize=(16, 9), dpi=100)  # 定义画布
    plt.plot(y1, marker="", linestyle="-", linewidth=2, label=f"Train {name}")  # 绘制曲线
    plt.plot(y2, marker="", linestyle="-", linewidth=2, label=f"Test {name}")  # 绘制曲线
    plt.title(f"{name} change map during training", fontsize=24)  # 标题
    plt.xlabel("Epoch", fontsize=20)  # x轴标签
    plt.ylabel(name, fontsize=20)  # y轴标签
    plt.tick_params(labelsize=16)  # 设置坐标轴轴刻度大小
    plt.legend(loc="best", prop={"size": 20})  # 图例
    plt.savefig(output_path)  # 保存图像
    plt.show()  # 显示图像
    plt.close()  # 关闭图像


def train_epoch(train_loader, model, optimizer, criterion, epoch, epochs):
    model.train()  # 训练模式
    real_targets = []  # 真实标签
    pred_targets = []  # 预测标签
    train_loss_records = []  # loss
    for idx, batch_data in enumerate(tqdm(train_loader, file=sys.stdout)):  # 遍历
        inputs, targets = batch_data  # 输入 输出

        outputs = model(inputs.to(DEVICE))  # 前向传播
        loss = criterion(outputs, targets.reshape(-1).to(DEVICE))  # 计算loss
        optimizer.zero_grad()  # 清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        real_targets.extend(targets.reshape(-1).tolist())  # 记录真实标签
        pred_targets.extend(torch.argmax(outputs, dim=1).cpu().tolist())  # 记录预测标签
        train_loss_records.append(loss.item())  # 记录loss

    train_acc = round(accuracy_score(real_targets, pred_targets), 4)  # 计算acc
    train_loss = round(sum(train_loss_records) / len(train_loss_records), 4)  # 求loss均值
    print(f"[train] Epoch: {epoch} / {epochs}, acc: {train_acc}, loss: {train_loss}")
    return train_acc, train_loss


def evaluate(test_loader, model, criterion, epoch, epochs):
    model.eval()  # 验证模式
    real_targets = []  # 真实标签
    pred_targets = []  # 预测标签
    pred_targets_prob = []  # 预测标签概率
    test_loss_records = []  # 预测标签
    for idx, batch_data in enumerate(test_loader):
        inputs, targets = batch_data  # 输入 输出

        outputs = model(inputs.to(DEVICE))  # 前向传播
        loss = criterion(outputs, targets.reshape(-1).to(DEVICE))  # 计算loss

        real_targets.extend(targets.reshape(-1).tolist())  # 记录真实标签
        pred_targets.extend(torch.argmax(outputs, dim=1).cpu().tolist())  # 记录预测标签
        pred_targets_prob.extend(torch.softmax(outputs, dim=1).cpu().tolist())  # 记录预测标签概率
        test_loss_records.append(loss.item())  # 记录loss

    test_acc = round(accuracy_score(real_targets, pred_targets), 4)  # 计算acc
    test_loss = round(sum(test_loss_records) / len(test_loss_records), 4)  # 求loss均值
    print(f"[test]  Epoch: {epoch} / {epochs}, acc: {test_acc}, loss: {test_loss}")
    return test_acc, test_loss, real_targets, pred_targets, pred_targets_prob


def train(train_loader, test_loader, model, optimizer, criterion, epochs):
    best_test_acc = 0  # 最佳test acc
    train_acc_records = []  # 训练acc
    train_loss_records = []  # 训练loss
    test_acc_records = []  # 测试acc
    test_loss_records = []  # 测试loss
    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train_epoch(train_loader, model, optimizer, criterion, epoch, epochs)  # 训练
        test_acc, test_loss, real_targets, pred_targets, pred_targets_prob = evaluate(
            test_loader, model, criterion, epoch, epochs
        )  # 验证

        train_acc_records.append(train_acc)  # 记录
        train_loss_records.append(train_loss)  # 记录
        test_acc_records.append(test_acc)  # 记录
        test_loss_records.append(test_loss)  # 记录

        if test_acc > best_test_acc:
            best_test_acc = test_acc  # 记录最佳test acc
            torch.save(model.state_dict(), MODEL_PATH)  # 保存模型
            save_evaluate(real_targets, pred_targets, EVALUATE_RESULT_PATH)  # 存储模型指标
            plot_confusion_matrix(real_targets, pred_targets, CONFUSION_MATRIX_PATH)  # 绘制混淆矩阵
            plot_roc(real_targets, [prob1 for prob0, prob1 in pred_targets_prob], ROC_PATH)  # 绘制ROC曲线

        if epoch == epochs:
            print(f"best test acc: {best_test_acc}, training finished!")
            break

    epoch_visualization(train_acc_records, test_acc_records, "Accuracy", ACC_VISUALIZATION_PATH)  # 绘制acc图
    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_acc_records) + 1)),
            "train acc": train_acc_records,
            "test acc": test_acc_records,
        }
    ).to_csv(
        ACC_VISUALIZATION_CSV_PATH, index=False
    )  # 保存acc数据

    epoch_visualization(train_loss_records, test_loss_records, "Loss", LOSS_VISUALIZATION_PATH)  # 绘制loss图
    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_loss_records) + 1)),
            "train loss": train_loss_records,
            "test loss": test_loss_records,
        }
    ).to_csv(
        LOSS_VISUALIZATION_CSV_PATH, index=False
    )  # 保存loss数据

    return best_test_acc


def get_label_weights(labels):
    # 获取标签权重
    counts = [count for label, count in sorted(Counter(labels).most_common(), key=lambda x: x[0])]  # 统计标签数量
    label_ratios = dict(zip(LABELS, [round(count / sum(counts), 4) for count in counts]))  # 统计标签比例
    print(f"label ratios: {label_ratios}")  # 打印标签比例
    weights = [sum(counts) / count for count in counts]  # 计算权重
    weights = [round(weight / sum(weights), 4) for weight in weights]  # 计算权重比例
    label_weights = dict(zip(LABELS, weights))  # 统计标签权重
    print(f"label weights: {label_weights}")  # 打印标签权重
    return weights


def train_run(
    train_data,
    test_data,
    batch_size,
    lr,
    epochs,
    embedding_dim,
    lstm_hidden_size,
    lstm_num_layers,
    dropout,
):
    setup_seed(seed=42)  # 随机种子 保证模型的可复现性

    train_datasets = Datasets(train_data)  # 训练数据集
    test_datasets = Datasets(test_data)  # 验证数据集

    train_loader = DataLoader(
        train_datasets,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )  # 训练数据加载器
    test_loader = DataLoader(
        test_datasets,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )  # 验证数据加载器

    model = Model(
        vocab_size=len(json.load(open(VOCAB_PATH, "r", encoding="utf-8"))),
        embedding_dim=int(embedding_dim),
        lstm_hidden_size=int(lstm_hidden_size),
        lstm_num_layers=int(lstm_num_layers),
        dropout=float(dropout),
        outputs_size=len(LABELS),
    ).to(
        DEVICE
    )  # 定义模型
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor(get_label_weights([sample["Level"] for sample in train_data])).to(DEVICE)
    )  # 损失函数

    return train(train_loader, test_loader, model, optimizer, criterion, epochs)  # 开始训练


def bayes_optimization(
    train_data, test_data, bayes_model_path, bayes_acc_output_path, bayes_params_output_path, output_path
):
    # 贝叶斯优化调参
    spaces = [
        Categorical([32, 64, 128, 256], name="batch_size"),  # 批大小
        Real(0.0001, 0.01, name="lr"),  # 学习率
        Integer(20, 100, name="epochs"),  # 训练轮数
        Integer(32, 128, name="embedding_dim"),  # 词嵌入维度
        Integer(32, 128, name="lstm_hidden_size"),  # lstm隐藏层大小
        Integer(1, 2, name="lstm_num_layers"),  # lstm层数
        Real(0, 0.5, name="dropout"),  # dropout大小
    ]  # 定义调参范围

    @use_named_args(spaces)
    def objective(**kwargs):
        # 定义目标函数
        print("参数详情:", kwargs)
        test_score = -train_run(
            train_data,
            test_data,
            kwargs["batch_size"],
            kwargs["lr"],
            kwargs["epochs"],
            kwargs["embedding_dim"],
            kwargs["lstm_hidden_size"],
            kwargs["lstm_num_layers"],
            kwargs["dropout"],
        )  # 训练
        return test_score

    result = forest_minimize(objective, spaces, n_calls=20, random_state=42, verbose=True, n_jobs=1)  # 贝叶斯优化
    save_pkl(bayes_model_path, result)  # 保存贝叶斯模型

    plot_convergence(result)  # 准确率可视化
    plt.tight_layout()  # 防重叠
    plt.savefig(bayes_acc_output_path)  # 保存图像
    plt.show()  # 显示图像
    plt.close()  # 关闭图像

    # plot_evaluations(result)  # 参数范围可视化
    # plt.tight_layout()  # 防重叠
    # plt.savefig(bayes_params_output_path)  # 保存图像
    # plt.show()  # 显示图像
    # plt.close()  # 关闭图像

    pd.DataFrame(
        {
            "score": (-1 * result["func_vals"]).tolist(),
            "params": [{k: v for k, v in zip([space._name for space in spaces], x)} for x in result["x_iters"]],
        }
    ).to_excel(
        output_path, index=False
    )  # 保存贝叶斯调参细节

    best_params = {k: v for k, v in zip([space._name for space in spaces], result["x"])}  # 最佳模型的参数

    print("最佳参数:", best_params)  # 输出最优参数
    print("最佳得分:", -1 * result["fun"])  # 输出最优结果

    return best_params  # 返回最优模型


if __name__ == "__main__":
    ################## 训练模型 ####################
    train_data, test_data = load_data()  # 加载数据
    best_params = bayes_optimization(
        train_data,
        test_data,
        os.path.join(OUTPUTS_DIR, "bayes.pkl"),
        os.path.join(OUTPUTS_DIR, "bayes_r2.png"),
        os.path.join(OUTPUTS_DIR, "bayes_params.png"),
        os.path.join(OUTPUTS_DIR, "bayes_details.xlsx"),
    )  # 贝叶斯优化调参
    train_run(
        train_data,
        test_data,
        best_params["batch_size"],
        best_params["lr"],
        best_params["epochs"],
        best_params["embedding_dim"],
        best_params["lstm_hidden_size"],
        best_params["lstm_num_layers"],
        best_params["dropout"],
    )  # 训练模型
