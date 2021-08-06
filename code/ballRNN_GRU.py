import os
import sys

import numpy
import torch
import gzip
import itertools
from torch import nn
from matplotlib import pyplot
import random
from ballThrowingPhysicalModel import ballNextState
import matplotlib.pyplot as plt
import threading

class pltThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        plt.figure(figsize=(10, 6))
        plt.show()


SUFFIX="GRU_8x3_tanh"
MODEL_PATH = f"ball_rnn_model_{SUFFIX}.pt"
INIT_BALL_POSITION = [0.0, 1, 0.0]
BATCH_SIZE = 64
EPOCH_SIZE = 100
MAX_SEQ_LEN = 100
DT = 0.1
RNN_INPUT_SIZE = 3
RNN_HIDDEN_SIZE = 8
RNN_NUM_LAYERS = 3
RNN_OUT_FEATURES = 3
ADAM_LEARN_RATE = 1e-4

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(
            input_size = RNN_INPUT_SIZE,
            hidden_size = RNN_HIDDEN_SIZE,
            num_layers = RNN_NUM_LAYERS,
            batch_first = True,
            # nonlinearity = 'relu'
        )
        self.linear = nn.Linear(in_features=RNN_HIDDEN_SIZE, out_features=RNN_OUT_FEATURES)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.linear(unpacked)
        return y


def save_tensor(tensor, path):
    """保存 tensor 对象到文件"""
    torch.save(tensor, gzip.GzipFile(path, "wb"))

def load_tensor(path):
    """从文件读取 tensor 对象"""
    return torch.load(gzip.GzipFile(path, "rb"))


def randomBallInitVelocity():
    return [random.random() * 4 - 2, random.random() * 20 - 10, random.random() * 4 - 2]


def formAFixedLenBatch(batchSize, maxSeqLen):
    x_batch = []
    y_batch = []
    seq_list = []
    for i in range(0, batchSize):
        nextPosition = INIT_BALL_POSITION
        nextVelocity = randomBallInitVelocity()
        x_seq = []
        y_seq = []
        seq_list.append(maxSeqLen)
        for j in range(0, maxSeqLen):
            x_seq.append(nextPosition)
            nextPosition, nextVelocity = ballNextState(nextPosition, nextVelocity, DT)
            y_seq.append(nextPosition)
        x_batch.append(x_seq)
        y_batch.append(y_seq)
    return torch.tensor(x_batch, dtype=torch.float), torch.tensor(y_batch, dtype=torch.float), seq_list

def formARandomLenBatch(batchSize, maxSeqLen):
    x_batch = []
    y_batch = []
    seq_list = []
    for i in range(0, batchSize):
        nextPosition = INIT_BALL_POSITION
        nextVelocity = randomBallInitVelocity()
        x_seq = []
        y_seq = []
        seqLen = round(random.random() * (maxSeqLen - 2)) + 2
        seq_list.append(seqLen)
        for j in range(0, maxSeqLen):
            if j < seqLen:
                x_seq.append(nextPosition)
                nextPosition, nextVelocity = ballNextState(nextPosition, nextVelocity, DT)
                y_seq.append(nextPosition)
            elif j == seqLen:
                nextPosition, nextVelocity = ballNextState(nextPosition, nextVelocity, DT)
                y_seq.append(nextPosition)
                x_seq.append([0, 0, 0])
            else:
                x_seq.append([0, 0, 0])
                y_seq.append([0, 0, 0])
        x_batch.append(x_seq)
        y_batch.append(y_seq)
    return torch.tensor(x_batch, dtype=torch.float), torch.tensor(y_batch, dtype=torch.float), seq_list

# 计算正确率的工具函数
def calc_accuracy(actual, predicted):
    return 1 - ((actual - predicted).abs() / (actual.abs() + 0.0001)).mean().item()

def calc_batch_accuracy(actual, predicted):
    accuracy_list = []
    for i in range(0, actual.size(0)):
        accuracy_list.append(1 - ((actual[i] - predicted[i]).abs() / (actual[i].abs() + 0.0001)).mean().item())
    return numpy.average(accuracy_list)

def print_a_batch_of_predicted_and_actual_values(y, predicted):
    plt.clf()
    time_seq = []
    time = 0
    y_seq = []
    predicted_seq = []
    for j in range(0, len(y[-1])):
        time_seq.append(time)
        time += DT
        y_seq.append(y[-1][j][1])
        predicted_seq.append(predicted[-1][j][1])
        print(f"actual   : {y[-1][j]}")
        print(f"predicted: {predicted[-1][j]}")
        print(f"--------------------------------------------------------------")
    plt.xlabel("Time(s)")
    plt.ylabel("Height(m)")
    plt.plot(time_seq, predicted_seq, 'r-o', label='predicted')
    plt.plot(time_seq, y_seq, 'b-o', label='target')
    plt.legend(["predicted", "target"])
    for j in range(0, len(time_seq)):
        plt.plot([time_seq[j], time_seq[j]], [float(y_seq[j]), float(predicted_seq[j])], color='grey')
    plt.draw()


def train():
    """开始训练"""
    # 创建模型实例
    model = MyModel()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(load_tensor(MODEL_PATH))
        print(f"Model is loaded")

    # 创建损失计算器
    loss_function = torch.nn.MSELoss()

    # 创建参数调整器
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LEARN_RATE)

    # 记录训练集和验证集的正确率变化

    model.train()


    for epoch in range(1, 10000):
        print(f"epoch: {epoch}")
        train_accuracy_history = []
        highest_accuracy = -1000000
        for i in range(1, EPOCH_SIZE):
            x_batch, y_batch, seq_list = formAFixedLenBatch(BATCH_SIZE, MAX_SEQ_LEN)
            seq_len = torch.tensor(seq_list, dtype=torch.int64)
            y_packed = nn.utils.rnn.pack_padded_sequence(y_batch, seq_len, batch_first=True, enforce_sorted=False)
            y_unpacked, _ = nn.utils.rnn.pad_packed_sequence(y_packed, batch_first=True)
            predicted = model(x_batch, seq_len)

            predicted_cut = predicted[:, 2:, :]
            y_cut = y_unpacked[:, 2:, :]
            loss = loss_function(predicted, y_unpacked)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_accuraccy = calc_batch_accuracy(y_cut, predicted_cut)
            if i == EPOCH_SIZE - 1:
                print_a_batch_of_predicted_and_actual_values(y_unpacked, predicted)
                print(f"loss: {loss}")
                train_accuracy_history.append(train_accuraccy)
        mean_accuracy = numpy.mean(train_accuracy_history)
        print(f"training accuracy: {mean_accuracy}")
        save_tensor(model.state_dict(), MODEL_PATH)
        if mean_accuracy > highest_accuracy:
            highest_accuracy = mean_accuracy
            highest_epoch = epoch
        elif epoch - highest_epoch > 100:
            print("stop training because highest validating accuracy not updated in 100 epoches")
            break

if __name__ == "__main__":
    thread = pltThread()
    thread.start()
    train()