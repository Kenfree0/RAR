import enum
import math

import numpy as np
import torch as th
import sys
sys.path.append('.')

import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

# 模拟数据生成
# 假设输入数据为长度为100的序列，每个元素从0到999的整数中随机选择
# 输出标签为0或1，表示两个类别
data_size = 1000
input_dim = 1000
sequence_length = 100

train_data = torch.randint(0, input_dim, (data_size, sequence_length))
train_labels = torch.randint(0, 2, (data_size,))

# 创建数据集和数据加载器
class CustomDataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

# 创建数据集和数据加载器
batch_size = 32
train_dataset = CustomDataset(train_data, train_labels)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建模型
input_dim = 1000
emb_dim = 256
nhead = 8
num_encoder_layers = 4
num_cnn_layers = 3
kernel_size = 3
out_channels = 64

class CustomConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CustomConvLayer, self).__init()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x

class TransformerCNNModel(nn.Module):
    def __init__(self, input_dim, emb_dim, nhead, num_encoder_layers, in_channels, kernel_size, out_channels):
        super(TransformerCNNModel, self).__init__()  # 正确的super()调用

        # 降维
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead),
            num_layers=num_encoder_layers
        )

        # 多层CNN，确保卷积核的通道数与输入通道数相匹配


        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.fc = nn.Linear(out_channels, 128)

    def forward(self, x):
        # 降维
        embedded = self.embedding(x)
        print("embedded:",embedded.shape)
        # Transformer编码
        transformed = self.transformer(embedded)
        print("transformed:",transformed.shape)
        # CNN特征提取
        convolved = transformed.unsqueeze(1)
        print("convolved:",convolved.shape)

        convolved = self.conv(convolved)
        print(convolved.shape)
        pooled = F.max_pool2d(convolved, (convolved.shape[2], 1)).squeeze(2)
        output = self.fc(pooled)

        return output

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()  # 正确的super()调用
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64,128)

    def forward(self, x):
        x = self.fc1(x)
        #print(x.shape)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x


'''# 其他部分保持不变，包括数据生成、数据集和数据加载器、损失函数、优化器以及训练循环。

# 创建模型实例
input_dim = 1000  # 输入词汇表大小
emb_dim = 256    # 词嵌入维度
nhead = 8        # Transformer头数
num_encoder_layers = 4  # Transformer编码器层数
num_cnn_layers = 3  # 多层CNN
kernel_size = 8  # CNN内核大小
out_channels = 64  # CNN输出通道数

model = TransformerCNNModel(input_dim, emb_dim, nhead, num_encoder_layers, num_cnn_layers, kernel_size, out_channels)

# 打印模型结构
print(model)

model = TransformerCNNModel(input_dim, emb_dim, nhead, num_encoder_layers, num_cnn_layers, kernel_size, out_channels)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        print("input:",inputs)
        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # 每10个小批量打印一次损失
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10}')
            running_loss = 0.0

print('Finished Training')
'''