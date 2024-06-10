import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module): # 定义GCN模型
    def __init__(self, num_node_features, num_classes): # 初始化函数
        super(GCN, self).__init__() # 调用父类的初始化函数
        hidden_channels = 32 # 隐藏层特征数为32
        # 隐藏层特征数是指在神经网络中隐藏层的神经元数量。隐藏层特征数的选择对神经网络的性能和表达能力有很大影响。隐藏层特征数的主要作用是决定神经网络的复杂度和表示能力。较少的隐藏层特征数可能导致神经网络无法捕捉到输入数据中的复杂模式和关系，从而导致欠拟合。而较多的隐藏层特征数可能导致神经网络过度拟合训练数据，无法泛化到新的未见过的数据。
        self.conv1 = GCNConv(num_node_features, hidden_channels) # 输入特征数为num_node_features，输出特征数为hidden_channels
        self.conv2 = GCNConv(hidden_channels, num_classes) # 输入特征数为hidden_channels，输出特征数为num_classes
        self.norm = torch.nn.BatchNorm1d(hidden_channels) # 一维批标准化层，输入特征数为hidden_channels

    def forward(self, data): # 前向传播函数
        x, edge_index = data.x, data.edge_index # 获取输入特征x和边索引edge_index
        x = self.conv1(x, edge_index) # 第一层GCN 通过GCNConv层对输入特征x和边索引edge_index进行卷积操作
        x = self.norm(x) # 批标准化 通过BatchNorm1d层对输出特征x进行批标准化操作
        x = F.relu(x) # ReLU激活函数 通过ReLU激活函数对输出特征x进行非线性变换
        x = F.dropout(x, training=self.training) # dropout层 通过Dropout层对输出特征x进行随机失活操作
        x = self.conv2(x, edge_index) # 第二层GCN
        return x


def load_data(name): #加载数据
    dataset = Planetoid(root='./' + name + '/', name=name)
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(_device)
    num_node_features = dataset.num_node_features
    return data, num_node_features, dataset.num_classes

#定义训练函数
def train(model, data, device):  # 模型，数据，设备
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)  # Adam优化器 学习率0.01  权重衰减1e-4 根据定义的优化算法和损失函数，自动调整模型的参数，以最小化损失函数的值。在深度学习中，优化器的目标是通过反向传播算法来计算梯度，并使用梯度下降的方法来更新模型的参数。model.parameters()表示要优化的模型的参数，lr表示学习率，即每次更新参数时的步长。另外，weight_decay参数用于控制L2正则化项的权重衰减。
    loss_function = torch.nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数计算损失值 将softmax函数和负对数似然损失（negative log likelihood loss）结合在一起。在计算损失时，它首先对模型的原始输出（也就是logits）应用softmax函数，将输出转换为概率分布，然后计算真实标签和预测概率分布之间的负对数似然损失
    model.train() # 训练模式
    for epoch in range(200): # 迭代200次
        out = model(data)  # 前向传播
        optimizer.zero_grad()  # 梯度清零
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])  #  计算损失值
        loss.backward()  # 反向传播 
        optimizer.step()  # 更新参数

        print('Epoch {:03d} loss {:.4f}'.format(epoch, loss.item()))

# 测试函数
def test(model, data):
    model.eval()  # 测试模式
    _, pred = model(data).max(dim=1) # 使用模型对数据进行预测，并返回预测结果中的最大值。具体来说，model(data)表示将数据data输入模型进行预测，返回的结果是一个张量。然后，.max(dim=1)表示在结果张量的第一个维度上取最大值，并返回最大值和对应的索引。在这里，我们使用下划线_来表示我们不关心最大值的具体值，只关心对应的索引值，所以将最大值赋值给了pred变量
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())  #获取预测正确的数量
    acc = correct / int(data.test_mask.sum())
    print('GCN Accuracy: {:.4f}'.format(acc))


def main():
    print('Cora' + ' dataset:')
    data, num_node_features, num_classes = load_data('Cora')
    print(data, num_node_features, num_classes)
    _device = 'cpu'
    device = torch.device(_device)
    model = GCN(num_node_features, num_classes).to('cpu')
    train(model, data, device)
    test(model, data)

if __name__ == '__main__':
    main()
    
