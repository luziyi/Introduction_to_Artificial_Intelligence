import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义名为 LeNet5 的类，该类继承自 nn.Module
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        # C1层：该层是一个卷积层。使用6个大小为5*5的卷积核，步长为1，对输入层进行卷积运算，特征图尺寸为32-5+1=28，因此产生6个大小为28*28的特征图。这么做够防止原图像输入的信息掉到卷积核边界之外。
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),	# 卷积
            # 第一个参数1，表示输入图像的通道数。在这个例子中，输入图像是灰度图像，所以通道数为1。
            # 第二个参数6，表示卷积层的输出通道数，也就是卷积核的数量。6个卷积核。
            # kernel_size=5定义了卷积核的大小，这里是5x5的卷积核。
            # stride=1定义了卷积核移动的步长，这里每次移动1个像素点。
            # padding=0定义了在输入图像周围填充0的层数，这里没有填充。
            nn.BatchNorm2d(6),		# 批归一化
            nn.ReLU(),)
        
        
        
        # S2层：该层是一个池化层（pooling，也称为下采样层）。这里采用max_pool（最大池化），池化的size定为2*2，经池化后得到6个14*14的特征图，作为下一层神经元的输入。
        self.subsampel1 = nn.MaxPool2d(kernel_size = 2, stride = 2)		# 最大池化  使用max提取特征
        
        
        
        
        # C3层：该层仍为一个卷积层，我们选用大小为5*5的16种不同的卷积核。这里需要注意：C3中的每个特征图，都是S2中的所有6个或其中几个特征图进行加权组合得到的。输出为16个10*10的特征图。
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),)
        
        
        
        # S4层：该层仍为一个池化层，size为2*2，仍采用max_pool。最后输出16个5*5的特征图，神经元个数也减少至16*5*5=400。
        self.subsampel2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        
        # F5层：该层我们继续用5*5的卷积核对S4层的输出进行卷积，卷积核数量增加至120。这样C5层的输出图片大小为5-5+1=1。最终输出120个1*1的特征图。这里实际上是与S4全连接了，但仍将其标为卷积层，原因是如果LeNet-5的输入图片尺寸变大，其他保持不变，那该层特征图的维数也会大于1*1。
        self.L1 = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        
        # F6层：该层与C5层全连接，输出84张特征图。
        self.L2 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        
        # 输出层：该层与F6层全连接，输出长度为10的张量，代表所抽取的特征属于哪个类别。（例如[0,0,0,1,0,0,0,0,0,0]的张量，1在index=3的位置，故该张量代表的图片属于第三类）
        self.L3 = nn.Linear(84, num_classes)
    
    # 前向传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.subsampel1(out)
        out = self.layer2(out)
        out = self.subsampel2(out)
        
        out = out.reshape(out.size(0), -1) # 将上一步输出的16个5×5特征图中的400个像素展平成一维向量，以便下一步全连接
        
        # 全连接
        out = self.L1(out)
        out = self.relu(out)
        out = self.L2(out)
        out = self.relu1(out)
        out = self.L3(out)
        
        
        # 找到概率最大的元素的索引
        max_index = torch.argmax(out, dim=1)
        # 创建一个新的张量，其中只有这个索引位置的元素为1，其他元素都为0
        out1 = F.one_hot(max_index, num_classes=10)
        # print(out1)
        return out

# 加载训练集
train_dataset = torchvision.datasets.MNIST(root = './data',	# 数据集保存路径
                                           train = True,	# 是否为训练集
                                           # 数据预处理
                                           transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1307,), 
												                     std = (0.3081,))]),
                                           download = True)	#是否下载
# transforms.Resize((32,32)) 将每个图像调整为32x32像素。
# transforms.ToTensor() 将图像数据转换为PyTorch张量。
# transforms.Normalize(mean = (0.1307,), std = (0.3081,)) 对图像数据进行标准化，这里的均值和标准差是根据MNIST数据集的特性设定的。
# 从图像的每个通道中减去对应的均值。
# 然后将结果除以对应的标准差。
# 这样处理后，每个通道的数据都会变成均值为0，标准差为1的分布，这有助于神经网络的训练。
 
# 加载测试集
test_dataset = torchvision.datasets.MNIST(root = './data',
                                          train = False,
                                          transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1325,), 
												                     std = (0.3105,))]),
                                          download=True)





batch_size = 64
# 定义了一个变量batch_size，并将其值设置为128。在机器学习和深度学习中，batch_size通常用于指定在一次迭代中用于更新模型权重的样本数量。
# 选择合适的batch_size是优化模型性能的关键。如果batch_size太小，模型可能会在训练过程中遇到噪声，导致权重更新不稳定。如果batch_size太大，模型可能会需要更多的内存，并且可能会导致模型在训练过程中过早地收敛到一个局部最优解，而不是全局最优解。

# 加载训练数据
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
# 加载测试数据
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

num_classes = 10 # 0-9数字，共10个类别

model = LeNet5(num_classes).to(device)  # 实例化模型，并将其移动到设备上

cost = nn.CrossEntropyLoss()    # 定义损失函数 交叉熵损失 度量模型的预测概率分布与真实分布之间的差距 

learning_rate = 0.001 # 学习率
# 如果学习率设置得过大，可能会导致模型在训练过程中震荡不定，难以收敛。如果学习率设置得过小，模型训练的速度可能会非常慢，需要更多的时间才能收敛。
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # 定义优化器 用于更新权重

# torch.optim.Adam是一个实现了Adam优化算法的类。Adam优化算法是一种自适应学习率的优化算法，它结合了RMSProp算法和Momentum算法的优点。
# model.parameters()是一个函数，它返回模型中所有的可训练参数。这些参数是需要优化的对象，因此我们将它们传递给优化器。
# lr=learning_rate设置了优化器的学习率。学习率是一个超参数，它决定了模型参数更新的速度。如果学习率太高，模型可能会在最优解附近震荡而无法收敛；如果学习率太低，模型收敛的速度可能会非常慢。
# 创建一个Adam优化器，用于管理和更新模型的参数，以便在训练过程中改进模型的性能。

total_step = len(train_loader)

# 设置一共训练几轮（epoch）
num_epochs = 10


# 外部循环用于遍历轮次
for epoch in range(num_epochs):
    # 内部循环用于遍历每轮中的所有批次
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device) # 将加载的图像和标签移动到设备上
        labels = labels.to(device) 
 
        # 前向传播
        outputs = model(images)   # 通过模型进行前向传播，得到模型的预测结果 outputs
        loss = cost(outputs, labels)	# 计算模型预测与真实标签之间的损失
 
        # 反向传播和优化
        optimizer.zero_grad()	# 清零梯度，以便在下一次反向传播中不累积之前的梯度
        loss.backward()		# 进行反向传播，计算梯度
        optimizer.step()	# 根据梯度更新（优化）模型参数
 
        # 定期输出训练信息
        # 在每经过一定数量的批次后，输出当前训练轮次、总周轮数、当前批次、总批次数和损失值
        if (i+1) % 300 == 0:
            print('训练轮次 [{:2d}/{:2d}], 批次 [{}/{}], 损失值: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))




# 测试数据集
with torch.no_grad():	# 指示 PyTorch 在接下来的代码块中不要计算梯度
    # 初始化计数器
    correct = 0		# 正确分类的样本数
    total = 0		# 总样本数
 
    # 遍历测试数据集的每个批次
    for images, labels in test_loader:
        # 将加载的图像和标签移动到设备上
        images = images.to(device)
        labels = labels.to(device)
 
        # 模型预测
        outputs = model(images)
 
        # 计算准确率
        # 从模型输出中获取每个样本预测的类别
        _, predicted = torch.max(outputs.data, 1)
        # 累积总样本数
        total += labels.size(0)
        # 累积正确分类的样本数
        correct += (predicted == labels).sum().item()
 
    # 输出准确率，正确的 / 总的
    print('测试总数: {} '.format(total))
    print('测试准确率: {} %'.format(100 * correct / total))
    