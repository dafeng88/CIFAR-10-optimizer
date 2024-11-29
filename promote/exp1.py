
# 来源：https://cto.eguidedog.net/node/1337
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

toPitImage = ToPILImage()  # 可以把Tensor转成Image，方便可视化

# 参考：https://www.cs.toronto.edu/~kriz/cifar.html
# 第一次运行程序torchvision会自动下载CIFAR-10数据集，原始尺寸是32x32
# 大约100M，需花费一定的时间，
# 如果已经下载有CIFAR-10，可通过root参数指定

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    # 训练集
    trainset = tv.datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=transform)

    trainloader = t.utils.data.DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=2)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    (data, label) = trainset[100]
    print(classes[label])

    # (data + 1) / 2是为了还原被归一化的数据
    # image = toPitImage((data + 1) / 2)
    # image.show()

    dataiter = iter(trainloader)
    images, labels = next(dataiter)  # 返回4张图片及标签
    print(' '.join('%11s' % classes[labels[j]] for j in range(4)))
    # image = toPitImage(tv.utils.make_grid((images+1)/2))
    # image.show()

    t.set_num_threads(8)
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # 输入数据
            inputs, labels = data

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 打印log信息
            # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个batch打印一下训练状态
                print('[%d, %5d] loss: %.3f' \
                      % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')


def test():
    # 测试集
    testset = tv.datasets.CIFAR10(
        '../data',
        train=False,
        download=True,
        transform=transform)

    testloader = t.utils.data.DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=2)

    dataiter = iter(testloader)
    images, labels = next(dataiter)  # 一个batch返回4张图片
    print('实际的label: ', ' '.join( \
        '%08s' % classes[labels[j]] for j in range(4)))
    # image = toPitImage(tv.utils.make_grid(images / 2 - 0.5))
    # image.show()

    # 计算图片在每个类别上的分数
    outputs = net(images)
    # 得分最高的那个类
    _, predicted = t.max(outputs.data, 1)

    print('预测结果: ', ' '.join('%5s' \
                                 % classes[predicted[j]] for j in range(4)))

    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数

    # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
    with t.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = t.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))


# 原书的模型识别率53%
# net = Net()

# resnet50 在2015年的比赛中准确率为76%
# net = tv.models.resnet34() # 准确率39%
# net = tv.models.resnet18() # 正确率50%

# https://pytorch.org/hub/pytorch_vision_googlenet/
# net = t.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True) # 正确率85%
# net.eval()

# https://pytorch.org/vision/main/models/generated/torchvision.models.googlenet.html
# net = tv.models.googlenet(pretrained=True)
# net.eval()

# https://pytorch.org/vision/main/models/efficientnet.html?highlight=efficientnet
# 正确率94%，可重现
# net = tv.models.efficientnet_b1(pretrained=True)
# net.eval()

# 不行，不知道怎么用
# https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/
# net = t.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
# net.eval()

# 不行，lost nan不知道怎么用
# net = tv.models.efficientnet_v2_s(pretrained=True)
# net.eval()

# 目前业界对CIFAR10数据集的最高正确率是96.53%。该准确率是由使用一种称为EfficientNet的深度学习模型在CIFAR10数据集上实现的。
# 正确率96%，但运行起来比b1慢很多，光测试就要花17分钟
net = tv.models.efficientnet_b6(pretrained=True)
net.eval()
# net = t.load('efficientnetb6.pth') # 173M

# print(net)
train()
test()
# t.save(net, 'efficientnetv2s.pth')