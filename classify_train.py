import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.preprocessing import label_binarize
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, \
    auc
from sklearn.metrics import roc_curve
from tqdm import tqdm
from torchvision import models
from net.lenet import LeNet

# GPU计算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
# 数据预处理
train_transform = transforms.Compose([
    transforms.ToTensor()
    , transforms.RandomCrop(32, padding=4)  # 先四周填充0，再把图像随机裁剪成32*32
    , transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转 选择一个概率概率
    ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
])

# 测试集通常不做数据增强，只归一化）
test_transform = transforms.Compose([
    transforms.ToTensor()
    , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)


# 使用同一个日志目录
writer = SummaryWriter("logs/")
# 现在得到的result是用matlab画的，没有采用tensorboard


# 训练函数
def train_and_test(net,optimizer_name,optimizer,num_epochs,i):
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 训练和验证
    best_acc = 0.0  # 初始化最佳准确率
    train_losses = []  # 用于保存训练损失
    test_losses = []  # 用于保存测试损失
    train_accs = []  # 用于保存训练准确率
    test_accs = []  # 用于保存测试准确率
    # 用于记录损失值未发生变化batch数
    counter = 0
    # 初始学习率
    Lr = 0.1
    #用来记录每次较好的精确率
    with open("accuracy_log.txt", "a") as file:  # "a" 表示追加模式
        file.write(f"traning:{optimizer_name}\n")  # 写入一行数据
    #开始训练
    for epoch in range(num_epochs): #一个epoch跑完一遍测试集
        #只有SGD是动态调整
        if i==0:
            #动态调整学习率
            if counter / 10 == 1:
                counter = 0
                Lr = Lr * 0.5
            #重新设置优化器
            optimizer = optim.SGD(net.parameters(), lr=Lr, momentum=0.9, weight_decay=5e-4)
        net.train()  # 设置网络为训练模式
        running_loss = 0.0  #记录一轮的损失
        correct = 0
        total = 0
        # 使用tqdm显示训练进度条
        train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in train_bar:    #每一轮训练一个批次,每个批次大小为设置的128
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数
            #loss.item()是一个batch的平均损失
            running_loss += loss.item()
            _, predicted = outputs.max(1)  # 获取预测结果
            #样本总数
            total += labels.size(0)    #注意是labels.size
            #正确个数
            correct += predicted.eq(labels).sum().item()

            # 更新进度条信息
            train_bar.set_postfix(loss=loss.item(),
                                  acc=100.0 * correct / total)



        #计算每一轮的损失度和准确率并添加进数组
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        writer.add_scalar(f"{optimizer_name}/TrainLoss", train_loss, epoch)
        writer.add_scalar(f"{optimizer_name}/TrainAcc", train_acc, epoch)

        #评估模型
        net.eval()  # 设置网络为评估模式
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # 禁用梯度计算
            # 使用tqdm显示验证进度条
            test_bar = tqdm(test_loader, desc=f'Validating Epoch {epoch + 1}/{num_epochs}')
            for inputs, labels in test_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                test_bar.set_postfix(loss=loss.item(),
                                     acc=100.0 * correct / total)

        test_loss = test_loss / len(test_loader)
        test_acc = 100.0 * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        writer.add_scalar(f"{optimizer_name}/Valid Loss", train_loss, epoch)
        writer.add_scalar(f"{optimizer_name}/Valid Acc", train_acc, epoch)
        # 每一轮结束后调用学习率调度器
        #scheduler.step(test_loss)
        save_path="./model"
        # 保存最好的模型
        if test_acc > best_acc:
            best_acc = test_acc
            # 将数据追加到文件中
            with open("accuracy_log.txt", "a") as file:  # "a" 表示追加模式
                file.write(f"{epoch + 1},{best_acc:.2f}\n")  # 写入一行数据
            torch.save(net.state_dict(), os.path.join(save_path, f"{optimizer_name}_model.pth"))
            counter=0
        else:
            counter += 1

    return net,train_losses,train_accs,test_losses,test_accs

# 损失度和精确率可视化
def plot_loss_acc(train_losses,train_accs,all_test_losses,all_test_accs):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    #遍历可视化各个算法的ltrain_loss
    for name, loss_values in train_losses.items():
        plt.plot(loss_values, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves for Different Algorithms')
    plt.legend()

    plt.subplot(2, 2, 2)
    for name, correct_values in train_accs.items():
        plt.plot(correct_values, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy Curves for Different Algorithms')
    plt.legend()

    plt.subplot(2, 2, 3)
    for name, loss_values in all_test_losses.items():
        plt.plot(loss_values, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Valid Loss')
    plt.title('Valid Loss Curves for Different Algorithms')
    plt.legend()

    plt.subplot(2, 2, 4)
    for name, correct_values in all_test_accs.items():
        plt.plot(correct_values, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Valid Accuracy')
    plt.title('Valid Accuracy Curves for Different Algorithms')
    plt.legend()

    # 保存图表
    plt.savefig('./results/plots/loss_accuracy_curves1.png')
    plt.show()

#主函数
def main():
    # Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.错误解决方法
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    #训练轮数
    num_epochs = 150

    optimizer_names=["SGD+Momentum","RMSprop","Adam","AdamW","Adagrad","Adadelta"]
    # 保存路径
    if not os.path.exists('./model'):
        os.makedirs('./model')
    if not os.path.exists('results/plots'):
        os.makedirs('results/plots')
    if not os.path.exists('results/roc_curves'):
        os.makedirs('results/roc_curves')
    with open("accuracy_log.txt", "w") as file:
        file.write("Epoch,Accuracy\n")  # 写入文件头

    #创建字典记录各个算法的损失度、准确率
    all_train_losses = {name: [] for name in optimizer_names}
    all_train_accs = {name: [] for name in optimizer_names}
    all_test_losses = {name: [] for name in optimizer_names}
    all_test_accs = {name: [] for name in optimizer_names}

    for i in range(len(optimizer_names)):
        # # 加载预训练的 ResNet18
        # net = models.resnet18(pretrained=True)
        # # 替换最后一层，以匹配你的数据集的类别数
        # num_classes = 10  # CIFAR-10 数据集有 10 个类别
        # net.fc = nn.Linear(net.fc.in_features, num_classes)
        # net = net.to(device)
        # 定义模型
        model_ft = torchvision.models.resnet18(pretrained=False)
        # 修改模型
        model_ft.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
        model_ft.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
        num_ftrs = model_ft.fc.in_features  # 获取（fc）层的输入的特征数
        # 替换最后一层，以匹配你的数据集的类别数
        model_ft.fc = nn.Linear(num_ftrs, 10)
        net = model_ft.to(device)
        #优化器名称
        optimizer_name = optimizer_names[i]
        #初始化优化器
        if i==0:
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        elif i==1:
            optimizer = optim.RMSprop(net.parameters(), lr=0.001)
        elif i==2:
            optimizer = optim.Adam(net.parameters(), lr=0.001)
        elif i==3:
            optimizer = optim.AdamW(net.parameters(), lr=0.001)
        elif i==4:
            optimizer = optim.Adagrad(net.parameters(), lr=0.01)
        elif i==5:
            optimizer = optim.Adadelta(net.parameters(), lr=1.0)

        print(f'Training with {optimizer_name}')
        #保存各个优化器的训练损失，训练精确率到数组
        net,train_losses, train_accs ,test_losses,test_accs= train_and_test(net,optimizer_name,optimizer, num_epochs,i)  # 训练网络
        all_train_losses[optimizer_name]= train_losses
        all_train_accs[optimizer_name]= train_accs
        all_test_losses[optimizer_name]= test_losses
        all_test_accs[optimizer_name]= test_accs

    #绘制损失、精确率
    plot_loss_acc(all_train_losses,all_train_accs,all_test_losses,all_test_accs)



if __name__ == '__main__':
    main()