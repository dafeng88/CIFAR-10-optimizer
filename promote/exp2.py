import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time


# ../input/cifar10-python

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_data = torchvision.datasets.CIFAR10("../data", train=True, transform=torchvision.transforms.Compose(
    [torchvision.transforms.ColorJitter(0.5), torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()]))
test_data = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor())

train_dataloader = DataLoader(train_data, batch_size=128)
test_dataloader = DataLoader(test_data, batch_size=128)
# print(len(train_dataloader)) #781
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_data_size = len(test_data)
train_data_size = len(train_data)
print(f'测试集大小为：{test_data_size}')
print(f'训练集大小为：{train_data_size}')
writer = SummaryWriter("../model_logs")

loss_fn = nn.CrossEntropyLoss(reduction='mean')
loss_fn = loss_fn.to(device)
time_able = True  # True


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = conv_block(3, 32)  # 3,32,32
        self.conv2 = conv_block(32, 64, pool=True)  # 64,16,16
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))  # 64, 16, 16
        self.conv3 = conv_block(64, 128)  # 128, 16, 16
        self.conv4 = conv_block(128, 256, pool=True)  # 256, 8, 8
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))  # 256,8,8
        self.conv5 = conv_block(256, 512)  # 512, 8, 8
        self.conv6 = conv_block(512, 1024, pool=True)  # 1024, 4, 4
        self.res3 = nn.Sequential(conv_block(1024, 1024), conv_block(1024, 1024))  # 1024, 4, 4
        self.linear1 = nn.Sequential(nn.MaxPool2d(4),  # 1024,1,1
                                     nn.Flatten(),
                                     nn.Dropout(0.2),
                                     nn.Linear(1024, 10))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out) + out
        out = self.linear1(out)
        return out


model = Model()
model = model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
epoch = 11
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=epoch,
                                            steps_per_epoch=len(train_dataloader))
running_loss = 0
total_train_step = 0
total_test_step = 0
if time_able:
    str_time = time.time()
for i in range(epoch):
    print(f'第{i + 1}次epoch')
    model.train()
    lrs = []
    total_accuracy1 = 0
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = model(imgs)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sched.step()
        lrs.append(get_lr(optimizer))
        total_train_step += 1
        print(f"loss:{loss}")
        if total_train_step % 200 == 0:
            if time_able:
                end_time = time.time()
                print(f'{end_time - str_time}')
            print(f'第{total_train_step}次训练，loss = {loss.item()},lr_last = {lrs[-1]}')
            writer.add_scalar("train_loss", loss.item(), total_train_step)
        accuracy1 = (output.argmax(1) == targets).sum()
        total_accuracy1 += accuracy1

    # 测试
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        model.eval()
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    total_test_loss = total_test_loss / test_data_size
    print(f'整体测试集上的loss = {total_test_loss}')
    print(f'整体测试集正确率 = {total_accuracy / test_data_size}')
    print(f'整体训练集正确率 = {total_accuracy1 / train_data_size}')
    writer.add_scalar("test_loss", total_test_loss.item(), total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    writer.add_scalar("train_accuracy", total_accuracy1 / train_data_size, total_test_step)  # test_step == epoch
    total_test_step += 1

writer.close()

#来源：https://blog.csdn.net/weixin_42037511/article/details/124201187?spm=1001.2014.3001.5502