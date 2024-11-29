# 导入数据集
from torchvision import datasets
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trunk')
cifar_train = datasets.CIFAR10(root="../data",train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(root="../data",train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=32, shuffle=False)
Resnet50 = torchvision.models.resnet50(pretrained=True)
Resnet50.fc.out_features=10
# 定义损失函数和优化器
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Resnet50.parameters(), lr=0.001)
epoch = 10
Resnet50 = Resnet50.to(device)
total_step = len(train_loader)
train_all_loss = []
test_all_loss=[]
val_all_loss = []
for i in range(epoch):
    Resnet50.train()
    train_total_loss = 0
    train_total_num = 0
    train_total_correct = 0

    for iter, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = Resnet50(images)
        loss = criterion(outputs, labels)
        train_total_correct += (outputs.argmax(1) == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_total_num += labels.shape[0]
        train_total_loss += loss.item()
        print("Epoch [{}/{}], Iter [{}/{}], train_loss:{:4f}".format(i + 1, epoch, iter + 1, total_step,
                                                                     loss.item() ))
    Resnet50.eval()
    test_total_loss = 0
    test_total_correct = 0
    test_total_num = 0
    for iter, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = Resnet50(images)
        loss = criterion(outputs, labels)
        test_total_correct += (outputs.argmax(1) == labels).sum().item()
        test_total_loss += loss.item()
        test_total_num += labels.shape[0]
    print("Epoch [{}/{}], train_loss:{:.4f}, train_acc:{:.4f}%, test_loss:{:.4f}, test_acc:{:.4f}%".format(
        i + 1, epoch, train_total_loss / len(train_loader), train_total_correct / train_total_num * 100,
        test_total_loss / len(test_loader), test_total_correct / test_total_num * 100

    ))
    train_all_loss.append(np.round(train_total_loss / train_total_num, 4))
    test_all_loss.append(np.round(test_total_loss / test_total_num, 4))
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.title("Train Loss and Test Loss Curve")
plt.xlabel('plot_epoch')
plt.ylabel('loss')
plt.plot(train_all_loss)
plt.plot(test_all_loss)
plt.legend(['train loss', 'test loss'])

