# 导入必要的库
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

from keras.src.metrics import F1Score
from matplotlib import pyplot as plt
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, \
    auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.metrics import roc_curve
from terminaltables import AsciiTable

from  net.lenet import LeNet
# 计算Top-k准确率
def top_k_accuracy(output, target, k=5):
    pred = output.topk(k, 1, True, True)[1]
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    correct = correct[0].view(-1).float().sum()
    return correct / output.size(0)
# 测试函数
def test_model(net, device,test_loader,k=5):
    net.eval()
    correct = 0
    total=0
    all_preds = []
    all_labels = []
    all_probs=[]
    correct_1 = 0
    correct_k = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            # Top-1 准确率
            _, pred_1 = outputs.max(1)
            correct_1 += pred_1.eq(labels).sum().item()

            # Top-k 准确率
            _, pred_k = outputs.topk(k, dim=1)
            correct_k += (pred_k.t() == labels.view(1, -1).expand_as(pred_k.t())).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
    accuracy = 100.0 *correct / total
    top1_acc = 100. * correct_1 / total
    topk_acc = 100. * correct_k / total
    # print(f'Top-1 Acc: {top1_acc:.3f}%, Top-{k} Acc: {topk_acc:.3f}%')
    # print(f'Accuracy: {accuracy:.3f}')
    return top1_acc,topk_acc,accuracy, all_labels, all_preds,all_probs

# 绘制混淆矩阵、计算精确率、召回率、F1分数
def evaluate(net, device, optimizer_name,testloader, y_true, y_pred):
    #混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',cmap='Blues')
    plt.title(f"{optimizer_name}_Confusion Matrix")
    plt.savefig(f"./results/confusion_matrix/{optimizer_name}_confusion_matrix.png")
    plt.show()
    #精确率
    precision = precision_score(y_true, y_pred, average='weighted')
    #召回率
    recall = recall_score(y_true, y_pred, average='weighted')
    #F1分数
    f1 = f1_score(y_true, y_pred, average='weighted')
    #print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    return precision, recall, f1

#绘制ROC曲线
def plot_auc_roc(all_labels,all_probs,optimizer_name):
    # 计算ROC曲线
    #将标签二值化
    y_test_binarized = label_binarize(all_labels, classes=np.arange(10))
    #print(y_test_binarized)
    #设置种类
    n_classes = y_test_binarized.shape[1]
    #print(n_classes)
    y_score = np.array(all_probs)[:, :n_classes]
    #print(y_score)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 计算每个类别的FPR、TPR和AUC
    for i in range(n_classes):
        # 计算第i个类别的FPR、TPR和AUC
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制所有类别的ROC曲线在同一张图上
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='Class {} (area = {:.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{optimizer_name}_Receiver Operating Characteristic for CIFAR-10')
    plt.legend(loc="lower right")
    plt.savefig(f"./results/roc_curves/{optimizer_name}_roc_curve.png")
    plt.show()

def EModel():
    # 测试集通常不做数据增强，只归一化）
    test_transform = transforms.Compose([
        transforms.ToTensor()
        , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)
    # GPU计算
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 定义模型
    model_ft = torchvision.models.resnet18(pretrained=False)
    # 修改模型
    model_ft.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
    model_ft.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
    num_ftrs = model_ft.fc.in_features  # 获取（fc）层的输入的特征数
    model_ft.fc = nn.Linear(num_ftrs, 10)

    optimizer_names = ["SGD+Momentum", "RMSprop", "Adam", "AdamW", "Adagrad", "Adadelta"]
    #
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    top1_acc=0.0
    topk_acc=0.0
    precisions = {name: 0.0 for name in optimizer_names}
    recalls = {name: 0.0 for name in optimizer_names}
    f1s = {name: 0.0 for name in optimizer_names}
    with open('output_dict.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Top-1 Acc', 'Top-5 Acc', 'Accuracy','Mean Precision', 'Mean Recall', 'Mean F1 Score']  # 定义字段名
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 写入表头
        writer.writeheader()

    for i in range(6):
        # 评估模型
        model_ft.eval()
        optimizer_name= optimizer_names[i]
        state_dict = torch.load("./model/SGD+Momentum_model.pth", map_location=torch.device('cpu'))
        if i == 0:
            state_dict = torch.load("./model/SGD+Momentum_model.pth", map_location=torch.device('cpu'))
        elif i == 1:
            state_dict = torch.load("./model/RMSprop_model.pth", map_location=torch.device('cpu'))
        elif i == 2:
            state_dict = torch.load("./model/Adam_model.pth", map_location=torch.device('cpu'))
        elif i == 3:
            state_dict = torch.load("./model/Adam_model.pth", map_location=torch.device('cpu'))
        elif i == 4:
            state_dict = torch.load("./model/Adagrad_model.pth", map_location=torch.device('cpu'))
        elif i == 5:
            state_dict = torch.load("./model/Adadelta_model.pth", map_location=torch.device('cpu'))
        model_ft.load_state_dict(state_dict)
        #评估模型
        top1_acc,topk_acc,accuracy, all_labels,all_preds,all_probs = test_model(model_ft,device,test_loader)
        # roc曲线
        plot_auc_roc(all_labels, all_probs, optimizer_name)
        #精确率、召回率、f1分数
        precision, recall, f1 = evaluate(model_ft,device,optimizer_name,test_loader,all_labels, all_preds)
        precisions[optimizer_name] = precision  #精确度
        recalls[optimizer_name] = recall  #召回率
        f1s[optimizer_name] = f1   #F1分数

        TITLE = f'  {optimizer_name}Total Results'
        TABLE_DATA_2 = [
            ['Top-1 Acc', 'Top-5 Acc', 'Accuracy','Mean Precision', 'Mean Recall', 'Mean F1 Score'],
            ['{:.4f}'.format(top1_acc),
            '{:.4f}'.format(topk_acc),
            '{:.4f}'.format(accuracy),
            '{:.4f}'.format(precision),
             '{:.4f}'.format(recall),
             '{:.4f}'.format(f1)],
        ]
        # 创建表格实例
        table_instance = AsciiTable(TABLE_DATA_2, TITLE)
        # table_instance.justify_columns[2] = 'right'
        print(table_instance.table)
        # writer.writerows(TABLE_DATA_2)
        # writer.writerow([])
        print()



if __name__ == '__main__':
    EModel()