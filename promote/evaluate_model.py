# 导入必要的库
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from matplotlib import pyplot as plt
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, \
    auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.metrics import roc_curve

from  net.lenet import LeNet

def EModel():
    transform_test = transforms.Compose([  # 测试验证用
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 测试集加载
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=2)


    #加载训练模型
    net = LeNet()
    net.load_state_dict(torch.load('./model/test.pth'))
    # 评估模型，评价指标
    net.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():  # 在评估阶段不需要计算梯度
        for data in test_loader:
            images, labels = data
            outputs = net(images)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 计算
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
    # 计算混淆矩阵、精确率、召回率和F1分数
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f"SDG_Confusion_Matrix.png")  # 保存混淆矩阵图

    # 计算AUC-ROC曲线
    y_test_binarized = label_binarize(all_labels, classes=np.arange(10))
    print(y_test_binarized)
    n_classes = y_test_binarized.shape[1]
    print(n_classes)
    y_score = np.array(all_probs)[:, :n_classes]
    print(y_score)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 计算每个类别的FPR、TPR和AUC
    for i in range(n_classes):
        # 计算第i个类别的FPR、TPR和AUC
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制AUC-ROC曲线
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for class {}'.format(i))
        plt.legend(loc="lower right")
        plt.show()


    # 计算Top-k准确率
    def top_k_accuracy(output, target, k=5):
        pred = output.topk(k, 1, True, True)[1]
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct = correct[0].view(-1).float().sum()
        return correct / output.size(0)


    top1_acc = top_k_accuracy(net(images), labels, 1)
    top5_acc = top_k_accuracy(net(images), labels, 5)

    print(f'Top-1 Accuracy: {top1_acc:.3f}')
    print(f'Top-5 Accuracy: {top5_acc:.3f}')

    print('Accuracy on the test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    EModel()