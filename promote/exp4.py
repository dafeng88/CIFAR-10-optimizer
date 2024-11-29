import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch import optim

mytrans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.45, 0.4), (0.23, 0.22, 0.21))])

train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=mytrans)
train_set = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)

test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=mytrans)
test_set = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True, num_workers=0)

model = torchvision.models.resnet50(weights=True)
nums = model.fc.in_features
model.fc = nn.Linear(nums, 10)

device = torch.device("cuda:0")
# device = torch.device("cpu")
model = model.to(device)

myoptim = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
myloss = nn.CrossEntropyLoss()

n_epoch = 10
for epoch in range(1, n_epoch + 1):
    print("epoch {}/{}".format(epoch, n_epoch))
    for step, test_data in enumerate(train_set):
        image, label = test_data[0].to(device), test_data[1].to(device)
        predict_label = model.forward(image)
        loss = myloss(predict_label, label)

        myoptim.zero_grad()
        loss.backward()
        myoptim.step()

        end = "" if step != len(train_set) - 1 else "\n"
        print("\rtrain iteration {}/{} - training loss: {:.3f}".format(step + 1, len(train_set), loss.item()), end=end)

    model.eval()
    total = 0
    correct = 0
    for step, test_data in enumerate(test_set):
        image = test_data[0].to(device)
        label = test_data[1].to(device)

        outputs = model(image)
        _, pred = torch.max(outputs.data, 1)

        total += label.size(0)
        correct += (pred == label).sum().item()

        end = "" if step != len(test_set) - 1 else "\n"
        print(
            '\rtest iteration {}/{} - testing accuracy: {:.3f}%'.format(step + 1, len(test_set), 100 * correct / total),
            end=end)
