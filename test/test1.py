import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from microexpression_dataset import MicroExpressionDataset, load_dataset_with_padding

# 定义3DCNN模型
class CNN3D(nn.Module):
    def __init__(self, num_classes):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc1 = nn.Linear(128*16*16*16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        # 16*128*16*16*16
        # 获取张量的形状

        x = torch.flatten(x, start_dim=1)  # Efficient flattening
        x = self.relu(self.fc1(x))
        x=self.fc2(x)
        return x

# 数据路径和超参数
# excel_file = "E:/桌面/论文写作/微表情识别/数据集/CASME2/CASME2-coding.xlsx"
# dataset_root = "E:/桌面/论文写作/微表情识别/数据集/CASME2/Cropped-updated/Cropped"
excel_file = "/root/autodl-tmp/CASME2/CASME2-coding.xlsx"
dataset_root = "/root/autodl-tmp/CASME2/Cropped-updated/Cropped"
seq_len = 64
batch_size = 16
epochs =120
learning_rate = 0.001  # SGD初始学习率
momentum = 0.9
num_classes = 5  # 根据实际情感类别数量调整

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 默认使用第0块显卡

# 定义数据增强的转换
base_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 定义旋转增强转换
rotate_transform_5_ccw = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(degrees=(-5, -5)),
    transforms.ToTensor()
])

rotate_transform_10_ccw = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(degrees=(-10, -10)),
    transforms.ToTensor()
])

rotate_transform_5_cw = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(degrees=(5, 5)),
    transforms.ToTensor()
])

rotate_transform_10_cw = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(degrees=(10, 10)),
    transforms.ToTensor()
])

# 加载原始数据集
train_df = load_dataset_with_padding(excel_file, dataset_root, seq_len=seq_len)

# 创建多个增强版本的数据集
base_dataset = MicroExpressionDataset(samples_df=train_df, transform=base_transform)
dataset_5_ccw = MicroExpressionDataset(samples_df=train_df, transform=rotate_transform_5_ccw)
dataset_10_ccw = MicroExpressionDataset(samples_df=train_df, transform=rotate_transform_10_ccw)
dataset_5_cw = MicroExpressionDataset(samples_df=train_df, transform=rotate_transform_5_cw)
dataset_10_cw = MicroExpressionDataset(samples_df=train_df, transform=rotate_transform_10_cw)

# 合并数据集
augmented_dataset = ConcatDataset([
    base_dataset,
    dataset_5_ccw,
    dataset_10_ccw,
    dataset_5_cw,
    dataset_10_cw
])

# 数据集划分为训练集和测试集 (80% 训练, 20% 测试)
train_size = int(0.8 * len(augmented_dataset))
test_size = len(augmented_dataset) - train_size
train_dataset, test_dataset = random_split(augmented_dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = CNN3D(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# 使用学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 每5个epoch学习率减少为原来的0.1

from sklearn.metrics import f1_score, recall_score

def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing")  # 使用tqdm显示测试进度条
        for videos, labels in progress_bar:
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total

    # 计算UF1和UAR
    uf1 = f1_score(all_labels, all_predictions, average='macro')
    uar = recall_score(all_labels, all_predictions, average='macro')

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"UF1: {uf1:.4f}")
    print(f"UAR: {uar:.4f}")

# 训练函数
#
# def train_and_test(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
#     model.train()
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         correct = 0
#         total = 0
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")  # 使用tqdm显示进度条
#         for videos, labels in progress_bar:
#             videos, labels = videos.to(device), labels.to(device)
#
#             # 前向传播
#             outputs = model(videos)
#             loss = criterion(outputs, labels)
#
#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # 统计损失和准确率
#             epoch_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#             # 更新进度条描述
#             progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
#
#         accuracy = 100 * correct / total
#         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
#
#         # 更新学习率
#         scheduler.step()
#
#         # 每轮训练后进行测试
#         print("Running test after epoch:")
#         test(model, test_loader)
#


def train_and_test(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    model.train()
    for epoch in range(epochs):
        # 在前5个epoch内进行warm-up，逐步增加学习率
        if epoch < 5:
            warmup_lr = learning_rate * (epoch + 1) / 5  # 线性增长
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Epoch {epoch+1}: Warm-up learning rate set to {warmup_lr:.6f}")
        else:
            # 从第6个epoch开始使用学习率调度器调整学习率
            scheduler.step()

        epoch_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")  # 使用tqdm显示进度条
        for videos, labels in progress_bar:
            videos, labels = videos.to(device), labels.to(device)

            # 前向传播
            outputs = model(videos)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条描述
            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # 每轮训练后进行测试
        print("Running test after epoch:")
        test(model, test_loader)
# 运行训练和测试
train_and_test(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs)