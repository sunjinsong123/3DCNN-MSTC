import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from microexpression_dataset import MicroExpressionDataset, load_dataset_with_padding
from model.cnn3d import CNN3D1 as CNN3D

# 数据路径和超参数
excel_file = "/root/autodl-tmp/CASME2/CASME2-coding.xlsx"
dataset_root = "/root/autodl-tmp/CASME2/Cropped-updated/Cropped"
seq_len = 64
batch_size = 16
epochs = 90
learning_rate = 0.001
momentum = 0.9
num_classes = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义数据增强的转换
base_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

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
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

from sklearn.metrics import f1_score, recall_score

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

count_parameters(model)

# 测试函数
def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing")
        for videos, labels in progress_bar:
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    uf1 = f1_score(all_labels, all_predictions, average='macro')
    uar = recall_score(all_labels, all_predictions, average='macro')

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"UF1: {uf1:.4f}")
    print(f"UAR: {uar:.4f}")

    return accuracy

# 训练和测试函数
def train_and_test(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    best_accuracy = 0.0
    best_model_path = "best_model.pth"

    model.train()
    for epoch in range(epochs):
        if epoch < 5:
            warmup_lr = learning_rate * (epoch + 1) / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Epoch {epoch + 1}: Warm-up learning rate set to {warmup_lr:.6f}")
        else:
            scheduler.step()

        epoch_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for videos, labels in progress_bar:
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        print("Running test after epoch:")
        test_accuracy = test(model, test_loader)

        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

train_and_test(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs)
