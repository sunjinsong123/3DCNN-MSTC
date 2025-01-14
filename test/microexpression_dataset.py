# -*- coding: utf-8 -*-
# @Time :2025/1/12 21:58
# @Author :孙劲松
# @File :microexpression_dataset.py
# @Software :PyCharm
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
def load_dataset_with_padding(excel_file, dataset_root, seq_len=64, classification="5-class"):
    """
    从Excel文件加载数据集，并将样本帧数不足的补充到指定长度。

    classification: 指定分类方式，"5-class" 或 "3-class"
    """
    try:
        data = pd.read_excel(excel_file)
    except Exception as e:
        raise FileNotFoundError(f"无法读取Excel文件: {e}")

    if classification == "5-class":
        # 标签映射：包含 others，舍弃 fear 和 sadness
        label_map = {
            "happiness": 0,
            "disgust": 1,
            "repression": 2,
            "surprise": 3,
            "others": 4  # 保留 others 分类
        }
    elif classification == "3-class":
        # 标签映射：3分类，将厌恶、压抑归为消极，舍弃 fear 和 sadness
        label_map = {
            "happiness": 0,  # 积极
            "disgust": 1,  # 消极
            "repression": 1,  # 消极
            "surprise": 2,  # 惊讶
            # "others": 2  # 归类到惊讶
        }
    else:
        raise ValueError("分类方式错误，请选择 '5-class' 或 '3-class'")

    samples = []

    for _, row in data.iterrows():
        subject = f"sub{int(row['Subject']):02d}"
        filename = row['Filename']
        emotion = row['Estimated Emotion']

        # 跳过标签不在映射中的样本
        if emotion not in label_map:
            continue

        label = label_map[emotion]
        sample_folder = os.path.join(dataset_root, subject, filename)

        if os.path.exists(sample_folder):
            frames = sorted([os.path.join(sample_folder, f) for f in os.listdir(sample_folder) if f.endswith(".jpg")])

            if len(frames) == 0:
                continue

            if len(frames) < seq_len:
                frames += [frames[-1]] * (seq_len - len(frames))

            frames = frames[:seq_len]

            sample_info = {
                "Subject": subject,
                "Filename": filename,
                "FolderPath": sample_folder,
                "Frames": frames,
                "Emotion": label  # 使用整数标签
            }
            samples.append(sample_info)

    # 打印每个类别的样本数量
    print("样本分布：")
    labels = [s["Emotion"] for s in samples]
    label_counts = pd.Series(labels).value_counts()
    print(label_counts)

    samples_df = pd.DataFrame(samples)
    return samples_df



class MicroExpressionDataset(Dataset):
    """
    微表情数据集的PyTorch自定义Dataset类。
    """
    def __init__(self, samples_df, transform=None):
        self.samples = samples_df
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        frames = sample["Frames"]
        label = sample["Emotion"]

        frame_tensors = []
        for frame_path in frames:
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frame_tensors.append(img)

        video_tensor = torch.stack(frame_tensors).permute(1, 0, 2, 3)
        return video_tensor, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    # Excel文件路径和数据集根目录路径
    excel_file = "E:/桌面/论文写作/微表情识别/数据集/CASME2/CASME2-coding.xlsx"
    dataset_root = "E:/桌面/论文写作/微表情识别/数据集/CASME2/Cropped-updated/Cropped"

    # 加载数据集并处理为固定长度序列
    seq_len = 64

    # 选择分类方式：5分类或3分类
    classification = "3-class"  # 改为 "5-class" 可切换到5分类
    dataset_df = load_dataset_with_padding(excel_file, dataset_root, seq_len=seq_len, classification=classification)

    # 准备PyTorch的数据集和数据加载器
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = MicroExpressionDataset(samples_df=dataset_df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
