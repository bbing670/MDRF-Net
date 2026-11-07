import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

# 自定义数据集类（单日版）
class DroughtResistanceDataset(Dataset):
    def __init__(self, experimental_dir, control_dir, label_file, transform=None):
        self.experimental_dir = experimental_dir
        self.control_dir = control_dir
        self.transform = transform
        
        # 读取标签文件
        self.labels_df = pd.read_csv(label_file)
        # 创建品种名到标签的映射字典
        self.label_map = {str(id): label for id, label in zip(self.labels_df['id'], self.labels_df['label'])}
        
        # 获取所有品种名（从实验组文件夹）
        self.species_list = [f for f in os.listdir(experimental_dir) 
                             if os.path.isfile(os.path.join(experimental_dir, f))]
        
        # 过滤掉没有对应对照组图像的样本
        self.species_list = [s for s in self.species_list 
                             if os.path.exists(os.path.join(control_dir, s))]
    
    def __len__(self):
        return len(self.species_list)
    
    def __getitem__(self, idx):
        # 获取品种名（文件名）
        species_name = self.species_list[idx]
        
        # 构建图像路径
        experimental_path = os.path.join(self.experimental_dir, species_name)
        control_path = os.path.join(self.control_dir, species_name)
        
        # 加载图像
        exp_img = Image.open(experimental_path).convert('RGB')
        ctrl_img = Image.open(control_path).convert('RGB')
        
        # 获取标签（去除文件扩展名以匹配品种名）
        species_id = os.path.splitext(species_name)[0]
        if species_id in self.label_map:
            label = self.label_map[species_id]
        else:
            # 如果找不到标签，默认设为0
            label = 0
            print(f"警告：未找到品种 {species_id} 的标签")
        
        # 应用变换
        if self.transform:
            exp_img = self.transform(exp_img)
            ctrl_img = self.transform(ctrl_img)
        
        return exp_img, ctrl_img, torch.tensor(label, dtype=torch.long)
    
    def get_labels(self):
        """获取数据集中所有样本的标签"""
        labels = []
        for idx in range(len(self)):
            _, _, label = self.__getitem__(idx)
            labels.append(label.item())
        return labels

# 自定义数据集类（多日版，支持多天数据）
class MultiDayDroughtResistanceDataset(Dataset):
    def __init__(self, base_dir, days, label_file, transform=None):
        self.base_dir = base_dir
        self.days = days  # 天数列表
        self.transform = transform
        
        # 读取标签文件
        self.labels_df = pd.read_csv(label_file)
        # 创建品种名到标签的映射字典
        self.label_map = {str(id): label for id, label in zip(self.labels_df['id'], self.labels_df['label'])}
        
        # 获取所有品种名（从第一个天的实验组文件夹）
        first_day_exp_dir = os.path.join(base_dir, f"{days[0]}experimental")
        self.species_list = [f for f in os.listdir(first_day_exp_dir) 
                             if os.path.isfile(os.path.join(first_day_exp_dir, f))]
        
        # 过滤掉没有对应所有天对照组和实验组图像的样本
        filtered_species = []
        for species in self.species_list:
            has_all_files = True
            for day in days:
                exp_path = os.path.join(base_dir, f"{day}experimental", species)
                ctrl_path = os.path.join(base_dir, f"{day}control", species)
                if not (os.path.exists(exp_path) and os.path.exists(ctrl_path)):
                    has_all_files = False
                    break
            if has_all_files:
                filtered_species.append(species)
        
        self.species_list = filtered_species
        print(f"过滤后的数据样本数量: {len(self.species_list)}")
    
    def __len__(self):
        return len(self.species_list)
    
    def __getitem__(self, idx):
        # 获取品种名（文件名）
        species_name = self.species_list[idx]
        
        # 加载所有天的图像
        exp_images = []
        ctrl_images = []
        
        for day in self.days:
            # 构建图像路径
            experimental_path = os.path.join(self.base_dir, f"{day}experimental", species_name)
            control_path = os.path.join(self.base_dir, f"{day}control", species_name)
            
            # 加载图像
            exp_img = Image.open(experimental_path).convert('RGB')
            ctrl_img = Image.open(control_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                exp_img = self.transform(exp_img)
                ctrl_img = self.transform(ctrl_img)
            
            exp_images.append(exp_img)
            ctrl_images.append(ctrl_img)
        
        # 将图像列表转换为张量，形状为 [C, T, H, W]
        exp_images = torch.stack(exp_images, dim=1)  # [C, T, H, W]
        ctrl_images = torch.stack(ctrl_images, dim=1)  # [C, T, H, W]
        
        # 获取标签（去除文件扩展名以匹配品种名）
        species_id = os.path.splitext(species_name)[0]
        if species_id in self.label_map:
            label = self.label_map[species_id]
        else:
            # 如果找不到标签，默认设为0
            label = 0
            print(f"警告：未找到品种 {species_id} 的标签")
        
        return exp_images, ctrl_images, torch.tensor(label, dtype=torch.long)
    
    def get_labels(self):
        """获取数据集中所有样本的标签"""
        labels = []
        for idx in range(len(self)):
            _, _, label = self.__getitem__(idx)
            labels.append(label.item())
        return labels

def create_dataloaders(experimental_dir, control_dir, label_file, batch_size=4, 
                     train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, num_workers=2, seed=42):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    dataset = DroughtResistanceDataset(
        experimental_dir=experimental_dir,
        control_dir=control_dir,
        label_file=label_file,
        transform=transform
    )
    
    # 设置随机种子以确保分割的可重复性
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 获取所有标签
    labels = np.array(dataset.get_labels())
    
    # 分离正样本和负样本的索引
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    
    print(f"原始数据集中正样本数量: {len(positive_indices)}, 负样本数量: {len(negative_indices)}")
    print(f"原始正负样本比例: {len(positive_indices)}:{len(negative_indices)}")
    
    # 对正样本和负样本分别进行分层采样
    def split_indices(indices, train_r, val_r, test_r):
        total = len(indices)
        train_size = int(train_r * total)
        val_size = int(val_r * total)
        test_size = total - train_size - val_size
        
        # 打乱索引
        np.random.shuffle(indices)
        
        return indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]
    
    # 分割正样本
    pos_train, pos_val, pos_test = split_indices(positive_indices, train_ratio, val_ratio, test_ratio)
    # 分割负样本
    neg_train, neg_val, neg_test = split_indices(negative_indices, train_ratio, val_ratio, test_ratio)
    
    # 合并正负样本索引
    train_indices = np.concatenate([pos_train, neg_train])
    val_indices = np.concatenate([pos_val, neg_val])
    test_indices = np.concatenate([pos_test, neg_test])
    
    # 创建数据集子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # 计算各数据集的正负样本比例
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]
    
    print(f"\n训练集大小: {len(train_dataset)} ({train_ratio*100:.1f}%)")
    print(f"训练集正样本数量: {np.sum(train_labels == 1)}, 负样本数量: {np.sum(train_labels == 0)}")
    print(f"训练集正负样本比例: {np.sum(train_labels == 1)}:{np.sum(train_labels == 0)}")
    
    print(f"\n验证集大小: {len(val_dataset)} ({val_ratio*100:.1f}%)")
    print(f"验证集正样本数量: {np.sum(val_labels == 1)}, 负样本数量: {np.sum(val_labels == 0)}")
    print(f"验证集正负样本比例: {np.sum(val_labels == 1)}:{np.sum(val_labels == 0)}")
    
    print(f"\n测试集大小: {len(test_dataset)} ({test_ratio*100:.1f}%)")
    print(f"测试集正样本数量: {np.sum(test_labels == 1)}, 负样本数量: {np.sum(test_labels == 0)}")
    print(f"测试集正负样本比例: {np.sum(test_labels == 1)}:{np.sum(test_labels == 0)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, dataset

# 创建多日数据加载器的函数
def create_multi_day_dataloaders(base_dir, days, label_file, batch_size=4, 
                               train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, num_workers=2, seed=42):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建多日数据集
    dataset = MultiDayDroughtResistanceDataset(
        base_dir=base_dir,
        days=days,
        label_file=label_file,
        transform=transform
    )
    
    # 设置随机种子以确保分割的可重复性
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 获取所有标签
    labels = np.array(dataset.get_labels())
    
    # 分离正样本和负样本的索引
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    
    print(f"原始数据集中正样本数量: {len(positive_indices)}, 负样本数量: {len(negative_indices)}")
    print(f"原始正负样本比例: {len(positive_indices)}:{len(negative_indices)}")
    
    # 对正样本和负样本分别进行分层采样
    def split_indices(indices, train_r, val_r, test_r):
        total = len(indices)
        train_size = int(train_r * total)
        val_size = int(val_r * total)
        test_size = total - train_size - val_size
        
        # 打乱索引
        np.random.shuffle(indices)
        
        return indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]
    
    # 分割正样本
    pos_train, pos_val, pos_test = split_indices(positive_indices, train_ratio, val_ratio, test_ratio)
    # 分割负样本
    neg_train, neg_val, neg_test = split_indices(negative_indices, train_ratio, val_ratio, test_ratio)
    
    # 合并正负样本索引
    train_indices = np.concatenate([pos_train, neg_train])
    val_indices = np.concatenate([pos_val, neg_val])
    test_indices = np.concatenate([pos_test, neg_test])
    
    # 创建数据集子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # 计算各数据集的正负样本比例
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]
    
    print(f"\n训练集大小: {len(train_dataset)} ({train_ratio*100:.1f}%)")
    print(f"训练集正样本数量: {np.sum(train_labels == 1)}, 负样本数量: {np.sum(train_labels == 0)}")
    print(f"训练集正负样本比例: {np.sum(train_labels == 1)}:{np.sum(train_labels == 0)}")
    
    print(f"\n验证集大小: {len(val_dataset)} ({val_ratio*100:.1f}%)")
    print(f"验证集正样本数量: {np.sum(val_labels == 1)}, 负样本数量: {np.sum(val_labels == 0)}")
    print(f"验证集正负样本比例: {np.sum(val_labels == 1)}:{np.sum(val_labels == 0)}")
    
    print(f"\n测试集大小: {len(test_dataset)} ({test_ratio*100:.1f}%)")
    print(f"测试集正样本数量: {np.sum(test_labels == 1)}, 负样本数量: {np.sum(test_labels == 0)}")
    print(f"测试集正负样本比例: {np.sum(test_labels == 1)}:{np.sum(test_labels == 0)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, dataset
