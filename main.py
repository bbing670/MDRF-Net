import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from torchvision import transforms
from model import ImagePairClassifier
from dataload import create_multi_day_dataloaders
from train import train, test

# 设置随机种子函数
def set_random_seed(seed=42):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 可能会降低速度，但确保结果一致

def main():
    # 设置随机种子
    seed = 42  # 可以根据需要修改
    set_random_seed(seed)
    print(f"设置随机种子: {seed}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    base_dir = '/home/gaoxt/dm/datatime_kejianguang'
    days = ['5.9', '5.13', '5.16', '5.19', '5.22', '5.25', '5.28', '5.29', '6.3', '6.6', '6.9']  # 11天的数据
    label_file = r'D:\app\dm\MM\data\2025label.csv'  # 标签文件
    
    # 使用多日数据加载器
    train_loader, val_loader, test_loader, _ = create_multi_day_dataloaders(
        base_dir=base_dir,
        days=days,
        label_file=label_file,
        batch_size=4,
        train_ratio=0.6,
        val_ratio=0.2,  # 验证集20%
        test_ratio=0.2,  # 测试集20%
        num_workers=2,
        seed=seed
    )
    
    model = ImagePairClassifier(time_steps=len(days)).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # 设置早停参数
    patience = 50
    checkpoint_path = 'best_drought_resistance_model.pth'
    
    # 训练模型 
    print("开始训练...")
    trained_model, best_model_paths = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=1000,
        patience=patience,
        checkpoint_path=checkpoint_path
    )
    
    # 测试每个最佳模型
    print("\n在测试集上测试最佳模型...")
    best_models_results = []
    
    for i, model_path in enumerate(best_model_paths, 1):
        print(f"\n测试第 {i} 个最佳模型: {model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            continue
        
        # 创建新模型实例并加载权重
        best_model = ImagePairClassifier(time_steps=len(days)).to(device)
        try:
            best_model.load_state_dict(torch.load(model_path))
            # 测试模型
            test_loss, test_acc, test_prec, test_rec, test_f1 = test(
                best_model, test_loader, criterion, device
            )
            # 记录结果
            best_models_results.append({
                'path': model_path,
                'loss': test_loss,
                'acc': test_acc,
                'prec': test_prec,
                'rec': test_rec,
                'f1': test_f1
            })
        except Exception as e:
            print(f"加载或测试模型时出错: {e}")
    
    # 找出性能最好的模型（可以根据F1分数或其他指标）
    if best_models_results:
        # 按F1分数排序
        best_models_results.sort(key=lambda x: x['f1'], reverse=True)
        
        print("\n===== 最佳模型测试结果汇总 =====")
        for i, result in enumerate(best_models_results, 1):
            print(f"模型 {i}: {result['path']}")
            print(f"  F1分数: {result['f1']:.4f}, 准确率: {result['acc']:.4f}, 精确率: {result['prec']:.4f}, 召回率: {result['rec']:.4f}, 损失: {result['loss']:.4f}")
        
        # 打印最佳模型
        best_overall = best_models_results[0]
        print(f"\n整体最佳模型: {best_overall['path']}")
        print(f"  F1分数: {best_overall['f1']:.4f}")
    else:
        print("\n警告: 没有成功测试任何模型")

    print(f"模型已保存到 {checkpoint_path}")

if __name__ == "__main__":
    main()