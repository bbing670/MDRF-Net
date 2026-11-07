import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from torchinfo import summary

# 早停类 - 修改为正确保存最好的3个模型
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', num_best_models=3):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.num_best_models = num_best_models
        # 用于存储最佳模型的损失值和路径
        self.best_models = []  # 格式: [(val_loss, model_path), ...]
        
        # 确保bestmodel目录存在
        self.bestmodel_dir = '/home/gaoxt/dm/MM/bestmodel'
        os.makedirs(self.bestmodel_dir, exist_ok=True)
        
        # 确保主模型保存路径也在bestmodel目录下
        if not os.path.dirname(self.path).endswith('bestmodel'):
            self.path = os.path.join(self.bestmodel_dir, os.path.basename(self.path))
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        # 检查是否是新的最佳模型或需要添加到最佳模型列表
        self._update_best_models(val_loss, model)
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def _update_best_models(self, val_loss, model):
        # 只有当这个损失值比当前列表中的某些模型好时才保存
        # 或者列表还没满时
        should_save = False
        
        # 如果列表未满，应该保存
        if len(self.best_models) < self.num_best_models:
            should_save = True
        else:
            # 如果当前损失比列表中最差的模型好，应该保存
            worst_loss = max([loss for loss, _ in self.best_models])
            should_save = val_loss < worst_loss
        
        if should_save:
            # 获取基础文件名（不包含扩展名）
            base_name = os.path.splitext(os.path.basename(self.path))[0]
            ext = os.path.splitext(self.path)[1]
            
            # 创建新的模型路径，保存到bestmodel目录下
            model_filename = f"{base_name}_loss_{val_loss:.6f}{ext}"
            model_path = os.path.join(self.bestmodel_dir, model_filename)
            
            # 保存模型
            torch.save(model.state_dict(), model_path)
            if self.verbose:
                print(f'保存模型到: {model_path} )')
            
            # 添加到最佳模型列表
            self.best_models.append((val_loss, model_path))
            
            # 按损失值排序（升序）
            self.best_models.sort(key=lambda x: x[0])
            
            # 如果列表超过num_best_models，删除最差的模型
            if len(self.best_models) > self.num_best_models:
                worst_model = self.best_models.pop()
                try:
                    if os.path.exists(worst_model[1]):
                        os.remove(worst_model[1])
                        if self.verbose:
                            print(f"删除最差模型: {worst_model[1]}")
                except Exception as e:
                    if self.verbose:
                        print(f"无法删除旧模型文件: {worst_model[1]}, 错误: {e}")
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
    
    def get_best_model_paths(self):
        # 过滤掉不存在的文件路径
        valid_paths = []
        for _, model_path in self.best_models[:self.num_best_models]:
            if os.path.exists(model_path):
                valid_paths.append(model_path)
            else:
                if self.verbose:
                    print(f"警告: 模型文件不存在: {model_path}")
        return valid_paths

def update_sfs_temperature(model):
    net = model.module if hasattr(model, "module") else model

    if hasattr(net, "attention") and hasattr(net.attention, "update_temperature"):
        old_temperature = net.attention.get_temperature()
        net.attention.update_temperature()
        new_temperature = net.attention.get_temperature()
        return old_temperature, new_temperature

    return None, None


# 训练函数 
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=100,
          patience=10, checkpoint_path='best_model.pth', fold=None):
    model.train()

    # 展示模型结构
    try:
        if len(train_loader) > 0:
            for img1, img2, _ in train_loader:
                input_size1 = img1.shape
                input_size2 = img2.shape
                break

            print("\n模型结构详情:")
            summary(
                model,
                input_data=[
                    torch.zeros(input_size1).to(device),
                    torch.zeros(input_size2).to(device)
                ],
                depth=4,
                col_names=['input_size', 'output_size', 'num_params', 'trainable']
            )
            print("\n")
        else:
            print("训练数据加载器为空，无法展示模型结构")
    except Exception as e:
        print(f"展示模型结构时出错: {e}")

    # 初始化早停机制
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=checkpoint_path,
        num_best_models=3
    )

    # 创建 TensorBoard 记录器
    log_dir = f'runs/training{"_fold_" + str(fold) if fold is not None else ""}'
    writer = SummaryWriter(log_dir=log_dir)

    # 获取初始学习率
    fixed_lr = optimizer.param_groups[0]['lr']
    print(f"使用固定学习率: {fixed_lr:.6f}")

    # 打印初始 SFS 温度
    net = model.module if hasattr(model, "module") else model
    if hasattr(net, "attention") and hasattr(net.attention, "get_temperature"):
        print(f"Initial SFS temperature: {net.attention.get_temperature():.1f}")

    for epoch in range(epochs):
        # ==================== 训练阶段 ====================
        running_loss = 0.0
        correct = 0
        total = 0

        train_true_positive = 0
        train_false_positive = 0
        train_false_negative = 0

        model.train()
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(img1, img2)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_true_positive += ((predicted == 1) & (labels == 1)).sum().item()
            train_false_positive += ((predicted == 1) & (labels == 0)).sum().item()
            train_false_negative += ((predicted == 0) & (labels == 1)).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        train_prec = (
            train_true_positive / (train_true_positive + train_false_positive)
            if (train_true_positive + train_false_positive) > 0 else 0
        )
        train_rec = (
            train_true_positive / (train_true_positive + train_false_negative)
            if (train_true_positive + train_false_negative) > 0 else 0
        )
        train_f1 = (
            2 * (train_prec * train_rec) / (train_prec + train_rec)
            if (train_prec + train_rec) > 0 else 0
        )

        # ==================== 验证阶段 ====================
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, device
        )

        current_lr = fixed_lr

        # ==================== TensorBoard 记录 ====================
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Precision/Train', train_prec, epoch)
        writer.add_scalar('Precision/Validation', val_prec, epoch)
        writer.add_scalar('Recall/Train', train_rec, epoch)
        writer.add_scalar('Recall/Validation', val_rec, epoch)
        writer.add_scalar('F1/Train', train_f1, epoch)
        writer.add_scalar('F1/Validation', val_f1, epoch)

        # 记录当前 SFS 温度
        net = model.module if hasattr(model, "module") else model
        if hasattr(net, "attention") and hasattr(net.attention, "get_temperature"):
            writer.add_scalar('SFS/Temperature', net.attention.get_temperature(), epoch)

        # ==================== 输出训练信息 ====================
        fold_info = f" (折 {fold})" if fold is not None else ""

        temp_info = ""
        if hasattr(net, "attention") and hasattr(net.attention, "get_temperature"):
            temp_info = f", SFS Temp: {net.attention.get_temperature():.1f}"

        print(
            f'Epoch {epoch + 1}/{epochs}{fold_info}, '
            f'Train Loss: {epoch_loss:.4f}, '
            f'Val Loss: {val_loss:.4f}, '
            f'Val Acc: {val_acc:.4f}, '
            f'Val Prec: {val_prec:.4f}, '
            f'Val Rec: {val_rec:.4f}, '
            f'Val F1: {val_f1:.4f}, '
            f'Train F1: {train_f1:.4f}'
            f'{temp_info}'
        )

        # ==================== 早停检查 ====================
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            print(f"已保存的最佳模型路径: {early_stopping.get_best_model_paths()}")
            break

        # ==================== 温度退火更新 ====================
        old_temp, new_temp = update_sfs_temperature(model)
        if old_temp is not None:
            print(f"SFS temperature annealed: {old_temp:.1f} -> {new_temp:.1f}")

    writer.close()

    # 加载最佳模型
    model.load_state_dict(torch.load(early_stopping.path))

    # 返回模型和最佳模型路径列表
    return model, early_stopping.get_best_model_paths()

# 验证函数（保持不变）
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 初始化混淆矩阵的元素
    true_positive = 0  # 正确预测的正例
    false_positive = 0  # 错误预测为正例的负例
    false_negative = 0  # 错误预测为负例的正例
    
    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            
            # 计算统计信息
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 计算混淆矩阵元素（假设1是正类）
            true_positive += ((predicted == 1) & (labels == 1)).sum().item()
            false_positive += ((predicted == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted == 0) & (labels == 1)).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = correct / total  # 改为小数形式，不再乘以100
    
    # 计算精确率、召回率和F1分数（改为小数形式）
    # 避免除零错误
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return val_loss, val_acc, precision, recall, f1

# 测试函数（保持不变）
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 初始化混淆矩阵的元素
    true_positive = 0
    false_positive = 0
    false_negative = 0
    
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 计算混淆矩阵元素（假设1是正类）
            true_positive += ((predicted == 1) & (labels == 1)).sum().item()
            false_positive += ((predicted == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted == 0) & (labels == 1)).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = correct / total  # 改为小数形式，不再乘以100
    
    # 计算精确率、召回率和F1分数（改为小数形式）
    # 避免除零错误
    test_prec = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    test_rec = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    test_f1 = 2 * (test_prec * test_rec) / (test_prec + test_rec) if (test_prec + test_rec) > 0 else 0
    
    # 按照要求的格式输出结果
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Prec: {test_prec:.4f}, Test Rec: {test_rec:.4f}, Test F1: {test_f1:.4f}')
    
    return test_loss, test_acc, test_prec, test_rec, test_f1

# 推理函数：用于单个样本的预测
def inference(model, exp_img_path, ctrl_img_path, transform, device):
    model.eval()
    
    # 加载并预处理图像
    from PIL import Image
    exp_img = Image.open(exp_img_path).convert('RGB')
    ctrl_img = Image.open(ctrl_img_path).convert('RGB')
    
    exp_img = transform(exp_img).unsqueeze(0).to(device)
    ctrl_img = transform(ctrl_img).unsqueeze(0).to(device)
    
    # 进行预测
    with torch.no_grad():
        output = model(exp_img, ctrl_img)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class, probabilities