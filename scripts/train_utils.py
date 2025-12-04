import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm




def load_transforms():
    """
    Load the data transformations
    """
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def load_data(data_dir, batch_size):
    """
    Load the data from the data directory and split it into training and validation sets
    This function is similar to the cell 2. Data Preparation in 04_model_training.ipynb

    Args:
        data_dir: The directory to load the data from
        batch_size: The batch size to use for the data loaders
    Returns:
        train_loader: The training data loader
        val_loader: The validation data loader
    """
    # Define data transformations: resize, convert to tensor, and normalize
    data_transforms = load_transforms()

    # Load the train dataset from the augmented data directory
    train_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

    # Load the validation dataset from the raw data directory
    val_dataset = datasets.ImageFolder(root=data_dir + "/../../raw/val", transform=data_transforms)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Print dataset summary
    print(f"Dataset loaded from: {data_dir}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    return train_loader, val_loader


# def define_loss_and_optimizer(model: nn.Module, lr: float, weight_decay: float):
#     """
#     Define the loss function and optimizer
#     This function is similar to the cell 3. Model Configuration in 04_model_training.ipynb
#     Args:
#         model: The model to train
#         lr: Learning rate
#         weight_decay: Weight decay
#     Returns:
#         criterion: The loss function
#         optimizer: The optimizer
#         scheduler: The scheduler
#     """
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
#     return criterion, optimizer, scheduler
def define_loss_and_optimizer(model, lr, weight_decay):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   # 0.1 平滑
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5),   # 0→0.1
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=55)                   # 0.1→0
        ],
        milestones=[5]
    )
    
    return criterion, optimizer, scheduler


# def train_epoch(model, dataloader, criterion, optimizer, device):
#     """
#     Train the model for one epoch
#     Args:
#         model: The model to train
#         dataloader: DataLoader for training data
#         criterion: Loss function
#         optimizer: Optimizer
#         device: Device to train on
#     Returns:
#         Average loss and accuracy for the epoch
#     """
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     progress_bar = tqdm(dataloader, desc="Training", leave=False)

#     for inputs, labels in progress_bar:
#         inputs, labels = inputs.to(device), labels.to(device)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()


#         # Statistics
#         running_loss += loss.item() * inputs.size(0)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#         # Update progress bar
#         progress_bar.set_postfix(
#             {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
#         )

#     epoch_loss = running_loss / total
#     epoch_acc = 100.0 * correct / total

#     return epoch_loss, epoch_acc


def train_epoch(model, dataloader, criterion, optimizer, device):
    import torch.nn.functional as F
    from torch.nn import KLDivLoss

    kl = KLDivLoss(reduction='batchmean')
    T = 3.0
    max_alpha = 0.1      # 最终蒸馏权重
    warmup_epoch = 3     # 前 3 epoch 纯分类
    warmup_alpha_step = 5  # 第 4 epoch 的前 5 batch 用最小 alpha 过渡

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    global_batch_idx = 0   # 跨 batch 计数

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        global_batch_idx += 1
        current_epoch = global_batch_idx / len(dataloader)   # 浮点 epoch
        optimizer.zero_grad()

        # 前向
        outputs = model(inputs)
        loss_cls = criterion(outputs, labels)

        # ---------- 蒸馏项 ----------
        loss_kd = 0.
        alpha = 0.
        if current_epoch >= warmup_epoch:
            # 线性爬升 alpha
            alpha = max_alpha * min(1., (current_epoch - warmup_epoch) /
                                    (warmup_alpha_step / len(dataloader)))
            with torch.no_grad():
                teacher_logit = model(inputs) / T   # 自蒸馏
            student_logit = outputs / T
            loss_kd = kl(F.log_softmax(student_logit, dim=1),
                         F.softmax(teacher_logit, dim=1)) * (T * T)

        loss = (1 - alpha) * loss_cls + alpha * loss_kd
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%",
             "α": f"{alpha:.3f}"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
    Returns:
        Average loss and accuracy for the validation set
    """

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(state, filename):
    """
    Save model checkpoint
    Args:
        state: Checkpoint state
        filename: Path to save checkpoint
    """
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    Args:
        filename: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
    Returns:
        Checkpoint state
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint

def save_metrics(metrics: str, filename: str = "training_metrics.txt"):
    """
    Save training metrics to a file
    Args:
        metrics: Metrics string to save
        filename: Path to save metrics
    """
    with open(filename, 'w') as f:
        f.write(metrics)
# import os, torch, torch.nn as nn, torch.optim as optim, numpy as np
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from torchvision.transforms import RandAugment
# from tqdm import tqdm

# # -------------------  EMA（不依赖 torch-ema） -------------------
# class EMA:
#     def __init__(self, model, decay=0.9992):
#         self.decay = decay
#         self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}

#     def update(self, model):
#         with torch.no_grad():
#             for name, param in model.named_parameters():
#                 self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)

#     def apply_shadow(self, model):
#         backup = {n: p.clone() for n, p in model.named_parameters()}
#         for name, param in model.named_parameters():
#             param.data.copy_(self.shadow[name])
#         return backup

#     def restore(self, model, backup):
#         for name, param in model.named_parameters():
#             param.data.copy_(backup[name])


# # -------------------  自动区分 train/val 的 transform  -------------------
# def load_transforms(train=True):
#     if train:
#         return transforms.Compose([
#             transforms.Resize(64),
#             RandAugment(2, 15),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#     else:
#         return transforms.Compose([
#             transforms.Resize(64),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])


# # -------------------  CutMix  -------------------
# def rand_bbox(size, lam):
#     W, H = size[2], size[3]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
#     cx, cy = np.random.randint(W), np.random.randint(H)
#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)
#     return bbx1, bby1, bbx2, bby2

# def cutmix_data(x, y, alpha=1.0):
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#     index = torch.randperm(x.size(0)).to(x.device)
#     bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
#     x_mixed = x.clone()
#     x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
#     y_a, y_b = y, y[index]
#     return x_mixed, y_a, y_b, lam


# # -------------------  数据加载（接口不变） -------------------
# def load_data(data_dir, batch_size):
#     # 根据路径关键字自动决定是不是训练集
#     train = 'train' in data_dir and 'val' not in data_dir
#     tf = load_transforms(train=train)
#     ds = datasets.ImageFolder(root=data_dir, transform=tf)
#     loader = DataLoader(ds, batch_size=batch_size, shuffle=train,
#                         num_workers=4, pin_memory=True)
#     print(f"Dataset loaded from: {data_dir}  train={train}  size={len(ds)}")
#     return loader


# # -------------------  optimizer & scheduler  -------------------
# def define_loss_and_optimizer(model, lr, weight_decay):
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
#                           weight_decay=weight_decay, nesterov=True)
#     scheduler = optim.lr_scheduler.SequentialLR(
#         optimizer,
#         schedulers=[
#             optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5),
#             optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=55)
#         ],
#         milestones=[5]
#     )
#     return criterion, optimizer, scheduler


# # -------------------  全局 EMA 对象（单例） -------------------
# _ema = None

# def get_ema(model):
#     global _ema
#     if _ema is None:
#         _ema = EMA(model)
#     return _ema


# # -------------------  train_epoch（签名不变） -------------------
# def train_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=2):
#     model.train()
#     running_loss, correct, total = 0.0, 0, 0
#     ema = get_ema(model)
#     pbar = tqdm(dataloader, desc="Train", leave=False)
#     for i, (x, y) in enumerate(pbar):
#         x, y = x.to(device), y.to(device)
#         x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
#         out = model(x)
#         loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
#         loss = loss / accumulation_steps               # ① 缩放
#         loss.backward()
#         if (i + 1) % accumulation_steps == 0:          # ② 累积 N 步再更新
#             optimizer.step()
#             optimizer.zero_grad()
#             ema.update()
#         running_loss += loss.item() * x.size(0) * accumulation_steps
#         _, pred = out.max(1)
#         total += y.size(0)
#         correct += (lam * pred.eq(y_a).sum().item()
#                     + (1 - lam) * pred.eq(y_b).sum().item())
#         pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}",
#                           "acc": f"{100. * correct / total:.2f}%"})
#     return running_loss / total, 100. * correct / total


# # -------------------  validate_epoch（签名不变） -------------------
# def validate_epoch(model, dataloader, criterion, device):
#     model.eval()
#     running_loss, correct, total = 0.0, 0, 0
#     ema = get_ema(model)
#     backup = ema.apply_shadow(model)          # 验证用 EMA 权重
#     with torch.no_grad():
#         for x, y in tqdm(dataloader, desc="Validation", leave=False):
#             x, y = x.to(device), y.to(device)
#             out = model(x)
#             loss = criterion(out, y)
#             running_loss += loss.item() * x.size(0)
#             _, pred = out.max(1)
#             total += y.size(0)
#             correct += pred.eq(y).sum().item()
#     ema.restore(model, backup)                # 恢复原始权重
#     return running_loss / total, 100. * correct / total


# # -------------------  checkpoint & metrics  -------------------
# def save_checkpoint(state, filename):
#     torch.save(state, filename)

# def save_metrics(metrics: str, filename="training_metrics.txt"):
#     with open(filename, "w") as f:
#         f.write(metrics)

# # ------------------- 兼容旧接口 -------------------
# def load_data(data_dir, batch_size):
#     """返回 train & val loader，验证集用 raw/val 兜底"""
#     train_dir = data_dir
#     val_dir   = data_dir.replace('/augmented/train', '/raw/val')   # 回退到原始验证集
#     train_loader = load_data_single(train_dir, batch_size, train=True)
#     val_loader   = load_data_single(val_dir,   batch_size, train=False)
#     print(f"Train size: {len(train_loader.dataset)}  Val size: {len(val_loader.dataset)}")
#     return train_loader, val_loader


# def load_data_single(data_dir, batch_size, train=True):
#     """内部真实实现，只返回一个 loader"""
#     tf = load_transforms(train=train)
#     ds = datasets.ImageFolder(root=data_dir, transform=tf)
#     loader = DataLoader(ds, batch_size=batch_size, shuffle=train,
#                         num_workers=4, pin_memory=True)
#     return loader