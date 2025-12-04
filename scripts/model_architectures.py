#还需要一个cifar100 baseline 0.7030 提升后0.7372
#cifar10 baseline 0.8993 提升后0.9206





#Quantitative analysis （cifar100）
#ResNet18
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 50.99%
#Top-5 Accuracy (Correct in Top 5): 76.07%
#ResNet50
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 38.61%
#Top-5 Accuracy (Correct in Top 5): 67.58%
#DenseNet-121
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 58.92%
#Top-5 Accuracy (Correct in Top 5): 82.16%
#MobileNet-v2
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 52.54%
#Top-5 Accuracy (Correct in Top 5): 77.45%
#ShuffleNet-v2 (1.0×)
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 42.14%
#Top-5 Accuracy (Correct in Top 5): 72.90%
#SE-ResNet-152
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 49.74%
#Top-5 Accuracy (Correct in Top 5): 75.31%
#EfficientNet-B0
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 48.70%
#Top-5 Accuracy (Correct in Top 5): 72.15%
#AlexNet
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 13.92%
#Top-5 Accuracy (Correct in Top 5): 37.56%
#GoogLeNet
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 54.23%
#Top-5 Accuracy (Correct in Top 5): 78.80%
#SDResNet101
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 38.96%
#Top-5 Accuracy (Correct in Top 5): 68.58%
#Ours
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 73.72%
#Top-5 Accuracy (Correct in Top 5): 91.94%





#Quantitative analysis （cifar10）
#ResNet18
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 81.59%
#Top-5 Accuracy (Correct in Top 5): 97.00%
#ResNet50
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 72.43%
#Top-5 Accuracy (Correct in Top 5): 96.78%
#DenseNet-121
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 85.54%
#Top-5 Accuracy (Correct in Top 5): 97.61%
#MobileNet-v2
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 79.99%
#Top-5 Accuracy (Correct in Top 5): 98.63%
#ShuffleNet-v2 (1.0×)
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 74.23%
#Top-5 Accuracy (Correct in Top 5): 97.84%
#SE-ResNet-152
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 71.26%
#Top-5 Accuracy (Correct in Top 5): 97.25%
#EfficientNet-B0
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 69.89%
#Top-5 Accuracy (Correct in Top 5): 97.38%
#AlexNet
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 58.78%
#Top-5 Accuracy (Correct in Top 5): 94.02%
#GoogLeNet
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 81.52%
#Top-5 Accuracy (Correct in Top 5): 98.85%
#SDResNet101
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 70.02%
#Top-5 Accuracy (Correct in Top 5): 97.41%
#Ours
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 92.06%
#Top-5 Accuracy (Correct in Top 5): 98.94%



#Ablation Study
# baseline 0.7030
#Dropout 0.7207
#PromptLite 0.6962
#DFS-SE串并联 0.7239=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 72.39%
#Top-5 Accuracy (Correct in Top 5): 91.37%
#DFS-SE+Dropout
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 73.36%
#Top-5 Accuracy (Correct in Top 5): 91.73%
#Prompt+Dropout
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 71.61%
#Top-5 Accuracy (Correct in Top 5): 90.32%
#Prompt+DFS-SE
#=== Top-K Accuracy ===
#Top-1 Accuracy (Exact Match): 72.69%
#Top-5 Accuracy (Correct in Top 5): 91.54%
#Dropout+PromptLite+SE 0.7338
#Dropout+PromptLite+SE-DFS只有串联部分 0.7274
#完整框架Dropout+PromptLite+DFS-SE 74.10% 0.7372



import torch
import torch.nn as nn
import torch.nn.functional as F

# 补充缺失的DropBlock2D类（原代码依赖）
class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=5):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        N, C, H, W = x.size()
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand(N, C, H - self.block_size + 1, W - self.block_size + 1, device=x.device) < gamma).float()
        mask = F.pad(mask, [self.block_size//2]*4, mode='constant', value=0)
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
        mask = 1 - mask
        x = x * mask * mask.numel() / mask.sum()
        return x

# ==================== 核心修改：新增并联融合模块 ====================
class Parallel_DFS_SE(nn.Module):
    """
    双分支并联融合：
    - 分支1：SE+DFS串联（保留原有核心逻辑，负责细节挖掘）
    - 分支2：单独SE（负责快速提分，弥补串联分支涨分效率短板）
    - 融合方式：可学习权重融合（动态调整双分支贡献）
    """
    def __init__(self, C, T=21, hidden=128, se_reduction=16):
        super().__init__()
        self.C = C
        
        # ---------------------- 分支1：原有SE+DFS串联模块 ----------------------
        self.branch_dfs_se = DFS_SE(C, T, hidden, se_reduction, fusion_weight=0.5)
        
        # ---------------------- 分支2：单独SE模块（新增） ----------------------
        self.branch_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C // se_reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // se_reduction, C, 1),
            nn.Sigmoid()
        )
        
        # ---------------------- 可学习融合权重（新增） ----------------------
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))  # 初始权重：串联0.5，SE0.5

    def forward(self, x):
        # 双分支特征计算
        feat_dfs_se = self.branch_dfs_se(x)  # 串联分支输出：[N,C,H,W]
        att_se = self.branch_se(x)            # 单独SE注意力：[N,C,1,1]
        feat_se = x * att_se                  # 单独SE分支输出：[N,C,H,W]
        
        # 权重归一化（确保权重和为1）
        norm_weight = F.softmax(self.fusion_weight, dim=0)
        
        # 加权融合
        feat_fusion = feat_dfs_se * norm_weight[0] + feat_se * norm_weight[1]
        return feat_fusion

# ==================== 原有DFS_SE串联模块（保持不变） ====================
class DFS_SE(nn.Module):
    """
    DFS与SE串联融合（SE先粗筛→DFS再精筛，固定权重）：
    - 输入: N,C,H,W  输出: N,C,H,W
    - 逻辑：SE先过滤无效通道，DFS在SE处理后的特征上探索像素，最后融合两者注意力
    """
    def __init__(self, C, T=21, hidden=128, se_reduction=16, fusion_weight=0.5):
        super().__init__()
        self.T = T
        self.hidden = hidden
        self.C = C
        self.fusion_weight = fusion_weight  # 固定融合权重（SE和DFS各占比）
        
        # ---------------------- 1. SE模块（先做通道粗筛） ----------------------
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化：N,C,H,W → N,C,1,1
            nn.Conv2d(C, C // se_reduction, 1),  # 压缩通道
            nn.ReLU(inplace=True),
            nn.Conv2d(C // se_reduction, C, 1),  # 恢复通道
            nn.Sigmoid()  # SE通道注意力：N,C,1,1
        )
        
        # ---------------------- 2. DFS模块（在SE粗筛后的特征上精筛像素） ----------------------
        # 节点嵌入（输入通道=C，SE处理后仍为C，映射到hidden=128）
        self.node_emb = nn.Conv2d(C, hidden, 1)
        # 策略网络（输入通道=hidden=128，和你原代码一致）
        self.policy = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 9, 1)  # 8邻居+stop
        )
        # DFS注意力生成（从hidden→C，匹配输入通道）
        self.dfs_att_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, C, 1), nn.Sigmoid()
        )

    # 保留你原有的邻接表生成逻辑（无修改）
    @staticmethod
    def _make_adj(H, W):
        idx = torch.arange(H * W).view(H, W)
        dirs = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),          (0, 1),
                (1, -1),  (1, 0), (1, 1)]
        adj = torch.zeros(H * W, 8, dtype=torch.long)
        for k, (dy, dx) in enumerate(dirs):
            y = torch.arange(H).view(-1, 1) + dy
            x = torch.arange(W).view(1, -1) + dx
            neighbor = idx[y.clamp(0, H - 1), x.clamp(0, W - 1)]
            adj[:, k] = neighbor.view(-1)
        return adj

    def forward(self, x):
        N, C, H, W = x.size()
        device = x.device

        # ---------------------- 第一步：SE先做通道粗筛 ----------------------
        att_se = self.se(x)  # SE注意力：[N, C, 1, 1]
        x_se = x * att_se  # SE处理后的特征（过滤无效通道）：[N, C, H, W]

        # ---------------------- 第二步：DFS在SE粗筛后的特征上精筛像素 ----------------------
        # 1. 节点特征（基于SE处理后的x_se，而非原始x，这是串联核心）
        node = self.node_emb(x_se)  # [N, hidden, H, W]（SE已过滤背景通道，更高效）
        node_flat = node.permute(0, 2, 3, 1).reshape(N, H * W, self.hidden)  # [N, H*W, hidden]

        # 2. 邻接表（和你原代码一致）
        adj = self._make_adj(H, W).to(device)  # [H*W, 8]

        # 3. 搜索初始化（和你原代码一致）
        curr = torch.zeros(N, dtype=torch.long, device=device)
        visited = torch.zeros(N, H * W, dtype=torch.bool, device=device)
        path_feat = torch.zeros_like(node_flat)  # [N, H*W, hidden]

        # 4. T步搜索（在SE处理后的特征上探索，背景干扰更少）
        for t in range(self.T):
            h_map = node_flat[torch.arange(N), curr].view(N, self.hidden, 1, 1)  # 通道=128，匹配policy
            logits = self.policy(h_map).squeeze(-1).squeeze(-1)  # [N, 9]
            a = F.gumbel_softmax(logits, tau=1, hard=False)

            # 下一步索引计算（和你原代码一致）
            stop_mask = a[:, 8]
            nei_mask = a[:, :8]
            next_idx = torch.gather(adj[curr], 1, torch.argmax(nei_mask, dim=1, keepdim=True)).squeeze(1)
            next_idx = torch.where(visited[torch.arange(N), next_idx], curr, next_idx)
            next_idx = torch.where(stop_mask.bool(), curr, next_idx)

            # 累积路径特征（SE已过滤背景，这里累积的特征更纯）
            visited[torch.arange(N), next_idx] = True
            path_feat[torch.arange(N), next_idx] += node_flat[torch.arange(N), next_idx] * (1 - stop_mask).unsqueeze(1)
            curr = next_idx

        # 5. DFS注意力生成（基于SE处理后的特征，精度更高）
        path_feat_avg = path_feat.mean(dim=1).view(N, self.hidden, 1, 1)  # [N, hidden, 1, 1]
        att_dfs = self.dfs_att_gen(path_feat_avg)  # [N, C, 1, 1]

        # ---------------------- 第三步：固定权重融合SE和DFS注意力 ----------------------
        # 两者都是[N,C,1,1]，直接按比例相加（默认各50%）
        att_fusion = att_se * self.fusion_weight + att_dfs * (1 - self.fusion_weight)

        # 应用融合注意力到原始x（保留原始特征，仅用融合注意力加权）
        return x * att_fusion

# ==================== 1. PromptIR 压缩版（保持不变） ====================
class PromptLite(nn.Module):
    def __init__(self, C, prompt_len=8):
        super().__init__()
        # ① 单尺度 1×1，groups=4 降参
        self.pgm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C//4, 1, groups=4), nn.ReLU(),
            nn.Conv2d(C//4, prompt_len, 1, groups=4)
        )
        # ② 无 BN、无 Dropout、无 Temperature
        self.pim = nn.Conv2d(C + prompt_len, C, 1, groups=4)

    def forward(self, x):
        b, c, h, w = x.size()
        prompt = self.pgm(x).expand(-1, -1, h, w)
        fusion = torch.cat([x, prompt], 1)
        att = torch.sigmoid(self.pim(fusion)).clamp_min(0.5)   # 下限 0.5，无归一
        return x * att

# ==================== 2. ReduNet 正交投影版（保持不变） ====================
class ReduLayer(nn.Module):
    """显式正交投影 + 能量衰减，C→C 尺寸不变"""
    def __init__(self, C, decay=0.9):
        super().__init__()
        self.decay = decay
        self.proj = nn.Conv2d(C, C, 1, groups=4, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C // 4, C, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x_proj = self.proj(x)
        gate = self.gate(x)
        return x + self.decay * gate * x_proj

# ==================== 3. 缝合残差块（保持不变） ====================
class PromptReduBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, drop_prob=0.1, block_size=5,
                 use_prompt=True, use_redu=True):
        super().__init__()
        # 原主分支
        self.block = BasicBlock(in_planes, planes, stride, drop_prob, block_size)
        self.prompt = PromptLite(planes) if use_prompt else nn.Identity()
        self.redu   = ReduLayer(planes)           if use_redu   else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        x = self.prompt(x)
        x = self.redu(x)
        return x

# ---------- 原 BasicBlock（修改：使用并联融合模块） ----------
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, drop_prob=0.1, block_size=5,
                 dfs_T=21, dfs_hidden=128):  # 保留DFS参数
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        # 核心修改：替换原DFS_SE为并联融合模块
        self.parallel_fusion = Parallel_DFS_SE(planes, T=dfs_T, hidden=dfs_hidden)
        self.dropblock = DropBlock2D(drop_prob, block_size)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropblock(out)
        out = self.bn2(self.conv2(out))
        out = self.parallel_fusion(out)  # 使用并联融合模块
        out += self.shortcut(x)
        return F.relu(out)

# ---------- WideResNet（保持不变，参数传递正常） ----------
class WideResNet(nn.Module):
    block = BasicBlock          # 可被工厂函数覆盖

    def __init__(self, depth=34, widen_factor=10, num_classes=100,
                 drop_prob=0.1, block_size=5, dfs_T=21, dfs_hidden=128):
        super().__init__()
        # 保存DFS参数，用于构建block
        self.dfs_T = dfs_T
        self.dfs_hidden = dfs_hidden
        
        n = (depth - 4) // 6
        k = widen_factor
        stages = [16, 16*k, 32*k, 64*k]
        self.conv1 = nn.Conv2d(3, stages[0], 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(stages[0], stages[1], n, stride=1,
                                       drop_prob=drop_prob, block_size=block_size)
        self.layer2 = self._make_layer(stages[1], stages[2], n, stride=2,
                                       drop_prob=drop_prob, block_size=block_size)
        self.layer3 = self._make_layer(stages[2], stages[3], n, stride=2,
                                       drop_prob=drop_prob, block_size=block_size)
        self.bn  = nn.BatchNorm2d(stages[3])
        self.fc  = nn.Linear(stages[3], num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, in_planes, planes, blocks, stride=1,
                    drop_prob=0.1, block_size=5):
        # 传递DFS参数到BasicBlock
        layers = [self.block(in_planes, planes, stride,
                             drop_prob=drop_prob, block_size=block_size,
                             dfs_T=self.dfs_T, dfs_hidden=self.dfs_hidden)]
        for _ in range(1, blocks):
            layers.append(self.block(planes, planes,
                                     drop_prob=drop_prob, block_size=block_size,
                                     dfs_T=self.dfs_T, dfs_hidden=self.dfs_hidden))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

# ---------- 工厂函数（保持不变，支持DFS参数配置） ----------
def create_model(num_classes=100, device='cuda', drop_prob=0.1,
                 use_parb=False, use_prompt=True, use_redu=False,
                 dfs_T=21, dfs_hidden=128):  # 保留DFS参数
    # 1. 选基础 block
    if use_parb:
        raise NotImplementedError("ParBBlock未定义，如需使用请补充实现")
    else:
        base_block = BasicBlock
    WideResNet.block = base_block
    model = WideResNet(
        num_classes=num_classes,
        depth=34,
        widen_factor=10,
        drop_prob=drop_prob,
        dfs_T=dfs_T,          # 传递DFS搜索步数
        dfs_hidden=dfs_hidden # 传递隐藏层维度
    ).to(device)

    # 2. 把每个 stage 的最后一层换成缝合块
    for stage in [model.layer1, model.layer2, model.layer3]:
        last_blk = stage[-1]
        stage[-1] = PromptReduBlock(
            in_planes=last_blk.conv1.in_channels,
            planes=last_blk.conv1.out_channels,
            stride=1,
            drop_prob=drop_prob,
            use_prompt=use_prompt,
            use_redu=use_redu
        ).to(device)
    return model
