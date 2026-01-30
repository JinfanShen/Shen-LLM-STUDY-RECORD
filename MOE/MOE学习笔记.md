# MOE（Mixture of Experts）架构学习笔记

## 1. 理论背景

### 为什么被提出？

在传统的 Transformer 模型中（如 BERT、GPT），**每一个 token 都会经过完全相同的前馈网络（FFN）**：

```
输入 token → 同一个 FFN → 输出
输入 token → 同一个 FFN → 输出
输入 token → 同一个 FFN → 输出
```

这有两个问题：

| 问题 | 说明 |
|------|------|
| **参数效率低** | 模型要学会所有知识，但不同token可能需要不同的知识（如"苹果"可能是水果也可能是公司） |
| **计算成本高** | 参数量越大，推理时计算量越大，成本指数上升 |

### 为什么要这样做？

**核心洞察**：并非所有参数对所有输入都同等重要。

想象一个翻译模型：
- 翻译技术文档时，需要专业术语知识
- 翻译日常对话时，需要口语表达知识
- 翻译诗歌时，需要语言美感知识

**解决方案**：让模型"按需激活"——不同的输入走不同的专家网络。

### 解决了什么问题？

| 原来 | MOE |
|------|-----|
| 所有 token 经过相同 FFN | 只有部分专家被激活 |
| 参数量 ↑ = 计算量 ↑ | 参数量可以很大，但计算量可控 |
| 知识全部"压缩"到一个网络 | 知识"分布式"存储在多个专家中 |

**核心价值**：**条件计算（Conditional Computation）**

> 稀疏激活：模型参数量可以扩展到万亿级，但推理时只计算一小部分。

### 原理是什么？

```
输入 token
    │
    ▼
┌─────────────────────────┐
│      Gate Network       │  ← 门控网络（决定激活哪些专家）
│   (可学习的路由器)       │
└───────────┬─────────────┘
            │
    ┌───────┼───────┬───────┐
    ▼       ▼       ▼       ▼
 ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
 │专家1│ │专家2│ │专家3│ │专家4│  ← 多个独立的FFN
 │FFN1 │ │FFN2 │ │FFN3 │ │FFN4 │
 └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
    │       │       │       │
    └───────┼───────┼───────┘
            │
            ▼
       加权求和 → 输出
```

**工作流程**：

1. **门控计算**：输入经过门控网络，输出每个专家的权重
2. **专家选择**：选择权重最高的 K 个专家（通常是 1-2 个）
3. **稀疏激活**：只有选中的专家参与计算
4. **加权输出**：专家输出按权重加权求和

**关键公式**：

```
y = Σ(g_i(x) * E_i(x))

其中：
- g_i(x) = Softmax(W_g * x)[i]  ← 门控权重
- E_i(x) = FFN_i(x)              ← 专家输出
- K      = 2                     ← 通常选择 top-2
```

---

## 2. 代码实现

### 核心代码（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Experts(nn.Module):
    """
    专家模块：多个独立的FFN
    每个专家是一个简单的两层网络
    """
    def __init__(self, num_experts, expert_dim, hidden_dim):
        super().__init__()
        self.num_experts = num_experts

        # 每个专家有自己的权重
        self.w1 = nn.ModuleList([
            nn.Linear(expert_dim, hidden_dim) for _ in range(num_experts)
        ])
        self.w2 = nn.ModuleList([
            nn.Linear(hidden_dim, expert_dim) for _ in range(num_experts)
        ])

    def forward(self, x, expert_indices):
        """
        x: 输入 [batch, expert_dim]
        expert_indices: 选中的专家索引 [batch, top_k]
        """
        batch_size = x.shape[0]
        top_k = expert_indices.shape[1]

        outputs = []

        for b in range(batch_size):
            expert_outputs = []
            for k in range(top_k):
                expert_id = expert_indices[b, k].item()
                h = F.relu(self.w1[expert_id](x[b]))
                out = self.w2[expert_id](h)
                expert_outputs.append(out)
            combined = torch.stack(expert_outputs).mean(0)
            outputs.append(combined)

        return torch.stack(outputs)


class GateNetwork(nn.Module):
    """
    门控网络：决定哪些专家被激活
    """
    def __init__(self, num_experts, expert_dim, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(expert_dim, num_experts)

    def forward(self, x):
        logits = self.gate(x)  # [batch, num_experts]
        weights = F.softmax(logits, dim=-1)
        _, expert_indices = torch.topk(weights, self.top_k, dim=-1)
        return expert_indices, weights


class MoELayer(nn.Module):
    """
    MOE 层：整合门控和专家
    """
    def __init__(self, num_experts, expert_dim, hidden_dim, top_k=2):
        super().__init__()
        self.gate = GateNetwork(num_experts, expert_dim, top_k)
        self.experts = Experts(num_experts, expert_dim, hidden_dim)

    def forward(self, x):
        expert_indices, weights = self.gate(x)
        expert_outputs = self.experts(x, expert_indices)

        selected_weights = torch.gather(weights, 1, expert_indices)
        selected_weights = F.normalize(selected_weights, p=1, dim=1)

        output = (expert_outputs * selected_weights.unsqueeze(-1)).sum(dim=1)
        return output


# 测试
if __name__ == "__main__":
    batch_size = 4
    expert_dim = 64
    hidden_dim = 128
    num_experts = 8
    top_k = 2

    moe = MoELayer(num_experts, expert_dim, hidden_dim, top_k)
    x = torch.randn(batch_size, expert_dim)
    output = moe(x)
    print(f"输入形状: {x.shape}, 输出形状: {output.shape}")
```

---

## 3. 学习资源

**必读论文**：
- [Switch Transformer](https://arxiv.org/abs/2101.03961) - Google 2021
- [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) - Mistral AI 2024

**代码参考**：
- fairscale / Megatron-LM 中的 MoE 实现
- PyTorch MoE 教程

---

## 4. 练习建议

1. 修改 top_k 参数，观察效果变化
2. 增加专家数量，对比计算量变化
3. 实现负载均衡损失函数
4. 在实际NLP任务上测试对比

---

## 5. 关键概念速查

| 概念 | 含义 |
|------|------|
| Expert | 独立的FFN网络 |
| Gate Network | 路由器，决定激活哪些专家 |
| Top-K | 选择权重最高的K个专家（通常K=2） |
| 稀疏激活 | 只计算部分专家，降低计算量 |
| 负载均衡 | 避免某些专家被过度使用 |
