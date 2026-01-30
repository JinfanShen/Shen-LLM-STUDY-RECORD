# ============================================================
# Switch Load Balancing Loss 详细调试代码
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def switch_load_balancing_loss_detailed(router_logits: torch.Tensor, num_experts: int, top_k: int = 2):
    """
    计算 Switch Transformers 的负载均衡损失 - 详细打印版
    """
    print("=" * 60)
    print("Step 1: 输入 router_logits")
    print(f"  Shape: {router_logits.shape}")  # [b * s, num_experts]
    print(f"  Values (前3个token):\n{router_logits[:3]}")
    
    # Step 2: 计算路由概率
    router_probs = torch.softmax(router_logits, dim=-1)
    print("\n" + "=" * 60)
    print("Step 2: 计算路由概率 (softmax)")
    print(f"  Shape: {router_probs.shape}")
    print(f"  每个token对各专家的分配概率 (前3个token):\n{router_probs[:3]}")
    print(f"  每行概率和 (验证=1): {router_probs[:3].sum(dim=-1)}")
    
    # Step 3: Top-K 选择
    router_probs_sorted, selected_experts = torch.topk(router_probs, top_k, dim=-1)
    print("\n" + "=" * 60)
    print(f"Step 3: Top-{top_k} 专家选择")
    print(f"  selected_experts shape: {selected_experts.shape}")
    print(f"  每个token选择的专家索引 (前5个token):\n{selected_experts[:5]}")
    print(f"  对应的路由权重 (前5个token):\n{router_probs_sorted[:5]}")
    
    # Step 4: One-Hot 编码
    mask = F.one_hot(selected_experts, num_classes=num_experts).float()
    print("\n" + "=" * 60)
    print("Step 4: One-Hot 编码 (expert × token)")
    print(f"  Mask shape: {mask.shape} [batch_tokens, top_k, num_experts]")
    print(f"  Mask 示例 (前3个token, 前4个专家):\n{mask[:3, :, :4]}")
    
    # Step 5: 计算实际负载
    actual_load = mask.mean(dim=0)  # [1, top_k, num_experts]  在b*s维度上求平均
    print("\n" + "=" * 60)
    print("Step 5: 计算实际负载分布")
    print(f"  Shape: {actual_load.shape}")
    print(f"  每个专家被选中的频率: {actual_load.numpy()}")
    print(f"  期望负载 (1/{num_experts}): {1/num_experts:.4f}")
    
    # Step 6: 计算期望负载
    expected_load = torch.ones_like(router_probs) / num_experts
    print("\n" + "=" * 60)
    print("Step 6: 期望负载 (均匀分布)")
    print(f"  Shape: {expected_load.shape}")
    print(f"  期望值: {expected_load[0].numpy()}")
    
    # Step 7: 计算 router_probs.mean (所有token的平均路由概率)
    router_probs_mean = router_probs.mean(dim=0)
    print("\n" + "=" * 60)
    print("Step 7: 所有token的平均路由概率")
    print(f"  Shape: {router_probs_mean.shape}")
    print(f"  平均概率: {router_probs_mean.numpy()}")
    
    # Step 8: Auxiliary Loss
    aux_loss = torch.sum(actual_load * router_probs_mean) * num_experts
    print("\n" + "=" * 60)
    print("Step 8: 计算 Auxiliary Loss")
    print(f"  actual_load: {actual_load.numpy()}")
    print(f"  router_probs_mean: {router_probs_mean.numpy()}")
    print(f"  actual_load * router_probs_mean: {(actual_load * router_probs_mean).numpy()}")
    print(f"  Sum: {torch.sum(actual_load * router_probs_mean).item():.6f}")
    print(f"  aux_loss (Sum * num_experts): {aux_loss.item():.6f}")
    
    # Step 9: Z-Loss Router Logits 的平方均值
    # router_logits 越大 → softmax输出越极端 → 某些专家负载越不均衡
    z_loss = torch.mean(torch.square(router_logits))
    print("\n" + "=" * 60)
    print("Step 9: 计算 Z-Loss")
    print(f"  router_logits mean: {router_logits.mean().item():.6f}")
    print(f"  router_logits std: {router_logits.std().item():.6f}")
    print(f"  z_loss: {z_loss.item():.6f}")
    
    # Step 10: 总损失
    z_loss_weight = 0.001
    total_loss = aux_loss + z_loss * z_loss_weight
    print("\n" + "=" * 60)
    print("Step 10: 总损失计算")
    print(f"  z_loss_weight: {z_loss_weight}")
    print(f"  z_loss * weight: {z_loss.item() * z_loss_weight:.6f}")
    print(f"  total_loss = aux_loss + z_loss*weight: {total_loss.item():.6f}")
    
    print("\n" + "=" * 60)
    print("最终输出:")
    print(f"  total_loss: {total_loss.item():.6f}")
    print(f"  aux_loss: {aux_loss.item():.6f}")
    print(f"  z_loss: {z_loss.item():.6f}")
    print("=" * 60)
    
    return total_loss, aux_loss, z_loss


# ============================================================
# 运行测试
# ============================================================

print("\n" + "🚗" * 30)
print("Switch Load Balancing Loss 详细调试")
print("🚗" * 30 + "\n")

# 设置随机种子以确保可复现
torch.manual_seed(42)
np.random.seed(42)

# 参数设置
batch_size = 8        # batch大小
seq_len = 4           # 序列长度  
hidden_dim = 16       # 隐藏层维度
num_experts = 4       # 专家数量
top_k = 2             # 每个token选择的专家数量

# 模拟 router_logits (batch_size * seq_len, num_experts)
# 模拟不均匀分布的情况 (专家0和1被更频繁选择)
router_logits = torch.randn(batch_size * seq_len, num_experts) * 2.0
router_logits[:, 0] += 1.0  # 让专家0的logits更高
router_logits[:, 1] += 0.5  # 让专家1的logits稍高

print(f"📊 参数设置:")
print(f"  batch_size: {batch_size}")
print(f"  seq_len: {seq_len}")
print(f"  总token数: {batch_size * seq_len}")
print(f"  num_experts: {num_experts}")
print(f"  top_k: {top_k}")
print()

# 调用详细版损失函数
total_loss, aux_loss, z_loss = switch_load_balancing_loss_detailed(
    router_logits, 
    num_experts,
    top_k
)

# ============================================================
# 额外分析: 专家负载分布可视化
# ============================================================

print("\n📈 专家负载分布分析:")
print("-" * 40)

# 计算每个专家被选中的次数
_, selected = torch.topk(torch.softmax(router_logits, dim=-1), top_k, dim=-1)
expert_counts = torch.bincount(selected.flatten(), minlength=num_experts)
total_assignments = expert_counts.sum()

for exp_idx in range(num_experts):
    count = expert_counts[exp_idx].item()
    pct = count / total_assignments * 100
    bar = "█" * int(pct / 2)
    print(f"  专家 {exp_idx}: {count:3d} 次 ({pct:5.1f}%) {bar}")

print("-" * 40)
print(f"  总分配次数: {total_assignments}")
print(f"  理想每次分配: {total_assignments / num_experts:.1f}")
