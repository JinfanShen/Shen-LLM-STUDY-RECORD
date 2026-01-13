# RoPE 旋转机制详解：为何交换前半与后半?
虽然 RoPE 的原始论文公式通常写作**相邻元素配对** $(x_{2i}, x_{2i+1})$，但在主流代码实现（如 HuggingFace Transformers, LLaMA, PaLM）中，通过**交换前半与后半部分** $(x_i, x_{i+d/2})$ 来实现旋转。

这就导致了你看到的现象：`rotate_half` 函数将向量的前半和后半交换，并对后半取负。

本文将通过数学推导证明这两种方式本质上是等价的二维旋转，并解释为什么代码要这样实现。

## 1. 核心数学原理：二维旋转

无论如何配对，RoPE 的核心都是将一个二维向量 $\begin{pmatrix} u \\ v \end{pmatrix}$ 旋转角度 $\theta$：

$$
\begin{pmatrix} u' \\ v' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix} = \begin{pmatrix} u\cos\theta - v\sin\theta \\ u\sin\theta + v\cos\theta \end{pmatrix}
$$

## 2. 两种配对方式对比

假设我们有一个维度 $d=4$ 的向量 $x = [x_0, x_1, x_2, x_3]$。我们需要应用两个旋转角 $\theta_0, \theta_1$。

### 方式 A：理论上的相邻配对 (Adjacent Pairs)
论文通常将向量看作复数序列：$(x_0+ix_1), (x_2+ix_3)$。
配对为：$(x_0, x_1)$ 和 $(x_2, x_3)$。

*   **对 $(x_0, x_1)$ 旋转 $\theta_0$**:
    $$x'_0 = x_0 \cos\theta_0 - x_1 \sin\theta_0$$
    $$x'_1 = x_0 \sin\theta_0 + x_1 \cos\theta_0$$
*   **对 $(x_2, x_3)$ 旋转 $\theta_1$**:
    $$x'_2 = x_2 \cos\theta_1 - x_3 \sin\theta_1$$
    $$x'_3 = x_2 \sin\theta_1 + x_3 \cos\theta_1$$

### 方式 B：代码中的半半配对 (Half-Half Pairs)
代码将向量分为前半部分 $X_{half1} = [x_0, x_1]$ 和后半部分 $X_{half2} = [x_2, x_3]$。
配对为：$(x_0, x_2)$ 和 $(x_1, x_3)$。即索引 $i$ 和 $i+d/2$ 配对。

代码逻辑如下：
1.  **构造旋转角向量**：
    由于是拼接的，前半部分和后半部分对应同一个 $\theta$。
    `cos` 向量结构为：$[\cos\theta_0, \cos\theta_1, \cos\theta_0, \cos\theta_1]$ (注意这里通常是广播或者拼接，使得 $i$ 和 $i+d/2$ 用同一个 $\theta$)。
    在你的代码中，`freqs` 长度是 $d/2$，
    `angles` 是 $[ \theta_0, \theta_1 ]$。
    `cos` 是 `cat([cos, cos])` $\rightarrow [\cos\theta_0, \cos\theta_1, \cos\theta_0, \cos\theta_1]$。
    `sin` 是 `cat([sin, sin])` $\rightarrow [\sin\theta_0, \sin\theta_1, \sin\theta_0, \sin\theta_1]$。

2.  **执行 `rotate_half(x)`**：
    原始 $x = [x_0, x_1, x_2, x_3]$
    前半 $x_{h1} = [x_0, x_1]$，后半 $x_{h2} = [x_2, x_3]$
    `rotate_half(x)` 返回 $[-x_2, -x_3, x_0, x_1]$。

3.  **计算最终结果**：
    公式：$x_{new} = x \cdot \cos + \text{rotate\_half}(x) \cdot \sin$

    我们来看看第 0 个位置（对应原 $x_0$）和第 2 个位置（对应原 $x_2$）：

    *   **位置 0 ($i=0$)**:
        对应 $\cos$ 值：$\cos\theta_0$
        对应 $\sin$ 值：$\sin\theta_0$
        对应 $x$ 值：$x_0$
        对应 $rotate\_half$ 值：$-x_2$ (来自后半部分取负)
        $$x'_{0} = x_0 \cdot \cos\theta_0 + (-x_2) \cdot \sin\theta_0 = \mathbf{x_0 \cos\theta_0 - x_2 \sin\theta_0}$$

    *   **位置 2 ($i=2$)**:
        对应 $\cos$ 值：$\cos\theta_0$ (因为是 cat 重复的)
        对应 $\sin$ 值：$\sin\theta_0$
        对应 $x$ 值：$x_2$
        对应 $rotate\_half$ 值：$x_0$ (来自前半部分)
        $$x'_{2} = x_2 \cdot \cos\theta_0 + x_0 \cdot \sin\theta_0 = \mathbf{x_2 \cos\theta_0 + x_0 \sin\theta_0}$$

    **结论**：
    你可以清晰地看到，方式 B 实际上就是对 **$(x_0, x_2)$** 这一对数值执行了标准的二维旋转！
    
    $$
    \begin{pmatrix} x'_0 \\ x'_{2} \end{pmatrix} = \begin{pmatrix} \cos\theta_0 & -\sin\theta_0 \\ \sin\theta_0 & \cos\theta_0 \end{pmatrix} \begin{pmatrix} x_0 \\ x_{2} \end{pmatrix}
    $$

## 3. 为什么要用“方式 B”？

既然数学上等价，为什么不直接用相邻配对？主要原因是**计算效率**。

在 GPU 编程中，内存访问模式对性能影响巨大。

*   **相邻配对** $(x_{2i}, x_{2i+1})$：需要访问 `x[0], x[1]`, `x[2], x[3]`。在 Tensor 操作中，如果要并行化处理，通常需要 `reshape` 成 `(..., d/2, 2)` 然后操作，或者使用 stride slice `x[..., ::2]` 和 `x[..., 1::2]`。这种 stride 访问在内存中是不连续的。
*   **半半配对** $(x_i, x_{i+d/2})$：只需要将向量“切两半”。
    `x1 = x[..., :d/2]` （前半段，内存连续）
    `x2 = x[..., d/2:]` （后半段，内存连续）
    这种大块的连续内存切片（Slice）和拼接（Concat）操作在现代深度学习框架（PyTorch, TensorFlow, JAX）和底层硬件（GPU）上非常高效。

因此，**半半配对**成为了 RoPE 的标准工程实现。
