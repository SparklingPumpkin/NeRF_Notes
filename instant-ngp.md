# Instant Neural Graphics Primitives with a Multiresolution Hash Encoding

本篇不是专门用于改善NeRF的工作, 可以用于SDF/ Gigapixel image/ NeRF

然而, 本偏主要的优化方式是通过定制cuda (包含了大量的cuda源码), 使得硬件利用效率最大化. 这很好地利用了硬件上限, 但大大降低了可拓展性 -- 1, 优化空间不多; 2, 其他开发者很难在此工作基础上进行修改. 

## 1 特色

作者评价了此前研究者的工作, 认为最成功的一个是特化训练任务的一个方法 (用于改进NeRF)-- 然而此类数据结构依赖于启发式和结构修改（例如修剪、拆分或合并），这可能会使训练过程复杂化，将方法限制为特定任务，或限制 GPU 上的性能，因为控制流和指针追逐成本很高。

作者通过 **多分辨率哈希编码** 解决了这个问题. 该方法有自适应性和高效性, 它仅由两个值（参数数量 T 和所需的最高分辨率 Nmax）进行配置，经过几秒钟的训练后, 即可得到超越大多数相似研究的渲染重建质量. 

### 1.1 自适应性

作者将需要重建的 NeRF网格 映射到相应的固定大小的特征向量数组。
- 在**粗略分辨率**下，从网格点到数组条目的映射为 1：1
- 在**精细分辨率**下，数组被视为**哈希表**，并使用空间哈希函数进行索引，其中多个网格点为每个数组条目添加别名。这种**哈希冲突**会导致**碰撞训练梯度平均**，这意味着最大的梯度（与损失函数最相关的梯度）将占主导地位。因此，哈希表会自动对稀疏区域进行优先级排序，并具有最重要的精细比例细节。

与以前的工作不同, 得益于哈希冲突的自动排序机制, instant-ngp 在训练期间的任何时候都不需要对数据结构进行结构更新。

### 1.2 高效性

- 哈希表查找操作的时间复杂度是常数时间 $O(1)$ (无论哈希表的大小如何，查找操作的时间都是恒定的)
- 不需要控制流 (即循环或条件判断等). 这可以高效地利用 GPU. (避免了树遍历中固有的执行分歧和串行指针追逐, 可以**并行查询**所有分辨率的哈希表)

## 2 多分辨率哈希编码

给定一个 FNN -- $m(\mathbf{y};\Phi)$, 其中很重要的一部分是是输入 $\mathbf{y}=\mathrm{enc}(\mathbf{x};\theta)$ 的编码, 能在不增加性能开销的情况下加速训练提升渲染重建质量. 

所以, instant-ngp 不仅训练了权重参数 $\Phi$, 还训练了编码参数 $\theta$

这些超参数分为$L$级，每个级别包含最多$T$个维度为$F$的特征向量. 每个级别 (其中两个在下图中显示为红色和蓝色) 都是独立的. 

特征向量存储在网格的顶点处, 其分辨率被选为最粗 (Nmin) 和最细 (Nmax)分辨率之间的**几何级数**. 


![Alt text](image-7.png)

*以 2D 形式展现多分辨率哈希编码过程*

1. 对于给定的输入坐标 $x$，以 $L$级别的分辨率 水平找到周围的体素, 并通过散列它的整数坐标将索引分配给它周围的网格顶点坐标. 
2. 对于这些 $L$分辨率 下的网格坐标, 从哈希表 $θ_l$  中查找相应的 $F$- 维特征向量
3. 根据 $x$ 的位置在 $L$分辨率 (?这里不是很懂, 应该是更细分辨率的级别插值?) 对应的体素下线性插值. 
4. 将每个级别的结果组合 (简单拼接), 并进行辅助的额外输入 $\xi\in\mathbb{R}^{E}$ (这个辅助输入暂时不知道什么意思?), 拼接成最终输入 $y\in\mathbb{R}^{LF+E}$, 完成编码, 进入MLP. 
5. 为了训练编码, 梯度下降损失 被backpropagated 通过 MLP (5) -> the concatenation (4) -> the linear interpolation (3), 然后累计到查找的特征向量. 


## 3 架构

模型由两个串联的 MLP 组成：密度 MLP 和 颜色 MLP 






### 4 实验

原文:

一方面，我们的方法在具有高几何细节的场景中表现最佳，如榕树、鼓、船和乐高，实现了所有方法中最好的PSNR。另一方面，mip-NeRF和NSVF在具有复杂、与视图相关的反射的场景中表现优于我们的方法，例如材质;我们将其归因于我们必须采用的更小的 MLP，以便与这些竞争实现相比获得几个数量级的加速。







































