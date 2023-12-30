# SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single Image

- [项目地址](https://vita-group.github.io/SinNeRF/)  
- [github](https://github.com/VITA-Group/SinNeRF)  

## 简介


### 特点/ 关键词

- 单一视图
- 半监督框架 semi-supervised
- 语义和几何规则 semantic pseudo labels & geometry pseudo labels 
- 卷积

### 类似工作比较

- pixelNeRF 等 -- 针对大规模数据集进行充分预训练
- 或要求 -- 不同视图的颜色和几何形状的各种正则化
- 或各种特定物体先验知识, 如人体.

且至少3视图. pixelNeRF可以实现单视图, 不过因其预训练的特性, 只适用于简单物体. 

### 挑战

对于NeRF来说, 直接在单一视图重建会导致严重过拟合, 导致其他视角画面崩坏. 

SinNeRF 对此, 构建半监督框架, 对看不见的视角提供特殊的约束 -- 几何约束和语义约束, 而不是图像. 

![Alt text](SinNeRF.drawio.01f837d9d69b1db62c00.jpg)

* **几何伪标签 & 深度一致性**：通过使用现有的3D重建方法从单一图像中估计出**粗略的几何结构**，产生所谓的**伪3D几何标签**。这些伪标签不是完全精确的，但提供了足够的几何信息来指导NeRF的训练。例如其中的深度一致性约束, 渲染的图像在结构上应该与初始估计的3D几何结构相匹配。
* **语义伪标签 & 语义一致性**：使用预训练的语义分割网络从单一视图中提取语义信息。这些语义标签（如物体边界、类别信息）被用作额外的指导信息，帮助NeRF理解场景中的不同物体和区域。确保NeRF模型在渲染新视角时，保持语义信息的一致性。例如，如果原始图像中的某个区域被标记为“树木”，那么从新视角渲染的相同区域也应当被识别为“树木”。




