# K-Planes: Explicit Radiance Fields in Space, Time, and Appearance

## 1 特点
- 白盒模型, 可解释
- 4D volumes分解. 用3个平面代表空间+3个平面代表时间变化. 

## 2 主要流程

4D volumes可以分解为6个平面, 3个平面代表空间+3个平面代表时间变化. 如下图:

![Alt text](image-9.png)

为了得到一个四维点 $q =(x，y，z，t)$, 
- a. 将该点投影到每个平面上
- b. 多尺度双线性插值
- c. 内插值相乘，然后在S尺度上进行连接
- d. 这些特征可以用一个小的MLP  或作者提供的显式线性解码器进行解码。
- e. 标准的体积渲染公式 预测光线的颜色和密度
- f. 在时空上的简单正则化来最小化重建损失进行优化


## 3 K-planes 模型

*有点类似于TensoRF的张量分解*

简单说, 这是一个可解释的, 用于分解表示**任意维度**场景的模型. 对 $d$ 维场景, 该模型将其分解为 $k=\bigl(\begin{smallmatrix}d\\2\end{smallmatrix}\bigr)$ 个平面进行处理

- 对于3D场景 (静态), 分解 $k=\bigl(\begin{smallmatrix}3\\2\end{smallmatrix}\bigr)=3$ 平面, 分别为 $xy, xz, yz$.
- 对于4D场景 (动态), 分解 $k=\bigl(\begin{smallmatrix}4\\2\end{smallmatrix}\bigr)=6$ 平面, 分别为 $xy, xz, yz, xt, yt, zt$.
- 对于5D...

### 3.1 平面

假设时间/ 空间分辨率 皆为 $N$, 特征维度为 $M$, 于是每个平面的形状为 $N{\times} N {\times} M$ 

对于一个4D点 $q=(i,j,k,\tau)$, 正则化后, 投影到平面:
$$f(\boldsymbol{q})_c=\psi(\mathbf{P}_c,\pi_c(\boldsymbol{q}))$$

$\pi_c$ 表示将点 $q$ 投影到 $P_c$平面上, $\psi$ 表示插值到规则二维网格. 最终用 Hadamard product 产生最后的长度为 M 的特征向量:
$$f(\boldsymbol{q})=\prod_{c\in C}f(\boldsymbol{q})_c$$

*为什么选择 Hadamard product 分解而不是像TensoRF VM分解中的加法乘法混合呢? 因为乘法组合更能体现局部的空间特征*


### 3.2 可解释性

**时空分离**的特性使得模型**可解释**, 并可以根据需求添加**特定维度的先验知识**. 特别是对于不随时间变化的点, 可以很方便的压缩. 

- **多尺度平面（Multiscale planes）** -- 较大尺度的平面捕捉粗略的全局特征，而较小尺度的平面负责细节。这在很多其他模型都有相应的思想. * 不对时间尺度做分离.  
- **空间的总变异（Total variation in space）** -- 空间的总变异作为一种正则化手段，帮助平滑空间平面并减少噪声。这种方法在保持细节的同时，确保了场景的整体平滑性和自然感。
- **时间的平滑性（Smoothness in time）** -- 在时间维度上，模型采用平滑约束来处理时间变化，确保动态场景中的连续性和自然过渡。这有助于生成在时间上一致和流畅的动态效果，特别是在处理视频和动态场景时非常关键。
- **稀疏瞬变（Sparse transients）** -- 为了有效地处理动态场景中的短暂变化（如物体的快速移动或外观的突然变化），K-Planes 模型引入了稀疏瞬变的概念。这种方法允许模型在保持整体连贯性的同时，快速适应场景的局部变化。

### 3.3 特征解码










