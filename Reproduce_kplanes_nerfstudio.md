# 复现

*本篇大体参照 NeRF, Nerfstudio 与 k-planes 的论文以及代码, 用于记录代码理解, 在 nerfstudio 框架下复现 k-planes 方法. 包含构建python package大体流程, 其中重点在于记录理解模型核心要点*

## 1. 环境

- win11/ python3.8/ pycharm/ cuda11.8/ torch2.0.1+cu118

## 2. 前期准备

- 创建项目文件夹fnspkg (FJN Nerfstudio Package)
- 创建子文件夹兼python包 fnspkg
- `__init__.py` 文件中加入版本控制说明 `__version__ = "0.1.0"`
- `__init__.py` 文件同级目录创建 `fnspkg.py`, `fnspkg_configs.py`, `fnspkg_field.py` 三个文件
    - `fnspkg.py` 为主体模型
    - `fnspkg_config.py` 为配置文件, 用于控制模型参数
    - `fnspkg_field.py` 为模型组件, 用于关联**空间信息与物体特征**. NeRF原文中, 前者为 3D 位置和 2D 观看方向 $x = (x, y, z) , d = (θ, φ)$, 后者为 颜色和体积密度 $c = (R, G, B), σ$ ; field文件在此主体就是求解器 MLP
    
    - 补充: 创建 `fnspkg_encoding.py` 用于更改编码策略

## 3. 模型主体

### 3.1. field

*部分重要函数/ 类方法*

```
# Field类方法
def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
    '''计算, 返回密度'''

    positions(光线位置) = ray_samples获得位置
    positions正则化, 确定是否有时间变量
    positions_flat展平
    position_flat - 编码/ 插值等操作 - 输入求解器(MLP等)\
    - (解码) 得到features - 分割feartures中的density部分\
    - (激活等操作)

    return density, features
```

```
# Field类方法
def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, Tensor]:
    '''返回自定义输出, 一般是RGB'''

    directions(光线方向) 获取 (通过raysample)
    正则化 directions
    编码 directions
    colorfeature -> 在 density_embedding 中提前

    '''外观嵌入, 低维表示相机视角或已知照明条件, 可以作为神经网络的输入'''
    if 使用embedded_appearance:
        张量- 通过输入信息获取embedded_appearance. (最优)
    elif 平均:
        张量 - 平均嵌入 (使用所有训练样本的外观嵌入的平均值)
    else:
        张量 - 全零嵌入 (忽略了外观变化)

    合并颜色特征

    返回特征
```


### 3.2. model

*部分重要函数/ 类方法*

```
# Model类方法
def get_outputs(self, ray_bundle: RayBundle):
    # 与field同名方法类似, 但专注于求单点特征
    # 此处省略
```


```
# Model类方法
def get_metrics_dict(self, outputs, batch)
    '''用于收集整合各种性能指标'''

    图像 = 从批次中获取"image"并传输到模型设备
    if 图像为RGBA格式: 
        将图像与背景混合

    初始化一个空字典：metrics_dict
    计算并添加"psnr"指标到metrics_dict

    if 训练模式:
        计算interlevel损失并添加到metrics_dict
        计算distortion损失并添加到metrics_dict

        从设定网络中获取plane coefficients作为prop_grids
        从field中获取plane coefficients作为field_grids

        计算plane_tv & plane_tv_proposal_net损失, 
        都添加到metrics_dict

        if grid_base_resolution的长度为4 (即考虑时间):
            计算l1_time_planes &l1_time_planes_proposal_net &
            time_smoothness &&
            time_smoothness_proposal_net损失, 
            都添加到metrics_dict

    return metrics_dict
```


```
# Model 函数
def compute_plane_tv(t: torch.Tensor, only_w: bool = False) -> float:
    '''计算分解的平面total varation损失'''

    # 分别计算宽 & 高相邻元素 的差平方 之和 之平均
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).mean()
    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).mean()

    return h_tv + w_tv

# 空间tv
def space_tv_loss(multi_res_grids: List[torch.Tensor]) -> float:
    '''遍历每个网格引用compute_plane_tv()即可'''
    return total / num_planes

# 时间平面类似
def l1_time_planes(multi_res_grids: List[torch.Tensor]) -> float:
    略

# 此外还有 compute_plane_smoothness() & time_smoothness() 用于计算平滑度, 均在类模型中使用, 此处略. 
```

### 3.3. Config

*用于配置各项参数, 略*

### 3.4. Encodings

*主要用于定义编码类, 一般在 Field 文件中使用*


## 4. 收尾

- 编写 pyproject.toml
- pip install e . 到conda环境
- ns-train ... 即可使用







