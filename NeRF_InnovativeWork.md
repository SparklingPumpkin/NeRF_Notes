# NeRF_InnovativeWork

近年来，NeRF相关论文主要往两两方向发展：NeRF本身的分析和改进；基于NeRF框架的扩展。

Filter criteria: literature review/ most cited by papers/ newest

## 1. Pre_NeRF

*2019_CVPR/ ICCV*

1. Occupancy networks 
2. IM-Net
3. DeepSDF
4. PIFu


## 2. NeRF

NeRF原论文效果好，并且模型简单 —— 仅仅使用MLP接受5D坐标并输出密度和颜色。因此，也留下了许多可以改进之处，如：  
* 训练、渲染慢
* 仅能表示静态场景
* it bakes lighting，失去了动态光的灵活性
* A trained NeRF 无法泛化

## 3. NeRF_Improved

### 3.1 Performance
*Performance-改善性能。以下分类只是大致分类，许多文章都在很多方面有改进，有交叉部分*   

#### 3.1.1 加快渲染Faster Inference
1. [DyLiN: Making Light Field Networks Dynamic](https://dylin2023.github.io/) | CVPR2023_2(被引用次数) | [github](https://github.com/Heng14/DyLiN)   
    * DyLiN是对传统的光场网络的扩展，旨在处理动态场景中的非刚性变形，包括拓扑变化。光场网络是辐射场的重构，用于处理定向光线，相比于它们的坐标网络对应物，光场网络在表示2D观察中的3D结构方面更快速且保真度更高。然而，传统光场网络的限制在于它们仅适用于整体静态场景。DyLiN通过学习从输入光线到规范光线的变形场，并将它们提升到更高维空间以处理不连续性，从而克服了这一限制​​。


2. [R2L: Distilling Neural Radiance Field to Neural Light Field for Efficient Novel View Synthesis](https://arxiv.org/abs/2203.17261) | ECCV2022_36 | [github](https://github.com/snap-research/R2L)   
    * 这项研究通过将神经辐射场（NeRF）的信息提炼到神经光场（NeLF）中，以实现更高效的新视角图像合成。NeRF用于3D场景计算量较大，NeLF则是一种更高效的2D图像表示方法。通过将复杂的3D信息提炼为2D表示，R2L能够更快速地生成新视角的图像。


3. [KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs](https://arxiv.org/abs/2103.13744) | ICCV2021_443 | [github](https://github.com/creiser/kilonerf)   
    * KiloNeRF**将三维空间分成小块，每块由一个独立的小型MLP处理**（约有数千个小型MLP）。这种划分使得每个MLP只需要处理较小的数据量，从而降低了计算复杂度，同时可以更快地并行处理，大大提高了渲染速度。KiloNeRF对每个小型MLP进行了优化，以确保它们在处理速度和渲染质量之间达到良好的平衡。
    * KiloNeRF实现了一种适应性渲染策略，根据场景的复杂性动态调整MLP的数量和分布。这意味着对于简单的场景部分，可以使用更少的MLP，而对于复杂的部分则使用更多，从而在保持渲染质量的同时提高了效率。这与NeRF原文思想类似，但应用方式有变化。


4. [AutoInt: Automatic Integration for Fast Neural Volume Rendering](https://arxiv.org/abs/2012.01714) | CVPR2021_182 | [github](https://github.com/computational-imaging/automatic-integration)      
    * AutoInt通过**自动化积分**过程来加速神经体积渲染。AutoInt的快速渲染能力使其在许多需要实时或近实时渲染的应用中具有潜在价值，如虚拟现实、增强现实和计算机游戏。


5. [Neural Sparse Voxel Fields](https://github.com/facebookresearch/NSVF) | NeurIPS2020_733 | [github](https://github.com/facebookresearch/NSVF)  
    * SVF提出了一种新的隐式表示方法，用于高效的自由视角渲染。它由一组**体素边界隐式场**组成，这些场被组织在一个**稀疏的体素八叉树**中。这种方法的优点在于它可以通过**跳过不包含相关场景内容的体素**，从而**加快新视角的渲染**。与大多数显式的几何表示不同，神经隐函数是平滑且连续的，并且理论上可以实现高空间分辨率​​。
    * 在体素八叉树中，每个节点代表一个体素，而该体素可以被进一步细分为八个更小的体素（子节点）。这种结构非常适合用于表示稀疏的3D数据，因为它允许在具有复杂几何或属性的区域使用更细的分辨率，同时在空旷或简单的区域使用较粗的分辨率。体素八叉树常用于体积渲染和场景的几何优化，因为它可以显著减少渲染和处理所需的数据量。



#### 3.1.2 加快训练

1. [TensoRF: Tensorial Radiance Fields](https://apchenstu.github.io/TensoRF/) | ECCV2022_405 | [github](https://github.com/apchenstu/TensoRF)  
    * TensoRF将场景的辐射场视为一个4D张量，并提出将这个张量分解为多个紧凑的低秩张量成分，用于高效的场景建模​​。TensoRF的张量分解显著降低了空间复杂性，从而减少了内存占用。

2. [Plenoxels Radiance Fields without Neural Networks](https://alexyu.net/plenoxels/) | CVPR2022_353 | [github](https://github.com/sxyu/svox2)  
    * Plenoxel模型是一种基于视角依赖的稀疏体素模型，它能够以与神经辐射场（NeRF）相同的保真度进行优化，但**不需要任何神经网络**。这一模型在单GPU上的典型优化时间仅为11分钟，比NeRF快两个数量级​​​​。

3. [Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction](https://arxiv.org/abs/2111.11215.pdf) | CVPR2022_401 | [github](https://github.com/sunset1995/DirectVoxGO)  
    * 此文提出了一种用于重建每个场景的辐射场的超快收敛方法。这种方法的关键是直接优化在密集体素网格中建模的体积密度。与NeRF和它的变体不同，这种方法不需要从任何训练过的隐式模型转换步骤或跨场景预训练，而是可以直接且高效地从头开始为每个场景进行训练​​。


#### 3.1.3 一般化Generalization

1. [Rodin: A generative model for sculpting 3d digital avatars using diffusion](https://3d-avatar-diffusion.microsoft.com/) | CVPR2023_44 | [github](https://3d-avatar-diffusion.microsoft.com/)  
    * 使用扩散模型自动生成以神经辐射场（Neural Radiance Fields, NeRF）表示的3D数字化头像。 
    * 为了解决在生成高品质头像时所需的丰富细节所带来的巨大内存和计算成本，论文提出了卷展扩散网络（Rodin）。Rodin将神经体积表示为多个2D特征图，并将这些图卷展到一个单一的2D特征平面中，然后在该平面内进行3D感知的扩散。这使得Rodin模型在保持扩散在3D中的完整性的同时，带来了急需的计算效率​​。
    * 3D感知卷积——明确考虑到一个2D特征在三面体平面（tri-plane）中的一个平面上是从3D数据中投影出来的，与其他两个平面上相同数据的投影特征内在相关，借此跨平面通信，同步细节的合成，以反映它们在3D中的关系。  
    * Rodin模型使用潜在向量来调控特征生成，使其在整个3D体积中全局一致，从而产生更高质量的头像，并支持基于文本提示的语义编辑。这是通过使用训练数据集中的头像来训练一个额外的图像编码器实现的，该编码器提取一个语义潜在向量作为扩散模型的条件输入。此外，模型采用了冻结的CLIP图像编码器，该编码器与文本提示共享潜在空间​​。

2. [Point-NeRF: Point-based Neural Radiance Fields](https://xharlie.github.io/projects/project_sites/pointnerf/index.html) | CVPR2022_187 | [github](https://github.com/Xharlie/pointnerf)  
    * 作者结合了NeRF（神经辐射场）和基于点云的深度多视图立体视觉（Multi-View Stereo, MVS）技术的优势，以解决现有方法在重建和渲染**高质量头像**时面临的一些挑战。以下是该论文的主要改进点：
    * 点基神经辐射场（Point-NeRF）表示：Point-NeRF使用**神经3D点云**及其相关的神经特征来模拟辐射场，每个神经点编码了其周围局部3D场景的几何和外观信息。这种表示形式不仅提高了渲染质量，而且使得模型能够更有效地重建和渲染场景，尤其是在避免在空场景空间中进行射线采样方面​​。
    * 基于点云的高效初始化和优化：Point-NeRF可以通过预训练的深度神经网络直接推断来初始化，生成一个神经点云。与仅依赖于每个场景拟合的NeRF不同，这种方法允许在短时间内对每个场景进一步优化，实现了更快速和精确的头像重建和渲染​​。
    * 处理重建点云中的错误和离群值：在处理像COLMAP这样的重建技术生成的点云时，Point-NeRF引入了点增长和剪枝机制。这一机制有效地改进了最终的重建和渲染质量，特别是在填补大型空洞和产生逼真渲染方面​​。
    * 更精细的局部场景几何和外观建模：与全局MLP编码整个场景空间的传统NeRF方法相比，Point-NeRF利用精细的神经点来模拟复杂的局部场景几何和外观，从而提供了更好的渲染质量​​。
    * 适应实际表面的点基表示：与基于体素网格的局部神经辐射表示相比，Point-NeRF的点基表示更好地适应了实际表面，从而提供了更高的质量。此外，该模型直接预测良好的初始神经点特征，绕过了大多数基于体素的方法所需的每个场景优化​​。
    * 神经点与空间变化的神经特征结合：Point-NeRF利用具有空间变化神经特征的神经点来编码其辐射场。这种局部化表示可以比纯粹的MLP（具有有限的网络容量）模拟更复杂的场景内容。更重要的是，Point-NeRF展示了其点基神经场可以通过预训练的深度神经网络高效初始化，并在场景中普遍实现高效的辐射场重建​​。

3. [SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single Image](https://vita-group.github.io/SinNeRF/) | ECCV2022_81 | [github](https://github.com/VITA-Group/SinNeRF)  
    * 该研究聚焦于使用单张图像来训练复杂场景的NeRF。
    * 通过创新的网络架构和训练策略，SinNeRF能够从二维图像中推断出三维空间的信息，这使得从单一视角图像生成的三维场景具有逼真的深度感和细节。尽管只依赖单一图像，SinNeRF依然能够处理和重建复杂的场景。这包括丰富的纹理、复杂的光照条件和细节丰富的几何结构。

4. [MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo](https://apchenstu.github.io/mvsnerf/) | ICCV2021_378 | [github](https://github.com/apchenstu/mvsnerf)  
    * 将多视角立体视觉（Multi-View Stereo, MVS）和NeRF结合。传统的NeRF方法在处理新场景时需要长时间的训练。MVSNeRF通过特殊的网络设计和训练策略（设计了一种高效的数据处理流程，能够快速处理输入的多视角图像），实现了对新场景的快速适应，即使是在只有有限视角数据的情况下也能快速重建出高质量的三维场景。作者在多种场景下都显示出了良好的适应性和泛化能力，例如室内外场景、历史遗迹、自然景观等，显示了其在实际应用中的潜力。
 
5. [IBRNet: Learning Multi-View Image-Based Rendering](https://ibrnet.github.io/static/paper.pdf) | CVPR202_459 | [github](https://github.com/googleinterns/IBRNet)   
    * 论文提出了一种自适应特征聚合机制，能够从多个输入图像中有效地提取和融合信息。这种机制使得网络能够根据不同场景自动调整其处理方式，提高了渲染质量和泛化能力。

6. [pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](https://marcoamonteiro.github.io/pi-GAN-website/) | CVPR2021_572  
    * pi-GAN是一种用于3D感知图像合成的生成对抗网络。
    * pi-GAN专注于生成3D感知的图像，这意味着它不仅能产生高质量的2D图像，还能捕捉这些图像背后的3D几何结构。这一点在虚拟现实、增强现实和3D建模等领域尤为重要。
    * pi-GAN利用了生成对抗网络的架构，其中生成器试图产生越来越逼真的图像，而鉴别器则试图区分生成的图像和真实图像。这种机制使得pi-GAN能够生成高度逼真的图像。

7. [pixelNeRF: Neural Radiance Fields from One or Few Images](https://arxiv.org/abs/2012.02190) | CVPR2021_868 | [github](https://github.com/sxyu/pixel-nerf)   
    * 此文提出pixelNeRF，能够根据一个或几个输入图像预测连续的神经场景表示。
    * 从少量视角生成NeRF。pixelNeRF能够从极少量的视图（**甚至一个视图**）中预测出NeRF——这是通过将NeRF条件化在输入图像的**全卷积图像特征网格**上来实现的。这样的设计使得网络能够在多个场景上进行训练，学习到场景先验（pixelNeRF利用与每个像素对齐的空间图像特征作为输入），从而能够以前馈方式从稀疏的视图集合中进行新视角合成​​。
    * pixelNeRF在**输入图像的相机坐标系而非规范坐标框架**中预测NeRF表示。这不仅对于泛化到未见过的场景和对象类别至关重要，而且由于在有多个对象或真实场景中没有明确的规范坐标系，这种方法也更具灵活性。

8. [GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis](https://arxiv.org/abs/2007.02442) | NeurIPS2020_599 | [github](https://github.com/autonomousvision/graf)
    * GRAF将NeRF技术与GAN框架结合起来。NeRF用于创建连续的3D场景表示，而GAN则用于提高生成图像的真实性。这种结合允许GRAF同时捕捉3D场景的几何细节和视觉真实感。


#### 3.1.4 Video

1. [UV Volumes for Real-time Rendering of Editable Free-view Human Performance](https://fanegg.github.io/UV-Volumes/) | CVPR2023_10 | [github](https://github.com/fanegg/UV-Volumes)  
    * **UV体积技术**是一种新型的数据结构，它将3D空间中的人体表演编码为一系列2D纹理图，这些图通过UV映射关联到3D空间。这种方法相比传统的3D模型或点云表示，可以更高效地处理和渲染复杂的人体动作。该技术支持实时渲染，能够在没有预先渲染的情况下快速生成高质量的图像。这对于需要实时反馈的应用场景（如虚拟现实或交互式游戏）非常重要。

2. [Neural 3D Video Synthesis from Multi-view Video](https://neural-3d-video.github.io/) | CVPR2022_156
    * 多视角观察动态场景视频。通过分析这些视角，算法能够理解场景的3D结构。多视角视频合成三维视频，不仅包含时间上的动态变化，还包含空间上的深度和体积感。
    * 这种从多视角视频合成3D视频的技术在电影制作、虚拟现实、远程会议以及游戏开发等多个领域都有广泛的应用潜力。

3. [Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes](http://www.cs.cornell.edu/~zl548/NSFF/) | CVPR2021_440 | [github](https://github.com/zhengqili/Neural-Scene-Flow-Fields)  
    * NSFF引入了场景流场的概念，这是一种用于描述场景中每个点随时间变化的3D向量场。这使得模型能够捕捉和理解场景中物体的运动和变化。NSFF特别关注在不同时间点生成视图的一致性，确保随时间变化时视觉效果的平滑过渡，这在处理视频和动态内容时尤为重要。
    * NSFF展示了在各种复杂动态场景下的强大性能，包括人物运动、自然环境变化等，表明其具有广泛的应用潜力。

4. [Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans](https://zju3dv.github.io/neuralbody/) | CVPR2021_415 | [github](https://github.com/zju3dv/neuralbody)  
    * Neural Body使用隐式神经网络来表示动态人体。这种表示方法不依赖于传统的3D网格或点云模型，而是通过连续的函数来描述人体的3D结构。论文引入了结构化潜码的概念，这些潜码能够捕捉动态人体的不同部位的动态变化。通过这种方式，模型能够更好地理解和重现人体动态。




#### 3.1.5 多分辨率表示

1. [Variable Bitrate Neural Fields](https://nv-tlabs.github.io/vqad/) | SIGGRAPH2022_50 | [github](https://github.com/nv-tlabs/vqad)  
    **Vector-Quantized Auto-Decoder (向量量化自解码器VQ-AD)** 直接学习没有直接监督的信号的压缩特征网格。这种方法支持数据的渐进式、可变比特率流，允许根据可用带宽或所需的细节级别调整质量。  
    VQ-AD使用自解码器框架，专注于学习**压缩**表示。这包括用学习到的代码本中的索引替换庞大的特征向量，这些索引可以小到4位，而以前的方法中特征向量消耗512位。  
    不使用单一分辨率特征网格，而是支持不同细节级别的流。这种方法使得可以同时训练多个细节级别，这对于需要同时过滤和压缩的实时图形系统非常重要​。  
    压缩特征网格的字典方法，将其内存消耗降低了高达100倍。
 







### 3.2 Application
*扩展NeRF的应用*

#### 3.2.1 机器人

1. [3D Neural Scene Representations for Visuomotor Control](https://3d-representation-learning.github.io/nerf-dy/) | CoRL2021_80   
    * 这项研究集中于将3D神经场景表示应用于机器人和自动控制系统，以提高它们对环境的理解和交互能力。


## Review

1. [2020-2021](https://dellaert.github.io/NeRF/)
2. [ICCV 2021](https://dellaert.github.io/NeRF21/)
3. [CVPR 2022](https://dellaert.github.io/NeRF22/)
4. [ECCV 2022](https://markboss.me/post/nerf_at_eccv22/)
5. [CVPR 2023](https://markboss.me/post/nerf_at_cvpr23/)






