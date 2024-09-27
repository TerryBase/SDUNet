# SDUNet
This is a official implementation of SDUNet: A Spatial Deformable Kernel-based U-Net

## Introduction
Spatial Deformable U-Net (SDUNet) is proposed based on U-Net, which serve as a backbone of semantic segmentaion for computer vision in this paper. The effectiveness and limitation of offsets is mainly representated with Spatial Deformable Convolution.
Spatial Deformable Convolution (Spatial DConv) is proposed with an activate function layer and linear cumulative additional layer associated to Deformable Convolution, which prevent the offsets out of focus as well as enrich more spatial connection 
![image](https://github.com/TerryBase/SDUNet/blob/main/figures/deformable_convolution.png)
