# Spatial Frequency Modulation Network for Efficient Image Dehazing

[Hao Shen](https://www.haoshen.site/), [Henghui Ding](https://henghuiding.github.io/), [Yulun Zhang](https://yulunzhang.com/), [Zhong-Qiu Zhao](http://faculty.hfut.edu.cn/zzq123456/zh_CN/index.htm), [Xudong Jiang]([Xudong Jiang](https://personal.ntu.edu.sg/exdjiang/default.htm))

Spatial Frequency Modulation Network for Efficient Image Dehazing, IEEE TIP, 2025. 

<hr />

> **Abstract:** Currently, two main research lines in efficient context modeling for image dehazing are tailoring effective feature modulation mechanisms and utilizing the Fourier transform more precisely. The former is usually based on self-scale features that ignore complementary cross-scale/level features, and the latter tends to overlook regions with pronounced haze degradation and intricate structures. This paper introduces a novel spatial and frequency modulation perspective to synergistically investigate contextual feature modeling for efficient image dehazing. Specifically, we delicately develop a Spatial Frequency Modulator (SFM) equipped with a Cross-Scale Modulator (CSM) and Frequency Modulator (FM) to implement intra-block feature modulation. The CSM progressively aggregates hierarchical features across different scales, employing them for spatial self-modulation, and the FM subsequently adopts a dual-branch design to focus more on the crucial areas with severe haze and complex structures for reconstruction. Further, we propose a Cross-Level Modulator (CLM) to facilitate inter-block feature mutual modulation, enhancing seamless interaction between features at different depths and layers. Integrating the above-developed modules into the U-Net architecture, we construct a two-stage spatial frequency modulation network (SFMN). Extensive quantitative and qualitative evaluations showcase the superior performance and efficiency of the proposed SFMN over recent state-of-the-art image dehazing methods.

## Training
## Evaluation
### Testing on ITS dataset

```shell
python evaluate_its.py
```

### Testing on OTS dataset

```shell
python evaluate_ots.py
```

### Testing on Dense-Haze dataset

```shell
python evaluate_dense.py
```

### Testing on NH-HAZE dataset

```powershell
python evaluate_nh.py
```

### Testing on O-HAZE dataset

```shell
python evaluate_ohaze.py
```

