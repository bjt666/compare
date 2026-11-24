# GF 16 m 马尾藻分割对比实验

五个模型独立训练。共用 TIFF 数据读取、训练入口和评估流程。先在 AutoDL 的 Python 3.10 环境中运行 `setup.sh`。脚本创建项目内的 `.venv`，优先复用实例已有的 PyTorch 2.1.2、torchvision 0.16.2、CUDA 11.8 运行库。一次安装可用于五个模型。

## 准备数据

`--data-root` 指向含有 `images/` 和 `masks/` 的目录。可以直接使用已有的 `/root/segformer/dataset`。

```text
dataset/
  images/{train,val,test}/  四通道 TIFF，波段顺序 B/G/R/NIR
  masks/{train,val,test}/   与影像同名，单通道 TIFF，背景 0，马尾藻 1 或 255
```

影像按 NIR/R/G 读取，以训练集均值和标准差归一化。默认将 256×256 样本双线性插到 512×512。训练和计分对齐到原始标注网格。

## 安装与检查

```bash
cd /root/gf16m_benchmark
nvidia-smi
bash setup.sh all
bash run.sh all /root/segformer/dataset 42 --check
```

`--check` 用真实数据试一次优化器更新，加一次验证样本前向，并报告显存峰值。检查过程不会生成实验输出。如果只准备一个模型，可以把 `all` 换成 `unet`、`deeplabv3plus`、`unetformer`、`pugnet` 或 `mseonet`。

首次安装需要访问 PyPI、PyTorch 权重服务器、GitHub 和 OpenMMLab。PUGNet、MSEONet 的官方源码由准备脚本按固定提交获取，预训练权重也缓存到本地。正式训练只读取本地文件。如果 AutoDL 上下载失败，可在能联网的、已安装项目依赖的机器运行 `python prepare_assets.py all`，再把生成的 `weights/`、`vendor/pugnet/` 和 `vendor/mseonet/` 上传到服务器项目对应位置。服务器已有这些文件时，可用 `SKIP_ASSETS=1 bash setup.sh all` 只安装依赖。

## 正式训练

```bash
cd /root/gf16m_benchmark
bash run.sh unet          /root/segformer/dataset 42
bash run.sh deeplabv3plus /root/segformer/dataset 42
bash run.sh unetformer    /root/segformer/dataset 42
bash run.sh pugnet        /root/segformer/dataset 42
bash run.sh mseonet       /root/segformer/dataset 42
```

也可用 `bash run.sh all /root/segformer/dataset 42`，按上述顺序单卡串行运行。`0` 或 `1` 可替换 `42`，结果会分别保存在不同文件夹。每组默认 100 轮，有效 batch 8。默认物理 batch 分别为 `8/4/4/2/2`，通过梯度累积对齐更新预算。遇到显存不足，先用较小物理 batch 做 `--check`。例如：

```bash
bash run.sh deeplabv3plus /root/segformer/dataset 42 --batch-size 2 --check
bash run.sh mseonet /root/segformer/dataset 42 --amp --check
```

检查通过后，用相同参数去掉 `--check` 即可训练。`--amp` 启用 FP16 混合精度训练，验证和测试统一用 FP32。更改物理 batch 或训练精度时要在实验表中记录。PUGNet 在批内计算对比损失，物理 batch 至少为 2。显存允许时用更大的物理 batch 可以增加实际参与对比的样本数。

## 训练口径与模型实现

| 模型 | 骨干和实现 | 训练损失 |
|---|---|---|
| `unet` | ImageNet-1K ResNet-18，U-Net 跳连和卷积解码 | CE + 0.1 × 前景 Lovász |
| `deeplabv3plus` | ImageNet-1K ResNet-50，OS16，ASPP 加浅层特征解码 | CE + 0.1 × 前景 Lovász |
| `unetformer` | GeoSeg 解码器，监督式 ImageNet-1K ResNet-18 | 原生 SoftCE + Dice + 0.4 × 辅助 SoftCE |
| `pugnet` | 官方单时相模型，ImageNet-1K PVT-v2-B4 | 四级 CE 求和 + 原生批内对比损失 |
| `mseonet` | 官方 MSCA、EPFO 和 OS8 ResNet-101 V1c，PyTorch 适配 | MSCA 像元 CE + EPFO 采样点 CE |

输入首层的 ImageNet 权重按 NIR/R/G 适配，NIR 通道用 RGB 卷积权重均值初始化。采用 AdamW、余弦衰减，骨干初始学习率 `6e-5`，其余 `6e-4`，权重衰减 `0.01`。训练保存验证集马尾藻 IoU 最高的权重，结束后自动评估验证集和测试集。不同模型的物理 batch 与原生损失均记录在 `config.json` 中。

MSEONet 采用原作者的 MSCA、EPFO 层和原配 ResNet-101 V1c 权重。适配器用 PyTorch 实现两阶段的采点、监督和推理回填，并修正官方源码的推理调用参数不匹配及边缘候选不足时的回退问题。论文实验设置中可将这组标为 **MSEONet–R101（GF 二分类适配）**。对比结果使用本项目的共同训练预算和测试口径。

## 结果位置与续训

每组输出到 `outputs/<模型名>_s<种子>/`：

- `evaluation/test/metrics.json` 保存测试集 IoU、F1、精确率、召回率和 mIoU。
- `evaluation/test/per_image_metrics.csv` 保存逐图指标。
- `metrics.csv` 保存训练损失、每轮验证指标和实际更新数。
- `config.json`、`dataset_manifest.json` 和 `model_summary.json` 保存训练设置、数据指纹与参数量。
- `checkpoints/best.pt` 保存验证最优权重，`checkpoints/last.pt` 保存最近一个完成轮次的续训状态。

中断后使用原运行参数加 `--resume`。以下示例适用于已有至少一个完整轮次的 U-Net：

```bash
bash run.sh unet /root/segformer/dataset 42 --resume
```

如果 100 轮尚未完成，续训使用原计划的 100 轮。需要改变总轮数时，以新输出目录另起实验。例如 `--epochs 150 --output-dir outputs/unet_150_s42`。

## 上游实现

- UNetFormer 使用 GeoSeg 的固定版本解码器，源码来源与许可见 `vendor/geoseg/SOURCE.md` 和 `vendor/geoseg/LICENSE`。[GeoSeg](https://github.com/WangLibo1995/GeoSeg)
- PUGNet 使用官方单时相 PUGNet，源码版本与校验值见 `bench_models/pugnet.py`。[PUGNet](https://github.com/Henryjiepanli/PU_RS)
- MSEONet 的原始头部版本与校验值见 `bench_models/mseonet.py`。[MSEONet](https://github.com/HuangWBill/MSEONet)
