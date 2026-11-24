# GeoSeg UNetFormer decoder

`UNetFormer.py` is the unchanged source file from
[`WangLibo1995/GeoSeg`, commit `9453fe48209c4626b29e35e61bab93b61212c4b1`](https://github.com/WangLibo1995/GeoSeg/blob/9453fe48209c4626b29e35e61bab93b61212c4b1/geoseg/models/UNetFormer.py).
It retains the upstream GPL-3.0 license, included in `LICENSE`.

The benchmark uses the original `Decoder` with a torchvision ResNet-18
initialized from supervised ImageNet-1K weights, two output categories,
and GeoSeg's smoothed CE, Dice, and auxiliary CE terms.
