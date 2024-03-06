# Data Preparation

## Scannet200
1\) Create data folder:
```python
mkdir -p data/Scannet200
# mkdir -p data/replica
# mkdir -p data/S3DIS
# mkdir -p data/Scannetpp

```
2\) Data tree
```
Open3DIS <-- (you are here)
Dataset
├── ScannetV2
│   ├── ScannetV2_2D_5interval
│   │    ├── train
│   │    ├── val
│   │    |    scene0011_00
│   │    |    ├──color
│   │    |    ├──depth
│   │    |    ├──pose
│   │    |    |  instrinsic.txt (image intrinsic)
│   │    |    .... scenes
│   │    | intrinsic_depth.txt (depth intrinsic)
│   │    ├── original_ply_files # the _vh_clean_2.ply file from Scannet raw data.
├── Scannet200
|   ├── train 
│   ├── val
│   ├── test
│   ├── superpoints
│   ├── isbnet_clsagnostic_scannet200 # class agnostic 3D proposals
│   ├── dc_feat_scannet200 # 3D deep feature map from backbone of 3D proposals network
├── Replica, S3DIS, Scannetpp,...
pretrains
├── foundation_models
|   ├── groundingdino_swint_ogc.pth
|   ├── sam_vit_h_4b8939.pth
```

3\) Generate Superpoint

...updating

4\) Superpoint Deep 3D Features

...updating