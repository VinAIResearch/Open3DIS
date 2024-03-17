# Data Preparation

## Scannet200
1\) Create data folder:
```python
mkdir -p Dataset/Scannet200
# mkdir -p Dataset/replica
# mkdir -p Dataset/S3DIS
# mkdir -p Dataset/Scannetpp

```
2\) Data tree

For Scannet200, we construct data tree directory as follow and consider only for validation set:

```
Open3DIS <-- (you are here)
Dataset
├── Scannet200
############## 2D root folder with default image sampling factor: 5 ##############
│    ├── Scannet200_2D_5interval 
│    │    ├── val                                       <- validation set
│    │    |    ├──scene0011_00
│    │    |    │   ├──color                             <- folder with image RGB
│    │    │    │   │    00000.jpg
│    │    │    │   │    ...
│    │    |    │   ├──depth                             <- folder with image depth
│    │    │    │   │    00000.png
│    │    │    │   │    ...
│    │    |    │   ├──pose                              <- folder with camera poses
│    │    │    │   │    00000.txt
│    │    │    │   │    ...
│    │    |    |   instrinsic.txt (image intrinsic)
│    │    |    ....
│    │    |    intrinsic_depth.txt (depth intrinsic)    <- Scannet intrinsic ~ depth img
│    │    ├── train
│    │    ├── test 
############## 3D root folder with point cloud and annotation ##############
|    ├── Scannet200_3D
│    │    ├── val                                       <- validation set
│    │    │    ├── original_ply_files                   <- the _vh_clean_2.ply point cloud file from Scannet raw data.
│    │    │    │     scene0011_00_vh_clean_2.ply
│    │    │    │     ...
|    │    │    ├── groundtruth                          <- normalized point cloud, color from PLY + ann (for 3D backbone)
|    │    │    │     scene0011_00_inst_nostuff.pth           
|    │    │    │     ...
|    │    │    ├── superpoints                          <- superpoints directory
|    │    │    │     scene0011_00.pth
|    │    │    │     ...
|    │    │    ├── isbnet_clsagnostic_scannet200        <- class agnostic 3D proposals
|    │    │    │     scene0011_00.pth
|    │    │    │     ...
|    │    │    ├── dc_feat_scannet200                   <- 3D deep feature of 3D proposals network
|    │    │    │    scene0011_00.pth
|    │    │    │    ...
│    │    ├── train
│    │    ├── test 
####################################################################################
pretrains
├── foundation_models
|   ├── groundingdino_swint_ogc.pth
|   ├── sam_vit_h_4b8939.pth
```

3\) Generating RGB-D images, camera poses, original PLY, superpoints and inst_nostuff files

* Download the [ScannetV2 dataset](http://www.scan-net.org/)

* Please refer to [RGB-D images, camera poses and original PLY](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python)

* Please refer to [Superpoints and inst_nostuff](https://github.com/VinAIResearch/ISBNet/tree/master/dataset/scannet200)

4\) Generating class-agnostic 3D proposals and 3D deep features

* ISBNet

```python
# Download the pretrained ISBNet on Scannet200 or V2 to acquire deep level feature
cd segmenter3d/ISBNet/
python3 tools/test.py configs/scannet200/isbnet_scannet200.yaml pretrains/scannet200/head_scannetv2_200_val.pth
```

* More 3DIS backbone will be updated...