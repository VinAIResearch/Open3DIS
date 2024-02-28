# Run the code

1\) Extract class-agnostic 3D proposals and 3D feature map from ISBNet:

```
cd ISBNet/
python3 tools/test.py configs/scannet200/isbnet_scannet200.yaml pretrains/scannet200/head_scannetv2_200_val.pth
```

2\) Extract 2D masks from RGB-D sequences:

```
sh scripts/grounding_2d.sh
```

3\) Generate open-vocab 3D instances:

```
sh scripts/generate_3d_inst.sh
```

4\) Run interactive visualization (required Open3D):

```
cd visualization
python3 vis_gui.py
```