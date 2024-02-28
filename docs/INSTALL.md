# Installation Guide

1\) Environment requirements

The following installation guide supposes ``python=3.8``, ``pytorch=1.12.1``, ``cuda=11.3``, ``torch-scatter-2.1.0+pt112cu113``, and ``spconv-cu113==2.1.25``. You may change them according to your system.

Create conda and install core packages:
```
conda create -n Open3DIS python=3.8
conda activate Open3DIS
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip3 install spconv-cu113==2.1.25
```

Install torch_scatter: the version must match your ``python`` version. It is recommended to visit [torch_scatter](https://data.pyg.org/whl/torch-1.12.1+cu113.html) and download the corresponding version.
In our case, we use:
```
pip install torch_scatter-2.1.0+pt112cu113-cp38-cp38-linux_x86_64.whl
```

General case, might lead to Segmentaion Core Dump:
```
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
```

2\) Install Detectron2:  or you can refer to this link for cloning the [source](https://detectron2.readthedocs.io/en/latest/tutorials/install.html):

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

3\) Install [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), we modified the models to work with ours Open3DIS, so install the repo locally is needed: 
```
cd GroundingDINO/
pip install -e .
cd ../
cd segment_anything
pip install -e .
cd ../
```

4\) Install [ISBNet](https://github.com/VinAIResearch/): 
```
cd ISBNet
pip install -r requirements.txt
# Install build requirement
sudo apt-get install libsparsehash-dev
# Setup PointNet operator for DyCo3D
cd isbnet/pointnet2
python3 setup.py bdist_wheel
cd ./dist
pip3 install <.whl>
cd ../../../
# Setup build environment
python3 setup.py build_ext develop
cd ../
```

5\) Finally, install other dependencies:
```
pip install scikit-image opencv-python open3d imageio plyfile
pip install -r requirements.txt
```

# Download pretrained foundation models

Download the pretrained foundation models:
```
mkdir -p pretrains/foundation_models
cd pretrains/foundation_models

# Grounding DINO and Segment Anything model (Grounded-SAM + SAM)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# DETIC
...
# SEEM
...
```