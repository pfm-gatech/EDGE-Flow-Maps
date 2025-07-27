# EDGE: Epsilon-Difference Gradient Evolution for Buffer-Free Flow Maps

[[project page]](https://pearseven.github.io/EDGEProject/)

## Overview

This repo provides implementation of several advection methods mentioned in the paper.

- `run_efm.py`: Eulerian Flow Maps (EFM)
- `run_ed8.py`: 8-Point Epsilon Differnece Method
- `run_ed4.py`: 4-Point Epsilon Differnece Method.
- `run_ge.py`: Gradient Evolution Method, inspired by Gradient-Augmented Reference Maps (referred to as `garm` in the code)
- `run_edge.py`: Epsilon-Difference Gradient Evolution Method (EDGE)

## Installation

The code has been tested on Windows 11 with CUDA 12.8, Python 3.9.23, and Taichi 1.7.3. Other dependencies are listed in `requirement.txt`.

To set up the environment, run:

```powershell
conda create -n edge python=3.9
conda activate edge

pip install taichi
pip install matplotlib
pip install vtk
pip install pyevtk
pip install trimesh
pip install scipy
```

## Running Examples

```powershell
python run_efm.py [-c 0]
```

Available cases:

- case 0: Leapfrogging Vortices
- case 1: Four Vortices Collision
- case 2: Delta Wing

Results would be saved in the `logs\` directory. We use [ParaView](https://www.paraview.org/) for data visualization.

## Citation

If you find our work useful in your research, please consider citing:

```
@article{li2025edge,
title = {EDGE: Epsilon-Difference Gradient Evolution for Buffer-Free Flow Maps},
author = {Li, Zhiqi and Wang, Ruicheng and Li, Junlin and Chen, Duowen and Wang, Sinan and Zhu, Bo},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {44},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3731193},
doi = {10.1145/3731193},
journal = {ACM Trans. Graph.},
articleno = {96},
numpages = {11}
}
```