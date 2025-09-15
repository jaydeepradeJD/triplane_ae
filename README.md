# Tri-plane Autoencoder

A PyTorch Lightning implementation of a tri-plane autoencoder for 3D point cloud reconstruction.

## Overview

This project implements a neural autoencoder that uses tri-plane representations to encode and decode 3D point clouds. The tri-plane representation uses three orthogonal 2D feature planes (XY, YZ, XZ) to represent 3D geometry, which is more memory efficient than traditional 3D voxel grids while maintaining high quality reconstructions.

## Dataset
Using ShapeNetCore Dataset [HF Dataset link](https://huggingface.co/datasets/ShapeNet/ShapeNetCore).
The dataset was processed to generate 3D point clouds for each object and occupancy for random 1M points on and off the surface as discussed in [NFD Paper](https://arxiv.org/pdf/2311.09217).
The processed dataset contains `.npy` files where each file contains:
- Shape: `(N, 4)` where N is the number of points
- Columns: `[x, y, z, occupancy]`
  - `x, y, z`: 3D coordinates
  - `occupancy`: Binary occupancy value (0 or 1)

Currently it only has one object class: `aeroplane`.

## TODOs:
- [ ] Add more object classes
- [ ] Add triplane visualization

## Acknowledgments

- Code Reference: https://github.com/JRyanShue/NFD/tree/main
