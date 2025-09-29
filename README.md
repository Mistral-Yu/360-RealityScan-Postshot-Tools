# 360-RealityScan-Postshot-Tools

A toolkit for processing and optimizing **360 Camera 3DGS Scans** aligned with RealityScan, preparing them for **PostShot/3DGS** workflows.

---

## Overview

RealityScan outputs raw PLY point clouds and 360° images/videos which often need cleaning, optimization, and alignment before they can be effectively used in PostShot or other 3D Gaussian Splatting pipelines. The tools in this repo cover the typical preprocessing steps for **360° camera-based scanning**:

1. **Point cloud optimization** (downsampling, merging) — initialization point clouds for 3DGS, exported from RealityScan
2. **360° video → image frame extraction**
3. **360° image sequence → image selection for SfM/3DGS**
4. **360° image → perspective cutout generation**

Together, they allow for a smoother end-to-end pipeline.

---

## Tools

* **rs2ps_PlyOptimizer**: Downsampling, merging, and optimizing PLY point clouds
* **rs2ps_Video2Frames**: Extracting 360° images from 360° videos
* **rs2ps_FrameSelector**: Selecting suitable frames from 360° image sequences for SfM/3DGS
* **rs2ps_360PerspCut: Cutting perspective images from 360° equirectangular images

---

## rs2ps_PlyOptimizer

A small utility to **optimize PLY point clouds from RealityScan for PostShot/3DGS**. Supports XYZ/RGB loading, **voxel downsampling (fixed size / target points)**, **appending multiple PLY files**, and **binary (little-endian) PLY export**.

### Features

* **Robust PLY (XYZ+RGB) loading**
* **Statistics reporting** (point count, AABB, volume, etc.)
* **Downsampling** (fixed voxel size / target points / percent ratio)
* **Appending multiple PLY files**
* **Binary little-endian output** (fast & compact)

### Requirements

* Python **3.7**
* `numpy`, `plyfile`

### Usage (CLI)

```bash
rs2ps_PlyOptimizer -i input.ply -o output.ply --target-points 500000
```

#### Main Options

| Option             | Description                                           |
| ------------------ | ----------------------------------------------------- |
| `--voxel-size`     | Fixed voxel size                                      |
| `--target-points`  | Target number of points                               |
| `--target-percent` | Target points as percentage of input                  |
| `--keep-strategy`  | Representative point: `centroid` / `center` / `first` |
| `--append-ply`     | Append other PLY files                                |

---

## License

* (To be defined according to the project)
