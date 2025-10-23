# 360-RealityScan-Postshot-Tools

A consolidated toolkit for turning 360° video captures into RealityScan-friendly datasets and optimized point clouds that seed PostShot/3D Gaussian Splatting (3DGS) projects. The scripts cover every stage from frame extraction to PLY refinement, and the desktop GUI orchestrates the full flow.

---

## Installation & Environment Setup

### Requirements
- Python **3.9** or newer
- `pip` for installing Python packages
- [FFmpeg](https://ffmpeg.org/) and `ffprobe` available on your `PATH`
- A GPU is not required, but fast storage/CPU cores benefit multi-threaded exports

### Setup Steps
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/360-RealityScan-Postshot-Tools.git
   cd 360-RealityScan-Postshot-Tools
   ```
2. **(Optional) create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Verify FFmpeg**
   ```bash
   ffmpeg -version
   ffprobe -version
   ```

---

## Workflow at a Glance
1. Sample equirectangular frames from a 360° video.
2. Score and retain the sharpest frames for Structure-from-Motion (SfM).
3. Convert panoramas into perspective/fisheye views.
4. Reconstruct with RealityScan and export the PLY point cloud plus camera data.
5. Downsample/merge the PLY for PostShot initialization and feed both assets into your 3DGS pipeline.

The **rs2ps_360GUI** provides a launch pad for each CLI stage, letting you preview perspective camera layouts and run exports without memorising command-line options.

---

## Toolchain Overview

### rs2ps_360GUI (Desktop Launcher)
Tkinter/Pillow interface that wraps `rs2ps_360PerspCut` and the other scripts. It previews camera rigs, edits presets, triggers FFmpeg jobs, and exposes context help for every option.

Launch the GUI:
```bash
python rs2ps_360GUI.py --input-dir path/to/frames
```
Use the tabs to:
- Preview and tweak perspective camera layouts before exporting.
- Kick off frame extraction, sharp-frame selection, and perspective cuts with your saved presets.
- Run PLY downsampling once RealityScan reconstruction is complete.

### CLI Stage 1: rs2ps_Video2Frames
Sample frames from 360° footage at controllable frame rates, automatically normalising colour space/bit depth and handling time-range trimming.

```bash
python rs2ps_Video2Frames.py \
    --input path/to/video.mp4 \
    --output-dir frames/360_raw \
    --fps 2
```

### CLI Stage 2: rs2ps_FrameSelector
Rank and filter equirectangular stills using hybrid sharpness metrics (Laplacian, Tenengrad, FFT) with segment-based quotas, brightness/motion guards, and optional CSV round-tripping.

```bash
python rs2ps_FrameSelector.py \
    --input-dir frames/360_raw \
    --output-dir frames/360_selected \
    --segment-size 50 --select-per-segment 5
```

### CLI Stage 3: rs2ps_360PerspCut
Convert panoramas or selected frames into perspective/fisheye outputs using presets or ad-hoc camera definitions, with multi-process FFmpeg execution and graceful cancellation.

```bash
python rs2ps_360PerspCut.py \
    --input-dir frames/360_selected \
    --output-dir frames/perspective \
    --preset default --size 2048 --jobs auto
```

### CLI Stage 4: rs2ps_PlyOptimizer
Prepare RealityScan PLY point clouds for PostShot/3DGS by reporting statistics, voxel-downsampling to a target point count, optionally merging additional clouds, and exporting binary little-endian files.

```bash
python rs2ps_PlyOptimizer.py \
    --input realityscan_output.ply \
    --output optimized.ply \
    --target-points 500000
```

Feed the optimised PLY and RealityScan camera bundle into PostShot to finish the 3DGS pipeline.

---

## License

(To be defined according to the project.)
