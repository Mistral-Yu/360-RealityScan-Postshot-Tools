# 360-RealityScan-Postshot-Tools

A consolidated toolkit for turning 360° video captures into RealityScan-friendly datasets and optimized point clouds that seed PostShot/3D Gaussian Splatting (3DGS) projects. The scripts cover every stage from frame extraction to PLY refinement, and the desktop GUI orchestrates the full flow.

> **Project status:** actively edited and expanded. Expect interface polish, richer presets, and additional documentation updates in upcoming revisions.

---

## Installation & Environment Setup

### Requirements
- Python **3.7** or newer
- `pip` for installing Python packages
- [FFmpeg](https://ffmpeg.org/) and `ffprobe` available on your `PATH`
- [PyTorch](https://pytorch.org/get-started/locally/) (`torch` **1.10+** and matching `torchvision` build) for the human masking tool
- A GPU is not required, but fast storage/CPU cores benefit multi-threaded exports. CUDA acceleration is optional for masking but speeds up large batches.

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
   > **Note:** PyTorch wheels are platform-specific. If `pip` cannot find a build for your OS/Python combo, follow the [official selector](https://pytorch.org/get-started/locally/) and install `torch`/`torchvision` manually before rerunning the command.
4. **Verify FFmpeg**
   ```bash
   ffmpeg -version
   ffprobe -version
   ```

---

## Workflow at a Glance
1. Sample equirectangular frames from a 360° video.
2. Score and retain the sharpest frames for Structure-from-Motion (SfM).
3. *(Optional)* Remove or isolate bystanders with `rs2ps_HumanMaskTool` to avoid reconstruction ghosts.
4. Convert panoramas into perspective/fisheye views.
5. Align in RealityScan and export the PLY point cloud plus camera data.
6. Downsample/merge the PLY for PostShot initialization and feed both assets into your 3DGS pipeline.

### Recommended RealityScan-to-PostShot Flow
- After RealityScan completes **Reconstruction** and **Colorize**, export the colourised mesh as a PLY and grab the accompanying camera **CSV** (not the Bundler bundle); that pairing feeds best into `rs2ps_PlyOptimizer`.
- Start with a conservative `--target-points` value around **100,000** to keep PostShot responsive; you can rerun the optimizer with higher counts if needed.

The **rs2ps_360GUI** provides a launch pad for each CLI stage, letting you preview perspective camera layouts and run exports without memorising command-line options.

---

## Toolchain Overview

- **rs2ps_360GUI (Desktop Launcher)** — Tkinter/Pillow interface that wraps the CLI scripts. Preview camera rigs, edit presets, trigger FFmpeg jobs, and surface inline help for every option.
  - Launch with `python rs2ps_360GUI.py` and use the tabs to configure exports, kick off batch operations, and hand RealityScan outputs to the optimiser.
- **rs2ps_Video2Frames** — Samples frames from 360° footage at controllable rates, normalises colour space/bit depth, and trims time ranges.
  - Example: `python rs2ps_Video2Frames.py --input path/to/video.mp4 --output-dir frames/360_raw --fps 2`
- **rs2ps_FrameSelector** — Ranks equirectangular stills with hybrid sharpness metrics, segment quotas, and CSV round-tripping to curate SfM-ready sets.
  - Example: `python rs2ps_FrameSelector.py --input-dir frames/360_raw --output-dir frames/360_selected --segment-size 50 --select-per-segment 5`
- **rs2ps_360PerspCut** — Converts panoramas, selected frames, **or a 360° video file** into perspective/fisheye outputs via presets or ad-hoc camera definitions. Supports multi-process FFmpeg execution, `--fps` for video sampling, optional `--keep-rec709` colour handling, and emits focal-length logs you can copy into RealityScan or Metashape to minimise alignment errors.
  - Example: `python rs2ps_360PerspCut.py --input-dir frames/360_selected --output-dir frames/perspective --preset default --size 2048 --jobs auto`
- **rs2ps_PlyOptimizer** — Prepares RealityScan PLY point clouds (exported after the **Reconstruction → Colorize → PLY** sequence) for PostShot/3DGS by reporting statistics, voxel-downsampling to a target point count, optionally merging additional clouds, and exporting binary little-endian files. Its voxel-based strategy is ideal when you need a uniform detail distribution before pushing splats into 3DGS.
  - Example: `python rs2ps_PlyOptimizer.py --input realityscan_output.ply --output optimized.ply --target-points 100000`
- **rs2ps_HumanMaskTool** — Batch-detects people with Mask R-CNN to generate binary mattes, RGBA cut-outs, or inpainted plates before photogrammetry. Supports optional CUDA inference and shadow expansion to keep reflected silhouettes intact.
  - Example: `python rs2ps_HumanMaskTool.py --in frames/360_selected --mode alpha --include_shadow`

Feed the optimised PLY and RealityScan camera CSV into PostShot to finish the 3DGS pipeline.

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

### Optional Masking: rs2ps_HumanMaskTool
Remove or isolate people before reconstruction by generating mattes, alpha cut-outs, or inpainted plates. The script scans a directory (non-recursive) with Mask R-CNN and writes results to an `_mask/` folder unless `--out` is provided.

```bash
python rs2ps_HumanMaskTool.py \
    --in frames/360_selected \
    --mode keep_person \
    --include_shadow
```

Tips:
- Switch `--mode` between `mask`, `alpha`, `cutout`, `keep_person`, `remove_person`, and `inpaint` depending on the downstream need.
- Add `--cpu` if CUDA is unavailable; GPU inference is automatically used when possible.
- Use `--dilate 1.5` (percent) to slightly grow masks and avoid halos, or pair with `--include_shadow` to capture nearby shadows/reflections.

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

This project is released under the [MIT License](LICENSE). Copyright (c) 2025 Yu.

---

## TODO / Roadmap
- Flesh out a full GUI walkthrough (tab descriptions, launch parameters, screenshot gallery).
- Document advanced CLI recipes for batch jobs and automation.
- Publish sample datasets and recommended RealityScan export settings.
- Finalize licensing and contribution guidelines.
(To be defined according to the project.)
