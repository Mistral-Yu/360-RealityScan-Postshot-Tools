# 360-RealityScan-Postshot-Tools

A consolidated toolkit for turning 360° video captures into RealityScan-friendly datasets and optimized point clouds that seed PostShot/3D Gaussian Splatting (3DGS) projects. The scripts cover every stage from frame extraction to PLY refinement, and the desktop GUI orchestrates the full flow.

> **Project status:** actively edited and expanded. Expect interface polish, richer presets, and additional documentation updates in upcoming revisions.
> The code in this repository was generated with the assistance of OpenAI Codex.

---

## Installation & Environment Setup

### Requirements
- Python **3.7** or newer
- `pip` for installing Python packages
- [FFmpeg](https://ffmpeg.org/)  available on your `PATH`
- [PyTorch](https://pytorch.org/get-started/locally/) (`torch` **1.10+** and matching `torchvision` build) for the human masking tool
- A GPU is not required, but fast storage/CPU cores benefit multi-threaded exports. CUDA acceleration is optional for masking but speeds up large batches.

### Setup Steps
1. **Clone the repository**
   ```bash

   git clone https://github.com/Mistral-Yu/360-RealityScan-Postshot-Tools.git
   cd 360-RealityScan-Postshot-Tools
   ```
2. **Create and activate a Conda environment**
   ```bash
   conda create -n rs2ps python=3.7
   conda activate rs2ps
   ```
   > **Tip:** pick any supported Python version (3.7+) that matches your GPU drivers/CUDA toolkits.
3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:** PyTorch wheels are platform-specific. If `pip` cannot find a build for your OS/Python combo, follow the [official selector](https://pytorch.org/get-started/locally/) and install `torch`/`torchvision` manually inside the Conda environment before rerunning the command.
4. **Verify FFmpeg**
   ```bash
   ffmpeg -version
   ```

---

## Workflow at a Glance
1. Sample equirectangular frames from a 360° video.
2. Score and retain the sharpest frames for Structure-from-Motion (SfM).
3. Convert panoramas into perspective views.
4. (Optional) Remove or isolate bystanders with `rs2ps_HumanMaskTool` to avoid reconstruction ghosts.
5. Align in RealityScan and export the PLY point cloud and camera CSV data.
6. (Optional) Downsample(and merge) the PLY for PostShot initialization and feed both assets into your 3DGS pipeline.

### Recommended RealityScan-to-PostShot Flow
- After RealityScan completes **Reconstruction** and **Colorize**, export the colourised mesh as a PLY and grab the accompanying camera **CSV** ; that pairing feeds best into `rs2ps_PlyOptimizer`.
- Start with a conservative `--target-points` value around **100,000** to keep PostShot responsive; you can rerun the optimizer with higher counts if needed.

The **rs2ps_360GUI** provides a launch pad for each CLI stage, letting you preview perspective camera layouts and run exports without memorising command-line options.

---

## Toolchain Overview

- **rs2ps_360GUI (Desktop Launcher)** — Tkinter/Pillow interface that wraps the CLI scripts. Preview camera rigs, edit presets, trigger FFmpeg jobs, and surface inline help for every option.
  - Launch with `python rs2ps_360GUI.py` and use the tabs to configure exports, kick off batch operations, and hand RealityScan outputs to the optimiser.
- **rs2ps_Video2Frames** — Samples frames from 360° footage at controllable rates, normalises colour space/bit depth, and trims time ranges.
  - Example: `python rs2ps_Video2Frames.py --input path/to/video.mp4 --output-dir frames/360_raw --fps 2`
- **rs2ps_FrameSelector** — Ranks equirectangular stills with hybrid sharpness metrics, segment quotas, and CSV round-tripping to curate SfM-ready sets. After filtering, a dedicated button reveals frames that may still contain blur so you can manually audit borderline candidates.
  - Example: `python rs2ps_FrameSelector.py --input-dir frames/360_raw --output-dir frames/360_selected --segment-size 50 --select-per-segment 5`
- **rs2ps_360PerspCut** — Converts panoramas, selected frames, or a 360° video file into perspective/fisheye outputs via presets or ad-hoc camera definitions. Supports multi-process FFmpeg execution, `--fps` for video sampling, optional `--keep-rec709` colour handling, and emits **focal-length logs you can copy into RealityScan or Metashape to minimise alignment errors**. The default cut preset is the safest choice when you want the most alignment-stable export, while the `fisheyelike` cut yields the highest alignment fidelity and best 3DGS quality when captured carefully, at the cost of being more sensitive to stitching artefacts.
  - Example: `python rs2ps_360PerspCut.py --input-dir frames/360_selected --output-dir frames/perspective --preset default --size 1600 --jobs auto`
- **rs2ps_PlyOptimizer** — Prepares RealityScan PLY point clouds (exported after the **Alignment/Reconstruction → Colorize → PLY** ) for PostShot/3DGS by reporting statistics, voxel-downsampling to a target point count, optionally merging additional clouds, and exporting binary little-endian files. In addition to the voxel strategy, an adaptive octree-based downsampler is available for evaluation when you need detail that follows geometry more closely before pushing splats into 3DGS.
  - Example: `python rs2ps_PlyOptimizer.py --input realityscan_output.ply --output optimized.ply --target-points 100000`
- **rs2ps_HumanMaskTool** — Batch-detects people with Mask R-CNN to generate binary mattes, RGBA cut-outs, or inpainted plates before photogrammetry. Supports optional CUDA inference and shadow expansion to keep reflected silhouettes intact.
  - Example: `python rs2ps_HumanMaskTool.py --in frames/360_selected --mode mask --include_shadow`

Feed the optimised PLY and RealityScan camera CSV into PostShot to finish the 3DGS pipeline.


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
