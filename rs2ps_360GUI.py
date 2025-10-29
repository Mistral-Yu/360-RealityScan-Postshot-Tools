#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI preview and execution tool for rs2ps_360PerspCut."""

import argparse
import csv
import copy
import itertools
import math
import pathlib
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover - environment guard
    print("[ERR] Pillow (PIL) is required: pip install Pillow", file=sys.stderr)
    raise SystemExit(1) from exc

import rs2ps_360PerspCut as cutter

COLOR_CYCLE = [
    "#ff6b6b",
    "#4ecdc4",
    "#ffd166",
    "#5a189a",
    "#118ab2",
    "#f9844a",
    "#06d6a0",
    "#ef476f",
    "#073b4c",
    "#9b5de5",
]

PRESET_CHOICES = ["default", "2views", "fisheyelike", "evenMinus30", "evenPlus30", "fisheyeXY"]

HUMAN_MODE_CHOICES = ["mask", "alpha", "cutout", "keep_person", "remove_person", "inpaint"]



class ToolTip:
    """Simple tooltip that appears near widgets on hover."""

    def __init__(self, widget: tk.Widget, text: str, delay: int = 400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _event=None) -> None:
        self._schedule()

    def _schedule(self) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, _event=None) -> None:
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self.tipwindow or not self.text:
            return
        x, y = self.widget.winfo_pointerxy()
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x + 12}+{y + 12}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("TkDefaultFont", 9),
            wraplength=320,
        )
        label.pack(ipadx=6, ipady=4)

    def _hide(self) -> None:
        if self.tipwindow is not None:
            try:
                self.tipwindow.destroy()
            except Exception:
                pass
            self.tipwindow = None

HELP_TEXT = """
- preset: Output preset (default / fisheyelike / 2views / evenMinus30 / evenPlus30 / fisheyeXY)
- count: Horizontal divisions (360 deg / count gives yaw step)
- addcam: Extra cameras, e.g. B, D:U15, H:D20
- addcam delta (deg): Default delta when U/D omit a value
- delcam: Remove baseline cameras, e.g. B,D
- setcam: Override/adjust cameras, e.g. A=U30, B:-5, A_U=10 (upper), A_D:-5 (lower)
- Size(px): Output square size per view
- HFOV (deg): Horizontal FOV (overrides focal length)
- Focal(mm): Focal length when HFOV is not set
- Sensor(mm): Sensor width/height (e.g. 36 24 or 36x24)
- Add top/bottom: Include cube-map style top/bottom views
- Video input: Use Browse... to select a video file (mp4/mov/etc.); choose preview FPS and ensure ffmpeg path is configured.

Update: Refresh preview only with the current parameters
Run Export: Execute FFmpeg jobs and write outputs (with confirmation)
"""

FIELD_HELP_TEXT = {
    "preset": "Choose a preset layout for virtual cameras. Defaults to an 8-view configuration.",
    "addcam": "Add extra cameras using codes like B, D:U20, H:D to offset pitch or duplicate slots.",
    "addcam_deg": "Default pitch delta (degrees) when U/D is specified without a value in addcam/setcam.",
    "delcam": "Remove baseline cameras by letter, e.g. B,D to skip those outputs.",
    "setcam": "Override existing camera pitch. Use A=U20, C:+15, D:-5, or target extras with A_U=10 / A_D:-5.",
    "size": "Square output size (pixels) for each perspective view.",
    "focal_mm": "Virtual focal length (mm). Used when HFOV is blank.",
    "hfov": "Horizontal field-of-view override (degrees). Takes priority over focal length when provided.",
    "sensor_mm": "Sensor width/height (mm). Example: 36 24 or 36x24.",
    "count": "Number of evenly spaced horizontal cameras (360° / count determines yaw step).",
    "add_topdown": "Enable additional top (+90° pitch) and bottom (-90° pitch) views.",
    "input_path": "Select an image folder or Browse to choose a video file (mp4/mov/etc.). For videos, set preview FPS and ensure ffmpeg is configured.",
    "show_seam_overlay": "Overlay a translucent band along the panorama seam to visualise potential stitching artifacts.",
    "ffmpeg": "Path to the ffmpeg executable. Leave blank to use the system PATH.",
    "jobs": "Number of parallel ffmpeg processes. 'auto' uses approximately half the CPU cores.",
    "ffthreads": "Internal ffmpeg threads per process. Set greater than 1 for multi-threaded encoding.",
    "fps": "Frame extraction rate (fps) required when processing a video source.",
    "keep_rec709": "Keep Rec.709 transfer characteristics for video inputs instead of converting to sRGB.",
}


def parse_arguments() -> argparse.Namespace:
    """Build the preview CLI parser and return parsed arguments."""

    parser = cutter.create_arg_parser()
    for action in parser._actions:
        if action.dest == "input_dir":
            action.required = False
    parser.description = "Visualize and execute rs2ps_360PerspCut camera layouts."
    parser.add_argument(
        "--image",
        help="Specific panorama to preview. Defaults to the first supported file in the selected directory.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional fixed scale factor (0 < scale <= 1). Overrides --max-width/--max-height.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Maximum display width when --scale is not supplied.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="Maximum display height when --scale is not supplied.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=90,
        help="Edge sampling count per side when tracing polygons (higher = smoother).",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide overlay labels if you prefer a clean view.",
    )
    return parser.parse_args()


def normalize_vector(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Normalize a 3D vector."""

    length = math.sqrt(x * x + y * y + z * z)
    if length <= 0.0:
        return 0.0, 0.0, 1.0
    return x / length, y / length, z / length


def rotate_pitch(vec: Sequence[float], pitch_rad: float) -> Tuple[float, float, float]:
    """Rotate vector around the X-axis by pitch (radians)."""

    x, y, z = vec
    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)
    return (
        x,
        cos_p * y + sin_p * z,
        -sin_p * y + cos_p * z,
    )


def rotate_yaw(vec: Sequence[float], yaw_rad: float) -> Tuple[float, float, float]:
    """Rotate vector around the Y-axis by yaw (radians)."""

    x, y, z = vec
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    return (
        cos_y * x + sin_y * z,
        y,
        -sin_y * x + cos_y * z,
    )


def direction_from_uv(
    u: float,
    v: float,
    hfov_rad: float,
    vfov_rad: float,
    yaw_rad: float,
    pitch_rad: float,
) -> Tuple[float, float]:
    """Convert normalized viewport coordinates to longitude/latitude in radians."""

    x = math.tan(hfov_rad / 2.0) * u
    y = math.tan(vfov_rad / 2.0) * (-v)
    z = 1.0
    vec = normalize_vector(x, y, z)
    vec = rotate_pitch(vec, pitch_rad)
    vec = rotate_yaw(vec, yaw_rad)
    lon = math.atan2(vec[0], vec[2])
    lat = math.asin(max(-1.0, min(1.0, vec[1])))
    return lon, lat


def unwrap_angles(values: Iterable[float]) -> List[float]:
    """Unwrap angle list to maintain continuity across the 2*pi seam."""

    iterator = iter(values)
    try:
        first = next(iterator)
    except StopIteration:
        return []
    unwrapped = [first]
    prev = first
    for angle in iterator:
        candidate = angle
        while candidate - prev > math.pi:
            candidate -= 2.0 * math.pi
        while candidate - prev < -math.pi:
            candidate += 2.0 * math.pi
        unwrapped.append(candidate)
        prev = candidate
    return unwrapped


def lonlat_to_xy(lon: float, lat: float, width: int, height: int) -> Tuple[float, float]:
    """Convert spherical lon/lat (radians) into equirectangular pixel coordinates."""

    x = ((lon / (2.0 * math.pi)) + 0.5) * width
    y = (0.5 - (lat / math.pi)) * height
    return x, y




def sample_view_segments(
    spec: cutter.ViewSpec,
    pano_width: int,
    pano_height: int,
    samples: int,
) -> Tuple[List[List[Tuple[float, float]]], Tuple[float, float]]:
    """Return edge segments and centroid (both in pixel space)."""

    hfov_rad = math.radians(min(max(spec.hfov_deg, 1e-3), 179.9))
    vfov_rad = math.radians(min(max(spec.vfov_deg, 1e-3), 179.9))
    yaw_rad = math.radians(spec.yaw_deg)
    pitch_rad = math.radians(spec.pitch_deg)

    per_side = max(8, samples)

    def sample_edge(
        u_start: float,
        u_end: float,
        v_start: float,
        v_end: float,
        count: int,
    ) -> List[List[Tuple[float, float]]]:
        if count <= 0:
            return []
        lon_samples: List[float] = []
        lat_samples: List[float] = []
        for i in range(count + 1):
            t = i / count
            u = u_start + (u_end - u_start) * t
            v = v_start + (v_end - v_start) * t
            lon, lat = direction_from_uv(u, v, hfov_rad, vfov_rad, yaw_rad, pitch_rad)
            lon_samples.append(lon)
            lat_samples.append(lat)
        lon_samples = unwrap_angles(lon_samples)
        segments: List[List[Tuple[float, float]]] = []
        current: List[Tuple[float, float]] = []
        current_wrap: Optional[int] = None
        for lon, lat in zip(lon_samples, lat_samples):
            x_unwrapped = ((lon / (2.0 * math.pi)) + 0.5) * pano_width
            wrap_index = math.floor(x_unwrapped / pano_width)
            x = x_unwrapped - wrap_index * pano_width
            if x < 0.0:
                x += pano_width
            elif x >= pano_width:
                x -= pano_width
            y = max(0.0, min(float(pano_height), (0.5 - (lat / math.pi)) * pano_height))
            if current_wrap is None:
                current_wrap = wrap_index
            elif wrap_index != current_wrap:
                if current:
                    segments.append(current)
                current = []
                current_wrap = wrap_index
            current.append((x, y))
        if current:
            segments.append(current)
        return segments

    segments: List[List[Tuple[float, float]]] = []
    segments.extend(sample_edge(-1.0, 1.0, -1.0, -1.0, per_side))
    segments.extend(sample_edge(1.0, 1.0, -1.0, 1.0, per_side))
    segments.extend(sample_edge(1.0, -1.0, 1.0, 1.0, per_side))
    segments.extend(sample_edge(-1.0, -1.0, 1.0, -1.0, per_side))

    center_lon, center_lat = direction_from_uv(0.0, 0.0, hfov_rad, vfov_rad, yaw_rad, pitch_rad)
    center_x = ((center_lon / (2.0 * math.pi)) + 0.5) * pano_width
    center_x = center_x % pano_width
    center_y = (0.5 - (center_lat / math.pi)) * pano_height
    center_y = max(0.0, min(float(pano_height), center_y))

    return segments, (center_x, center_y)

def compute_scale(
    width: int,
    height: int,
    scale_override: Optional[float],
    max_width: int,
    max_height: int,
) -> float:
    """Compute the display scale factor."""

    if scale_override:
        if scale_override <= 0:
            raise ValueError("--scale must be > 0")
        return min(1.0, float(scale_override))
    scale_w = max_width / float(width)
    scale_h = max_height / float(height)
    return min(1.0, scale_w, scale_h)


def build_info_lines(specs: Sequence[cutter.ViewSpec]) -> List[str]:
    """Create textual summaries for each view."""

    lines: List[str] = []
    for spec in specs:
        lines.append(
            f"{spec.view_id:>6}: yaw={spec.yaw_deg:+6.1f}°  "
            f"pitch={spec.pitch_deg:+6.1f}°  hfov={spec.hfov_deg:5.1f}°  "
            f"vfov={spec.vfov_deg:5.1f}°"
        )
    return lines


def clone_namespace(ns: argparse.Namespace) -> argparse.Namespace:
    """Return a deep copy of an argparse namespace."""

    return copy.deepcopy(ns)


def ensure_explicit_flags(args: argparse.Namespace) -> None:
    """Ensure *_explicit attributes exist for shared CLI."""

    for attr in ("size", "hfov", "focal_mm"):
        flag = f"{attr}_explicit"
        if not hasattr(args, flag):
            setattr(args, flag, False)


class PreviewApp:
    """Interactive GUI to tweak rs2ps_360 cuts and visualise overlays."""

    VIDEO_PERSIST_FIELDS = {"fps", "keep_rec709"}
    FIELD_DEFS = [
        {"name": "preset", "label": "Preset", "type": "choice", "choices": PRESET_CHOICES, "width": 14},
        {"name": "addcam", "label": "AddCam", "type": "str", "width": 18},
        {"name": "addcam_deg", "label": "CamDeg", "type": "float", "width": 10},
        {"name": "delcam", "label": "DelCam", "type": "str", "width": 16},
        {"name": "setcam", "label": "SetCam", "type": "str", "width": 18},
        {"name": "size", "label": "Size", "type": "int", "width": 10},
        {"name": "focal_mm", "label": "Focal", "type": "float", "width": 10},
        {"name": "hfov", "label": "HFOV", "type": "float_optional", "width": 10},
        {"name": "sensor_mm", "label": "Sensor", "type": "str", "width": 14},
        {"name": "count", "label": "Count", "type": "int", "width": 8},
        {"name": "ext", "label": "Ext", "type": "str", "width": 8},
        {
            "name": "add_topdown",
            "label": "Add top/bottom",
            "type": "bool",
            "align_with": "ext",
            "col_shift": 2,
        },
        {
            "name": "fps",
            "label": "FPS",
            "type": "float_optional",
            "width": 8,
            "video_only": True,
            "align_with": "count",
            "col_shift": 0,
            "row_shift": 1,
            "required_if_video": True,
        },
        {
            "name": "keep_rec709",
            "label": "Keep Rec.709",
            "type": "bool",
            "video_only": True,
            "align_with": "ext",
            "col_shift": 0,
            "row_shift": 1,
        },
        {
            "name": "show_seam_overlay",
            "label": "Show seam hint",
            "type": "bool",
            "align_with": "ext",
            "col_shift": 0,
            "row_shift": 2,
        },
    ]

    EXPLICIT_FIELDS = {"size", "hfov", "focal_mm"}

    def __init__(self, args: argparse.Namespace):
        ensure_explicit_flags(args)
        self.original_args = args
        self.current_args = clone_namespace(args)
        if not hasattr(self.current_args, "input_is_video"):
            setattr(self.current_args, "input_is_video", False)
        if not hasattr(self.current_args, "video_bit_depth"):
            setattr(self.current_args, "video_bit_depth", 8)
        if not hasattr(self.current_args, "show_seam_overlay"):
            setattr(self.current_args, "show_seam_overlay", False)
        self.defaults = clone_namespace(self.current_args)
        ensure_explicit_flags(self.current_args)
        self.form_snapshot: Dict[str, str] = {}
        self.field_vars: Dict[str, tk.Variable] = {}
        self.field_widgets: Dict[str, tk.Widget] = {}
        self.video_only_fields: Set[str] = {
            definition["name"]
            for definition in self.FIELD_DEFS
            if definition.get("video_only")
        }
        self.result: Optional[cutter.BuildResult] = None
        self.matched_specs: List[cutter.ViewSpec] = []
        self.is_executing = False
        self.source_is_video = False
        self._video_preview_signature: Optional[Tuple[pathlib.Path, bool]] = None
        self.video_persist_state: Dict[str, Any] = {"fps": 2.0, "keep_rec709": False}

        self.scale_override = args.scale
        self.max_width_limit = int(args.max_width)
        self.max_height_limit = int(args.max_height)

        self.in_dir: Optional[pathlib.Path] = None
        self.files: List[pathlib.Path] = []
        self.image_path: Optional[pathlib.Path] = None
        self.out_dir: Optional[pathlib.Path] = None

        self.samples = max(8, int(args.samples))
        self.hide_labels = bool(args.hide_labels)

        self.pano_image: Optional[Image.Image] = None
        self.display_image: Optional[Image.Image] = None
        self.pano_width = 0
        self.pano_height = 0
        self.scale = 1.0
        self.display_width = 640
        self.display_height = 360

        self.root = tk.Tk()
        self.root.withdraw()
        self.base_dir = Path(__file__).resolve().parent
        screen_w = max(1, self.root.winfo_screenwidth())
        screen_h = max(1, self.root.winfo_screenheight())
        self.view_max_width = max(1, min(self.max_width_limit, screen_w // 3))
        self.view_max_width = max(1, int(math.ceil(self.view_max_width * 1.1)))
        self.view_max_height = max(1, min(self.max_height_limit, screen_h // 3))

        self.photo: Optional[ImageTk.PhotoImage] = None
        self.preview_frame: Optional[tk.LabelFrame] = None
        self.controls_frame: Optional[tk.LabelFrame] = None
        self.left_frame: Optional[tk.Frame] = None
        self.canvas: Optional[tk.Canvas] = None
        self.canvas_image_id: Optional[int] = None
        self.folder_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self._tooltips: List[ToolTip] = []
        self.ffmpeg_path_var = tk.StringVar(value=str(getattr(self.current_args, "ffmpeg", "ffmpeg")))
        self.ffthreads_var = tk.StringVar(value=str(getattr(self.current_args, "ffthreads", "1")))
        self.jobs_var = tk.StringVar(value=str(getattr(self.current_args, "jobs", "auto")))
        self.log_text: Optional[tk.Text] = None
        self.help_text: Optional[tk.Text] = None
        self.update_button: Optional[tk.Button] = None
        self.execute_button: Optional[tk.Button] = None
        self.right_inner: Optional[tk.Frame] = None
        self.notebook: Optional[ttk.Notebook] = None

        self.video_vars: Dict[str, tk.Variable] = {}
        self.video_log: Optional[tk.Text] = None
        self.video_run_button: Optional[tk.Button] = None

        self.selector_vars: Dict[str, tk.Variable] = {}
        self.selector_log: Optional[tk.Text] = None
        self.selector_run_button: Optional[tk.Button] = None
        self.selector_show_score_button: Optional[tk.Button] = None

        self.human_vars: Dict[str, tk.Variable] = {}
        self.human_log: Optional[tk.Text] = None
        self.human_run_button: Optional[tk.Button] = None

        self.ply_vars: Dict[str, tk.Variable] = {}
        self.ply_log: Optional[tk.Text] = None
        self.ply_run_button: Optional[tk.Button] = None
        self.ply_append_text: Optional[tk.Text] = None
        self.ply_adaptive_weight_entry: Optional[tk.Entry] = None
        self.ply_keep_menu: Optional[tk.OptionMenu] = None

        self.video_stop_button: Optional[tk.Button] = None
        self.selector_stop_button: Optional[tk.Button] = None
        self.human_stop_button: Optional[tk.Button] = None
        self.ply_stop_button: Optional[tk.Button] = None
        self.preview_stop_button: Optional[tk.Button] = None

        self._process_lock = threading.Lock()
        self._active_processes: Dict[str, subprocess.Popen] = {}
        self._process_ui: Dict[str, Dict[str, Any]] = {}

        self._video_output_auto = True
        self._video_last_auto_output = ""
        self._video_output_updating = False
        self._video_prefix_auto = True
        self._video_last_auto_prefix = "out"
        self._video_prefix_updating = False
        self.selector_csv_entry: Optional[tk.Entry] = None
        self.selector_dry_run_check: Optional[tk.Checkbutton] = None
        self.selector_csv_button: Optional[tk.Button] = None

        self._preview_frame_padding = 0
        self._controls_frame_padding = 0

        self.root.title("rs2ps_360GUI")
        self.build_ui()
        self.set_form_values()
        self.root.update_idletasks()

        loaded = False
        if getattr(args, "input_dir", None):
            loaded = self.try_load_directory(args.input_dir, image_hint=args.image)
            if loaded:
                self.refresh_overlays(initial=True)

        if not loaded:
            self.set_log_text("Select an input folder to preview camera layouts.")

        self.root.deiconify()



    def build_ui(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        ttk.Separator(self.root, orient="horizontal").pack(fill="x")

        video_tab = tk.Frame(self.notebook)
        selector_tab = tk.Frame(self.notebook)
        preview_tab = tk.Frame(self.notebook)
        human_tab = tk.Frame(self.notebook)
        ply_tab = tk.Frame(self.notebook)

        self.notebook.add(video_tab, text="rs2ps_Video2Frames")
        self.notebook.add(selector_tab, text="rs2ps_FrameSelector")
        self.notebook.add(preview_tab, text="rs2ps_360PerspCut")
        self.notebook.add(human_tab, text="rs2ps_HumanMaskTool")
        self.notebook.add(ply_tab, text="rs2ps_PlyOptimizer")

        self._build_video_tab(video_tab)
        self._build_frame_selector_tab(selector_tab)
        self._build_preview_tab(preview_tab)
        self._build_human_mask_tab(human_tab)
        self._build_ply_tab(ply_tab)

        self.notebook.select(preview_tab)

    def _build_video_tab(self, parent: tk.Widget) -> None:
        container = tk.Frame(parent)
        container.pack(fill="both", expand=True)

        params = tk.LabelFrame(container, text="Parameters")
        params.pack(fill="x", padx=8, pady=8)

        self.video_vars = {
            "prefix": tk.StringVar(value="out"),
            "video": tk.StringVar(),
            "output": tk.StringVar(),
            "fps": tk.StringVar(value="5"),
            "ext": tk.StringVar(value="jpg"),
            "start": tk.StringVar(),
            "end": tk.StringVar(),
            "keep_rec709": tk.BooleanVar(value=False),
            "overwrite": tk.BooleanVar(value=False),
        }

        self.video_vars["video"].trace_add("write", self._on_video_input_changed)
        self.video_vars["output"].trace_add("write", self._on_video_output_changed)
        self.video_vars["fps"].trace_add("write", self._on_video_fps_changed)
        self.video_vars["prefix"].trace_add("write", self._on_video_prefix_changed)

        row = 0
        tk.Label(params, text="Input video").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        video_entry = tk.Entry(params, textvariable=self.video_vars["video"], width=52)
        video_entry.grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_file(self.video_vars["video"], title="Select video file"),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        tk.Label(params, text="Output folder").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        out_entry = tk.Entry(params, textvariable=self.video_vars["output"], width=52)
        out_entry.grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_directory(self.video_vars["output"], title="Select output folder"),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        fps_ext_frame = tk.Frame(params)
        fps_ext_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(fps_ext_frame, text="FPS").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(fps_ext_frame, textvariable=self.video_vars["fps"], width=10).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(fps_ext_frame, text="Extension").pack(side=tk.LEFT, padx=(0, 4))
        ext_combo = ttk.Combobox(
            fps_ext_frame,
            textvariable=self.video_vars["ext"],
            values=("jpg", "tif", "png"),
            width=8,
            state="readonly",
        )
        ext_combo.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(fps_ext_frame, text="Prefix").pack(side=tk.LEFT, padx=(12, 4))
        tk.Entry(fps_ext_frame, textvariable=self.video_vars["prefix"], width=12).pack(side=tk.LEFT, padx=(0, 4))

        row += 1
        range_frame = tk.Frame(params)
        range_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(range_frame, text="Start (s)").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(range_frame, textvariable=self.video_vars["start"], width=10).pack(side=tk.LEFT, padx=(0, 16))
        tk.Label(range_frame, text="End (s)").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(range_frame, textvariable=self.video_vars["end"], width=10).pack(side=tk.LEFT, padx=(0, 4))

        row += 1
        flags_frame = tk.Frame(params)
        flags_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        tk.Checkbutton(
            flags_frame,
            text="Keep Rec.709",
            variable=self.video_vars["keep_rec709"],
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(
            flags_frame,
            text="Overwrite output",
            variable=self.video_vars["overwrite"],
        ).pack(side=tk.LEFT, padx=(0, 12))

        for col in range(3):
            params.grid_columnconfigure(col, weight=1 if col == 1 else 0)

        actions = tk.Frame(container)
        actions.pack(fill="x", padx=8, pady=(0, 8))
        self.video_stop_button = tk.Button(
            actions,
            text="Stop",
            command=lambda: self._stop_cli_process("video"),
        )
        self.video_stop_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.video_stop_button.configure(state="disabled")
        self.video_run_button = tk.Button(
            actions,
            text="Run rs2ps_Video2Frames",
            command=self._run_video_tool,
        )
        self.video_run_button.pack(side=tk.RIGHT, padx=4, pady=4)

        log_frame = tk.LabelFrame(container, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.video_log = tk.Text(log_frame, wrap="word", height=18, cursor="arrow")
        self.video_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.video_log.bind("<Key>", self._block_text_edit)
        self.video_log.bind("<Button-1>", lambda event: self.video_log.focus_set())
        self._set_text_widget(self.video_log, "")

        self._update_video_default_output(force=False)

    def _on_video_input_changed(self, *_args) -> None:
        value = self.video_vars.get("video")
        if value is None:
            return
        if not value.get().strip():
            return
        self._update_video_default_output(force=True)

    def _on_video_fps_changed(self, *_args) -> None:
        self._update_video_default_output(force=False)

    def _on_video_output_changed(self, *_args) -> None:
        if self._video_output_updating:
            return
        self._video_output_auto = False

    def _on_video_prefix_changed(self, *_args) -> None:
        if self._video_prefix_updating:
            return
        self._video_prefix_auto = False


    def _format_fps_for_output(self, fps_value: str) -> Optional[str]:
        if not fps_value:
            return None
        try:
            fps = float(fps_value)
        except ValueError:
            return None
        fps = max(fps, 0.0)
        text = f"{fps}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    def _update_video_default_output(self, force: bool = False) -> None:
        if "video" not in self.video_vars or "output" not in self.video_vars:
            return
        video_value = self.video_vars["video"].get().strip()
        prefix_var = self.video_vars.get("prefix")
        if not video_value:
            if prefix_var is not None and (self._video_prefix_auto or not prefix_var.get().strip()):
                self._video_prefix_auto = True
                self._video_last_auto_prefix = "out"
                self._video_prefix_updating = True
                try:
                    prefix_var.set("out")
                finally:
                    self._video_prefix_updating = False
            return

        fps_text = self.video_vars["fps"].get().strip()
        fps_formatted = self._format_fps_for_output(fps_text) or "fps"

        path = Path(video_value).expanduser()
        try:
            if not path.is_absolute():
                path = path.resolve()
        except FileNotFoundError:
            return
        if not path.suffix:
            return

        parent = path.parent if path.parent != Path("") else Path.cwd()
        default_output = str(parent / f"{path.stem}_frames_{fps_formatted}fps")

        if prefix_var is not None:
            default_prefix = re.sub(r"\s+", "_", path.stem) if path.stem else "out"
            if not default_prefix:
                default_prefix = "out"
            current_prefix = prefix_var.get().strip()
            should_update_prefix = (
                force
                or self._video_prefix_auto
                or not current_prefix
                or current_prefix == self._video_last_auto_prefix
            )
            if should_update_prefix:
                self._video_prefix_auto = True
                self._video_last_auto_prefix = default_prefix
                self._video_prefix_updating = True
                try:
                    prefix_var.set(default_prefix)
                finally:
                    self._video_prefix_updating = False

        current_output = self.video_vars["output"].get().strip()
        should_update = force or self._video_output_auto or not current_output or current_output == self._video_last_auto_output
        if not should_update:
            return

        self._video_output_auto = True
        self._video_last_auto_output = default_output
        self._video_output_updating = True
        try:
            self.video_vars["output"].set(default_output)
        finally:
            self._video_output_updating = False

    def _build_frame_selector_tab(self, parent: tk.Widget) -> None:
        container = tk.Frame(parent)
        container.pack(fill="both", expand=True)

        params = tk.LabelFrame(container, text="Parameters")
        params.pack(fill="x", padx=8, pady=8)

        self.selector_vars = {
            "in_dir": tk.StringVar(),
            "segment_size": tk.StringVar(),
            "dry_run": tk.BooleanVar(value=True),
            "workers": tk.StringVar(),
            "ext": tk.StringVar(value="all"),
            "sort": tk.StringVar(value="lastnum"),
            "metric": tk.StringVar(value="hybrid"),
            "csv_mode": tk.StringVar(value="none"),
            "csv_path": tk.StringVar(),
            "crop_ratio": tk.StringVar(value="0.8"),
            "min_spacing_frames": tk.StringVar(),
            "prune_motion": tk.BooleanVar(value=False),
            "augment_gap": tk.BooleanVar(value=False),
            "augment_lowlight": tk.BooleanVar(value=False),
            "augment_motion": tk.BooleanVar(value=False),
        }

        self.selector_vars["csv_mode"].trace_add("write", self._on_selector_csv_mode_changed)

        row = 0
        tk.Label(params, text="Input folder").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(params, textvariable=self.selector_vars["in_dir"], width=52).grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_directory(self.selector_vars["in_dir"], title="Select input folder"),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        segment_frame = tk.Frame(params)
        segment_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(segment_frame, text="Segment size").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(segment_frame, textvariable=self.selector_vars["segment_size"], width=10).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(segment_frame, text="Workers").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(segment_frame, textvariable=self.selector_vars["workers"], width=10).pack(side=tk.LEFT, padx=(0, 12))
        self.selector_dry_run_check = tk.Checkbutton(
            segment_frame,
            text="Dry run",
            variable=self.selector_vars["dry_run"],
        )
        self.selector_dry_run_check.pack(side=tk.LEFT, padx=(0, 12))

        row += 1
        esm_frame = tk.Frame(params)
        esm_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(esm_frame, text="Extension").pack(side=tk.LEFT, padx=(0, 4))
        tk.OptionMenu(esm_frame, self.selector_vars["ext"], "all", "tif", "jpg", "png").pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(esm_frame, text="Sort").pack(side=tk.LEFT, padx=(0, 4))
        tk.OptionMenu(esm_frame, self.selector_vars["sort"], "lastnum", "firstnum", "name", "mtime").pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(esm_frame, text="Metric").pack(side=tk.LEFT, padx=(0, 4))
        tk.OptionMenu(esm_frame, self.selector_vars["metric"], "hybrid", "lapvar", "tenengrad", "fft").pack(side=tk.LEFT, padx=(0, 4))

        row += 1
        csv_frame = tk.Frame(params)
        csv_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(csv_frame, text="CSV mode").pack(side=tk.LEFT, padx=(0, 4))
        tk.OptionMenu(csv_frame, self.selector_vars["csv_mode"], "none", "write", "apply", "reselect").pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(csv_frame, text="Path").pack(side=tk.LEFT, padx=(0, 4))
        self.selector_csv_entry = tk.Entry(csv_frame, textvariable=self.selector_vars["csv_path"], width=40)
        self.selector_csv_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.selector_csv_button = tk.Button(csv_frame, text="Browse...", command=self._browse_selector_csv)
        self.selector_csv_button.pack(side=tk.LEFT, padx=(0, 4))

        row += 1
        tk.Label(params, text="Min spacing").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(params, textvariable=self.selector_vars["min_spacing_frames"], width=10).grid(row=row, column=1, sticky="w", padx=4, pady=4)

        row += 1
        augment_frame = tk.Frame(params)
        augment_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        tk.Checkbutton(augment_frame, text="Augment gap", variable=self.selector_vars["augment_gap"]).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(augment_frame, text="Augment lowlight", variable=self.selector_vars["augment_lowlight"]).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(augment_frame, text="Augment motion", variable=self.selector_vars["augment_motion"]).pack(side=tk.LEFT, padx=(0, 12))

        row += 1
        final_frame = tk.Frame(params)
        final_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Checkbutton(final_frame, text="Prune motion", variable=self.selector_vars["prune_motion"]).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(final_frame, text="Crop ratio").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(final_frame, textvariable=self.selector_vars["crop_ratio"], width=10).pack(side=tk.LEFT, padx=(0, 4))

        for col in range(3):
            params.grid_columnconfigure(col, weight=1 if col == 1 else 0)

        actions = tk.Frame(container)
        actions.pack(fill="x", padx=8, pady=(0, 8))
        self.selector_stop_button = tk.Button(
            actions,
            text="Stop",
            command=lambda: self._stop_cli_process("selector"),
        )
        self.selector_stop_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.selector_stop_button.configure(state="disabled")
        self.selector_run_button = tk.Button(
            actions,
            text="Run rs2ps_FrameSelector",
            command=self._run_frame_selector,
        )
        self.selector_run_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.selector_count_var = tk.StringVar(value="20")
        count_entry = tk.Entry(actions, textvariable=self.selector_count_var, width=6)
        count_entry.pack(side=tk.LEFT, padx=(4, 0), pady=4)
        count_entry.insert(0, "")
        if not self.selector_count_var.get():
            self.selector_count_var.set("20")
        self.selector_show_score_button = tk.Button(
            actions,
            text="Fetch Sharpness Scores From CSV",
            command=self._show_selector_scores,
        )
        self.selector_show_score_button.pack(side=tk.LEFT, padx=(4, 4), pady=4)

        log_frame = tk.LabelFrame(container, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.selector_log = tk.Text(log_frame, wrap="word", height=18, cursor="arrow")
        self.selector_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.selector_log.bind("<Key>", self._block_text_edit)
        self.selector_log.bind("<Button-1>", lambda event: self.selector_log.focus_set())
        self._set_text_widget(self.selector_log, "")

        self._on_selector_csv_mode_changed()

    def _build_human_mask_tab(self, parent: tk.Widget) -> None:
        container = tk.Frame(parent)
        container.pack(fill="both", expand=True)

        params = tk.LabelFrame(container, text="Parameters")
        params.pack(fill="x", padx=8, pady=8)

        self.human_vars = {
            "input": tk.StringVar(),
            "output": tk.StringVar(),
            "mode": tk.StringVar(value="mask"),
            "dilate": tk.StringVar(),
            "cpu": tk.BooleanVar(value=False),
            "include_shadow": tk.BooleanVar(value=False),
        }

        row = 0
        tk.Label(params, text="Input folder").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(params, textvariable=self.human_vars["input"], width=52).grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_directory(
                self.human_vars["input"],
                title="Select input folder",
                on_select=self._on_human_input_selected,
            ),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        tk.Label(params, text="Output folder").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(params, textvariable=self.human_vars["output"], width=52).grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_directory(self.human_vars["output"], title="Select output folder"),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        mode_frame = tk.Frame(params)
        mode_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(mode_frame, text="Mode").pack(side=tk.LEFT, padx=(0, 4))
        mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.human_vars["mode"],
            values=HUMAN_MODE_CHOICES,
            state="readonly",
            width=16,
        )
        mode_combo.pack(side=tk.LEFT, padx=(0, 12))
        mode_combo.set(self.human_vars["mode"].get())

        tk.Label(mode_frame, text="Dilation %").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(mode_frame, textvariable=self.human_vars["dilate"], width=10).pack(side=tk.LEFT, padx=(0, 12))

        row += 1
        flags_frame = tk.Frame(params)
        flags_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        tk.Checkbutton(flags_frame, text="Force CPU", variable=self.human_vars["cpu"]).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(
            flags_frame,
            text="Include shadow",
            variable=self.human_vars["include_shadow"],
        ).pack(side=tk.LEFT, padx=(0, 12))

        for col in range(3):
            params.grid_columnconfigure(col, weight=1 if col == 1 else 0)

        actions = tk.Frame(container)
        actions.pack(fill="x", padx=8, pady=(0, 8))
        self.human_stop_button = tk.Button(
            actions,
            text="Stop",
            command=lambda: self._stop_cli_process("human"),
        )
        self.human_stop_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.human_stop_button.configure(state="disabled")
        self.human_run_button = tk.Button(
            actions,
            text="Run rs2ps_HumanMaskTool",
            command=self._run_human_mask_tool,
        )
        self.human_run_button.pack(side=tk.RIGHT, padx=4, pady=4)

        log_frame = tk.LabelFrame(container, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.human_log = tk.Text(log_frame, wrap="word", height=18, cursor="arrow")
        self.human_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.human_log.bind("<Key>", self._block_text_edit)
        self.human_log.bind("<Button-1>", lambda event: self.human_log.focus_set())
        self._set_text_widget(self.human_log, "")

    def _on_human_input_selected(self, selected_path: str) -> None:
        if not self.human_vars:
            return
        try:
            base_path = Path(selected_path).expanduser()
        except Exception:
            return
        default_out = base_path / "_mask"
        self.human_vars["output"].set(str(default_out))

    def _build_ply_tab(self, parent: tk.Widget) -> None:
        container = tk.Frame(parent)
        container.pack(fill="both", expand=True)

        params = tk.LabelFrame(container, text="Parameters")
        params.pack(fill="x", padx=8, pady=8)

        self.ply_vars = {
            "input": tk.StringVar(),
            "output": tk.StringVar(),
            "target_points": tk.StringVar(),
            "target_percent": tk.StringVar(),
            "voxel_size": tk.StringVar(),
            "adaptive": tk.BooleanVar(value=False),
            "adaptive_weight": tk.StringVar(value="1.0"),
            "keep_strategy": tk.StringVar(value="centroid"),
        }
        self.ply_vars["adaptive"].trace_add("write", self._update_ply_adaptive_state)

        row = 0
        tk.Label(params, text="Input PLY").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(params, textvariable=self.ply_vars["input"], width=52).grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_file(self.ply_vars["input"], title="Select input PLY", filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        tk.Label(params, text="Output PLY").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        tk.Entry(params, textvariable=self.ply_vars["output"], width=52).grid(row=row, column=1, sticky="we", padx=4, pady=4)
        tk.Button(
            params,
            text="Browse...",
            command=lambda: self._select_save_file(self.ply_vars["output"], title="Select output PLY", defaultextension=".ply", filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]),
        ).grid(row=row, column=2, padx=4, pady=4)

        row += 1
        numbers_frame = tk.Frame(params)
        numbers_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Label(numbers_frame, text="Target points").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(numbers_frame, textvariable=self.ply_vars["target_points"], width=12).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(numbers_frame, text="Target percent").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(numbers_frame, textvariable=self.ply_vars["target_percent"], width=10).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(numbers_frame, text="Voxel size").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(numbers_frame, textvariable=self.ply_vars["voxel_size"], width=10).pack(side=tk.LEFT, padx=(0, 4))

        row += 1
        adaptive_frame = tk.Frame(params)
        adaptive_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        tk.Checkbutton(
            adaptive_frame,
            text="Adaptive sampling",
            variable=self.ply_vars["adaptive"],
            command=self._update_ply_adaptive_state,
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(adaptive_frame, text="Weight").pack(side=tk.LEFT, padx=(0, 4))
        self.ply_adaptive_weight_entry = tk.Entry(
            adaptive_frame,
            textvariable=self.ply_vars["adaptive_weight"],
            width=8,
        )
        self.ply_adaptive_weight_entry.pack(side=tk.LEFT, padx=(0, 4))

        row += 1
        tk.Label(params, text="Keep strategy").grid(row=row, column=0, sticky="e", padx=4, pady=4)
        self.ply_keep_menu = tk.OptionMenu(
            params,
            self.ply_vars["keep_strategy"],
            "centroid",
            "center",
            "first",
            "random",
        )
        self.ply_keep_menu.grid(row=row, column=1, sticky="w", padx=4, pady=4)

        params.grid_columnconfigure(1, weight=1)

        append_frame = tk.LabelFrame(container, text="Append PLY files (one per line)")
        append_frame.pack(fill="both", expand=False, padx=8, pady=(0, 8))
        self.ply_append_text = tk.Text(append_frame, height=4, wrap="none")
        self.ply_append_text.pack(fill="both", expand=True, padx=6, pady=4)

        actions = tk.Frame(container)
        actions.pack(fill="x", padx=8, pady=(0, 8))
        self.ply_stop_button = tk.Button(
            actions,
            text="Stop",
            command=lambda: self._stop_cli_process("ply"),
        )
        self.ply_stop_button.pack(side=tk.RIGHT, padx=4, pady=4)
        self.ply_stop_button.configure(state="disabled")
        self.ply_run_button = tk.Button(
            actions,
            text="Run rs2ps_PlyOptimizer",
            command=self._run_ply_optimizer,
        )
        self.ply_run_button.pack(side=tk.RIGHT, padx=4, pady=4)

        log_frame = tk.LabelFrame(container, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.ply_log = tk.Text(log_frame, wrap="word", height=18, cursor="arrow")
        self.ply_log.pack(fill="both", expand=True, padx=6, pady=4)
        self.ply_log.bind("<Key>", self._block_text_edit)
        self.ply_log.bind("<Button-1>", lambda event: self.ply_log.focus_set())
        self._set_text_widget(self.ply_log, "")
        self._update_ply_adaptive_state()

    def _build_preview_tab(self, parent: tk.Widget) -> None:
        main = tk.Frame(parent)
        main.pack(fill="both", expand=True)

        work_frame = tk.Frame(main)
        work_frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        left_right_wrapper = tk.Frame(work_frame)
        left_right_wrapper.pack(fill="both", expand=True)

        self.preview_frame = tk.LabelFrame(left_right_wrapper, text="Preview")
        self.preview_frame.pack(side=tk.LEFT, padx=(0, 6), pady=0, fill="y", expand=False, anchor="n")
        self.preview_frame.pack_propagate(False)
        self.preview_frame.configure(
            width=self.display_width + 16,
            height=self.display_height + 16,
        )

        self.left_frame = tk.Frame(self.preview_frame, bg=self.root.cget("bg"), width=self.display_width, height=self.display_height)
        self.left_frame.pack(fill="y", expand=False)
        self.left_frame.pack_propagate(False)

        self.canvas = tk.Canvas(
            self.left_frame,
            width=self.display_width,
            height=self.display_height,
            highlightthickness=0,
            bg="#202020",
        )
        self.canvas.pack(side=tk.TOP, fill="both", expand=True)
        self.canvas.create_text(
            self.display_width / 2,
            self.display_height / 2,
            text="Select an input folder",
            fill="#d0d0d0",
            tags=("placeholder",),
        )

        self.controls_frame = tk.LabelFrame(left_right_wrapper, text="Controls", width=520)
        self.controls_frame.pack(side=tk.LEFT, fill="y", expand=False, padx=(6, 0), pady=0)
        self.controls_frame.pack_propagate(False)
        self.right_inner = tk.Frame(self.controls_frame)
        self.right_inner.pack(fill="both", expand=True)

        folder_frame = tk.LabelFrame(self.right_inner, text="Input Folder")
        folder_frame.pack(fill="x")
        path_entry = tk.Entry(folder_frame, textvariable=self.folder_path_var, width=38)
        path_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(8, 4), pady=4)
        self._bind_help(path_entry, "input_path")
        tk.Button(
            folder_frame,
            text="Browse...",
            command=lambda: self.prompt_for_directory(False),
        ).pack(side=tk.RIGHT, padx=(4, 8), pady=4)
        tk.Button(
            folder_frame,
            text="Load",
            command=self.load_directory_from_entry,
        ).pack(side=tk.RIGHT, padx=(4, 4), pady=4)

        output_frame = tk.LabelFrame(self.right_inner, text="Output Folder")
        output_frame.pack(fill="x", pady=(4, 4))
        out_entry = tk.Entry(output_frame, textvariable=self.output_path_var, width=38)
        out_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(8, 4), pady=4)
        tk.Button(
            output_frame,
            text="Browse...",
            command=self.browse_output_directory,
        ).pack(side=tk.RIGHT, padx=(4, 8), pady=4)

        controls = tk.LabelFrame(self.right_inner, text="Crop Parameters")
        controls.pack(fill="x", pady=(8, 4))

        columns = 3
        field_positions: Dict[str, Tuple[int, int]] = {}
        for idx, definition in enumerate(self.FIELD_DEFS):
            name = definition["name"]
            field_type = definition["type"]
            base_row = idx // columns
            base_col = (idx % columns) * 2
            row = base_row + int(definition.get("row_shift", 0))
            col = base_col
            anchor = definition.get("align_with")
            if anchor:
                anchor_pos = field_positions.get(anchor)
                if anchor_pos:
                    row = anchor_pos[0] + int(definition.get("row_shift", 0))
                    col = anchor_pos[1] + int(definition.get("col_shift", 2))
            field_positions[name] = (row, col)
            video_only = bool(definition.get("video_only"))
            if field_type == "bool":
                var = tk.BooleanVar(value=False)
                widget = tk.Checkbutton(controls, text=definition["label"], variable=var, anchor="w")
                widget.configure(width=20)
                if name == "show_seam_overlay":
                    widget.configure(command=lambda v=var: self._on_show_seam_toggle(v))
                widget.grid(row=row, column=col, columnspan=2, sticky="w", padx=4, pady=2)
                self._bind_help(widget, name)
                self.field_vars[name] = var
                self.field_widgets[name] = widget
                if video_only and not self.source_is_video:
                    widget.configure(state="disabled")
                continue
            if field_type == "choice":
                var = tk.StringVar(value=definition["choices"][0])
                label = tk.Label(controls, text=definition["label"])
                label.grid(row=row, column=col, sticky="e", padx=4, pady=2)
                self._bind_help(label, name)
                if name == "preset":
                    option = tk.OptionMenu(controls, var, *definition["choices"], command=self.on_preset_changed)
                else:
                    option = tk.OptionMenu(controls, var, *definition["choices"])
                option.grid(row=row, column=col + 1, sticky="we", padx=4, pady=2)
                option.configure(width=max(definition.get("width", 12), 16))
                self._bind_help(option, name)
                self.field_vars[name] = var
                self.field_widgets[name] = option
                continue
            label = tk.Label(controls, text=definition["label"])
            label.grid(row=row, column=col, sticky="e", padx=4, pady=2)
            self._bind_help(label, name)
            var = tk.StringVar()
            entry = tk.Entry(controls, textvariable=var, width=definition.get("width", 12))
            entry.grid(row=row, column=col + 1, sticky="we", padx=4, pady=2)
            self._bind_help(entry, name)
            self.field_vars[name] = var
            self.field_widgets[name] = entry
            if video_only and not self.source_is_video:
                entry.configure(state="disabled")

        max_col_index = 0
        if field_positions:
            max_col_index = max(pos[1] for pos in field_positions.values())
        for col in range(max_col_index + 2):
            controls.grid_columnconfigure(col, weight=1 if col % 2 else 0)

        ffmpeg_frame = tk.LabelFrame(self.right_inner, text="ffmpeg Options")
        ffmpeg_frame.pack(fill="x", pady=(4, 4))
        ffmpeg_frame.columnconfigure(1, weight=1)

        ffmpeg_label = tk.Label(ffmpeg_frame, text="ffmpeg path")
        ffmpeg_label.grid(row=0, column=0, padx=4, pady=2, sticky="e")
        self._bind_help(ffmpeg_label, "ffmpeg")
        ffmpeg_entry = tk.Entry(ffmpeg_frame, textvariable=self.ffmpeg_path_var, width=30)
        ffmpeg_entry.grid(row=0, column=1, padx=4, pady=2, sticky="we")
        self._bind_help(ffmpeg_entry, "ffmpeg")
        browse_btn = tk.Button(ffmpeg_frame, text="Browse...", command=self.browse_ffmpeg)
        browse_btn.grid(row=0, column=2, padx=4, pady=2)
        self._bind_help(browse_btn, "ffmpeg")
        reset_btn = tk.Button(ffmpeg_frame, text="Reset", command=self.reset_ffmpeg_path)
        reset_btn.grid(row=0, column=3, padx=4, pady=2)
        self._bind_help(reset_btn, "ffmpeg")

        jobs_label = tk.Label(ffmpeg_frame, text="jobs")
        jobs_label.grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self._bind_help(jobs_label, "jobs")
        jobs_entry = tk.Entry(ffmpeg_frame, textvariable=self.jobs_var, width=10)
        jobs_entry.grid(row=1, column=1, padx=4, pady=2, sticky="w")
        self._bind_help(jobs_entry, "jobs")
        ffthreads_label = tk.Label(ffmpeg_frame, text="ffthreads")
        ffthreads_label.grid(row=1, column=2, padx=4, pady=2, sticky="e")
        self._bind_help(ffthreads_label, "ffthreads")
        ffthreads_entry = tk.Entry(ffmpeg_frame, textvariable=self.ffthreads_var, width=10)
        ffthreads_entry.grid(row=1, column=3, padx=4, pady=2, sticky="w")
        self._bind_help(ffthreads_entry, "ffthreads")

        buttons_frame = tk.LabelFrame(self.right_inner, text="Actions")
        buttons_frame.pack(fill="x", pady=(8, 12), ipady=6)

        buttons_inner = tk.Frame(buttons_frame)
        buttons_inner.pack(fill="both", expand=True, padx=14, pady=(10, 12))
        buttons_inner.columnconfigure(0, weight=1)
        buttons_inner.columnconfigure(1, weight=1)
        buttons_inner.columnconfigure(2, weight=0)
        buttons_inner.grid_rowconfigure(0, weight=1)
        self.update_button = tk.Button(
            buttons_inner,
            text="Update",
            command=self.on_update,
        )
        self.update_button.grid(row=0, column=0, padx=(0, 12), pady=0, sticky="nsew", ipadx=10, ipady=8)
        self.execute_button = tk.Button(
            buttons_inner,
            text="Run Export",
            command=self.on_execute,
        )
        self.execute_button.grid(row=0, column=1, padx=(12, 0), pady=0, sticky="nsew", ipadx=10, ipady=8)
        self.preview_stop_button = tk.Button(
            buttons_inner,
            text="Stop",
            command=self._stop_preview_execution,
        )
        self.preview_stop_button.grid(row=0, column=2, padx=(12, 0), pady=0, sticky="nsew", ipadx=10, ipady=8)
        self.preview_stop_button.configure(state="disabled")

        log_frame = tk.LabelFrame(main, text="Log")
        log_frame.pack(fill="both", expand=False, padx=8, pady=(0, 4))
        self.log_text = tk.Text(log_frame, wrap="word", cursor="arrow", height=14)
        self.log_text.pack(fill="both", expand=True, padx=6, pady=4)
        self.log_text.bind("<Key>", self._block_text_edit)
        self.log_text.bind("<Button-1>", lambda event: self.log_text.focus_set())
        self.set_log_text("")

        help_frame = tk.LabelFrame(main, text="Help")
        help_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.help_text = tk.Text(help_frame, wrap="word", height=8, cursor="arrow")
        self.help_text.pack(fill="both", expand=True, padx=6, pady=4)
        self.help_text.insert("1.0", HELP_TEXT)
        self.help_text.bind("<Key>", self._block_text_edit)
        self.help_text.bind("<Button-1>", lambda event: self.help_text.focus_set())
        self._sync_panel_heights()

    def _set_text_widget(self, widget: Optional[tk.Text], text: str) -> None:
        if widget is None:
            return
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.see("1.0")

    def _append_text_widget(self, widget: Optional[tk.Text], line: str) -> None:
        if widget is None:
            return
        widget.configure(state="normal")
        current = widget.get("1.0", tk.END).strip()
        prefix = "\n" if current else ""
        widget.insert(tk.END, prefix + line)
        widget.see(tk.END)

    def _run_cli_command(
        self,
        cmd: Sequence[str],
        log_widget: Optional[tk.Text],
        run_button: Optional[tk.Button],
        process_key: str,
        stop_button: Optional[tk.Button] = None,
        cwd: Optional[Path] = None,
    ) -> None:
        if log_widget is None:
            return
        if not process_key:
            raise ValueError("process_key is required for CLI commands")

        with self._process_lock:
            if process_key in self._active_processes:
                self._append_text_widget(log_widget, "[warn] Process already running. Stop it before launching a new one.")
                return

        command_text = "CLI> " + " ".join(shlex.quote(str(token)) for token in cmd)
        self._set_text_widget(log_widget, command_text)

        if run_button is not None:
            run_button.configure(state="disabled")
        if stop_button is not None:
            stop_button.configure(state="disabled")
            stop_button.configure(command=lambda key=process_key: self._stop_cli_process(key))

        def worker() -> None:
            try:
                process = subprocess.Popen(
                    list(cmd),
                    cwd=str(cwd) if cwd else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:
                self.root.after(0, lambda: self._append_text_widget(log_widget, f"[ERR] {exc}"))
                if run_button is not None:
                    self.root.after(0, lambda: run_button.configure(state="normal"))
                if stop_button is not None:
                    self.root.after(0, lambda: stop_button.configure(state="disabled"))
                return

            def reader() -> None:
                assert process.stdout is not None
                for raw in process.stdout:
                    line = raw.rstrip("\n")
                    self.root.after(0, lambda l=line: self._append_text_widget(log_widget, l))

            with self._process_lock:
                self._active_processes[process_key] = process
                self._process_ui[process_key] = {
                    "run": run_button,
                    "stop": stop_button,
                    "log": log_widget,
                }

            if stop_button is not None:
                self.root.after(0, lambda: stop_button.configure(state="normal"))

            stdout_thread = threading.Thread(target=reader, name=f"{process_key}-stdout", daemon=True)
            stdout_thread.start()

            rc = process.wait()
            stdout_thread.join(timeout=0.5)
            self.root.after(0, lambda rc=rc: self._on_cli_completed(process_key, rc if rc is not None else 0, stopped=False))

        threading.Thread(target=worker, name=f"{process_key}-runner", daemon=True).start()

    def _on_cli_completed(self, key: str, rc: int, stopped: bool) -> None:
        with self._process_lock:
            process = self._active_processes.pop(key, None)
            info = self._process_ui.pop(key, {})
        run_button = info.get("run")
        stop_button = info.get("stop")
        log_widget = info.get("log")
        if run_button is not None:
            run_button.configure(state="normal")
        if stop_button is not None:
            stop_button.configure(state="disabled")
        if log_widget is not None:
            message = "[stop] terminated" if stopped else f"[exit] return code {rc}"
            self._append_text_widget(log_widget, message)
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

    def _stop_cli_process(self, key: str) -> None:
        with self._process_lock:
            process = self._active_processes.get(key)
            info = self._process_ui.get(key, {})
        if process is None:
            return
        stop_button = info.get("stop")
        log_widget = info.get("log")
        if stop_button is not None:
            stop_button.configure(state="disabled")
        if log_widget is not None:
            self._append_text_widget(log_widget, "[stop] Terminating process...")

        def worker() -> None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
            except Exception as exc:
                if log_widget is not None:
                    self.root.after(0, lambda e=exc: self._append_text_widget(log_widget, f"[ERR] stop failed: {e}"))
            finally:
                rc = process.poll()
                self.root.after(0, lambda: self._on_cli_completed(key, rc if rc is not None else -1, stopped=True))

        threading.Thread(target=worker, name=f"{key}-stopper", daemon=True).start()

    def _stop_preview_execution(self) -> None:
        if not self.is_executing:
            return
        cutter.stop_event.set()
        if self.preview_stop_button is not None:
            self.preview_stop_button.configure(state="disabled")
        self.append_log_line("[EXEC] Stop requested...")

    def _run_human_mask_tool(self) -> None:
        if not self.human_vars:
            return
        input_dir = self.human_vars["input"].get().strip()
        if not input_dir:
            messagebox.showerror("rs2ps_HumanMaskTool", "Input folder is required.")
            return
        input_path = Path(input_dir).expanduser()
        if not input_path.exists() or not input_path.is_dir():
            messagebox.showerror("rs2ps_HumanMaskTool", f"Input folder not found:\n{input_dir}")
            return

        cmd: List[str] = [
            sys.executable,
            str(self.base_dir / "rs2ps_HumanMaskTool.py"),
            "-i",
            str(input_path),
        ]

        output_dir = self.human_vars["output"].get().strip()
        if output_dir:
            cmd.extend(["-o", output_dir])

        mode_value = self.human_vars["mode"].get().strip()
        if mode_value and mode_value in HUMAN_MODE_CHOICES:
            cmd.extend(["--mode", mode_value])

        dilate_value = self.human_vars["dilate"].get().strip()
        if dilate_value:
            try:
                float(dilate_value)
            except ValueError:
                messagebox.showerror("rs2ps_HumanMaskTool", "Dilation must be numeric.")
                return
            cmd.extend(["--dilate", dilate_value])

        if bool(self.human_vars["cpu"].get()):
            cmd.append("--cpu")
        if bool(self.human_vars["include_shadow"].get()):
            cmd.append("--include_shadow")

        self._run_cli_command(
            cmd,
            self.human_log,
            self.human_run_button,
            process_key="human",
            stop_button=self.human_stop_button,
            cwd=self.base_dir,
        )

    def _run_video_tool(self) -> None:
        if not self.video_vars:
            return
        video_path = self.video_vars["video"].get().strip()
        if not video_path:
            messagebox.showerror("rs2ps_Video2Frames", "Input video is required.")
            return

        fps_value = self.video_vars["fps"].get().strip()
        try:
            fps_float = float(fps_value)
            if fps_float <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("rs2ps_Video2Frames", "FPS must be a positive number.")
            return

        fps_formatted = self._format_fps_for_output(fps_value) or f"{fps_float}"

        cmd: List[str] = [
            sys.executable,
            str(self.base_dir / "rs2ps_Video2Frames.py"),
            "-i",
            video_path,
            "-f",
            fps_formatted,
        ]

        output_dir = self.video_vars["output"].get().strip()
        if output_dir:
            cmd.extend(["-o", output_dir])

        ext_value = self.video_vars["ext"].get().strip()
        if ext_value:
            cmd.extend(["--ext", ext_value])

        prefix_value = self.video_vars.get("prefix")
        if prefix_value is not None:
            prefix_text = prefix_value.get().strip()
            if prefix_text:
                cmd.extend(["--prefix", prefix_text])

        start_value = self.video_vars["start"].get().strip()
        if start_value:
            try:
                float(start_value)
            except ValueError:
                messagebox.showerror("rs2ps_Video2Frames", "Start time must be numeric.")
                return
            cmd.extend(["--start", start_value])

        end_value = self.video_vars["end"].get().strip()
        if end_value:
            try:
                float(end_value)
            except ValueError:
                messagebox.showerror("rs2ps_Video2Frames", "End time must be numeric.")
                return
            cmd.extend(["--end", end_value])

        if bool(self.video_vars["keep_rec709"].get()):
            cmd.append("--keep-rec709")
        if bool(self.video_vars["overwrite"].get()):
            cmd.append("--overwrite")

        ffmpeg_path = self.ffmpeg_path_var.get().strip()
        if ffmpeg_path:
            cmd.extend(["--ffmpeg", ffmpeg_path])

        self._run_cli_command(
            cmd,
            self.video_log,
            self.video_run_button,
            process_key="video",
            stop_button=self.video_stop_button,
            cwd=self.base_dir,
        )

    def _run_frame_selector(self) -> None:
        if not self.selector_vars:
            return

        in_dir = self.selector_vars["in_dir"].get().strip()
        if not in_dir:
            messagebox.showerror("rs2ps_FrameSelector", "Input folder is required.")
            return

        cmd: List[str] = [
            sys.executable,
            str(self.base_dir / "rs2ps_FrameSelector.py"),
            "-i",
            in_dir,
        ]

        segment_size = self.selector_vars["segment_size"].get().strip()
        if segment_size:
            try:
                int(segment_size)
            except ValueError:
                messagebox.showerror("rs2ps_FrameSelector", "Segment size must be an integer.")
                return
            cmd.extend(["-n", segment_size])

        dry_run_required = bool(self.selector_vars["dry_run"].get())

        workers = self.selector_vars["workers"].get().strip()
        if workers:
            try:
                int(workers)
            except ValueError:
                messagebox.showerror("rs2ps_FrameSelector", "Workers must be an integer.")
                return
            cmd.extend(["-w", workers])

        ext_choice = self.selector_vars["ext"].get().strip()
        if ext_choice:
            cmd.extend(["-e", ext_choice])

        sort_choice = self.selector_vars["sort"].get().strip()
        if sort_choice:
            cmd.extend(["-s", sort_choice])

        metric_choice = self.selector_vars["metric"].get().strip()
        if metric_choice:
            cmd.extend(["-m", metric_choice])

        csv_mode = self.selector_vars["csv_mode"].get().strip()
        csv_path = self.selector_vars["csv_path"].get().strip()
        if csv_mode and csv_mode != "none":
            if not csv_path:
                messagebox.showerror("rs2ps_FrameSelector", "CSV path is required for the selected mode.")
                return
            if csv_mode == "write":
                cmd.extend(["-c", csv_path])
            elif csv_mode == "apply":
                cmd.extend(["-a", csv_path])
            elif csv_mode == "reselect":
                cmd.extend(["-r", csv_path])
                dry_run_required = True

        if dry_run_required:
            cmd.append("--dry_run")

        crop_ratio = self.selector_vars["crop_ratio"].get().strip()
        if crop_ratio:
            try:
                float(crop_ratio)
            except ValueError:
                messagebox.showerror("rs2ps_FrameSelector", "Crop ratio must be numeric.")
                return
            cmd.extend(["--crop_ratio", crop_ratio])

        min_spacing = self.selector_vars["min_spacing_frames"].get().strip()
        if min_spacing:
            try:
                int(min_spacing)
            except ValueError:
                messagebox.showerror("rs2ps_FrameSelector", "Min spacing must be an integer.")
                return
            cmd.extend(["--min_spacing_frames", min_spacing])

        if bool(self.selector_vars["prune_motion"].get()):
            cmd.append("--prune_motion")
        if bool(self.selector_vars["augment_gap"].get()):
            cmd.append("--augment_gap")
        if bool(self.selector_vars["augment_lowlight"].get()):
            cmd.append("--augment_lowlight")
        if bool(self.selector_vars["augment_motion"].get()):
            cmd.append("--augment_motion")

        self._run_cli_command(
            cmd,
            self.selector_log,
            self.selector_run_button,
            process_key="selector",
            stop_button=self.selector_stop_button,
            cwd=self.base_dir,
        )

    def _update_ply_adaptive_state(self, *_args) -> None:
        adaptive_var = self.ply_vars.get("adaptive")
        active = bool(adaptive_var.get()) if adaptive_var is not None else False
        entry = self.ply_adaptive_weight_entry
        if entry is not None:
            state = "normal" if active else "disabled"
            try:
                entry.configure(state=state)
            except tk.TclError:
                pass

    def _run_ply_optimizer(self) -> None:
        if not self.ply_vars:
            return
        input_path = self.ply_vars["input"].get().strip()
        if not input_path:
            messagebox.showerror("rs2ps_PlyOptimizer", "Input PLY is required.")
            return

        cmd: List[str] = [
            sys.executable,
            str(self.base_dir / "rs2ps_PlyOptimizer.py"),
            "-i",
            input_path,
        ]

        output_path = self.ply_vars["output"].get().strip()
        if output_path:
            cmd.extend(["-o", output_path])

        target_points = self.ply_vars["target_points"].get().strip()
        if target_points:
            try:
                int(target_points)
            except ValueError:
                messagebox.showerror("rs2ps_PlyOptimizer", "Target points must be an integer.")
                return
            cmd.extend(["-t", target_points])

        target_percent = self.ply_vars["target_percent"].get().strip()
        if target_percent:
            try:
                float(target_percent)
            except ValueError:
                messagebox.showerror("rs2ps_PlyOptimizer", "Target percent must be numeric.")
                return
            cmd.extend(["-r", target_percent])

        voxel_size = self.ply_vars["voxel_size"].get().strip()
        if voxel_size:
            try:
                float(voxel_size)
            except ValueError:
                messagebox.showerror("rs2ps_PlyOptimizer", "Voxel size must be numeric.")
                return
            cmd.extend(["-v", voxel_size])

        adaptive_enabled = bool(self.ply_vars["adaptive"].get())
        if adaptive_enabled:
            cmd.append("--adaptive")
            weight_text = self.ply_vars["adaptive_weight"].get().strip()
            if weight_text:
                try:
                    float(weight_text)
                except ValueError:
                    messagebox.showerror("rs2ps_PlyOptimizer", "Adaptive weight must be numeric.")
                    return
                cmd.extend(["--adaptive-weight", weight_text])

        keep_strategy = self.ply_vars["keep_strategy"].get().strip()
        if keep_strategy:
            cmd.extend(["-k", keep_strategy])

        append_items: List[str] = []
        if self.ply_append_text is not None:
            raw = self.ply_append_text.get("1.0", tk.END).strip()
            if raw:
                append_items = [line.strip() for line in raw.splitlines() if line.strip()]
        for item in append_items:
            cmd.extend(["-a", item])

        self._run_cli_command(
            cmd,
            self.ply_log,
            self.ply_run_button,
            process_key="ply",
            stop_button=self.ply_stop_button,
            cwd=self.base_dir,
        )

    def _select_file(
        self,
        var: tk.Variable,
        title: str = "Select file",
        filetypes: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> None:
        current = var.get().strip() if hasattr(var, "get") else ""
        initialdir = Path(current).parent if current else self.base_dir
        try:
            initialdir = initialdir if initialdir.exists() else self.base_dir
        except Exception:
            initialdir = self.base_dir
        path = filedialog.askopenfilename(
            title=title,
            initialdir=str(initialdir),
            filetypes=filetypes or [("All files", "*.*")],
        )
        if path:
            var.set(path)

    def _on_selector_csv_mode_changed(self, *_args) -> None:
        mode = self.selector_vars.get("csv_mode")
        if mode is None:
            return
        mode_value = mode.get()
        if self.selector_csv_entry is not None:
            if mode_value == "none":
                self.selector_csv_entry.configure(state="disabled")
                self.selector_vars["csv_path"].set("")
            else:
                self.selector_csv_entry.configure(state="normal")
        if self.selector_csv_button is not None:
            if mode_value == "none":
                self.selector_csv_button.configure(state="disabled")
            else:
                self.selector_csv_button.configure(state="normal")
        if self.selector_dry_run_check is not None:
            if mode_value == "reselect":
                self.selector_vars["dry_run"].set(True)
                self.selector_dry_run_check.configure(state="disabled")
            elif mode_value == "apply":
                self.selector_vars["dry_run"].set(False)
                self.selector_dry_run_check.configure(state="disabled")
            else:
                self.selector_dry_run_check.configure(state="normal")

    def _browse_selector_csv(self) -> None:
        if not self.selector_vars:
            return
        mode_value = self.selector_vars.get("csv_mode", tk.StringVar(value="none")).get().strip()
        if mode_value == "none":
            messagebox.showinfo("FrameSelector CSV", "Select a CSV mode before browsing for a file.")
            return
        current_path = self.selector_vars.get("csv_path", tk.StringVar()).get().strip()
        if current_path:
            initial_dir = Path(current_path).expanduser().parent
        else:
            in_dir = self.selector_vars.get("in_dir", tk.StringVar()).get().strip()
            initial_dir = Path(in_dir).expanduser() if in_dir else Path.cwd()
        try:
            if not initial_dir.exists():
                initial_dir = Path.cwd()
        except Exception:
            initial_dir = Path.cwd()

        path: Optional[str]
        if mode_value == "write":
            path = filedialog.asksaveasfilename(
                title="Select CSV output path",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=str(initial_dir),
            )
        else:
            path = filedialog.askopenfilename(
                title="Select CSV file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=str(initial_dir),
            )
        if path:
            self.selector_vars["csv_path"].set(path)

    def _show_selector_scores(self) -> None:
        if not self.selector_vars:
            return
        csv_var = self.selector_vars.get("csv_path")
        csv_path_raw = csv_var.get().strip() if csv_var is not None else ""
        if not csv_path_raw:
            messagebox.showerror("FrameSelector", "Select a CSV file in the Path field.")
            return
        csv_path = Path(csv_path_raw).expanduser()
        try:
            with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                field_map = {name.lower(): name for name in headers if name}
                selected_key = field_map.get("selected(1=keep)") or field_map.get("selected")
                score_key = field_map.get("score")
                filename_key = field_map.get("filename")
                index_key = field_map.get("index")
                if not selected_key or not score_key:
                    raise ValueError("CSV must contain 'selected(1=keep)' (or 'selected') and 'score' columns.")
                selected_entries: List[Tuple[float, str, str]] = []
                for row in reader:
                    if not row:
                        continue
                    flag = str(row.get(selected_key, "")).strip().lower()
                    if flag not in {"1", "true", "yes", "keep"}:
                        continue
                    score_raw = row.get(score_key)
                    try:
                        score_val = float(score_raw)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(score_val):
                        continue
                    fname = row.get(filename_key, "") if filename_key else ""
                    idx_text = row.get(index_key, "") if index_key else ""
                    selected_entries.append((score_val, fname, idx_text))
        except FileNotFoundError:
            messagebox.showerror("FrameSelector", f"CSV file not found:\n{csv_path}")
            return
        except ValueError as exc:
            messagebox.showerror("FrameSelector", str(exc))
            return
        except Exception as exc:  # pragma: no cover - unexpected CSV error
            messagebox.showerror("FrameSelector", f"Failed to read CSV:\n{exc}")
            return

        if not selected_entries:
            self._append_text_widget(
                self.selector_log,
                f"[score] {csv_path.name}: no selected entries with valid scores.",
            )
            return

        sorted_entries = sorted(selected_entries, key=lambda item: item[0])
        try:
            limit_text = self.selector_count_var.get().strip()
            max_lines = int(limit_text) if limit_text else 20
        except (TypeError, ValueError):
            max_lines = 20
        max_lines = max(1, min(max_lines, 200))
        suspects = sorted_entries[:max_lines]

        self._append_text_widget(
            self.selector_log,
            (
                f"[score] {csv_path.name}: selected={len(selected_entries)} "
                f"showing lowest {len(suspects)} scores (limit={max_lines})"
            ),
        )
        if not suspects:
            self._append_text_widget(self.selector_log, "[score] No low-score selections detected.")
            return

        for score, fname, idx_text in suspects:
            label = fname or (f"index {idx_text}" if idx_text else "(unknown)")
            self._append_text_widget(
                self.selector_log,
                f"  - {label} (score={score:.4f})",
            )

    def _select_directory(
        self,
        var: tk.Variable,
        title: str = "Select folder",
        on_select: Optional[Callable[[str], None]] = None,
    ) -> None:
        current = var.get().strip() if hasattr(var, "get") else ""
        initialdir = Path(current) if current else self.base_dir
        try:
            initialdir = initialdir if initialdir.exists() else self.base_dir
        except Exception:
            initialdir = self.base_dir
        path = filedialog.askdirectory(
            title=title,
            initialdir=str(initialdir),
        )
        if path:
            var.set(path)
            if on_select is not None:
                try:
                    on_select(path)
                except Exception:
                    pass

    def _select_save_file(
        self,
        var: tk.Variable,
        title: str,
        defaultextension: str,
        filetypes: Sequence[Tuple[str, str]],
    ) -> None:
        current = var.get().strip() if hasattr(var, "get") else ""
        initialdir = Path(current).parent if current else self.base_dir
        try:
            initialdir = initialdir if initialdir.exists() else self.base_dir
        except Exception:
            initialdir = self.base_dir
        path = filedialog.asksaveasfilename(
            title=title,
            initialdir=str(initialdir),
            defaultextension=defaultextension,
            filetypes=filetypes,
        )
        if path:
            var.set(path)

    def _set_video_mode_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for name in self.video_only_fields:
            widget = self.field_widgets.get(name)
            if widget is not None:
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass
            var = self.field_vars.get(name)
            if isinstance(var, tk.BooleanVar):
                if not enabled:
                    persisted = bool(self.video_persist_state.get(name, False))
                    var.set(persisted)
            elif isinstance(var, tk.StringVar):
                if not enabled:
                    persisted = self.video_persist_state.get(name)
                    var.set("" if persisted in (None, "") else str(persisted))
        if not enabled:
            fps_text = self.field_vars["fps"].get().strip() if "fps" in self.field_vars else ""
            try:
                fps_value = float(fps_text) if fps_text else None
            except ValueError:
                fps_value = None
            if fps_value is not None and fps_value > 0:
                self.video_persist_state["fps"] = fps_value
            self.video_persist_state["keep_rec709"] = bool(self.field_vars["keep_rec709"].get())
            setattr(self.current_args, "fps", None)
            setattr(self.current_args, "keep_rec709", False)
            self._video_preview_signature = None

    def _on_show_seam_toggle(self, var: tk.BooleanVar) -> None:
        value = bool(var.get())
        setattr(self.current_args, "show_seam_overlay", value)
        self.refresh_overlays()

    def _generate_video_preview_image(self, video_path: pathlib.Path) -> Optional[Image.Image]:
        ffmpeg_path = self.ffmpeg_path_var.get().strip() or getattr(self.defaults, "ffmpeg", "ffmpeg")
        if not ffmpeg_path:
            messagebox.showerror("Video Preview", "ffmpeg path is not configured.")
            return None
        keep_rec709 = bool(getattr(self.current_args, "keep_rec709", False))
        colorspace_filter = "colorspace=iall=bt709:all=smpte170m"
        if not keep_rec709:
            colorspace_filter += ":trc=iec61966-2-1"
        colorspace_filter += ":format=yuv444p"
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
        except Exception as exc:
            messagebox.showerror("Video Preview", f"Failed to allocate temporary file.\n{exc}")
            return None
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(video_path),
            "-vf", colorspace_filter,
            "-frames:v", "1",
            str(tmp_path),
        ]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            messagebox.showerror("Video Preview", f"ffmpeg not found: {ffmpeg_path}")
            try:
                tmp_path.unlink()
            except OSError:
                pass
            return None
        except subprocess.CalledProcessError as exc:
            messagebox.showerror(
                "Video Preview",
                f"ffmpeg failed to extract preview frame.\nCommand: {' '.join(cmd)}\n{exc}",
            )
            try:
                tmp_path.unlink()
            except OSError:
                pass
            return None
        try:
            with Image.open(tmp_path) as img:
                return img.convert("RGB")
        except Exception as exc:
            messagebox.showerror("Video Preview", f"Failed to load preview frame.\n{exc}")
            return None
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def _prepare_video_preview(self, video_path: pathlib.Path) -> bool:
        resolved_path = video_path
        try:
            resolved_path = video_path.resolve()
        except FileNotFoundError:
            pass
        keep_rec709 = bool(getattr(self.current_args, "keep_rec709", False))
        signature = (resolved_path, keep_rec709)
        if self._video_preview_signature == signature and self.display_image is not None:
            return True
        preview_image = self._generate_video_preview_image(video_path)
        if preview_image is None:
            return False
        self.pano_image = preview_image
        self.pano_width, self.pano_height = preview_image.size
        try:
            self.scale = compute_scale(
                self.pano_width,
                self.pano_height,
                self.scale_override,
                self.view_max_width,
                self.view_max_height,
            )
        except ValueError as exc:
            messagebox.showerror("Video Preview", str(exc))
            self.scale = 1.0
        self.display_width = int(round(self.pano_width * self.scale))
        self.display_height = int(round(self.pano_height * self.scale))
        if self.scale != 1.0:
            self.display_image = preview_image.resize(
                (self.display_width, self.display_height),
                Image.LANCZOS,
            )
        else:
            self.display_image = preview_image.copy()
        self.update_canvas_image()
        self._video_preview_signature = signature
        self.image_path = video_path
        return True

    def _default_output_path(self) -> Optional[pathlib.Path]:
        if not self.in_dir:
            return None
        in_path = pathlib.Path(self.in_dir)
        try:
            if self.source_is_video or in_path.is_file():
                base_dir = in_path.parent
            else:
                base_dir = in_path
        except Exception:
            base_dir = pathlib.Path.cwd()
        return base_dir / "_geometry"

    def set_form_values(self) -> None:
        self.form_snapshot.clear()
        for definition in self.FIELD_DEFS:
            name = definition["name"]
            field_type = definition["type"]
            if definition.get("video_only") and not self.source_is_video:
                if field_type == "bool":
                    var = self.field_vars[name]
                    assert isinstance(var, tk.BooleanVar)
                    var.set(False)
                    self.form_snapshot[name] = "0"
                else:
                    var = self.field_vars[name]
                    assert isinstance(var, tk.StringVar)
                    var.set("")
                    self.form_snapshot[name] = ""
                continue
            value = getattr(self.current_args, name)
            if field_type == "bool":
                var = self.field_vars[name]
                assert isinstance(var, tk.BooleanVar)
                var.set(bool(value))
                self.form_snapshot[name] = str(int(bool(value)))
                continue
            if field_type == "choice":
                var = self.field_vars[name]
                assert isinstance(var, tk.StringVar)
                choice = str(value) if value else definition["choices"][0]
                if choice not in definition["choices"]:
                    choice = definition["choices"][0]
                var.set(choice)
                self.form_snapshot[name] = choice
                continue
            var = self.field_vars[name]
            assert isinstance(var, tk.StringVar)
            display = "" if value is None else str(value)
            var.set(display)
            self.form_snapshot[name] = display
        self.ffmpeg_path_var.set(str(getattr(self.current_args, "ffmpeg", getattr(self.defaults, "ffmpeg", "ffmpeg"))))
        self.ffthreads_var.set(str(getattr(self.current_args, "ffthreads", getattr(self.defaults, "ffthreads", "1"))))
        self.jobs_var.set(str(getattr(self.current_args, "jobs", getattr(self.defaults, "jobs", "auto"))))
        if self.out_dir:
            self.output_path_var.set(str(self.out_dir))
        elif self.in_dir:
            default_out = self._default_output_path()
            self.output_path_var.set(str(default_out) if default_out is not None else "")
        else:
            self.output_path_var.set("")


    def _bind_help(self, widget: tk.Widget, key: str) -> None:
        text = FIELD_HELP_TEXT.get(key)
        if not text:
            return
        tooltip = ToolTip(widget, text)
        self._tooltips.append(tooltip)


    def collect_updated_args(self) -> Optional[argparse.Namespace]:
        updated = clone_namespace(self.current_args)
        ensure_explicit_flags(updated)

        def previous_flag(attr: str) -> bool:
            return bool(getattr(self.current_args, f"{attr}_explicit", False))

        for definition in self.FIELD_DEFS:
            name = definition["name"]
            field_type = definition["type"]
            default_value = getattr(self.defaults, name, None)
            if definition.get("video_only") and not self.source_is_video:
                if field_type == "bool":
                    setattr(updated, name, False)
                else:
                    setattr(updated, name, None)
                continue
            if field_type == "bool":
                var = self.field_vars[name]
                assert isinstance(var, tk.BooleanVar)
                setattr(updated, name, bool(var.get()))
                continue
            if field_type == "choice":
                var = self.field_vars[name]
                assert isinstance(var, tk.StringVar)
                choice = var.get().strip() or definition["choices"][0]
                if choice not in definition["choices"]:
                    messagebox.showerror(
                        "Input Error",
                        f"{definition['label']}: invalid value '{choice}'",
                    )
                    return None
                setattr(updated, name, choice)
                self.form_snapshot[name] = choice
                continue
            var = self.field_vars[name]
            assert isinstance(var, tk.StringVar)
            raw = var.get().strip()
            snapshot = self.form_snapshot.get(name, "")
            try:
                if field_type == "int":
                    if not raw:
                        if default_value is None:
                            raise ValueError("Please enter a value")
                        value = int(default_value)
                    else:
                        value = int(raw)
                    if name == "count" and value <= 0:
                        raise ValueError("count must be >= 1")
                    setattr(updated, name, value)
                    self.form_snapshot[name] = str(value)
                elif field_type == "float":
                    if not raw:
                        if default_value is None:
                            raise ValueError("Please enter a value")
                        value = float(default_value)
                    else:
                        value = float(raw)
                    setattr(updated, name, value)
                    self.form_snapshot[name] = str(value)
                elif field_type == "float_optional":
                    if raw:
                        value = float(raw)
                        if name == "fps" and value <= 0:
                            raise ValueError("must be > 0")
                        setattr(updated, name, value)
                        setattr(
                            updated,
                            f"{name}_explicit",
                            previous_flag(name) if raw == snapshot else True,
                        )
                        self.form_snapshot[name] = raw
                    else:
                        if definition.get("required_if_video") and self.source_is_video:
                            raise ValueError("Please enter a value")
                        setattr(updated, name, default_value)
                        setattr(updated, f"{name}_explicit", False)
                        self.form_snapshot[name] = "" if default_value is None else str(default_value)
                else:
                    if not raw:
                        value = "" if default_value is None else str(default_value)
                        setattr(updated, name, value)
                        self.form_snapshot[name] = value
                    else:
                        setattr(updated, name, raw)
                        self.form_snapshot[name] = raw
            except ValueError as exc:
                messagebox.showerror("Input Error", f"{definition['label']}: {exc}")
                return None

            if field_type in {"int", "float"} and name in self.EXPLICIT_FIELDS:
                setattr(
                    updated,
                    f"{name}_explicit",
                    previous_flag(name) if raw == snapshot and raw else bool(raw),
                )

        ffmpeg_path = self.ffmpeg_path_var.get().strip()
        if not ffmpeg_path:
            ffmpeg_path = getattr(self.defaults, "ffmpeg", "ffmpeg")
        updated.ffmpeg = ffmpeg_path
        self.ffmpeg_path_var.set(ffmpeg_path)

        ffthreads_value = self.ffthreads_var.get().strip()
        if not ffthreads_value:
            ffthreads_value = str(getattr(self.defaults, "ffthreads", "1"))
        updated.ffthreads = ffthreads_value
        self.ffthreads_var.set(ffthreads_value)

        jobs_value = self.jobs_var.get().strip()
        if not jobs_value:
            jobs_value = str(getattr(self.defaults, "jobs", "auto"))
        updated.jobs = jobs_value
        self.jobs_var.set(jobs_value)

        out_dir_text = self.output_path_var.get().strip()
        if out_dir_text:
            out_dir_path = Path(out_dir_text).expanduser()
            self.out_dir = out_dir_path
            normalized = str(out_dir_path)
            updated.out_dir = normalized
            self.output_path_var.set(normalized)
        else:
            self.out_dir = None
            updated.out_dir = None
            self.output_path_var.set("")

        return updated

    def _apply_preset_defaults(self, preset_value: str) -> None:
        preset_defaults: Dict[str, Dict[str, Any]] = {
            "fisheyelike": {"count": 10, "focal_mm": 17.0},
            "2views": {"size": 3600, "focal_mm": 6.0},
            "fisheyeXY": {"count": 8},
        }
        values = preset_defaults.get(preset_value)
        if not values:
            return
        # Reset HFOV to allow focal-based presets to take effect.
        self.current_args.hfov = None
        if hasattr(self.current_args, "hfov_explicit"):
            setattr(self.current_args, "hfov_explicit", False)
        for field_name, field_value in values.items():
            setattr(self.current_args, field_name, field_value)
            explicit_flag = f"{field_name}_explicit"
            if hasattr(self.current_args, explicit_flag):
                setattr(self.current_args, explicit_flag, False)

    def on_preset_changed(self, selection: str) -> None:
        preset_value = selection or self.field_vars["preset"].get()
        current_ffmpeg = self.ffmpeg_path_var.get().strip() or getattr(self.defaults, "ffmpeg", "ffmpeg")
        current_ffthreads = self.ffthreads_var.get().strip() or str(getattr(self.defaults, "ffthreads", "1"))
        current_jobs = self.jobs_var.get().strip() or str(getattr(self.defaults, "jobs", "auto"))
        video_prev_values = {
            "fps": getattr(self.current_args, "fps", None),
            "keep_rec709": getattr(self.current_args, "keep_rec709", False),
        }
        seam_prev = getattr(self.current_args, "show_seam_overlay", False)
        self.current_args = clone_namespace(self.defaults)
        ensure_explicit_flags(self.current_args)
        self.current_args.preset = preset_value
        self.current_args.ffmpeg = current_ffmpeg
        self.current_args.ffthreads = current_ffthreads
        self.current_args.jobs = current_jobs
        self.current_args.show_seam_overlay = seam_prev
        self._apply_preset_defaults(preset_value)
        if getattr(self.current_args, "input_is_video", False) or self.source_is_video:
            fps_value = video_prev_values.get("fps")
            if fps_value is None or fps_value <= 0:
                fps_value = self.video_persist_state.get("fps", 2.0)
            self.current_args.fps = fps_value
            setattr(self.current_args, "fps_explicit", True)
            keep_val = bool(video_prev_values.get("keep_rec709", False))
            self.current_args.keep_rec709 = keep_val
        self.set_form_values()
        self.field_vars["preset"].set(preset_value)
        if getattr(self.current_args, "input_is_video", False) or self.source_is_video:
            fps_current = self.current_args.fps if getattr(self.current_args, "fps", None) else self.video_persist_state.get("fps", 2.0)
            self.field_vars["fps"].set(str(fps_current))
            self.field_vars["keep_rec709"].set(bool(self.current_args.keep_rec709))
        updated = self.collect_updated_args()
        if updated is None:
            return
        self.current_args = updated
        ensure_explicit_flags(self.current_args)
        self.refresh_overlays()

    def load_directory_from_entry(self) -> None:
        path_value = self.folder_path_var.get().strip()
        if not path_value:
            messagebox.showerror("Input Source", "Please enter a folder or video file path, or use Browse.")
            return
        if self.try_load_directory(path_value):
            self.refresh_overlays(initial=True)
            if self.out_dir:
                self.output_path_var.set(str(self.out_dir))
            elif self.in_dir:
                default_out = self._default_output_path()
                self.output_path_var.set(str(default_out) if default_out is not None else "")

    def browse_ffmpeg(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select ffmpeg executable",
            filetypes=[("ffmpeg", "ffmpeg.exe"), ("All files", "*.*")]
        )
        if filename:
            self.ffmpeg_path_var.set(filename)

    def reset_ffmpeg_path(self) -> None:
        default_value = getattr(self.defaults, "ffmpeg", "ffmpeg")
        self.ffmpeg_path_var.set(default_value)

    def browse_output_directory(self) -> None:
        if self.out_dir:
            initial_path = self.out_dir
        elif self.in_dir:
            initial_path = self.in_dir.parent if self.source_is_video else self.in_dir
        else:
            initial_path = Path.cwd()
        initial = str(initial_path)
        selected = filedialog.askdirectory(parent=self.root, initialdir=initial)
        if selected:
            path_obj = Path(selected).expanduser()
            self.out_dir = path_obj
            self.current_args.out_dir = str(path_obj)
            self.output_path_var.set(str(path_obj))

    def prompt_for_directory(self, initial: bool = False) -> None:
        initial_path = self.in_dir
        if initial_path is None:
            initial_path = pathlib.Path.cwd()
        else:
            try:
                initial_path = initial_path if initial_path.is_dir() else initial_path.parent
            except Exception:
                initial_path = pathlib.Path.cwd()
        selected_dir = filedialog.askdirectory(parent=self.root, initialdir=str(initial_path))
        if selected_dir:
            if self.try_load_directory(selected_dir):
                self.folder_path_var.set(str(self.in_dir))
                self.refresh_overlays(initial=True)
            return
        video_file = filedialog.askopenfilename(
            parent=self.root,
            initialdir=str(initial_path),
            title="Select input video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv *.wmv *.mpg *.mpeg *.mxf"),
                ("All files", "*.*"),
            ],
        )
        if not video_file:
            if initial and not self.files:
                self.set_log_text("No folder or video selected.")
            return
        if self.try_load_directory(video_file):
            self.folder_path_var.set(str(self.in_dir))
            self.refresh_overlays(initial=True)

    def _load_video_source(self, source: pathlib.Path) -> bool:
        video_path = pathlib.Path(source).expanduser()
        try:
            video_path = video_path.resolve()
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {source}")
            return False
        if not video_path.is_file():
            messagebox.showerror("Error", f"File not found: {video_path}")
            return False
        self.in_dir = video_path
        self.files = [video_path]
        self.current_args.input_dir = str(video_path)
        setattr(self.current_args, "input_is_video", True)
        setattr(
            self.current_args,
            "video_bit_depth",
            cutter.detect_input_bit_depth(video_path),
        )
        default_out = self._default_output_path()
        if getattr(self.current_args, "out_dir", None):
            try:
                self.out_dir = pathlib.Path(self.current_args.out_dir).expanduser().resolve()
            except Exception:
                self.out_dir = default_out
                if self.out_dir is not None:
                    self.current_args.out_dir = str(self.out_dir)
        else:
            self.out_dir = default_out
            if self.out_dir is not None:
                self.current_args.out_dir = str(self.out_dir)
        fps_current = getattr(self.current_args, "fps", None)
        if fps_current is None or fps_current <= 0:
            self.current_args.fps = 2.0
            setattr(self.current_args, "fps_explicit", True)
        if not self._prepare_video_preview(video_path):
            self.source_is_video = False
            self.files = []
            self.image_path = None
            self.pano_image = None
            self.display_image = None
            return False
        self.source_is_video = True
        self.folder_path_var.set(str(video_path))
        self._set_video_mode_controls(True)
        self.set_form_values()
        return True

    def try_load_directory(
        self,
        directory: pathlib.Path,
        image_hint: Optional[str] = None,
    ) -> bool:
        path = pathlib.Path(directory).expanduser()
        try:
            path = path.resolve()
        except FileNotFoundError:
            messagebox.showerror("Error", f"Path not found: {directory}")
            return False
        if path.is_file():
            return self._load_video_source(path)
        if not path.is_dir():
            messagebox.showerror("Error", f"Folder not found: {path}")
            return False

        files = [
            p
            for p in sorted(path.iterdir())
            if p.is_file() and p.suffix.lower() in cutter.EXTS
        ]
        if not files:
            messagebox.showerror("Error", "No panorama images found (tif/jpg/png).")
            return False

        self.source_is_video = False
        self.in_dir = path
        self.files = files
        self.current_args.input_dir = str(path)
        setattr(self.current_args, "input_is_video", False)
        setattr(self.current_args, "video_bit_depth", 8)

        if getattr(self.current_args, "out_dir", None):
            self.out_dir = pathlib.Path(self.current_args.out_dir).expanduser().resolve()
        else:
            default_out = self._default_output_path()
            self.out_dir = default_out
            if self.out_dir is not None:
                self.current_args.out_dir = str(self.out_dir)

        if image_hint:
            hint_path = path / pathlib.Path(image_hint).name
            self.image_path = hint_path if hint_path in files else files[0]
        else:
            self.image_path = files[0]

        self.folder_path_var.set(str(self.in_dir))
        self._set_video_mode_controls(False)
        self.load_image()
        self.set_form_values()
        return True

    def load_image(self) -> None:
        if self.image_path is None:
            return
        try:
            with Image.open(self.image_path) as img:
                pano = img.copy()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to open image:\n{self.image_path}\n{exc}")
            return
        if pano.mode not in {"RGB", "RGBA", "L"}:
            try:
                pano = pano.convert("RGB")
            except Exception as exc:
                messagebox.showerror("Error", f"Unsupported image mode '{pano.mode}' for {self.image_path.name}:\n{exc}")
                return
        elif pano.mode == "L":
            # Ensure consistent 8-bit grayscale data
            pano = pano.convert("L")

        self.pano_image = pano
        self.pano_width, self.pano_height = self.pano_image.size

        try:
            self.scale = compute_scale(
                self.pano_width,
                self.pano_height,
                self.scale_override,
                self.view_max_width,
                self.view_max_height,
            )
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            self.scale = 1.0

        self.display_width = int(round(self.pano_width * self.scale))
        self.display_height = int(round(self.pano_height * self.scale))
        if self.scale != 1.0:
            self.display_image = self.pano_image.resize(
                (self.display_width, self.display_height),
                Image.LANCZOS,
            )
        else:
            self.display_image = self.pano_image.copy()
        self.update_canvas_image()
        self._sync_panel_heights()

    def update_canvas_image(self) -> None:
        if self.canvas is None or self.display_image is None:
            return
        self.canvas.delete("all")
        self.photo = ImageTk.PhotoImage(self.display_image, master=self.root)
        self.canvas.configure(width=self.display_width, height=self.display_height)
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.image = self.photo
        self._sync_panel_heights()

    def build_cli_command_line(self) -> str:
        parts = ["python", "rs2ps_360PerspCut.py"]
        if self.in_dir:
            parts.extend(["-i", str(self.in_dir)])
        if self.out_dir:
            default_out = self._default_output_path()
            if self.current_args.out_dir:
                parts.extend(["-o", str(self.out_dir)])
            elif default_out is not None and self.out_dir != default_out:
                parts.extend(["-o", str(self.out_dir)])

        ffmpeg_value = getattr(self.current_args, "ffmpeg", "")
        ffmpeg_default = getattr(self.defaults, "ffmpeg", "ffmpeg")
        if ffmpeg_value and ffmpeg_value != ffmpeg_default:
            parts.extend(["--ffmpeg", ffmpeg_value])

        defaults = self.defaults
        option_map = [
            ("--preset", "preset"),
            ("--count", "count"),
            ("--addcam", "addcam"),
            ("--addcam-deg", "addcam_deg"),
            ("--delcam", "delcam"),
            ("--setcam", "setcam"),
            ("--size", "size"),
            ("--sensor-mm", "sensor_mm"),
            ("--ext", "ext"),
            ("-f", "fps"),
            ("--jobs", "jobs"),
            ("--ffthreads", "ffthreads"),
            ("--print-cmd", "print_cmd"),
        ]
        for flag, name in option_map:
            value = getattr(self.current_args, name, None)
            if value in (None, ""):
                continue
            if name == "addcam" and not str(value).strip():
                continue
            if name in {"delcam", "setcam"} and not str(value).strip():
                continue
            if name == "jobs" and str(value).lower() == "auto":
                continue
            if name == "print_cmd" and value == "once":
                continue
            if name == "ext" and str(value).lower() in {"jpg", "jpeg"}:
                continue
            if name == "ffthreads" and str(value).lower() in {"1", "auto"}:
                continue
            default_value = getattr(defaults, name, None)
            if name in self.EXPLICIT_FIELDS:
                if not getattr(self.current_args, f"{name}_explicit", False) and value == default_value:
                    continue
            else:
                if value == default_value:
                    continue
            parts.extend([flag, str(value)])

        if getattr(self.current_args, "hfov", None) is not None:
            parts.extend(["--hfov", str(self.current_args.hfov)])
        else:
            focal_default = getattr(self.original_args, "focal_mm", None)
            if getattr(self.current_args, "focal_mm_explicit", False) or self.current_args.focal_mm != focal_default:
                parts.extend(["--focal-mm", str(self.current_args.focal_mm)])

        if getattr(self.current_args, "add_topdown", False):
            parts.append("--add-topdown")
        if self.source_is_video and getattr(self.current_args, "keep_rec709", False):
            parts.append("--keep-rec709")
        if getattr(self.current_args, "dry_run", False):
            parts.append("--dry-run")

        return "CLI> " + " ".join(shlex.quote(str(token)) for token in parts)

    def refresh_overlays(self, initial: bool = False) -> None:
        if not self.in_dir or not self.files:
            return
        source_is_video = bool(self.source_is_video)
        if source_is_video:
            video_path = self.files[0]
            if not self._prepare_video_preview(video_path):
                return
            fps_value = getattr(self.current_args, "fps", None)
            if fps_value is None or fps_value <= 0:
                if self.canvas is not None:
                    self.canvas.delete("overlay")
                self.set_log_text("Please enter a positive FPS for the video source and press Update.")
                return
            try:
                target_path = video_path.resolve()
            except FileNotFoundError:
                target_path = video_path
        else:
            if self.image_path is None:
                return
            if self.display_image is None:
                self.load_image()
                if self.display_image is None:
                    return
            target_path = self.image_path.resolve()

        args_for_build = clone_namespace(self.current_args)
        ensure_explicit_flags(args_for_build)
        args_for_build.input_dir = str(self.in_dir)
        args_for_build.out_dir = str(self.out_dir) if self.out_dir else None
        setattr(args_for_build, "input_is_video", source_is_video)
        setattr(
            args_for_build,
            "video_bit_depth",
            getattr(self.current_args, "video_bit_depth", 8),
        )

        default_out = self._default_output_path()
        output_root = self.out_dir or default_out or self.in_dir

        self.result = cutter.build_view_jobs(
            args_for_build,
            self.files,
            output_root,
        )
        self.matched_specs = [
            spec
            for spec in self.result.view_specs
            if spec.source_path.resolve() == target_path
        ]

        if self.canvas is not None:
            self.canvas.delete("overlay")
            if self.display_image is not None:
                draw_items: List[Tuple[List[List[Tuple[float, float]]], Tuple[float, float], str, str]] = []
                for spec, color in zip(self.matched_specs, itertools.cycle(COLOR_CYCLE)):
                    segments, centre = sample_view_segments(
                        spec,
                        self.pano_width,
                        self.pano_height,
                        self.samples,
                    )
                    draw_items.append((segments, centre, color, spec.view_id))
                seam_line_specs: Optional[Tuple[List[float], int]] = None
                if getattr(self.current_args, "show_seam_overlay", False) and self.display_width > 0:
                    def yaw_to_x(yaw_deg: float) -> float:
                        yaw_norm = ((yaw_deg + 180.0) % 360.0) - 180.0
                        return (yaw_norm + 180.0) / 360.0 * self.display_width

                    mid_yaws = [90.0, -90.0]
                    line_width = max(4, int(self.display_width * 0.01))
                    seam_line_specs = ([yaw_to_x(yaw_center) for yaw_center in mid_yaws], line_width)

                for segments, centre, color, view_id in draw_items:
                    for segment in segments:
                        if len(segment) < 2:
                            continue
                        flat_coords: List[float] = []
                        for px, py in segment:
                            flat_coords.extend([px * self.scale, py * self.scale])
                        self.canvas.create_line(
                            flat_coords,
                            fill=color,
                            width=2,
                            tags=("overlay",),
                        )
                    if not self.hide_labels:
                        self.canvas.create_text(
                            centre[0] * self.scale,
                            centre[1] * self.scale,
                            text=view_id,
                            fill=color,
                            font=("TkDefaultFont", 10, "bold"),
                            tags=("overlay",),
                        )
                if seam_line_specs:
                    seam_xs, line_width = seam_line_specs
                    for x in seam_xs:
                        self.canvas.create_line(
                            x,
                            0,
                            x,
                            self.display_height,
                            fill="#000000",
                            width=line_width,
                            tags=("overlay", "seam_line"),
                        )

        info_lines: List[str] = [self.build_cli_command_line()]

        try:
            jobs_parallel = cutter.parse_jobs(self.current_args.jobs)
        except Exception:
            jobs_parallel = self.current_args.jobs
        info_lines.append(
            f"[INFO] parallel jobs: {jobs_parallel} "
            f"/ ffthreads: {self.current_args.ffthreads} / total: {len(self.result.jobs)}"
        )
        if self.result.preview_views_line:
            info_lines.append(self.result.preview_views_line)
        if self.result.sensor_line:
            info_lines.append(self.result.sensor_line)
        if self.result.realityscan_line:
            info_lines.append(self.result.realityscan_line)
        if self.result.metashape_line:
            info_lines.append(self.result.metashape_line)

        if self.matched_specs:
            info_lines.append("")
            info_lines.append(f"[{target_path.name}] view list:")
            info_lines.extend(build_info_lines(self.matched_specs))
        else:
            info_lines.append("")
            info_lines.append("No views defined for the current source with these settings.")

        self.set_log_text("\n".join(info_lines))

        if not initial:
            self.set_form_values()

        self._sync_panel_heights()

    def on_update(self) -> None:
        updated = self.collect_updated_args()
        if updated is None:
            return
        self.current_args = updated
        ensure_explicit_flags(self.current_args)
        self.refresh_overlays()

    def on_execute(self) -> None:
        if self.is_executing:
            return
        updated = self.collect_updated_args()
        if updated is None:
            return
        self.current_args = updated
        ensure_explicit_flags(self.current_args)
        self.result = None
        if self.result is None or not self.result.jobs:
            self.refresh_overlays()
        if self.result is None or not self.result.jobs:
            messagebox.showinfo("Information", "No jobs to run with the current settings.")
            return
        if not messagebox.askyesno(
            "Confirm",
            "Run export with the current settings?\nFFmpeg will write results to the output folder.",
        ):
            return
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        cutter.stop_event.clear()
        self.is_executing = True
        if self.update_button is not None:
            self.update_button.configure(state="disabled")
        if self.execute_button is not None:
            self.execute_button.configure(state="disabled")
        if self.preview_stop_button is not None:
            self.preview_stop_button.configure(state="normal")
        self.append_log_line("[EXEC] Starting export...")
        jobs_snapshot = list(self.result.jobs)
        thread = threading.Thread(
            target=self._run_execute_jobs,
            args=(jobs_snapshot,),
            daemon=True,
        )
        thread.start()

    def _run_execute_jobs(self, jobs: List[Tuple[List[str], str, str]]) -> None:
        workers = cutter.parse_jobs(getattr(self.current_args, "jobs", "auto"))
        total = len(jobs)
        if total == 0:
            self.root.after(0, self._on_execute_finished, 0, 0, 0, [])
            return
        ok = 0
        fail = 0
        errors: List[str] = []
        done = 0
        last_pct = -1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(cutter.run_one, cmd): (src, dst)
                for cmd, src, dst in jobs
            }
            for future in as_completed(future_map):
                rc, err = future.result()
                done += 1
                if rc == 0:
                    ok += 1
                else:
                    fail += 1
                    err_text = (err or "").strip()
                    if err_text:
                        errors.append(err_text)
                pct = int((done * 100) / total)
                if pct == 100 or last_pct < 0 or (pct - last_pct) >= cutter.PROGRESS_INTERVAL:
                    last_pct = pct
                    self.root.after(0, self._log_progress, pct, done, total)
        self.root.after(0, self._on_execute_finished, ok, fail, len(jobs), errors)

    def _log_progress(self, pct: int, done: int, total: int) -> None:
        self.append_log_line(f"Progress... {pct:3d}% ({done}/{total})")

    def _on_execute_finished(
        self,
        ok: int,
        fail: int,
        total: int,
        errors: List[str],
    ) -> None:
        self.is_executing = False
        if self.update_button is not None:
            self.update_button.configure(state="normal")
        if self.execute_button is not None:
            self.execute_button.configure(state="normal")
        if self.preview_stop_button is not None:
            self.preview_stop_button.configure(state="disabled")
        summary = f"[EXEC] Done: succeeded={ok}, failed={fail}, total={total}"
        self.append_log_line(summary)
        for line in errors:
            self.append_log_line(line)
        if errors:
            messagebox.showwarning("Complete", summary + "\nCheck the log for error details.")
        else:
            messagebox.showinfo("Complete", summary)


    def _ensure_window_min_dimensions(self) -> None:
        if self.root is None:
            return
        self.root.update_idletasks()
        if self.controls_frame is not None:
            controls_width = max(
                self.controls_frame.winfo_width(),
                self.controls_frame.winfo_reqwidth(),
                460,
            )
        else:
            controls_width = 460
        total_width = int(self.display_width + controls_width + 80)
        total_height = int(self.display_height + 320)
        current_w = self.root.winfo_width()
        current_h = self.root.winfo_height()
        new_w = max(current_w, total_width)
        self.root.minsize(total_width, total_height)
        if new_w != current_w or total_height > current_h:
            new_h = max(current_h, total_height)
            self.root.geometry(f"{new_w}x{new_h}")

    def _sync_panel_heights(self) -> None:
        if self.left_frame is None or self.right_inner is None:
            return
        self.root.update_idletasks()
        right_height = max(self.right_inner.winfo_height(), self.right_inner.winfo_reqheight())
        desired_inner_height = max(self.display_height, right_height)
        self.left_frame.configure(height=desired_inner_height, width=self.display_width)
        self.canvas.configure(height=self.display_height, width=self.display_width)
        if self.preview_frame is not None:
            self.preview_frame.configure(width=self.display_width + 16)
        if self.left_frame is not None:
            self.left_frame.configure(width=self.display_width)
        if self.canvas is not None:
            self.canvas.configure(width=self.display_width)
        if self.controls_frame is not None:
            frame_height = max(self.controls_frame.winfo_height(), self.controls_frame.winfo_reqheight())
            inner_height = max(self.right_inner.winfo_height(), self.right_inner.winfo_reqheight())
            self._controls_frame_padding = max(0, frame_height - inner_height)
            controls_height = desired_inner_height + self._controls_frame_padding
            self.controls_frame.configure(height=controls_height)

        if self.preview_frame is not None:
            frame_height = max(self.preview_frame.winfo_height(), self.preview_frame.winfo_reqheight())
            inner_height = max(self.left_frame.winfo_height(), self.left_frame.winfo_reqheight())
            self._preview_frame_padding = max(0, frame_height - inner_height)
            preview_height = desired_inner_height + self._preview_frame_padding
            self.preview_frame.configure(height=preview_height)
        self._ensure_window_min_dimensions()

    def set_log_text(self, text: str) -> None:
        if self.log_text is None:
            return
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert("1.0", text)
        self.log_text.see("1.0")

    def append_log_line(self, line: str) -> None:
        if self.log_text is None:
            return
        current = self.log_text.get("1.0", tk.END).strip()
        prefix = "\n" if current else ""
        self.log_text.insert(tk.END, prefix + line)
        self.log_text.see(tk.END)

    @staticmethod
    def _block_text_edit(event) -> Optional[str]:
        if getattr(event, "state", 0) & 0x4 and event.keysym.lower() == "c":
            return None
        return "break"

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_arguments()
    ensure_explicit_flags(args)
    app = PreviewApp(args)
    app.run()


if __name__ == "__main__":
    main()

 
