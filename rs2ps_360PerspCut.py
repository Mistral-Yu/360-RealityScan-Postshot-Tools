#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rs2ps_360PerspCut converts 360-degree panoramas into perspective or fisheye crops
by orchestrating ffmpeg runs.
"""

import argparse, json, math, pathlib, subprocess, sys, os, signal, threading, shlex, re, shutil
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

PROGRESS_INTERVAL = 5


class StoreWithFlag(argparse.Action):
    """Argparse action that records whether the value was explicitly set."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


@dataclass
class ViewSpec:
    """Perspective or fisheye view definition derived from CLI options."""

    source_path: pathlib.Path
    output_name: str
    view_id: str
    yaw_deg: float
    pitch_deg: float
    hfov_deg: float
    vfov_deg: float
    width: int
    height: int
    projection: str = "perspective"


@dataclass
class BuildResult:
    """Container for generated ffmpeg jobs and associated metadata."""

    jobs: List[Tuple[List[str], str, str]]
    view_specs: List[ViewSpec]
    focal_used_mm: float
    focal_35mm_equiv: Optional[float]
    hfov_deg: float
    vfov_deg: float
    preview_views_line: str
    sensor_line: str
    realityscan_line: str
    metashape_line: str

    @property
    def total(self) -> int:
        return len(self.jobs)

def update_progress(label: str, completed: int, total: int, last_pct: int) -> int:
    if total <= 0:
        return last_pct
    pct = int((completed * 100) / total)
    if last_pct < 0 or pct >= 100 or (pct - last_pct) >= PROGRESS_INTERVAL:
        sys.stdout.write(f"{label}... {pct:3d}% ({completed}/{total})\r")
        sys.stdout.flush()
        return pct
    return last_pct

def fov_from_focal_mm(f_mm: float, sensor_w_mm: float) -> float:
    return math.degrees(2.0 * math.atan(sensor_w_mm / (2.0 * f_mm)))

def focal_from_hfov_deg(hfov_deg: float, sensor_w_mm: float) -> float:
    return sensor_w_mm / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))

def v_fov_from_hfov(hfov_deg: float, w: int, h: int) -> float:
    hfov_rad = math.radians(hfov_deg)
    vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * (h / float(w)))
    return math.degrees(vfov_rad)

def letter_tag(idx: int) -> str:
    base = ord('A')
    return chr(base + idx) if idx < 26 else f"{idx+1:02d}"

def letter_to_index1(s: str) -> int:
    s = s.strip()
    if not s: raise ValueError("empty key")
    if s.isdigit(): return int(s)
    ch = s.upper()[0]
    if 'A' <= ch <= 'Z': return (ord(ch) - ord('A')) + 1
    raise ValueError("invalid key: " + s)

def normalize_angle_deg(a: float) -> float:
    a = ((a + 180.0) % 360.0) - 180.0
    return 180.0 if abs(a + 180.0) < 1e-6 else a

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def map_interp_for_v360(name: str) -> str:
    name = (name or "").lower()
    return {"bicubic":"cubic","bilinear":"linear","lanczos":"lanczos"}.get(name, "cubic")

def detect_input_bit_depth(in_path: pathlib.Path) -> int:
    """Detect the video's nominal bit depth using ffprobe metadata."""
    if not shutil.which("ffprobe"):
        return 8
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=bits_per_raw_sample,pix_fmt",
        "-of", "json",
        str(in_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads(result.stdout or "{}")
        streams = info.get("streams") or [{}]
        stream = streams[0]
        bits_raw = stream.get("bits_per_raw_sample")
        if isinstance(bits_raw, str) and bits_raw.isdigit():
            value = int(bits_raw)
            return value if value >= 9 else 8
        pix_fmt = stream.get("pix_fmt") or ""
        if any(token in pix_fmt for token in (
            "p10", "p12", "p14", "p16",
            "yuv420p10", "yuv422p10", "yuv444p10",
            "yuv420p12", "yuv422p12", "yuv444p12",
            "p010", "p012", "p016",
            "gbrp10", "gbrp12", "gbrp14", "gbrp16",
            "rgb48", "rgba64",
        )):
            return 10
    except Exception:
        pass
    return 8

def parse_sensor(s: str) -> float:
    s = s.lower().replace("\u00d7", "x").replace(",", " ").strip()
    w = s.split("x")[0].strip() if "x" in s else s.split()[0]
    return float(w)

def parse_sensor_dimensions(s: str) -> Tuple[float, ...]:
    """
    Return all numeric components from a sensor string such as '36 24' or '36x24'.
    """
    s_norm = s.lower().replace("\u00d7", "x").replace(",", " ").strip()
    if "x" in s_norm:
        tokens = [t.strip() for t in s_norm.split("x") if t.strip()]
    else:
        tokens = [t for t in s_norm.split() if t]
    dims: List[float] = []
    for token in tokens:
        try:
            dims.append(float(token))
        except ValueError:
            continue
    return tuple(dims)

def extra_suffix(delta_pitch: float, default_deg: float=30.0) -> str:
    sign = "_U" if delta_pitch > 0 else "_D"
    mag = abs(delta_pitch)
    if abs(mag - default_deg) < 1e-6:
        return sign
    if float(mag).is_integer():
        return f"{sign}{int(round(mag))}"
    return f"{sign}{mag:g}"

# ---- add/del/setcam parser ----
def parse_addcam_spec(spec: str, default_deg: float) -> Dict[int, List[float]]:

    out: Dict[int, List[float]] = {}
    if not spec:
        return out
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token or "=" in token:
            k, v = re.split(r"[:=]", token, maxsplit=1)
            idx1 = letter_to_index1(k)
            v = v.strip().upper()
            m = re.match(r'^([UD])\s*([+-]?\d+(?:\.\d+)?)?$', v)
            if m:
                # absolute tokens: U/D with optional magnitude
                deg = float(m.group(2)) if m.group(2) else default_deg
                delta = +deg if m.group(1) == 'U' else -deg
                out.setdefault(idx1, []).append(delta)
            else:
                # unsupported pattern such as ': +10'
                raise ValueError("invalid --addcam token: " + token)
        else:
            idx1 = letter_to_index1(token)
            out.setdefault(idx1, []).extend([+default_deg, -default_deg])
    return out

def parse_delcam_spec(spec: str) -> Set[int]:
    s: Set[int] = set()
    if not spec:
        return s
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        s.add(letter_to_index1(token))
    return s

def parse_setcam_spec(spec: str, default_deg: float):
    """
    Parse --setcam specification.

    Absolute examples: A=30, A=-10, A=U30, A=D, A:U15, A:D, A_U=5
    Relative examples: A:+10, A:-5, A_U:+5

    Returns:
        (abs_map, delta_map, extra_abs_map, extra_delta_map)
        where extra maps are keyed by (idx, suffix) for additional views.
    """
    abs_map: Dict[int, float] = {}
    delta_map: Dict[int, float] = {}
    extra_abs_map: Dict[Tuple[int, str], float] = {}
    extra_delta_map: Dict[Tuple[int, str], float] = {}
    if not spec:
        return abs_map, delta_map, extra_abs_map, extra_delta_map

    def split_key(raw: str) -> Tuple[int, Optional[str]]:
        raw = raw.strip()
        suffix: Optional[str] = None
        base = raw
        if "_" in raw:
            base, suffix_part = raw.split("_", 1)
            suffix = "_" + suffix_part.strip()
        idx1 = letter_to_index1(base)
        return idx1, suffix

    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        target_abs = abs_map
        target_delta = delta_map
        suffix: Optional[str] = None
        if ":" in token or "=" in token:
            k, v = re.split(r"[:=]", token, maxsplit=1)
            idx1, suffix = split_key(k)
            if suffix:
                target_abs = extra_abs_map
                target_delta = extra_delta_map
            key = (idx1, suffix) if suffix else idx1
            v2 = v.strip()
            mrel = re.match(r'^[+|-]\s*\d+(?:\.\d+)?$', v2)
            if mrel:
                target_delta[key] = float(v2.replace(" ", ""))
                continue
            up = re.match(r'^[Uu]\s*(\d+(?:\.\d+)?)?$', v2)
            dn = re.match(r'^[Dd]\s*(\d+(?:\.\d+)?)?$', v2)
            if up:
                deg = float(up.group(1)) if up.group(1) else default_deg
                target_abs[key] = +deg
            elif dn:
                deg = float(dn.group(1)) if dn.group(1) else default_deg
                target_abs[key] = -deg
            else:
                try:
                    target_abs[key] = float(v2.replace(" ", ""))
                except Exception as exc:
                    raise ValueError("invalid --setcam token: " + token) from exc
        else:
            raise ValueError("invalid --setcam token: " + token)
    return abs_map, delta_map, extra_abs_map, extra_delta_map

# ---- ffmpeg command builders ----
def build_ffmpeg_cmd(ffmpeg: str, inp: pathlib.Path, out: pathlib.Path,
                     w: int, h: int, yaw: float, pitch: float,
                     hfov: float, vfov: float, interp_v360: str,
                     ext: str, ffthreads: str, *,
                     video_mode: bool = False,
                     fps: Optional[float] = None,
                     keep_rec709: bool = False,
                     bit_depth: int = 8,
                     jpeg_quality_95: bool = False) -> List[str]:
    ext_lower = ext.lower()
    filters: List[str] = []
    if video_mode:
        if fps is None or fps <= 0:
            raise ValueError("fps must be specified and > 0 when processing a video input")
        filters.append(f"fps={fps}")
        colorspace_filter = "colorspace=iall=bt709:all=smpte170m"
        if not keep_rec709:
            colorspace_filter += ":trc=iec61966-2-1"
        if ext_lower in (".jpg", ".jpeg"):
            filters.append(f"{colorspace_filter}:range=jpeg:format=yuv444p")
        else:
            filters.append(f"{colorspace_filter}:format=yuv444p")
    filters.append(
        f"v360=input=equirect:output=rectilinear"
        f":w={w}:h={h}:yaw={yaw}:pitch={pitch}:roll=0"
        f":h_fov={hfov}:v_fov={vfov}:interp={interp_v360}"
    )
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(inp), "-vf", ",".join(filters)]
    if str(ffthreads).lower() != "auto":
        cmd += ["-threads", str(int(ffthreads))]
    if video_mode:
        cmd += ["-vsync", "vfr", "-start_number", "0"]
    else:
        cmd += ["-frames:v", "1"]
    if ext_lower in (".jpg", ".jpeg"):
        q_value = "1"
        if jpeg_quality_95:
            q_value = "2"
        cmd += [
            "-c:v", "mjpeg",
            "-q:v", q_value,
            "-qmin", q_value,
            "-qmax", q_value,
            "-pix_fmt", "yuvj444p",
            "-huffman", "optimal",
        ]
        if video_mode:
            cmd += [
                "-colorspace", "smpte170m",
                "-color_primaries", "smpte170m",
                "-color_trc", "smpte170m",
            ]
    elif video_mode and ext_lower in (".png", ".tif", ".tiff"):
        pix_fmt = "rgb48le" if bit_depth > 8 else "rgb24"
        cmd += ["-pix_fmt", pix_fmt]
    cmd.append(str(out))
    return cmd

def build_ffmpeg_equisolid_cmd(ffmpeg: str, inp: pathlib.Path, out: pathlib.Path,
                               w: int, h: int, yaw: float, pitch: float,
                               fov_deg: float, interp_v360: str,
                               ext: str, ffthreads: str, *,
                               video_mode: bool = False,
                               fps: Optional[float] = None,
                               keep_rec709: bool = False,
                               bit_depth: int = 8,
                               jpeg_quality_95: bool = False) -> List[str]:
    ext_lower = ext.lower()
    filters: List[str] = []
    if video_mode:
        if fps is None or fps <= 0:
            raise ValueError("fps must be specified and > 0 when processing a video input")
        filters.append(f"fps={fps}")
        colorspace_filter = "colorspace=iall=bt709:all=smpte170m"
        if not keep_rec709:
            colorspace_filter += ":trc=iec61966-2-1"
        if ext_lower in (".jpg", ".jpeg"):
            filters.append(f"{colorspace_filter}:range=jpeg:format=yuv444p")
        else:
            filters.append(f"{colorspace_filter}:format=yuv444p")
    filters.append(
        f"v360=input=equirect:output=fisheye"
        f":w={w}:h={h}:yaw={yaw}:pitch={pitch}:roll=0"
        f":d_fov={fov_deg}:interp={interp_v360}"
    )
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(inp), "-vf", ",".join(filters)]
    if str(ffthreads).lower() != "auto":
        cmd += ["-threads", str(int(ffthreads))]
    if video_mode:
        cmd += ["-vsync", "vfr", "-start_number", "0"]
    else:
        cmd += ["-frames:v", "1"]
    if ext_lower in (".jpg", ".jpeg"):
        q_value = "1"
        if jpeg_quality_95:
            q_value = "2"
        cmd += [
            "-c:v", "mjpeg",
            "-q:v", q_value,
            "-qmin", q_value,
            "-qmax", q_value,
            "-pix_fmt", "yuvj444p",
            "-huffman", "optimal",
        ]
        if video_mode:
            cmd += [
                "-colorspace", "smpte170m",
                "-color_primaries", "smpte170m",
                "-color_trc", "smpte170m",
            ]
    elif video_mode and ext_lower in (".png", ".tif", ".tiff"):
        pix_fmt = "rgb48le" if bit_depth > 8 else "rgb24"
        cmd += ["-pix_fmt", pix_fmt]
    cmd.append(str(out))
    return cmd


def create_arg_parser() -> argparse.ArgumentParser:
    """Create the shared argument parser for CLI usage and GUI preview tools."""

    ap = argparse.ArgumentParser(
        description=(
            "Batch convert equirectangular images with ffmpeg/v360, "
            "including optional virtual camera add/delete/set operations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Notes: presets can be overridden with --focal-mm / --size / --sensor-mm. "
            "Priority: --hfov overrides --focal-mm. "
            "Use --setcam to specify absolute or relative pitch values per camera."
        )
    )
    ap.add_argument(
        "-i", "--in", dest="input_dir", required=True,
        help="Input folder (equirectangular images) or a video file containing equirectangular frames"
    )
    ap.add_argument(
        "-o", "--out", dest="out_dir", default=None,
        help="Output folder. Defaults to <input>/_geometry if omitted"
    )

    ap.add_argument(
        "--preset",
        choices=["default", "fisheyelike", "2views", "evenMinus30", "evenPlus30", "fisheyeXY"],
        default="default",
        help=(
            "default=8-view baseline / "
            "fisheyelike=10-view mix (auto focal 17mm, custom deletions/additions) / "
            "2views=front/back only (6mm focal, 3600px) / "
            "evenMinus30=even slots pitch -30deg / "
            "evenPlus30=even slots pitch +30deg / "
            "fisheyeXY=fisheye X/Y pair only (Equisolid 3600px FOV180)"
        )
    )
    ap.add_argument("--count", type=int, default=8, help="Horizontal division count (4=90deg, 8=45deg)")
    ap.add_argument(
        "--addcam", default="",
        help="Add virtual cameras, e.g. 'B' (+/-default pitch), 'B:U', 'D:D20', 'F:U15' (comma separated)"
    )
    ap.add_argument(
        "--addcam-deg", type=float, default=30.0,
        help="Default magnitude in degrees when 'U/D' in --addcam/--setcam omit a value (default 30)"
    )
    ap.add_argument(
        "--add-top",
        action="store_true",
        help="Include cube-map style top view (pitch +90 deg)",
    )
    ap.add_argument(
        "--add-bottom",
        action="store_true",
        help="Include cube-map style bottom view (pitch -90 deg)",
    )
    ap.add_argument(
        "--add-topdown",
        action="store_true",
        dest="add_topdown",
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--delcam", default="", help="Remove baseline cameras by letter, e.g. 'B,D'")
    ap.add_argument(
        "--setcam", default="",
        help="Override/adjust baseline pitch. Absolute: 'A=30','A=U','A=D20'. Relative: 'A:+10','B:-5'."
    )

    ap.add_argument("--size", type=int, default=1600, action=StoreWithFlag, help="Square output size per view")
    ap.add_argument("--ext", default="jpg", help="Output extension (jpg=high quality mjpeg)")
    ap.add_argument(
        "--jpeg-quality-95",
        action="store_true",
        help="When set with --ext jpg, encode outputs at approximately 95% JPEG quality instead of maximum.",
    )
    ap.add_argument(
        "-f", "--fps", type=float, default=None,
        help="Frame extraction rate (fps) when input is a video file"
    )
    ap.add_argument(
        "--keep-rec709", action="store_true",
        help="Keep Rec.709 transfer characteristics for video inputs (default: convert to sRGB)"
    )
    ap.add_argument(
        "--hfov", type=float, default=None, action=StoreWithFlag,
        help="Horizontal FOV in degrees (overrides focal length)"
    )
    ap.add_argument(
        "--focal-mm", type=float, default=12.0, action=StoreWithFlag,
        help="Focal length in millimetres when --hfov is not set"
    )
    ap.add_argument(
        "--sensor-mm", default="36 36",
        help="Sensor width/height in millimetres, e.g. '36 36' or '36x24'"
    )

    ap.add_argument(
        "-j", "--jobs", default="auto",
        help="Concurrent ffmpeg processes (number or 'auto'=physical cores/2)"
    )
    ap.add_argument("--ffthreads", default="1", help="Internal ffmpeg threads per process (number or 'auto')")
    ap.add_argument(
        "--print-cmd", choices=["once", "none", "all"], default="once",
        help="How many ffmpeg commands to print: once/none/all"
    )
    ap.add_argument("--ffmpeg", default="ffmpeg", help="Path to the ffmpeg executable")
    ap.add_argument("--dry-run", action="store_true", help="Print all commands without executing them")
    return ap

# ---- Parallel execution and cancellation ----
stop_event = threading.Event()
procs_lock = threading.Lock()
running_procs = set()   # type: set

sig_hits = 0
def on_signal(sig, frame):
    global sig_hits
    sig_hits += 1
    if not stop_event.is_set():
        print("\n[INFO] Cancel requested. Stopping new jobs and terminating running processes...", file=sys.stderr)
        stop_event.set()
    with procs_lock:
        for p in list(running_procs):
            try:
                p.terminate() if sig_hits == 1 else p.kill()
            except Exception:
                pass
    if sig_hits >= 2:
        print("[INFO] Force exiting", file=sys.stderr)

try:
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    if os.name == "nt" and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, on_signal)
except Exception:
    pass

def parse_jobs(s: str) -> int:
    if str(s).lower() == "auto":
        cores = os.cpu_count() or 1
        return max(1, cores // 2)
    return max(1, int(s))

def run_one(cmd: List[str]) -> Tuple[int, str]:
    if stop_event.is_set():
        return 130, ""
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    with procs_lock:
        running_procs.add(proc)
    try:
        while True:
            try:
                rc = proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                if stop_event.is_set():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
        err_text = (proc.stderr.read() or b"").decode(errors="ignore")
        return rc, err_text
    finally:
        with procs_lock:
            running_procs.discard(proc)


def build_view_jobs(args, files: List[pathlib.Path], out_dir: pathlib.Path) -> BuildResult:
    """Compose ffmpeg job definitions and collect view specifications."""

    size_explicit = getattr(args, "size_explicit", False)
    hfov_explicit = getattr(args, "hfov_explicit", False)
    focal_explicit = getattr(args, "focal_mm_explicit", False)
    video_mode = bool(getattr(args, "input_is_video", False))
    fps_value = getattr(args, "fps", None)
    keep_rec709 = bool(getattr(args, "keep_rec709", False))
    video_bit_depth = int(getattr(args, "video_bit_depth", 8))

    even_pitch_all = None
    even_pitch_map: Dict[int, float] = {}

    add_top = bool(getattr(args, "add_top", False))
    add_bottom = bool(getattr(args, "add_bottom", False))
    if getattr(args, "add_topdown", False):
        add_top = True
        add_bottom = True
    setattr(args, "add_top", add_top)
    setattr(args, "add_bottom", add_bottom)
    preset_fisheye_xy = (args.preset == "fisheyeXY")
    preset_two_views = (args.preset == "2views")
    preset_fisheyelike = (args.preset == "fisheyelike")

    if preset_fisheyelike:
        if args.count != 10:
            args.count = 10
    elif preset_fisheye_xy:
        if args.count != 8:
            print("[INFO] preset 'fisheyeXY' forces count=8")
        args.count = 8
    elif args.preset == "evenMinus30" and even_pitch_all is None:
        even_pitch_all = -30.0
    elif args.preset == "evenPlus30" and even_pitch_all is None:
        even_pitch_all = +30.0
    preset_extra_even_pitches: List[float] = []

    if preset_two_views and not size_explicit:
        args.size = 3600
    if preset_two_views and not hfov_explicit and not focal_explicit:
        args.focal_mm = 6.0
    if preset_fisheyelike and not hfov_explicit and not focal_explicit:
        args.focal_mm = 17.0

    add_map = parse_addcam_spec(args.addcam, args.addcam_deg)
    del_set = parse_delcam_spec(args.delcam)
    user_addcam_supplied = bool(str(getattr(args, "addcam", "")).strip()) or bool(
        getattr(args, "addcam_explicit", False)
    )
    user_delcam_supplied = bool(str(getattr(args, "delcam", "")).strip()) or bool(
        getattr(args, "delcam_explicit", False)
    )
    if preset_fisheyelike:
        if not user_delcam_supplied:
            for ch in ("A", "C", "D", "F", "H", "I"):
                del_set.add(letter_to_index1(ch))
        if not user_addcam_supplied:
            for ch in ("A", "F"):
                idx = letter_to_index1(ch)
                slot = add_map.setdefault(idx, [])
                if not any(abs(val - float(args.addcam_deg)) < 1e-6 for val in slot):
                    slot.append(float(args.addcam_deg))
                if not any(abs(val + float(args.addcam_deg)) < 1e-6 for val in slot):
                    slot.append(-float(args.addcam_deg))
    if preset_two_views:
        for ch in ("B", "C", "D", "F", "G", "H"):
            del_set.add(letter_to_index1(ch))
    set_abs_map, set_delta_map, set_extra_abs_map, set_extra_delta_map = parse_setcam_spec(args.setcam, args.addcam_deg)

    sensor_w_mm = parse_sensor(args.sensor_mm)
    sensor_dims = parse_sensor_dimensions(args.sensor_mm)
    sensor_long_mm = max(sensor_dims) if sensor_dims else sensor_w_mm
    sensor_h_mm = None
    if sensor_dims:
        if len(sensor_dims) >= 2:
            sensor_h_mm = float(sensor_dims[1])
        else:
            sensor_h_mm = sensor_w_mm
    else:
        sensor_h_mm = sensor_w_mm
    if sensor_h_mm is not None and sensor_h_mm <= 0:
        sensor_h_mm = None
    if args.hfov is not None:
        hfov_deg = float(args.hfov)
        f_used_mm = focal_from_hfov_deg(hfov_deg, sensor_w_mm)
    else:
        f_used_mm = float(args.focal_mm)
        hfov_deg = fov_from_focal_mm(f_used_mm, sensor_w_mm)

    focal_35mm_equiv = None
    if sensor_long_mm and sensor_long_mm > 0 and abs(sensor_long_mm - 36.0) > 1e-6:
        focal_35mm_equiv = f_used_mm * (36.0 / sensor_long_mm)

    base_size = int(args.size)
    w = h = base_size
    if sensor_h_mm and f_used_mm > 1e-6:
        vfov_rad = 2.0 * math.atan(sensor_h_mm / (2.0 * f_used_mm))
        vfov_deg = max(1.0, min(179.9, math.degrees(vfov_rad)))
    else:
        vfov_deg = v_fov_from_hfov(hfov_deg, w, h)

    if preset_fisheye_xy:
        fisheye_size = base_size if size_explicit else 3600
        fisheye_fov_deg = hfov_deg if hfov_explicit else 180.0
    else:
        fisheye_size = base_size
        fisheye_fov_deg = hfov_deg

    count = int(args.count)
    if count <= 0:
        print("[ERR] --count must be >= 1", file=sys.stderr)
        sys.exit(1)
    yaw_step = 360.0 / count

    ffmpeg = args.ffmpeg
    ext_dot = "." + args.ext.lower().lstrip(".")
    interp_v360 = map_interp_for_v360("bicubic")

    jobs_list: List[Tuple[List[str], str, str]] = []
    view_specs: List[ViewSpec] = []
    existing_names: Set[str] = set()
    fisheye_letter_map = {1: "X", 5: "Y"} if preset_fisheye_xy else {}
    ffmpeg_common_kwargs = dict(
        video_mode=video_mode,
        fps=fps_value,
        keep_rec709=keep_rec709,
        bit_depth=video_bit_depth,
        jpeg_quality_95=args.jpeg_quality_95,
    )

    def make_out_path(view_id: str) -> pathlib.Path:
        if video_mode:
            return out_dir / f"{stem}_%07d_{view_id}{ext_dot}"
        return out_dir / f"{stem}_{view_id}{ext_dot}"

    for img in files:
        stem = img.stem
        xy_views: List[Tuple[str, float, float]] = []
        base_pitch0 = 0.0

        def record_view(out_path: pathlib.Path, yaw_val: float, pitch_val: float,
                        width: int, height: int, hfov_val: float, vfov_val: float,
                        projection: str = "perspective") -> None:
            out_stem = out_path.stem
            if video_mode:
                placeholder_prefix = f"{stem}_%07d_"
                if out_stem.startswith(placeholder_prefix):
                    view_id = out_stem[len(placeholder_prefix):]
                elif out_stem.startswith(f"{stem}_"):
                    view_id = out_stem[len(stem) + 1:]
                else:
                    view_id = out_stem
            else:
                if out_stem.startswith(f"{stem}_"):
                    view_id = out_stem[len(stem) + 1:]
                else:
                    view_id = out_stem
            view_specs.append(
                ViewSpec(
                    source_path=img,
                    output_name=out_path.name,
                    view_id=view_id,
                    yaw_deg=yaw_val,
                    pitch_deg=pitch_val,
                    hfov_deg=hfov_val,
                    vfov_deg=vfov_val,
                    width=width,
                    height=height,
                    projection=projection,
                )
            )

        for yi in range(count):
            if stop_event.is_set():
                break
            idx1 = yi + 1
            tag = letter_tag(yi)
            skip_base = (idx1 in del_set) or preset_fisheye_xy
            yaw = normalize_angle_deg(yi * yaw_step)
            pitch = base_pitch0
            if (idx1 % 2) == 0 and not preset_fisheye_xy:
                if even_pitch_all is not None:
                    pitch += float(even_pitch_all)
                if idx1 in even_pitch_map:
                    pitch += float(even_pitch_map[idx1])

            def apply_setcam_pitch(idx: int, base_pitch: float, suffix: Optional[str] = None) -> float:
                pitch_val = base_pitch
                if suffix:
                    key = (idx, suffix)
                    if key in set_extra_abs_map:
                        pitch_val = float(set_extra_abs_map[key])
                    elif idx in set_abs_map:
                        pitch_val = float(set_abs_map[idx])
                    if key in set_extra_delta_map:
                        pitch_val += float(set_extra_delta_map[key])
                    elif idx in set_delta_map:
                        pitch_val += float(set_delta_map[idx])
                else:
                    if idx in set_abs_map:
                        pitch_val = float(set_abs_map[idx])
                    if idx in set_delta_map:
                        pitch_val += float(set_delta_map[idx])
                return pitch_val

            pitch = apply_setcam_pitch(idx1, pitch)
            pitch = clamp(pitch, -90.0, 90.0)

            if preset_fisheye_xy and idx1 in fisheye_letter_map:
                xy_views.append((fisheye_letter_map[idx1], yaw, pitch))

            if not skip_base:
                out_path = make_out_path(tag)
                if out_path.name not in existing_names:
                    cmd = build_ffmpeg_cmd(
                        ffmpeg, img, out_path, w, h, yaw, pitch,
                        hfov_deg, vfov_deg, interp_v360, ext_dot, args.ffthreads,
                        **ffmpeg_common_kwargs,
                    )
                    jobs_list.append((cmd, img.name, out_path.name))
                    existing_names.add(out_path.name)
                    record_view(out_path, yaw, pitch, w, h, hfov_deg, vfov_deg)

            if not skip_base and (idx1 % 2) == 0 and preset_extra_even_pitches:
                for d in preset_extra_even_pitches:
                    p2 = clamp(pitch + d, -90.0, 90.0)
                    suf = extra_suffix(d, 30.0)
                    p2 = apply_setcam_pitch(idx1, p2, suffix=suf)
                    out_path2 = make_out_path(f"{tag}{suf}")
                    if out_path2.name not in existing_names:
                        cmd2 = build_ffmpeg_cmd(
                            ffmpeg, img, out_path2, w, h, yaw, p2,
                            hfov_deg, vfov_deg, interp_v360, ext_dot, args.ffthreads,
                            **ffmpeg_common_kwargs,
                        )
                        jobs_list.append((cmd2, img.name, out_path2.name))
                        existing_names.add(out_path2.name)
                        record_view(out_path2, yaw, p2, w, h, hfov_deg, vfov_deg)

            if not preset_fisheye_xy and idx1 in add_map:
                for d in add_map[idx1]:
                    p3 = clamp(pitch + d, -90.0, 90.0)
                    suf3 = extra_suffix(d, args.addcam_deg)
                    p3 = apply_setcam_pitch(idx1, p3, suffix=suf3)
                    out_path3 = make_out_path(f"{tag}{suf3}")
                    if out_path3.name not in existing_names:
                        cmd3 = build_ffmpeg_cmd(
                            ffmpeg, img, out_path3, w, h, yaw, p3,
                            hfov_deg, vfov_deg, interp_v360, ext_dot, args.ffthreads,
                            **ffmpeg_common_kwargs,
                        )
                        jobs_list.append((cmd3, img.name, out_path3.name))
                        existing_names.add(out_path3.name)
                        record_view(out_path3, yaw, p3, w, h, hfov_deg, vfov_deg)

        if preset_fisheye_xy:
            for xy_tag, yaw_xy, pitch_xy in xy_views:
                out_path_xy = make_out_path(xy_tag)
                if out_path_xy.name in existing_names:
                    continue
                cmd_xy = build_ffmpeg_equisolid_cmd(
                    ffmpeg, img, out_path_xy, fisheye_size, fisheye_size,
                    yaw_xy, pitch_xy, fisheye_fov_deg, interp_v360, ext_dot, args.ffthreads,
                    **ffmpeg_common_kwargs,
                )
                jobs_list.append((cmd_xy, img.name, out_path_xy.name))
                existing_names.add(out_path_xy.name)
                record_view(
                    out_path_xy, yaw_xy, pitch_xy,
                    fisheye_size, fisheye_size, fisheye_fov_deg, fisheye_fov_deg,
                    projection="equisolid"
                )

        extra_pitches: List[float] = []
        if add_top:
            extra_pitches.append(90.0)
        if add_bottom:
            extra_pitches.append(-90.0)
        if extra_pitches:
            td_index = count
            for td_pitch in extra_pitches:
                td_tag = letter_tag(td_index)
                td_index += 1
                pitch_td = clamp(td_pitch, -90.0, 90.0)
                idx_td = letter_to_index1(td_tag)
                pitch_td = apply_setcam_pitch(idx_td, pitch_td)
                out_path_td = make_out_path(td_tag)
                if out_path_td.name in existing_names:
                    continue
                cmd_td = build_ffmpeg_cmd(
                    ffmpeg, img, out_path_td, w, h, 0.0, pitch_td,
                    hfov_deg, vfov_deg, interp_v360, ext_dot, args.ffthreads,
                    **ffmpeg_common_kwargs,
                )
                jobs_list.append((cmd_td, img.name, out_path_td.name))
                existing_names.add(out_path_td.name)
                record_view(out_path_td, 0.0, pitch_td, w, h, hfov_deg, vfov_deg)

    preview_views_line = ""
    sensor_line = ""
    realityscan_line = ""
    metashape_line = ""

    if jobs_list:
        first_src = jobs_list[0][1]
        reference_stem = pathlib.Path(first_src).stem
        seen_views: List[str] = []
        for _, src_name, dst_name in jobs_list:
            if src_name != first_src:
                break
            stem_candidate = pathlib.Path(dst_name).stem
            if getattr(args, "input_is_video", False):
                placeholder_prefix = f"{reference_stem}_%07d_"
                if stem_candidate.startswith(placeholder_prefix):
                    view_id = stem_candidate[len(placeholder_prefix):]
                elif stem_candidate.startswith(f"{reference_stem}_"):
                    view_id = stem_candidate[len(reference_stem) + 1:]
                else:
                    view_id = stem_candidate
            else:
                if stem_candidate.startswith(f"{reference_stem}_"):
                    view_id = stem_candidate[len(reference_stem) + 1:]
                else:
                    view_id = stem_candidate
            if view_id and view_id not in seen_views:
                seen_views.append(view_id)
        if seen_views:
            preview_views_line = (
                f"[INFO] View summary ({first_src}): "
                + ", ".join(seen_views)
            )
            if preset_fisheye_xy:
                preview_views_line += (
                    f" | fisheye_fov={fisheye_fov_deg:.1f}deg | size={fisheye_size}x{fisheye_size}"
                )
            else:
                sensor_line = f"[INFO] Sensor={args.sensor_mm} mm | size={w}x{h}"
                focal_segment = f"focal length=**{f_used_mm:.3f}**mm"
                if focal_35mm_equiv is not None:
                    focal_segment += f" (35mm eq=**{focal_35mm_equiv:.3f}**mm)"
                realityscan_line = f"[INFO] For RealityScan: {focal_segment}"
                if w > 0:
                    pixel_size_mm = sensor_w_mm / float(w)
                    if pixel_size_mm > 0:
                        focal_px = f_used_mm / pixel_size_mm
                        metashape_line = (
                            "[INFO] For Metashape: Precalibrated f=**{:.5f}** | pixel_size=**{:.4f}** mm".format(
                                focal_px, pixel_size_mm
                            )
                        )

    return BuildResult(
        jobs=jobs_list,
        view_specs=view_specs,
        focal_used_mm=f_used_mm,
        focal_35mm_equiv=focal_35mm_equiv,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        preview_views_line=preview_views_line,
        sensor_line=sensor_line,
        realityscan_line=realityscan_line,
        metashape_line=metashape_line,
    )

# ---- main ----
def main():
    ap = create_arg_parser()
    args = ap.parse_args()
    for attr in ("size", "hfov", "focal_mm"):
        setattr(args, f"{attr}_explicit", getattr(args, f"{attr}_explicit", False))

    input_path = pathlib.Path(args.input_dir).expanduser().resolve()
    files: List[pathlib.Path] = []
    if input_path.is_dir():
        setattr(args, "input_is_video", False)
        setattr(args, "video_bit_depth", 8)
        out_dir = pathlib.Path(args.out_dir).resolve() if args.out_dir else (input_path / "_geometry")
        out_dir.mkdir(parents=True, exist_ok=True)
        files = [
            p for p in sorted(input_path.iterdir())
            if p.is_file() and p.suffix.lower() in EXTS
        ]
        if not files:
            print("[WARN] No target images found (tif/jpg/png)", file=sys.stderr)
            sys.exit(0)
    elif input_path.is_file():
        setattr(args, "input_is_video", True)
        if args.fps is None or args.fps <= 0:
            print("[ERR] -f/--fps must be specified for video inputs", file=sys.stderr)
            sys.exit(1)
        video_out_dir = pathlib.Path(args.out_dir).resolve() if args.out_dir else (
            input_path.parent / f"{input_path.stem}_geometry"
        )
        video_out_dir.mkdir(parents=True, exist_ok=True)
        setattr(args, "video_bit_depth", detect_input_bit_depth(input_path))
        files = [input_path]
        out_dir = video_out_dir
    else:
        print("[ERR] Input path not found:", input_path, file=sys.stderr)
        sys.exit(1)

    result = build_view_jobs(args, files, out_dir)
    jobs_list = result.jobs
    total = result.total

    if args.dry_run:
        for cmd, _, _ in jobs_list:
            print("$ " + " ".join(shlex.quote(c) for c in cmd))
        print(f"\n[DRY] Exiting without execution (total {total} commands)")
        return

    # command display policy
    if args.print_cmd == "all":
        for cmd, _, _ in jobs_list: print("$ " + " ".join(shlex.quote(c) for c in cmd))
    elif args.print_cmd == "once" and jobs_list:
        print("$ " + " ".join(shlex.quote(c) for c in jobs_list[0][0]))

    jobs = parse_jobs(args.jobs)
    last_progress_pct = -1
    progress_label = "Progress"
    print(f"[INFO] parallel jobs: {jobs} / ffthreads: {args.ffthreads} / total: {total}")
    if result.preview_views_line:
        print(result.preview_views_line)
        if result.sensor_line:
            print(result.sensor_line)
        if result.realityscan_line:
            print(result.realityscan_line)
        if result.metashape_line:
            print(result.metashape_line)

    ok = fail = done = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = [ex.submit(run_one, cmd) for cmd, _, _ in jobs_list]
        for (fut, (_, src, dst)) in zip(as_completed(futures), jobs_list):
            rc, err = fut.result()
            done += 1
            if rc == 0:
                ok += 1
                last_progress_pct = update_progress(progress_label, done, total, last_progress_pct)
            elif rc == 130:
                fail += 1
                if stop_event.is_set():
                    continue
                if total:
                    last_progress_pct = update_progress(progress_label, done, total, last_progress_pct)
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                print(f"[{done}/{total}] {dst} canceled", file=sys.stderr)
                if err.strip():
                    print(err.strip(), file=sys.stderr)
            else:
                fail += 1
                if stop_event.is_set():
                    continue
                if total:
                    last_progress_pct = update_progress(progress_label, done, total, last_progress_pct)
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                print(f"[{done}/{total}] {dst} failed", file=sys.stderr)
                if err.strip():
                    print(err.strip(), file=sys.stderr)
    if total and last_progress_pct >= 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if stop_event.is_set():
        print(f"[STOPPED] Interrupted: success={ok}, failed={fail}, total={total}")
        sys.exit(130)
    else:
        print(f"[OK] Completed: success={ok}, failed={fail}, total={total}")

if __name__ == "__main__":
    main()

