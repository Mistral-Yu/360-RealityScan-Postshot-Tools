#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rs2ps_360PerspCut converts 360-degree panoramas into perspective or fisheye crops
by orchestrating ffmpeg runs.
It accepts files or directories, derives camera orientations from presets or
manual modifiers, and emits view images with optional fisheye/top-down outputs.
Field-of-view math is resolved from sensor and focal parameters, while ffmpeg
jobs run in parallel with progress reporting and graceful cleanup.
Cancellation uses the default Ctrl+C (SIGINT) handler for graceful shutdowns.
"""

import argparse, math, pathlib, subprocess, sys, os, signal, threading, shlex, re
from typing import List, Dict, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

PROGRESS_INTERVAL = 5

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

def parse_sensor(s: str) -> float:
    s = s.lower().replace("\u00d7", "x").replace(",", " ").strip()
    w = s.split("x")[0].strip() if "x" in s else s.split()[0]
    return float(w)

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

    Absolute examples: A=30, A=-10, A=U30, A=D, A:U15, A:D
    Relative examples: A:+10, A:-5

    Returns:
        (abs_map, delta_map) where keys are 1-based indices.
    """
    abs_map: Dict[int, float] = {}
    delta_map: Dict[int, float] = {}
    if not spec:
        return abs_map, delta_map

    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token or "=" in token:
            k, v = re.split(r"[:=]", token, maxsplit=1)
            idx1 = letter_to_index1(k)
            v2 = v.strip()
            mrel = re.match(r'^[+|-]\s*\d+(?:\.\d+)?$', v2)
            if mrel:
                delta_map[idx1] = float(v2.replace(" ", ""))
                continue
            up = re.match(r'^[Uu]\s*(\d+(?:\.\d+)?)?$', v2)
            dn = re.match(r'^[Dd]\s*(\d+(?:\.\d+)?)?$', v2)
            if up:
                deg = float(up.group(1)) if up.group(1) else default_deg
                abs_map[idx1] = +deg
            elif dn:
                deg = float(dn.group(1)) if dn.group(1) else default_deg
                abs_map[idx1] = -deg
            else:
                try:
                    abs_map[idx1] = float(v2.replace(" ", ""))
                except Exception as exc:
                    raise ValueError("invalid --setcam token: " + token) from exc
        else:
            raise ValueError("invalid --setcam token: " + token)
    return abs_map, delta_map

# ---- ffmpeg command builders ----
def build_ffmpeg_cmd(ffmpeg: str, inp: pathlib.Path, out: pathlib.Path,
                     w: int, h: int, yaw: float, pitch: float,
                     hfov: float, vfov: float, interp_v360: str,
                     ext: str, ffthreads: str) -> List[str]:
    vfilter = (
        f"v360=input=equirect:output=rectilinear"
        f":w={w}:h={h}:yaw={yaw}:pitch={pitch}:roll=0"
        f":h_fov={hfov}:v_fov={vfov}:interp={interp_v360}"
    )
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(inp), "-vf", vfilter]
    if str(ffthreads).lower() != "auto":
        cmd += ["-threads", str(int(ffthreads))]
    cmd += ["-frames:v", "1"]
    if ext.lower() in (".jpg", ".jpeg"):
        cmd += ["-c:v","mjpeg","-q:v","1","-qmin","1","-qmax","1",
                "-pix_fmt","yuvj444p","-huffman","optimal"]
    cmd.append(str(out))
    return cmd

def build_ffmpeg_equisolid_cmd(ffmpeg: str, inp: pathlib.Path, out: pathlib.Path,
                               w: int, h: int, yaw: float, pitch: float,
                               fov_deg: float, interp_v360: str,
                               ext: str, ffthreads: str) -> List[str]:
    vfilter = (
        f"v360=input=equirect:output=fisheye"
        f":w={w}:h={h}:yaw={yaw}:pitch={pitch}:roll=0"
        f":d_fov={fov_deg}:interp={interp_v360}"
    )
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(inp), "-vf", vfilter]
    if str(ffthreads).lower() != "auto":
        cmd += ["-threads", str(int(ffthreads))]
    cmd += ["-frames:v", "1"]
    if ext.lower() in (".jpg", ".jpeg"):
        cmd += ["-c:v","mjpeg","-q:v","1","-qmin","1","-qmax","1",
                "-pix_fmt","yuvj444p","-huffman","optimal"]
    cmd.append(str(out))
    return cmd

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

# ---- main ----
def main():
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
    ap.add_argument("-i","--in", dest="input_dir", required=True,
        help="Input folder containing equirectangular images (.tif/.tiff/.jpg/.jpeg/.png)")
    ap.add_argument("-o","--out", dest="out_dir", default=None,
        help="Output folder. Defaults to <input>/_geometry if omitted")

    ap.add_argument("--preset",
        choices=["default","2views","10views","evenMinus30","evenPlus30","fisheyeXY"], default="default",
        help=("default=8-view baseline / "
              "2views=front/back only (6mm focal, 3600px) / "
              "10views=baseline + top/bottom (10 total views) / "
              "evenMinus30=even slots pitch -30deg / "
              "evenPlus30=even slots pitch +30deg / "
              "fisheyeXY=fisheye X/Y pair only (Equisolid 3600px FOV180)"))

    ap.add_argument("--count", type=int, default=8, help="Horizontal division count (4=90deg, 8=45deg)")

    ap.add_argument("--addcam", default="",
        help="Add virtual cameras, e.g. 'B' (+/-default pitch), 'B:U', 'D:D20', 'F:U15' (comma separated)")
    ap.add_argument("--addcam-deg", type=float, default=30.0,
        help="Default magnitude in degrees when 'U/D' in --addcam/--setcam omit a value (default 30)")
    ap.add_argument("--add-topdown", action="store_true",
        help="Include cube-map style top (pitch +90 deg) and bottom (pitch -90 deg) views")

    ap.add_argument("--delcam", default="",
        help="Remove baseline cameras by letter, e.g. 'B,D'")
    ap.add_argument("--setcam", default="",
        help="Override/adjust baseline pitch. Absolute: 'A=30','A=U','A=D20'. Relative: 'A:+10','B:-5'.")

    # geometry / output
    ap.add_argument("--size", type=int, default=1600, help="Square output size per view")
    ap.add_argument("--ext", default="jpg", help="Output extension (jpg=high quality mjpeg)")
    ap.add_argument("--hfov", type=float, default=None, help="Horizontal FOV in degrees (overrides focal length)")
    ap.add_argument("--focal-mm", type=float, default=12.0, help="Focal length in millimetres when --hfov is not set")
    ap.add_argument("--sensor-mm", default="36 36", help="Sensor width/height in millimetres, e.g. '36 36' or '36x24'")

    # parallelism / logging
    ap.add_argument("-j","--jobs", default="auto", help="Concurrent ffmpeg processes (number or 'auto'=physical cores/2)")
    ap.add_argument("--ffthreads", default="1", help="Internal ffmpeg threads per process (number or 'auto')")
    ap.add_argument("--print-cmd", choices=["once","none","all"], default="once",
        help="How many ffmpeg commands to print: once/none/all")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="Path to the ffmpeg executable")
    ap.add_argument("--dry-run", action="store_true", help="Print all commands without executing them")
    args = ap.parse_args()
    in_dir = pathlib.Path(args.input_dir).expanduser().resolve()
    if not in_dir.is_dir():
        print("[ERR] Input folder not found:", in_dir, file=sys.stderr); sys.exit(1)
    out_dir = pathlib.Path(args.out_dir).resolve() if args.out_dir else (in_dir / "_geometry")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(in_dir.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
    if not files:
        print("[WARN] No target images found (tif/jpg/png)", file=sys.stderr); sys.exit(0)

    size_explicit = any(arg == "--size" or arg.startswith("--size=") for arg in sys.argv[1:])
    hfov_explicit = any(arg == "--hfov" or arg.startswith("--hfov=") for arg in sys.argv[1:])
    focal_explicit = any(arg == "--focal-mm" or arg.startswith("--focal-mm=") for arg in sys.argv[1:])

    # ---- preset handling (pitch adjustments only) ----
    even_pitch_all = None
    even_pitch_map: Dict[int, float] = {}

    add_topdown = bool(args.add_topdown)
    preset_fisheye_xy = (args.preset == "fisheyeXY")
    preset_two_views = (args.preset == "2views")

    if args.preset == "10views":
        add_topdown = True
    elif preset_fisheye_xy:
        if args.count != 8:
            print("[INFO] preset 'fisheyeXY' forces count=8")
        args.count = 8
    elif args.preset == "evenMinus30" and even_pitch_all is None:
        even_pitch_all = -30.0
    elif args.preset == "evenPlus30" and even_pitch_all is None:
        even_pitch_all = +30.0
    preset_extra_even_pitches = []

    if preset_two_views and not size_explicit:
        args.size = 3600
    if preset_two_views and not hfov_explicit and not focal_explicit:
        args.focal_mm = 6.0

    # add/del/setcam parsing
    add_map = parse_addcam_spec(args.addcam, args.addcam_deg)   # idx1 -> [delta_pitch,...]
    del_set = parse_delcam_spec(args.delcam)                    # idx1 set
    if preset_two_views:
        for ch in ("B","C","D","F","G","H"):
            del_set.add(letter_to_index1(ch))
    set_abs_map, set_delta_map = parse_setcam_spec(args.setcam, args.addcam_deg if hasattr(args, "addcam_deeg") else args.addcam_deg)

    # ---- FOV / size / derived values ----
    sensor_w_mm = parse_sensor(args.sensor_mm)
    if args.hfov is not None:
        hfov_deg = float(args.hfov)
        f_used_mm = focal_from_hfov_deg(hfov_deg, sensor_w_mm)
    else:
        f_used_mm = float(args.focal_mm)
        hfov_deg = fov_from_focal_mm(f_used_mm, sensor_w_mm)

    base_size = int(args.size)
    w = h = base_size
    vfov_deg = v_fov_from_hfov(hfov_deg, w, h)

    if preset_fisheye_xy:
        fisheye_size = base_size if size_explicit else 3600
        fisheye_fov_deg = hfov_deg if hfov_explicit else 180.0
    else:
        fisheye_size = base_size
        fisheye_fov_deg = hfov_deg

    # ---- Angles & output naming ----
    count = int(args.count)
    if count <= 0:
        print("[ERR] --count must be >= 1", file=sys.stderr); sys.exit(1)
    yaw_step = 360.0 / count

    ffmpeg = args.ffmpeg
    ext_dot = "." + args.ext.lower().lstrip(".")
    interp_v360 = map_interp_for_v360("bicubic")

    def extra_suffix(delta_pitch: float, default_deg: float=30.0) -> str:
        sign = "_U" if delta_pitch > 0 else "_D"
        mag = abs(delta_pitch)
        if abs(mag - default_deg) < 1e-6:
            return sign
        if float(mag).is_integer():
            return f"{sign}{int(round(mag))}"
        else:
            return f"{sign}{mag:g}"

    jobs_list: List[Tuple[List[str], str, str]] = []
    existing_names: Set[str] = set()  # prevent duplicate jobs
    fisheye_letter_map = {1: "X", 5: "Y"} if preset_fisheye_xy else {}

    for img in files:
        stem = img.stem
        xy_views: List[Tuple[str, float, float]] = []
        base_pitch0 = 0.0
        for yi in range(count):
            if stop_event.is_set(): break
            idx1 = yi + 1
            tag = letter_tag(yi)

            # Skip baseline output when camera is deleted or preset suppresses it
            skip_base = (idx1 in del_set) or preset_fisheye_xy

            yaw = normalize_angle_deg(yi * yaw_step)

            # ---- Base view adjustments (preset -> setcam) ----
            pitch = base_pitch0
            if (idx1 % 2) == 0 and not preset_fisheye_xy:
                if even_pitch_all is not None: pitch += float(even_pitch_all)
                if idx1 in even_pitch_map:     pitch += float(even_pitch_map[idx1])

            # setcam: absolute values replace, relative values add
            if idx1 in set_abs_map:
                pitch = float(set_abs_map[idx1])
            if idx1 in set_delta_map:
                pitch += float(set_delta_map[idx1])

            pitch = clamp(pitch, -90.0, 90.0)
            if preset_fisheye_xy and idx1 in fisheye_letter_map:
                xy_views.append((fisheye_letter_map[idx1], yaw, pitch))

            # 1) Baseline image (skipped when requested)
            if not skip_base:
                out_path = out_dir / f"{stem}_{tag}{ext_dot}"
                if out_path.name not in existing_names:
                    cmd = build_ffmpeg_cmd(ffmpeg, img, out_path, w, h, yaw, pitch,
                                           hfov_deg, vfov_deg, interp_v360, ext_dot, args.ffthreads)
                    jobs_list.append((cmd, img.name, out_path.name))
                    existing_names.add(out_path.name)

            # 2) Preset extras (only when baseline exists)
            if not skip_base and (idx1 % 2) == 0 and preset_extra_even_pitches:
                for d in preset_extra_even_pitches:
                    p2 = clamp(pitch + d, -90.0, 90.0)
                    suf = extra_suffix(d, 30.0)
                    out_path2 = out_dir / f"{stem}_{tag}{suf}{ext_dot}"
                    if out_path2.name not in existing_names:
                        cmd2 = build_ffmpeg_cmd(ffmpeg, img, out_path2, w, h, yaw, p2,
                                                hfov_deg, vfov_deg, interp_v360, ext_dot, args.ffthreads)
                        jobs_list.append((cmd2, img.name, out_path2.name))
                        existing_names.add(out_path2.name)

            # 3) User-specified additions (applied regardless of --delcam)
            if not preset_fisheye_xy and idx1 in add_map:
                for d in add_map[idx1]:
                    p3 = clamp(pitch + d, -90.0, 90.0)
                    suf3 = extra_suffix(d, args.addcam_deg)
                    out_path3 = out_dir / f"{stem}_{tag}{suf3}{ext_dot}"
                    if out_path3.name not in existing_names:
                        cmd3 = build_ffmpeg_cmd(ffmpeg, img, out_path3, w, h, yaw, p3,
                                                hfov_deg, vfov_deg, interp_v360, ext_dot, args.ffthreads)
                        jobs_list.append((cmd3, img.name, out_path3.name))
                        existing_names.add(out_path3.name)



        if preset_fisheye_xy:
            for xy_tag, yaw_xy, pitch_xy in xy_views:
                out_path_xy = out_dir / f"{stem}_{xy_tag}{ext_dot}"
                if out_path_xy.name in existing_names:
                    continue
                cmd_xy = build_ffmpeg_equisolid_cmd(
                    ffmpeg, img, out_path_xy, fisheye_size, fisheye_size,
                    yaw_xy, pitch_xy, fisheye_fov_deg, interp_v360, ext_dot, args.ffthreads)
                jobs_list.append((cmd_xy, img.name, out_path_xy.name))
                existing_names.add(out_path_xy.name)

        if add_topdown:
            # Optional cube-map style vertical views
            td_index = count
            for td_pitch in (90.0, -90.0):
                td_tag = letter_tag(td_index)
                td_index += 1
                pitch_td = clamp(td_pitch, -90.0, 90.0)
                out_path_td = out_dir / f"{stem}_{td_tag}{ext_dot}"
                if out_path_td.name in existing_names:
                    continue
                cmd_td = build_ffmpeg_cmd(
                    ffmpeg, img, out_path_td, w, h, 0.0, pitch_td,
                    hfov_deg, vfov_deg, interp_v360, ext_dot, args.ffthreads)
                jobs_list.append((cmd_td, img.name, out_path_td.name))
                existing_names.add(out_path_td.name)

    total = len(jobs_list)

    preview_views_line = ""
    if jobs_list:
        first_src = jobs_list[0][1]
        reference_stem = pathlib.Path(first_src).stem
        seen_views = []
        for _, src_name, dst_name in jobs_list:
            if src_name != first_src:
                break
            stem_candidate = pathlib.Path(dst_name).stem
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
                preview_views_line += f" | fisheye_fov={fisheye_fov_deg:.1f}deg | size={fisheye_size}x{fisheye_size}"
            else:
                preview_views_line += f" | focal={f_used_mm:.3f}mm | sensor={args.sensor_mm} mm | size={w}x{h}"

        # dry-run: print commands only
    if args.dry_run:
        for cmd, _, _ in jobs_list: print("$ " + " ".join(shlex.quote(c) for c in cmd))
        print(f"\n[DRY] Exiting without execution (total {total} commands)"); return

    # command display policy
    if args.print_cmd == "all":
        for cmd, _, _ in jobs_list: print("$ " + " ".join(shlex.quote(c) for c in cmd))
    elif args.print_cmd == "once" and jobs_list:
        print("$ " + " ".join(shlex.quote(c) for c in jobs_list[0][0]))

    jobs = parse_jobs(args.jobs)
    last_progress_pct = -1
    progress_label = "Progress"
    print(f"[INFO] parallel jobs: {jobs} / ffthreads: {args.ffthreads} / total: {total}")
    if preview_views_line:
        print(preview_views_line)

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
