#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, pathlib, subprocess, sys, os, signal, threading, shlex, re, time
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

def parse_even_pitch_letters(s: str) -> Dict[int, float]:
    out = {}
    if not s: return out
    for part in s.split(","):
        part = part.strip()
        if not part: continue
        if ":" in part: k, v = part.split(":", 1)
        elif "=" in part: k, v = part.split("=", 1)
        else: raise ValueError("区切りが不正: " + part)
        out[letter_to_index1(k)] = float(v)
    return out

def normalize_angle_deg(a: float) -> float:
    a = ((a + 180.0) % 360.0) - 180.0
    return 180.0 if abs(a + 180.0) < 1e-6 else a

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def map_interp_for_v360(name: str) -> str:
    name = (name or "").lower()
    return {"bicubic":"cubic","bilinear":"linear","lanczos":"lanczos"}.get(name, "cubic")

def parse_sensor(s: str) -> float:
    s = s.lower().replace("×","x").replace(",", " ").strip()
    w = s.split("x")[0].strip() if "x" in s else s.split()[0]
    return float(w)

# ---- add/del/setcam パーサ ----
def parse_addcam_spec(spec: str, default_deg: float) -> Dict[int, List[float]]:
    """
    'B' → [+default, -default]
    'B:U' → [+default]
    'D:D20' → [-20]
    'F:U15' → [+15]
    'B, D:U15, F:D10'
    """
    out: Dict[int, List[float]] = {}
    if not spec: return out
    for token in spec.split(","):
        token = token.strip()
        if not token: continue
        if ":" in token or "=" in token:
            k, v = re.split(r"[:=]", token, maxsplit=1)
            idx1 = letter_to_index1(k)
            v = v.strip().upper()
            m = re.match(r'^([UD])\s*([+-]?\d+(?:\.\d+)?)?$', v)
            if m:
                # 絶対: U/D + (度数省略可)
                deg = float(m.group(2)) if m.group(2) else default_deg
                delta = +deg if m.group(1) == 'U' else -deg
                out.setdefault(idx1, []).append(delta)
            else:
                # ': +10' など（相対指定はここでは使わない）→ 無視
                raise ValueError("addcam書式エラー: " + token)
        else:
            idx1 = letter_to_index1(token)
            out.setdefault(idx1, []).extend([+default_deg, -default_deg])
    return out

def parse_delcam_spec(spec: str) -> Set[int]:
    s: Set[int] = set()
    if not spec: return s
    for token in spec.split(","):
        token = token.strip()
        if not token: continue
        s.add(letter_to_index1(token))
    return s

def parse_setcam_spec(spec: str, default_deg: float):
    """
    setcam: ベース角度の上書き/微調整（ピッチ）
      絶対:  A=30, A=-10, A=U30, A=D,  A:U15, A:D
      相対:  A:+10, A:-5
    返り値:
      (abs_map, delta_map)  # idx1 -> float
    """
    abs_map: Dict[int, float] = {}
    delta_map: Dict[int, float] = {}
    if not spec: return abs_map, delta_map

    for token in spec.split(","):
        token = token.strip()
        if not token: continue
        # 分割（= or : を許容）
        if ":" in token or "=" in token:
            k, v = re.split(r"[:=]", token, maxsplit=1)
            idx1 = letter_to_index1(k)
            v2 = v.strip()
            # 相対: +10 / -5
            mrel = re.match(r'^[+|-]\s*\d+(?:\.\d+)?$', v2)
            if mrel:
                delta_map[idx1] = float(v2.replace(" ", ""))
                continue
            # 絶対: U / D / U15 / D20 / 30 / -10
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
                except Exception:
                    raise ValueError("setcam書式エラー: " + token)
        else:
            raise ValueError("setcam書式エラー: " + token)
    return abs_map, delta_map

# ---- ffmpeg コマンド ----
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

# ---- 並列・中断制御 ----
stop_event = threading.Event()
procs_lock = threading.Lock()
running_procs = set()   # type: set


cancel_listener_thread = None
cancel_key_name = ""

def start_cancel_listener(key: str):
    global cancel_listener_thread, cancel_key_name
    key = (key or "").strip()
    if not key:
        return
    if len(key) != 1:
        print(f"[WARN] --cancel-key は1文字で指定してください: '{key}'", file=sys.stderr)
        return
    if cancel_listener_thread and cancel_listener_thread.is_alive():
        return
    if not sys.stdin.isatty():
        print("[WARN] --cancel-key は対話端末でのみ有効です", file=sys.stderr)
        return
    cancel_key_name = key.lower()

    def _listener():
        try:
            if os.name == "nt":
                try:
                    import msvcrt  # type: ignore
                except Exception:
                    print("[WARN] cancel-key listener を初期化できませんでした", file=sys.stderr)
                    return
                print(f"[INFO] Cancel key '{key}' で停止できます", file=sys.stderr)
                while not stop_event.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if not ch:
                            continue
                        try:
                            s = ch.decode("utf-8", "ignore")
                        except Exception:
                            s = ""
                        if s.lower() == cancel_key_name:
                            on_signal(None, None)
                            break
                    time.sleep(0.05)
            else:
                try:
                    import termios, tty, select
                except Exception:
                    print("[WARN] cancel-key listener を初期化できませんでした", file=sys.stderr)
                    return
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    print(f"[INFO] Cancel key '{key}' で停止できます", file=sys.stderr)
                    while not stop_event.is_set():
                        r, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if sys.stdin in r:
                            ch = sys.stdin.read(1)
                            if ch.lower() == cancel_key_name:
                                on_signal(None, None)
                                break
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
        finally:
            return

    cancel_listener_thread = threading.Thread(target=_listener, name="cancel-listener", daemon=True)
    cancel_listener_thread.start()

sig_hits = 0
def on_signal(sig, frame):
    global sig_hits
    sig_hits += 1
    if not stop_event.is_set():
        print("\n[INFO] Cancel requested. 新規ジョブ停止 & 実行中を終了中…", file=sys.stderr)
        stop_event.set()
    with procs_lock:
        for p in list(running_procs):
            try:
                p.terminate() if sig_hits == 1 else p.kill()
            except Exception:
                pass
    if sig_hits >= 2:
        print("[INFO] 強制終了", file=sys.stderr)

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

# ---- メイン ----
def main():
    ap = argparse.ArgumentParser(
        description="Equirect→Perspective batch (v360, 並列/中断/簡略ログ + 任意Cam追加/削除/上書き)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=("メモ: プリセット使用時も --focal-mm / --size / --sensor-mm で上書き可能。"
                " 優先順位: --hfov > --focal-mm。"
                " --setcam でベース角度を文字ごとに絶対/相対指定できます。")
    )
    ap.add_argument("-i","--in", dest="input_dir", required=True,
        help="入力フォルダ。equirect画像(.tif/.tiff/.jpg/.jpeg/.png)を含むパス")
    ap.add_argument("-o","--out", dest="out_dir", default=None,
        help="出力フォルダ。未指定時は input_dir/_geometry")

    ap.add_argument("--preset",
        choices=["default","10views","even30","evenPlus30","evenPlusMinus30"], default="default",
        help=("default=8-view baseline / "
              "10views=8-view baseline + top/bottom (計10枚) / "
              "even30=even slots pitch -30deg / "
              "evenPlus30=even slots pitch +30deg / "
              "evenPlusMinus30=8-view baseline with even slots +/-30deg extras"))

    ap.add_argument("--count", type=int, default=8, help="水平方向分割（4=90°, 8=45°）")

    ap.add_argument("--even-pitch", type=float, default=None,
        help="偶数(B,D,F,...)のピッチを一律オフセット[deg]（例: -30/+30）")
    ap.add_argument("--even-pitch-letters", type=str, default="",
        help="偶数個別（例: 'B:-20,D:+10' または '2:-20,4:+10'）")

    # 追加/削除/上書き
    ap.add_argument("--addcam", default="",
        help="任意カメラを追加：'B'（Bの上下±既定30°）、'B:U'、'D:D20'、'F:U15'、カンマ区切り可")
    ap.add_argument("--addcam-deg", type=float, default=30.0,
        help="--addcam／--setcam の 'U/D' で数値省略時に使う角度[deg]（既定30）")
    ap.add_argument("--add-topdown", action="store_true",
        help="Add cube-map style top (pitch +90deg) and bottom (pitch -90deg) views")

    ap.add_argument("--delcam", default="",
        help="ベース8方向（A..）から削除する文字。例: 'B,D'（B/Dのベース＆派生を全削除）")
    ap.add_argument("--setcam", default="",
        help="ベース角度の上書き/微調整：絶対 'A=30','A=U','A=D20'／相対 'A:+10','B:-5'。複数可")

    # 幾何・出力
    ap.add_argument("--size", type=int, default=1600, help="各ビュー出力サイズ（正方）")
    ap.add_argument("--ext", default="jpg", help="出力拡張子（jpg=最高画質固定）")
    ap.add_argument("--interp", choices=["lanczos","bicubic","bilinear"], default="bicubic",
        help="補間（'bicubic'/'bilinear' は v360内で 'cubic'/'linear' にマップ）")
    ap.add_argument("--hfov", type=float, default=None, help="水平FOV[deg]（指定時は最優先）")
    ap.add_argument("--focal-mm", type=float, default=12.0, help="焦点距離[mm]（--hfov未指定時）")
    ap.add_argument("--sensor-mm", default="36 36", help="センサー寸法[mm]（'36 36' / '36x36' / '36x24'）")

    # 並列・ログ
    ap.add_argument("-j","--jobs", default="auto", help="並列 ffmpeg プロセス数（数値 or 'auto'=論理コアの1/2）")
    ap.add_argument("--ffthreads", default="1", help="各 ffmpeg の内部スレッド数（数値 or 'auto'）")
    ap.add_argument("--print-cmd", choices=["once","none","all"], default="once",
        help="ffmpegフルコマンド表示：once=最初だけ / none=表示なし / all=毎回")
    ap.add_argument("--quiet", action="store_true", help="進捗も抑制（失敗のみエラー表示）")
    ap.add_argument("--cancel-key", default="",
        help="進行中に押すと停止する1文字キー。対話端末でのみ有効")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg 実行ファイルパス")
    ap.add_argument("--dry-run", action="store_true", help="実行せず全コマンド表示して終了")
    args = ap.parse_args()
    start_cancel_listener(args.cancel_key)

    in_dir = pathlib.Path(args.input_dir).expanduser().resolve()
    if not in_dir.is_dir():
        print("[ERR] 入力フォルダが見つかりません:", in_dir, file=sys.stderr); sys.exit(1)
    out_dir = pathlib.Path(args.out_dir).resolve() if args.out_dir else (in_dir / "_geometry")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(in_dir.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
    if not files:
        print("[WARN] 対象画像が見つかりません（tif/jpg/png）", file=sys.stderr); sys.exit(0)

    # ---- プリセット適用（ピッチ系のみ）----
    even_pitch_all = args.even_pitch
    even_pitch_map = parse_even_pitch_letters(args.even_pitch_letters)

    add_topdown = bool(args.add_topdown)

    if args.preset == "10views":
        add_topdown = True
    elif args.preset == "even30" and even_pitch_all is None:
        even_pitch_all = -30.0
    elif args.preset == "evenPlus30" and even_pitch_all is None:
        even_pitch_all = +30.0

    # p5：偶数に上下±30°の追加ビュー（ベースは8方向固定）
    preset_extra_even_pitches = []
    if args.preset == "evenPlusMinus30":
        if args.count != 8 and not args.quiet:
            print("[INFO] preset 'p5' は 8分割前提です。count=8 に上書きします。")
        args.count = 8
        preset_extra_even_pitches = [+30.0, -30.0]

    # add/del/setcam 指定
    add_map = parse_addcam_spec(args.addcam, args.addcam_deg)   # idx1 -> [delta_pitch,...]
    del_set = parse_delcam_spec(args.delcam)                    # idx1 set
    set_abs_map, set_delta_map = parse_setcam_spec(args.setcam, args.addcam_deg if hasattr(args, "addcam_deeg") else args.addcam_deg)

    # ---- FOV/サイズ・有効値 ----
    sensor_w_mm = parse_sensor(args.sensor_mm)
    if args.hfov is not None:
        hfov_deg = float(args.hfov)
        f_used_mm = focal_from_hfov_deg(hfov_deg, sensor_w_mm)
    else:
        f_used_mm = float(args.focal_mm)
        hfov_deg = fov_from_focal_mm(f_used_mm, sensor_w_mm)

    w = h = int(args.size)
    vfov_deg = v_fov_from_hfov(hfov_deg, w, h)

    if not args.quiet:
        print(f"[INFO] effective camera: sensor_w={sensor_w_mm:.2f} mm, focal={f_used_mm:.3f} mm, "
              f"HFOV={hfov_deg:.3f}°, size={w}x{h}")

    # ---- 角度・出力名 ----
    count = int(args.count)
    if count <= 0:
        print("[ERR] --count は 1 以上", file=sys.stderr); sys.exit(1)
    yaw_step = 360.0 / count

    ffmpeg = args.ffmpeg
    ext_dot = "." + args.ext.lower().lstrip(".")
    interp_v360 = map_interp_for_v360(args.interp)

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
    existing_names: Set[str] = set()  # 重複回避

    for img in files:
        stem = img.stem
        base_pitch0 = 0.0
        for yi in range(count):
            if stop_event.is_set(): break
            idx1 = yi + 1
            tag = letter_tag(yi)

            # 削除指定がある場合はベース生成のみスキップ
            skip_base = idx1 in del_set

            yaw = normalize_angle_deg(yi * yaw_step)

            # ---- ベース（プリセット→setcam）----
            pitch = base_pitch0
            if (idx1 % 2) == 0:
                if even_pitch_all is not None: pitch += float(even_pitch_all)
                if idx1 in even_pitch_map:     pitch += float(even_pitch_map[idx1])

            # setcam: 絶対があれば置き換え、相対は加算
            if idx1 in set_abs_map:
                pitch = float(set_abs_map[idx1])
            if idx1 in set_delta_map:
                pitch += float(set_delta_map[idx1])

            pitch = clamp(pitch, -90.0, 90.0)

            # 1) ベース画像（オプションでスキップ）
            if not skip_base:
                out_path = out_dir / f"{stem}_{tag}{ext_dot}"
                if out_path.name not in existing_names:
                    cmd = build_ffmpeg_cmd(ffmpeg, img, out_path, w, h, yaw, pitch,
                                           hfov_deg, vfov_deg, interp_v360, ext_dot, args.ffthreads)
                    jobs_list.append((cmd, img.name, out_path.name))
                    existing_names.add(out_path.name)

            # 2) プリセット extra（ベースがある場合のみ）
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

            # 3) ユーザ指定の追加（delcam有無に関わらず適用）
            if idx1 in add_map:
                for d in add_map[idx1]:
                    p3 = clamp(pitch + d, -90.0, 90.0)
                    suf3 = extra_suffix(d, args.addcam_deg)
                    out_path3 = out_dir / f"{stem}_{tag}{suf3}{ext_dot}"
                    if out_path3.name not in existing_names:
                        cmd3 = build_ffmpeg_cmd(ffmpeg, img, out_path3, w, h, yaw, p3,
                                                hfov_deg, vfov_deg, interp_v360, ext_dot, args.ffthreads)
                        jobs_list.append((cmd3, img.name, out_path3.name))
                        existing_names.add(out_path3.name)



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
                + f" | focal={f_used_mm:.3f}mm | sensor={args.sensor_mm} mm | size={w}x{h}"
            )

    # dry-run: 全コマンド表示して終了
    if args.dry_run:
        for cmd, _, _ in jobs_list: print("$ " + " ".join(shlex.quote(c) for c in cmd))
        print(f"\n[DRY] 実行せず終了（合計 {total} 枚）"); return

    # コマンド表示ポリシー
    if args.print_cmd == "all":
        for cmd, _, _ in jobs_list: print("$ " + " ".join(shlex.quote(c) for c in cmd))
    elif args.print_cmd == "once" and jobs_list:
        print("$ " + " ".join(shlex.quote(c) for c in jobs_list[0][0]))

    jobs = parse_jobs(args.jobs)
    last_progress_pct = -1
    progress_label = "進捗"
    if not args.quiet:
        print(f"[INFO] 並列ジョブ: {jobs} / ffthreads: {args.ffthreads} / 合計: {total}")
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
                if not args.quiet:
                    last_progress_pct = update_progress(progress_label, done, total, last_progress_pct)
            elif rc == 130:
                fail += 1
                if stop_event.is_set():
                    continue
                if not args.quiet and total:
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
                if not args.quiet and total:
                    last_progress_pct = update_progress(progress_label, done, total, last_progress_pct)
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                print(f"[{done}/{total}] {dst} failed", file=sys.stderr)
                if err.strip():
                    print(err.strip(), file=sys.stderr)
    if not args.quiet and total and last_progress_pct >= 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if stop_event.is_set():
        print(f"\n[STOPPED] 中断: success={ok}, failed={fail}, total={total}")
        sys.exit(130)
    else:
        if not args.quiet:
            print(f"\n[OK] 完了: success={ok}, failed={fail}, total={total}")

if __name__ == "__main__":
    main()
