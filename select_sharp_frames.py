#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_sharp_frames.py (listdir-based sharp frame selector)

Workflow:
  1) Gather images directly under the input directory (tif/png/jpg by default)
  2) Sort them according to the chosen rule
  3) Score sharpness in parallel (grayscale read + optional resize/crop)
  4) Split the list into --group batches, keep the sharpest frame(s) per batch,
     and move the rest into in_dir/blur/

Defaults:
  - ext: all (tif/png/jpg)
  - metric: hybrid (0.6 * laplacian variance + 0.4 * fft energy)
  - low group: off unless explicitly enabled (ratio controls extra keeps)
  - crop_ratio: 0.8 (evaluate the central 80%)
  - sort: lastnum (prefer trailing numbers, fallback to name)
  - use_exposure: off (enable with --use_exposure)
  - csv: off (enable with --csv)
  - max_long: 1280 px (0 disables downscale)
  - workers: min(8, os.cpu_count() or 4)
  - opencv_threads: 0 (leave OpenCV threading untouched)

Python 3.7+ / OpenCV 4.x
"""

import os
import sys
import csv
import argparse
import shutil
import re
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


# ---------- Collection and sorting (dedupe) ----------

EXTS = {
    "tif": {".tif", ".tiff"},
    "jpg": {".jpg", ".jpeg"},
    "png": {".png"},
}
ALL_EXTS = set().union(*EXTS.values())

_num_pat = re.compile(r'(\d+)')

def _extract_number_groups(stem):
    return _num_pat.findall(stem)

def sort_key_lastnum(path):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    gs = _extract_number_groups(stem)
    if gs:
        return (0, int(gs[-1]), base.lower())
    return (1, base.lower())

def sort_key_firstnum(path):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    gs = _extract_number_groups(stem)
    if gs:
        return (0, int(gs[0]), base.lower())
    return (1, base.lower())

def sort_key_name(path):
    return os.path.basename(path).lower()

def sort_key_mtime(path):
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

SORTERS = {
    "lastnum": sort_key_lastnum,
    "firstnum": sort_key_firstnum,
    "name": sort_key_name,
    "mtime": sort_key_mtime,
}

def positive_int(value):
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("--group must be a positive integer")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("--group must be a positive integer")
    return ivalue

def crop_ratio_arg(value):
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("--crop_ratio must be a number")
    if not (0.0 < fvalue <= 1.0):
        raise argparse.ArgumentTypeError("--crop_ratio must be in (0, 1]")
    return fvalue

def percentile_arg(value):
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("percentile must be in [0, 100]")
    if not (0.0 <= fvalue <= 100.0):
        raise argparse.ArgumentTypeError("percentile must be in [0, 100]")
    return fvalue

def fraction_arg(value):
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("fraction must be in [0, 1]")
    if not (0.0 <= fvalue <= 1.0):
        raise argparse.ArgumentTypeError("fraction must be in [0, 1]")
    return fvalue

HYBRID_LAPVAR_WEIGHT = 0.6
HYBRID_FFT_WEIGHT = 0.4
PROGRESS_INTERVAL = 5

def update_progress(label, completed, total, last_pct):
    if total <= 0:
        return last_pct
    pct = int((completed * 100) / total)
    if last_pct < 0 or pct >= 100 or pct - last_pct >= PROGRESS_INTERVAL:
        sys.stdout.write(f"{label}... {pct:3d}% ({completed}/{total})\r")
        sys.stdout.flush()
        return pct
    return last_pct

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def unique_path(dst_path):
    if not os.path.exists(dst_path):
        return dst_path
    base, ext = os.path.splitext(dst_path)
    k = 1
    while True:
        cand = "{}_{}{}".format(base, k, ext)
        if not os.path.exists(cand):
            return cand
        k += 1

def safe_move(src, dst):
    """Move a file safely, falling back to copy+delete on failure."""
    if not os.path.isfile(src):
        return None
    dst_final = unique_path(dst)
    ensure_dir(os.path.dirname(dst_final))
    try:
        shutil.move(src, dst_final)
        return dst_final
    except Exception:
        try:
            shutil.copy2(src, dst_final)
            try:
                os.remove(src)
            except Exception:
                pass
            return dst_final
        except Exception:
            return None

def gather_files(in_dir, ext_mode="all"):
    """
    Collect image files in the given directory without descending into subfolders.
    
    Args:
        in_dir (str): Directory to scan for image files.
        ext_mode (str): Extension key or "all" for every supported extension.
    
    Returns:
        list[str]: Absolute file paths that pass the extension filter.
    """
    target_exts = ALL_EXTS if ext_mode == "all" else EXTS[ext_mode]
    raw = []
    for name in os.listdir(in_dir):
        fp = os.path.join(in_dir, name)
        if not os.path.isfile(fp):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() in target_exts:
            raw.append(fp)

    # Deduplicate using normalized absolute paths (case-insensitive on Windows).
    seen = set()
    files = []
    for f in raw:
        key = os.path.normcase(os.path.abspath(f))
        if key in seen:
            continue
        seen.add(key)
        files.append(f)
    return files


# ---------- High-speed scoring helpers ----------

def downscale_gray(gray, max_long):
    """
    Optionally resize the grayscale frame so that the long side stays under max_long.
    
    Args:
        gray (np.ndarray): Input grayscale image.
        max_long (int): Maximum allowed long side length (0 disables scaling).
    
    Returns:
        np.ndarray: Resized (or original) grayscale image.
    """
    if not max_long or max_long <= 0:
        return gray
    h, w = gray.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long:
        return gray
    scale = float(max_long) / float(long_side)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

def crop_by_ratio_gray(gray, crop_ratio):

    """
    Center-crop the grayscale image by the given ratio.
    
    Args:
        gray (np.ndarray): Input grayscale image.
        crop_ratio (float | None): Portion to keep (0 < ratio <= 1).
    
    Returns:
        np.ndarray: Cropped grayscale image.
    """
    if crop_ratio is None:
        return gray
    if not (0.0 < crop_ratio <= 1.0):
        raise ValueError("crop_ratio must be in (0, 1]")
    if abs(crop_ratio - 1.0) < 1e-6:
        return gray
    h, w = gray.shape[:2]
    nh = max(1, int(h * crop_ratio))
    nw = max(1, int(w * crop_ratio))
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    return gray[y0:y0+nh, x0:x0+nw]

def lapvar32(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    # var = (std)^2
    _, std = cv2.meanStdDev(lap)
    return float(std[0,0] * std[0,0])

def tenengrad32(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = cv2.multiply(gx, gx) + cv2.multiply(gy, gy)
    m = cv2.mean(mag2)[0]
    return float(m)

def fft_energy_fast(gray):
    """
    Estimate sharpness from the mean magnitude of high-frequency FFT components.
    
    Args:
        gray (np.ndarray): Input grayscale image.
    
    Returns:
        float: Mean magnitude of the clipped FFT spectrum.
    """
    g = downscale_gray(gray, 512)
    f = np.fft.fft2(g.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = g.shape
    cy, cx = h//2, w//2
    r = max(1, min(h, w) // 8)  # Reject low frequencies around the center
    # Use a donut-shaped mask to keep only high-frequency energy
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy)**2 + (xx - cx)**2
    mask = (dist2 >= r*r).astype(np.float32)
    hf = fshift * mask
    return float(np.mean(np.abs(hf)))

    # Assume 8-bit input (IMREAD_GRAYSCALE).
    p0 = float(np.mean(gray <= 2))
    p255 = float(np.mean(gray >= 253))
    return p0, p255

def score_one_file(
        fp,
        metric,
        crop_ratio,
        use_exposure,
        clip_penalty,
        clip_thresh,
        max_long,
):
    """
    Compute the sharpness score (and optional exposure statistics) for one file.
    
    Args:
        fp (str): Path to the image file.
        metric (str): Selected metric name.
        crop_ratio (float | None): Ratio used for center cropping.
        use_exposure (bool): Whether to apply exposure penalties.
        clip_penalty (float): Multiplier applied when exposure is clipped.
        clip_thresh (float): Exposure clipping threshold.
        max_long (int): Optional maximum long edge for downscaling.
    
    Returns:
        tuple[float | None, float, float]: (score, black_ratio, white_ratio).
    """
    try:
        gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None, 0.0, 0.0
        gray = downscale_gray(gray, max_long)
        gray = crop_by_ratio_gray(gray, crop_ratio)

        if metric == "lapvar":
            sharp = lapvar32(gray)
        elif metric == "tenengrad":
            sharp = tenengrad32(gray)
        elif metric == "fft":
            sharp = fft_energy_fast(gray)
        elif metric == "hybrid":
            lap_score = lapvar32(gray)
            fft_score = fft_energy_fast(gray)
            sharp = (
                HYBRID_LAPVAR_WEIGHT * lap_score
                + HYBRID_FFT_WEIGHT * fft_score
            )
        else:
            return None, 0.0, 0.0

        if use_exposure:
            p0, p255 = exposure_clip_stats(gray)
            clip = p0 + p255
            if clip > clip_thresh:
                sharp *= clip_penalty
            return sharp, p0, p255
        else:
            return sharp, 0.0, 0.0

    except ValueError:
        raise
    except Exception:
        return None, 0.0, 0.0



def _score_or_negative_infinity(scores, index):
    """Return the score for an index, falling back to negative infinity.

    Args:
        scores (list[float | None]): Sharpness scores.
        index (int): Frame index.

    Returns:
        float: Score value or negative infinity when the score is missing.
    """
    value = scores[index]
    return float(value) if value is not None else float("-inf")

def _pick_even_candidate(
    existing_indices, initial_selected, scores, used, target_pos
):
    """Pick the best candidate near the target position for even spacing.

    Args:
        existing_indices (list[int]): Indices of frames that still exist on disk.
        initial_selected (set[int]): Indices selected during grouping.
        scores (list[float | None]): Sharpness scores for all frames.
        used (set[int]): Indices already chosen for the final selection.
        target_pos (int): Desired position in the existing_indices list.

    Returns:
        int | None: The chosen index or None when nothing is left.
    """
    length = len(existing_indices)
    if length == 0:
        return None

    best = None
    best_key = None
    radius = 0
    while radius < length:
        start_pos = max(0, target_pos - radius)
        end_pos = min(length, target_pos + radius + 1)
        for pos in range(start_pos, end_pos):
            idx = existing_indices[pos]
            if idx in used:
                continue
            key = (
                1 if idx in initial_selected else 0,
                _score_or_negative_infinity(scores, idx),
                -abs(pos - target_pos),
                -idx,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = idx
        if best is not None:
            return best
        radius += 1
    return None

def evenly_distribute_indices(existing_indices, initial_selected, scores):
    """Return an evenly spaced selection that still prefers sharp frames.

    Args:
        existing_indices (list[int]): Indices of frames that still exist on disk.
        initial_selected (set[int]): Indices selected during grouping.
        scores (list[float | None]): Sharpness scores for all frames.

    Returns:
        set[int]: Final indices to keep after spacing adjustments.
    """
    desired_count = len(initial_selected)
    if desired_count <= 0:
        return set()
    if desired_count >= len(existing_indices):
        return set(existing_indices)

    used = set()
    selected = []
    max_pos = len(existing_indices) - 1
    step = max_pos / max(desired_count - 1, 1)

    for order in range(desired_count):
        if desired_count == 1:
            target_pos = max_pos // 2
        else:
            target_pos = int(round(order * step))
        candidate = _pick_even_candidate(
            existing_indices,
            initial_selected,
            scores,
            used,
            target_pos,
        )
        if candidate is None:
            break
        used.add(candidate)
        selected.append(candidate)

    if len(selected) < desired_count:
        remaining = [idx for idx in existing_indices if idx not in used]
        remaining.sort(
            key=lambda idx: (
                1 if idx in initial_selected else 0,
                _score_or_negative_infinity(scores, idx),
                -idx,
            ),
            reverse=True,
        )
        needed = desired_count - len(selected)
        selected.extend(remaining[:needed])

    return set(selected)
# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Score frames, keep the sharp ones, and move the rest into in_dir/blur."
        )
    )
    ap.add_argument(
        "-i",
        "--in_dir",
        required=True,
        help="Input directory containing frames (non-recursive).",
    )
    ap.add_argument(
        "-g",
        "--group",
        type=positive_int,
        required=True,
        help="Number of frames per group before selection.",
    )
    ap.add_argument(
        "-e",
        "--ext",
        choices=["all", "tif", "jpg", "png"],
        default="all",
        help="File extension filter (default: all).",
    )
    ap.add_argument(
        "-s",
        "--sort",
        choices=["lastnum", "firstnum", "name", "mtime"],
        default="lastnum",
        help="Sorting rule applied before scoring.",
    )
    ap.add_argument(
        "-m",
        "--metric",
        choices=["hybrid", "lapvar", "tenengrad", "fft"],
        default="hybrid",
        help="Sharpness metric (default: hybrid 0.6*lapvar + 0.4*fft).",
    )
    ap.add_argument(
        "-P",
        "--low_group_percentile",
        type=percentile_arg,
        default=30.0,
        help=(
            "Percentile threshold for detecting low-quality groups (default: 30)."
        ),
    )
    ap.add_argument(
        "-r",
        "--low_group_keep_ratio",
        type=fraction_arg,
        default=0.4,
        help=(
            "Fraction of frames to keep in low-quality groups (0 disables)."
        ),
    )
    ap.add_argument(
        "-k",
        "--low_group_min_keep",
        type=positive_int,
        default=2,
        help="Minimum number of frames to keep when a group is flagged low.",
    )
    ap.add_argument(
        "-c",
        "--crop_ratio",
        type=crop_ratio_arg,
        default=0.8,
        help="Center crop ratio used during scoring (0.8 keeps 80%%).",
    )
    ap.add_argument(
        "-E",
        "--use_exposure",
        action="store_true",
        help="Apply exposure penalty when black/white clipping is detected.",
    )
    ap.add_argument(
        "-p",
        "--clip_penalty",
        type=float,
        default=0.5,
        help="Multiplier applied to scores when clipping exceeds the threshold.",
    )
    ap.add_argument(
        "-t",
        "--clip_thresh",
        type=float,
        default=0.25,
        help="Exposure clipping threshold for applying the penalty.",
    )
    ap.add_argument(
        "-M",
        "--max_long",
        type=int,
        default=0,
        help="Maximum long edge for scoring (0 keeps the original resolution).",
    )
    ap.add_argument(
        "-w",
        "--workers",
        type=int,
        help="Override the worker pool size (default: min(8, cpu or 4)).",
    )
    ap.add_argument(
        "-o",
        "--opencv_threads",
        type=int,
        default=0,
        help="Set OpenCV thread count (0 leaves the default).",
    )
    ap.add_argument(
        "-C",
        "--csv",
        help="Optional CSV output path relative to the input directory.",
    )
    args = ap.parse_args()

    # Keep OpenCV from competing with the Python thread pool.
    try:
        if args.opencv_threads and args.opencv_threads > 0:
            cv2.setNumThreads(args.opencv_threads)
    except Exception:
        pass

    files = gather_files(args.in_dir, args.ext)
    if not files:
        print(f"No input images found: {args.in_dir}")
        sys.exit(1)

    sorter = SORTERS[args.sort]
    files = sorted(files, key=sorter)

    blur_dir = os.path.join(args.in_dir, "blur")
    ensure_dir(blur_dir)

    # Score every file in parallel
    n = len(files)
    scores = [None] * n
    p0_arr = [0.0] * n
    p255_arr = [0.0] * n

    workers = (
        args.workers
        if (args.workers and args.workers > 0)
        else min(8, (os.cpu_count() or 4))
    )
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                score_one_file, files[i],
                args.metric, args.crop_ratio,
                args.use_exposure, args.clip_penalty, args.clip_thresh,
                args.max_long
            ): i for i in range(n)
        }
        completed = 0
        last_pct = -1
        for fut in as_completed(futs):
            i = futs[fut]
            s, p0, p255 = fut.result()
            scores[i] = s
            p0_arr[i] = p0
            p255_arr[i] = p255
            completed += 1
            last_pct = update_progress("Scoring", completed, n, last_pct)

    if n:
        print(f"Scoring... 100% ({n}/{n})")
    # Prepare optional CSV output
    csv_writer = None
    fcsv = None
    if args.csv:
        csv_path = (
            args.csv
            if os.path.isabs(args.csv)
            else os.path.join(args.in_dir, args.csv)
        )
        fcsv = open(csv_path, "w", newline="")
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(
            [
                "index",
                "filename",
                "score",
                "p0_black",
                "p255_white",
                "selected(1=keep)",
            ]
        )

    # Grouping and relocation
    total = n
    group_infos = []
    for grp_start in range(0, total, args.group):
        grp_end = min(total, grp_start + args.group)
        valid_idx = []
        group_sum = 0.0
        for i in range(grp_start, grp_end):
            s = scores[i]
            if s is None:
                continue
            valid_idx.append(i)
            if s > 0.0:
                group_sum += s
        group_infos.append(
            {
                "start": grp_start,
                "end": grp_end,
                "valid_idx": valid_idx,
                "group_sum": group_sum,
            }
        )

    group_sums = [info["group_sum"] for info in group_infos if info["valid_idx"]]
    low_threshold = None
    if args.low_group_keep_ratio > 0.0:
        if group_sums:
            low_threshold = float(
                np.percentile(group_sums, args.low_group_percentile)
            )
        else:
            low_threshold = 0.0

    initial_selected = set()
    for info in group_infos:
        grp_start = info["start"]
        grp_end = info["end"]
        group_range = range(grp_start, grp_end)
        existing_indices = [
            i for i in group_range if os.path.isfile(files[i])
        ]
        valid_indices = [
            i for i in existing_indices if scores[i] is not None
        ]

        is_low_group = False
        if args.low_group_keep_ratio > 0.0:
            if not valid_indices:
                is_low_group = True
            elif (
                low_threshold is None
                or info["group_sum"] <= low_threshold
            ):
                is_low_group = True

        if not valid_indices:
            selected_indices = set(existing_indices)
        else:
            sorted_valid = sorted(
                valid_indices,
                key=lambda idx: (scores[idx], idx),
                reverse=True,
            )
            keep_count = 1
            if is_low_group:
                desired = max(
                    args.low_group_min_keep,
                    int(
                        math.ceil(
                            len(sorted_valid) * args.low_group_keep_ratio
                        )
                    ),
                )
                keep_count = min(len(sorted_valid), max(1, desired))
            selected_indices = set(sorted_valid[:keep_count])

        initial_selected.update(selected_indices)

    existing_indices = [
        i for i in range(total) if os.path.isfile(files[i])
    ]
    initial_selected &= set(existing_indices)
    final_selected = evenly_distribute_indices(
        existing_indices,
        initial_selected,
        scores,
    )
    if not final_selected and initial_selected:
        final_selected = set(initial_selected)

    kept = 0
    moved = 0
    skipped = 0
    processed = 0
    last_group_pct = -1

    for i in range(total):
        s = scores[i]
        processed += 1
        file_exists = os.path.isfile(files[i])
        if not file_exists or s is None:
            skipped += 1
            if csv_writer:
                csv_writer.writerow(
                    [i, os.path.basename(files[i]), -1.0, 0.0, 0.0, 0]
                )
            last_group_pct = update_progress(
                "Grouping",
                processed,
                total,
                last_group_pct,
            )
            continue

        if i in final_selected:
            kept += 1
            if csv_writer:
                csv_writer.writerow(
                    [
                        i,
                        os.path.basename(files[i]),
                        s,
                        p0_arr[i],
                        p255_arr[i],
                        1,
                    ]
                )
        else:
            dst = os.path.join(blur_dir, os.path.basename(files[i]))
            if safe_move(files[i], dst) is None:
                skipped += 1
            else:
                moved += 1
            if csv_writer:
                csv_writer.writerow(
                    [
                        i,
                        os.path.basename(files[i]),
                        s,
                        p0_arr[i],
                        p255_arr[i],
                        0,
                    ]
                )
        last_group_pct = update_progress(
            "Grouping",
            processed,
            total,
            last_group_pct,
        )
    if total:
        print(f"Grouping... 100% ({total}/{total})")
    if fcsv:
        fcsv.close()

    print(f"Done: input {total} / kept {kept} / moved {moved} / skipped {skipped}")
    print("Blur directory:", blur_dir)
    print(
        f"workers={workers}, opencv_threads={args.opencv_threads}, "
        f"max_long={args.max_long}"
    )


if __name__ == "__main__":
    main()
