#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_sharp_frames.py (listdir-based sharp frame selector)

Workflow (after this update):
  1) Gather images directly under the input directory (tif/png/jpg by default)
  2) Sort them according to the chosen rule
  3) Score sharpness in parallel (grayscale read + optional resize/crop)
  4) Split the list into --segment_size batches and do an initial "sharpness-driven" selection
     (optionally more keeps for low-quality groups)
  5) Evenly distribute the initially selected indices over the whole sequence (min spacing aware)
  6) Gap augmentation: backfill when gaps between selected frames exceed --max_spacing
  7) Brightness+Sharpness in-group augmentation (NEW): after gap augmentation, add frames with
     high (score * brightness^power) inside each segment, respecting min spacing and a per-segment budget
  8) Motion enhancement (optional): if --enhance_motion, add extra frames in high-motion segments
  9) (Optional) Motion pruning: if --prune_motion, drop the lowest-motion subset (bottom percentile)
 10) Move non-selected images into in_dir/blur/ (or only write CSV in --dry_run)

Defaults:
  - ext: all (tif/png/jpg)
  - metric: hybrid (0.6 * laplacian variance + 0.3 * tenengrad + 0.2 * fft)  [normalized blending]
  - low group: off unless explicitly enabled (ratio controls extra keeps)
  - crop_ratio: 0.8 (evaluate the central 80%)
  - sort: lastnum (prefer trailing numbers, fallback to name)
  - use_exposure: off (enable with --use_exposure)
  - csv: off (enable with --csv)
  - max_long: 1280 px (0 disables downscale)
  - workers: min(8, os.cpu_count() or 4)
  - opencv_threads: 0 (leave OpenCV threading untouched)
  - augment_bs: ON by default (disable via --no_augment_bs)

Python 3.7+ / OpenCV 4.x
"""

import os
import sys
import csv
import argparse
import shutil
import re
import math
from bisect import bisect_left, insort
from concurrent.futures import ThreadPoolExecutor, as_completed

import signal
import threading

import cv2

# Detect whether stdout is an interactive TTY
IS_TTY = False
try:
    IS_TTY = bool(sys.stdout.isatty())
except Exception:
    IS_TTY = False
import numpy as np


cancel_event = threading.Event()


def _handle_sigint(signum, frame):
    if not cancel_event.is_set():
        print("\\nCancellation requested (Ctrl+C). Finishing current tasks...")
        cancel_event.set()


def start_cancel_listener():
    """Start a background listener that cancels on 'q' input."""
    if not sys.stdin or not sys.stdin.isatty():
        return None

    def _watch():
        try:
            while not cancel_event.is_set():
                line = sys.stdin.readline()
                if not line:
                    break
                if line.strip().lower() == "q":
                    print("\\nCancellation requested (q). Finishing current tasks...")
                    cancel_event.set()
                    break
        except Exception:
            pass

    thread = threading.Thread(target=_watch, name="cancel-listener", daemon=True)
    thread.start()
    return thread


# ---------- Collection and sorting (dedupe) ----------

EXTS = {
    "tif": {".tif", ".tiff"},
    "jpg": {".jpg", ".jpeg"},
    "png": {".png"},
}
ALL_EXTS = set().union(*EXTS.values())

_num_pat = re.compile(r'(\\d+)')

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
        raise argparse.ArgumentTypeError("--segment_size must be a positive integer")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("--segment_size must be a positive integer")
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


# Validators
def non_negative_int(value):
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("value must be >= 0")
    if ivalue < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return ivalue

HYBRID_LAPVAR_WEIGHT = 0.5
HYBRID_TENENGRAD_WEIGHT = 0.3
HYBRID_FFT_WEIGHT = 0.2
HYBRID_MOTION_REFERENCE = 5000.0
HYBRID_MOTION_PENALTY_WEIGHT = 0.4
MOTION_ANISO_WEIGHT = 0.5
FLOW_DOWNSCALE = 320
FLOW_MOTION_WEIGHT = 0.6
FLOW_HIGH_MOTION_THRESHOLD = 0.5
FLOW_HIGH_MOTION_RATIO = 0.4
FLOW_LOW_MOTION_PERCENTILE = 10.0
FLOW_MISSING_HIGH_VALUE = 9999.0
FLOW_CROP_RATIO = 0.5
FAST_SPACING_WINDOW = 64
FAST_SPACING_MULTIPLIER = 4.0
GROUP_BRIGHTNESS_POWER = 1.5
HYBRID_DARK_THRESHOLD = 0.35
HYBRID_DARK_PENALTY_WEIGHT = 0.5
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


def round_half_up(value):
    """Return the value rounded to the nearest integer (half up)."""
    return int(math.floor(value + 0.5))


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

def exposure_clip_stats(gray):
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
        enhance_motion,
):
    """Compute the sharpness score and ancillary metrics for one frame."""
    try:
        gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None, 0.0, 0.0, 0.0, 1.0, None, None, None, 1.0, 1.0
        gray = downscale_gray(gray, max_long)
        gray = crop_by_ratio_gray(gray, crop_ratio)

        brightness_mean = float(np.mean(gray) / 255.0)
        brightness_weight = 1.0
        lap_feature = None
        ten_feature = None
        fft_feature = None
        motion_factor = 1.0
        exposure_factor = 1.0

        if metric == "lapvar":
            lap_score = lapvar32(gray)
            sharp = lap_score
            lap_feature = lap_score * lap_score
        elif metric == "tenengrad":
            ten_score = tenengrad32(gray)
            sharp = ten_score
            ten_feature = ten_score
        elif metric == "fft":
            fft_score = fft_energy_fast(gray)
            sharp = fft_score
            fft_feature = fft_score
        elif metric == "hybrid":
            lap_score = lapvar32(gray)
            ten_score = tenengrad32(gray)
            fft_score = fft_energy_fast(gray)
            lap_energy = lap_score * lap_score
            lap_feature = lap_energy
            ten_feature = ten_score
            fft_feature = fft_score

            hybrid_raw = (
                HYBRID_LAPVAR_WEIGHT * lap_energy
                + HYBRID_TENENGRAD_WEIGHT * ten_score
                + HYBRID_FFT_WEIGHT * fft_score
            )

            motion_ratio = ten_score / (ten_score + HYBRID_MOTION_REFERENCE)
            motion_ratio = max(0.0, min(1.0, motion_ratio))
            motion_factor = 1.0 - HYBRID_MOTION_PENALTY_WEIGHT * (1.0 - motion_ratio)
            motion_factor = max(0.0, motion_factor)

            if enhance_motion:
                # Motion-sensitive enhancement is now handled during post-selection augmentation.
                pass

            if brightness_mean < HYBRID_DARK_THRESHOLD:
                dark_ratio = brightness_mean / HYBRID_DARK_THRESHOLD
            else:
                dark_ratio = 1.0
            dark_ratio = max(0.0, min(1.0, dark_ratio))
            brightness_weight = 1.0 - HYBRID_DARK_PENALTY_WEIGHT * (1.0 - dark_ratio)
            brightness_weight = max(0.0, brightness_weight)

            sharp = hybrid_raw * motion_factor
        else:
            return None, 0.0, 0.0, brightness_mean, brightness_weight, None, None, None, 1.0, 1.0

        if use_exposure:
            p0, p255 = exposure_clip_stats(gray)
            clip = p0 + p255
            if clip > clip_thresh:
                exposure_factor = clip_penalty
            sharp *= exposure_factor
        else:
            p0, p255 = 0.0, 0.0

        return (
            sharp,
            p0,
            p255,
            brightness_mean,
            brightness_weight,
            lap_feature,
            ten_feature,
            fft_feature,
            motion_factor,
            exposure_factor,
        )

    except ValueError:
        raise
    except Exception:
        return None, 0.0, 0.0, 0.0, 1.0, None, None, None, 1.0, 1.0





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

def _spacing_respects(sorted_selected, candidate, min_diff):
    """Return True when candidate keeps at least min_diff distance."""
    if min_diff <= 1 or not sorted_selected:
        return True
    pos = bisect_left(sorted_selected, candidate)
    if pos > 0 and candidate - sorted_selected[pos - 1] < min_diff:
        return False
    if pos < len(sorted_selected) and sorted_selected[pos] - candidate < min_diff:
        return False
    return True


def _pick_even_candidate(
    existing_indices,
    initial_selected,
    scores,
    used,
    target_pos,
    sorted_selected,
    min_diff,
    fast_window=FAST_SPACING_WINDOW,
):
    """Pick the best candidate near the target position for even spacing."""
    length = len(existing_indices)
    if length == 0:
        return None

    best_idx = None
    best_key = None
    window_start = max(0, target_pos - fast_window)
    window_end = min(length, target_pos + fast_window + 1)
    ranges = [range(window_start, window_end)]
    if window_start > 0 or window_end < length:
        ranges.append(range(0, length))

    seen_positions = set()
    for pos_range in ranges:
        for pos in pos_range:
            if pos in seen_positions:
                continue
            seen_positions.add(pos)
            idx = existing_indices[pos]
            if idx in used:
                continue
            score = scores[idx]
            if score is None:
                continue
            if min_diff > 1 and not _spacing_respects(sorted_selected, idx, min_diff):
                continue
            key = (
                1 if idx in initial_selected else 0,
                _score_or_negative_infinity(scores, idx),
                -abs(pos - target_pos),
                -idx,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        if best_idx is not None:
            break
    return best_idx


def _pick_best_between(
    existing_indices,
    scores,
    used,
    start_pos,
    end_pos,
    target_pos,
    initial_selected,
    sorted_selected,
    min_diff,
    fast_window=FAST_SPACING_WINDOW,
):
    """Pick a frame between two positions prioritizing sharpness and proximity."""
    if end_pos - start_pos <= 1:
        return None

    best_idx = None
    best_key = None
    window_start = max(start_pos + 1, target_pos - fast_window)
    window_end = min(end_pos, target_pos + fast_window + 1)
    ranges = [range(window_start, window_end)]
    if window_start > start_pos + 1 or window_end < end_pos:
        ranges.append(range(start_pos + 1, end_pos))

    seen_positions = set()
    for pos_range in ranges:
        for pos in pos_range:
            if pos <= start_pos or pos >= end_pos:
                continue
            if pos in seen_positions:
                continue
            seen_positions.add(pos)
            idx = existing_indices[pos]
            if idx in used:
                continue
            score = scores[idx]
            if score is None:
                continue
            if min_diff > 1 and not _spacing_respects(sorted_selected, idx, min_diff):
                continue
            key = (
                1 if idx in initial_selected else 0,
                score,
                -abs(pos - target_pos),
                -idx,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        if best_idx is not None:
            break
    return best_idx


def augment_spacing(
    final_selected,
    existing_indices,
    scores,
    initial_selected,
    max_spacing,
    min_diff,
    fast_window=FAST_SPACING_WINDOW,
):
    """Augment the selection by inserting frames when spacing exceeds the limit."""
    if max_spacing is None or max_spacing <= 0:
        return set(final_selected)
    position_map = {idx: pos for pos, idx in enumerate(existing_indices)}
    augmented = set(final_selected)
    used = set(final_selected)
    selected_sorted = sorted(augmented)

    changed = True
    while changed:
        changed = False
        for i in range(len(selected_sorted) - 1):
            left_idx = selected_sorted[i]
            right_idx = selected_sorted[i + 1]
            pos_left = position_map.get(left_idx)
            pos_right = position_map.get(right_idx)
            if pos_left is None or pos_right is None:
                continue
            gap = pos_right - pos_left
            if gap <= max_spacing:
                continue
            target_pos = int(round((pos_left + pos_right) / 2.0))
            candidate = _pick_best_between(
                existing_indices,
                scores,
                used,
                pos_left,
                pos_right,
                target_pos,
                initial_selected,
                selected_sorted,
                min_diff,
                fast_window,
            )
            if candidate is None:
                continue
            augmented.add(candidate)
            used.add(candidate)
            insort(selected_sorted, candidate)
            changed = True
            break
    return augmented


def _load_flow_gray(path, crop_ratio):
    """Load and optionally downscale a grayscale frame for flow."""
    flow_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if flow_gray is None:
        return None
    h, w = flow_gray.shape[:2]
    if crop_ratio and 0.0 < crop_ratio < 1.0:
        crop_h = max(1, int(round(h * crop_ratio)))
        crop_w = max(1, int(round(w * crop_ratio)))
        start_y = max(0, (h - crop_h) // 2)
        start_x = max(0, (w - crop_w) // 2)
        end_y = start_y + crop_h
        end_x = start_x + crop_w
        flow_gray = flow_gray[start_y:end_y, start_x:end_x]
        h, w = flow_gray.shape[:2]
    if FLOW_DOWNSCALE and max(h, w) > FLOW_DOWNSCALE:
        scale = FLOW_DOWNSCALE / float(max(h, w))
        flow_gray = cv2.resize(
            flow_gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return flow_gray





def _compute_pair_flow_magnitude(prev_path, curr_path, crop_ratio):
    prev_gray = _load_flow_gray(prev_path, crop_ratio)
    if prev_gray is None:
        return None
    curr_gray = _load_flow_gray(curr_path, crop_ratio)
    if curr_gray is None or prev_gray.shape != curr_gray.shape:
        return None
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 1, 15, 3, 5, 1.1, 0)
    except Exception:
        return None
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = float(np.mean(mag))
    if not np.isfinite(mean_mag):
        return None
    return mean_mag


def _compute_flow_magnitudes(files, flow_mag_arr, flow_crop_ratio, workers, label):
    """Compute mean optical-flow magnitudes between consecutive frames."""
    if len(files) < 2:
        return 0

    pair_indices = []
    prev_idx = None
    for idx, fp in enumerate(files):
        if cancel_event.is_set():
            break
        if not os.path.isfile(fp):
            prev_idx = None
            continue
        if prev_idx is not None:
            pair_indices.append((prev_idx, idx))
        prev_idx = idx

    total_pairs = len(pair_indices)
    if total_pairs == 0:
        return 0

    completed = 0
    last_pct = -1

    def _process_pair(pair):
        left_idx, right_idx = pair
        mean_mag = _compute_pair_flow_magnitude(files[left_idx], files[right_idx], flow_crop_ratio)
        if mean_mag is None or not math.isfinite(mean_mag):
            mean_mag = FLOW_MISSING_HIGH_VALUE
        return left_idx, right_idx, mean_mag

    with ThreadPoolExecutor(max_workers=workers) as flow_executor:
        futs = {flow_executor.submit(_process_pair, pair): pair for pair in pair_indices}
        try:
            for fut in as_completed(futs):
                if cancel_event.is_set():
                    break
                left_idx, right_idx = futs[fut]
                try:
                    _, _, mean_mag = fut.result()
                except Exception:
                    mean_mag = FLOW_MISSING_HIGH_VALUE
                if mean_mag is None or not math.isfinite(mean_mag):
                    mean_mag = FLOW_MISSING_HIGH_VALUE
                flow_mag_arr[right_idx] = max(flow_mag_arr[right_idx], mean_mag)
                flow_mag_arr[left_idx] = max(flow_mag_arr[left_idx], mean_mag)
                completed += 1
                last_pct = update_progress(label, completed, total_pairs, last_pct)
        except KeyboardInterrupt:
            cancel_event.set()

    return completed



def load_selection_from_csv(csv_path, files, scores, brightness_mean_arr, group_score_arr, flow_mag_arr):
    """Load selection flags and metrics from an existing CSV."""
    selection_flags = [0] * len(files)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header")
        fields_lower = {name.lower(): name for name in reader.fieldnames}
        if "selected(1=keep)" in fields_lower:
            selected_key = fields_lower["selected(1=keep)"]
        elif "selected" in fields_lower:
            selected_key = fields_lower["selected"]
        else:
            raise ValueError("CSV missing 'selected(1=keep)' column")
        index_key = fields_lower.get("index")
        score_key = fields_lower.get("score")
        brightness_key = fields_lower.get("brightness_mean")
        group_key = fields_lower.get("group_score")
        flow_key = fields_lower.get("flow_motion")

        for row in reader:
            if index_key is None:
                raise ValueError("CSV missing 'index' column")
            try:
                idx = int(row[index_key])
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(files):
                continue
            flag_raw = str(row.get(selected_key, "0")).strip()
            keep_flag = 1 if flag_raw in {"1", "true", "True"} else 0
            selection_flags[idx] = keep_flag

            if score_key and row.get(score_key) not in (None, ""):
                try:
                    score_val = float(row[score_key])
                except ValueError:
                    scores[idx] = None
                else:
                    scores[idx] = None if score_val < 0.0 else score_val
            if brightness_key and row.get(brightness_key) not in (None, ""):
                try:
                    brightness_mean_arr[idx] = float(row[brightness_key])
                except ValueError:
                    pass
            if group_key and row.get(group_key) not in (None, ""):
                try:
                    group_score_arr[idx] = float(row[group_key])
                except ValueError:
                    pass
            if flow_key and row.get(flow_key) not in (None, ""):
                try:
                    flow_mag_arr[idx] = float(row[flow_key])
                except ValueError:
                    pass
    return selection_flags

def augment_motion_segments(
    final_selected,
    group_infos,
    existing_indices,
    scores,
    flow_mag_arr,
    min_diff,
):
    """Add extra frames to high-motion groups after gap augmentation."""
    motion_values = [v for v in flow_mag_arr if v > 0.0 and np.isfinite(v)]
    if not motion_values:
        return set(final_selected)

    percentile_threshold = float(np.percentile(motion_values, 80.0))
    threshold = max(FLOW_HIGH_MOTION_THRESHOLD, percentile_threshold)
    augmented = set(final_selected)
    existing_set = set(existing_indices)
    ratio_limit = max(0.0, min(1.0, FLOW_HIGH_MOTION_RATIO))
    spacing = max(1, min_diff)

    for info in group_infos:
        start = info["start"]
        end = info["end"]
        segment_indices = [
            idx
            for idx in range(start, end)
            if idx in existing_set and scores[idx] is not None and np.isfinite(flow_mag_arr[idx])
        ]
        if not segment_indices:
            continue

        segment_motion = max(flow_mag_arr[idx] for idx in segment_indices)
        if not np.isfinite(segment_motion) or segment_motion < threshold:
            continue

        current_in_segment = [idx for idx in augmented if start <= idx < end]
        segment_span = max(1, end - start)
        spacing_limit = math.ceil(segment_span / spacing)
        available_budget = max(0, spacing_limit - len(current_in_segment))
        if available_budget <= 0:
            continue

        if ratio_limit > 0.0:
            ratio_cap = max(1, round_half_up(segment_span * ratio_limit))
            available_budget = min(available_budget, ratio_cap)
            if available_budget <= 0:
                continue

        candidates = [idx for idx in segment_indices if idx not in augmented]
        if not candidates:
            continue

        candidates.sort(
            key=lambda idx: (
                flow_mag_arr[idx],
                _score_or_negative_infinity(scores, idx),
                -idx,
            ),
            reverse=True,
        )

        added = 0
        for idx in candidates:
            if added >= available_budget:
                break
            if min_diff > 1 and any(abs(idx - sel) < min_diff for sel in augmented):
                continue
            augmented.add(idx)
            added += 1

    return augmented

def evenly_distribute_indices(existing_indices, initial_selected, scores, min_diff, fast_window):
    """Return an evenly spaced selection that still prefers sharp frames."""
    desired_count = len(initial_selected)
    if desired_count <= 0:
        return set()
    if desired_count >= len(existing_indices):
        return set(existing_indices)

    used = set()
    selected_sorted = []
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
            selected_sorted,
            min_diff,
            fast_window,
        )
        if candidate is None:
            break
        used.add(candidate)
        insort(selected_sorted, candidate)

    if len(selected_sorted) < desired_count:
        remaining = [idx for idx in existing_indices if idx not in used]
        remaining.sort(
            key=lambda idx: (
                1 if idx in initial_selected else 0,
                _score_or_negative_infinity(scores, idx),
                -idx,
            ),
            reverse=True,
        )
        for idx in remaining:
            if len(selected_sorted) >= desired_count:
                break
            if min_diff > 1 and not _spacing_respects(selected_sorted, idx, min_diff):
                continue
            used.add(idx)
            insort(selected_sorted, idx)

    return set(selected_sorted)


# ---------- New: Brightness+Sharpness in-group augmentation ----------

def augment_brightness_sharpness_segments(
    final_selected,
    group_infos,
    existing_indices,
    scores,
    brightness_mean_arr,
    min_diff,
    keep_ratio,
    min_keep,
):
    """
    After gap augmentation, add more frames within each segment using a brightness-weighted sharpness:

        bs_score = scores[idx] * (brightness_mean[idx] ** GROUP_BRIGHTNESS_POWER)

    - Only considers frames that exist, are scored, and are not already selected.
    - Enforces spacing by min_diff.
    - Per-segment budget = max(min_keep, round(span * keep_ratio)).
    """
    if keep_ratio <= 0.0 and min_keep <= 0:
        return set(final_selected)

    augmented = set(final_selected)
    existing_set = set(existing_indices)

    for info in group_infos:
        start = info["start"]
        end = info["end"]
        span = max(1, end - start)

        budget = max(int(round(span * max(0.0, min(1.0, keep_ratio)))), int(min_keep))
        if budget <= 0:
            continue

        # Candidates = frames in this segment that are valid and not yet selected
        candidates = [
            idx for idx in range(start, end)
            if (idx in existing_set) and (scores[idx] is not None) and (idx not in augmented)
        ]
        if not candidates:
            continue

        def bs_score(i):
            b = max(1e-6, float(brightness_mean_arr[i]))
            return float(scores[i]) * (b ** GROUP_BRIGHTNESS_POWER)

        # Sort by bs_score then raw score (both high-first)
        candidates.sort(key=lambda i: (bs_score(i), _score_or_negative_infinity(scores, i), -i), reverse=True)

        added = 0
        sorted_selected = sorted(augmented)
        for idx in candidates:
            if added >= budget:
                break
            if min_diff > 1 and not _spacing_respects(sorted_selected, idx, min_diff):
                continue
            augmented.add(idx)
            insort(sorted_selected, idx)
            added += 1

    return augmented


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
        "-n",
        "--segment_size",
        type=positive_int,
        default=None,
        help="Number of consecutive frames considered per segment (required unless --apply_csv).",
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
        help="Sharpness metric (default: hybrid 0.6*lapvar + 0.3*tenengrad + 0.2*fft).",
    )
    ap.add_argument(
        "-q",
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
        "-x",
        "--max_spacing",
        type=non_negative_int,
        default=0,
        help="Maximum allowed gap between kept frames before backfilling (0 uses 80%% of segment size).",
    )
    ap.add_argument(
        "-u",
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
        "-l",
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
        "-d",
        "--dry_run",
        action="store_true",
        help="Perform scoring and selection without moving files.",
    )
    ap.add_argument(
        "-a",
        "--apply_csv",
        help="Apply selections from an existing CSV produced during a dry run.",
    )
    ap.add_argument(
        "--enhance_motion",
        action="store_true",
        help="Apply additional motion blur penalty during hybrid scoring.",
    )
    ap.add_argument(
        "--allow_initial_adjacent",
        action="store_true",
        help="Allow adjacent frames during initial per-group selection.",
    )
    ap.add_argument(
        "-md",
        "--min_diff_ratio",
        type=float,
        default=0.2,
        help="Base ratio of --segment_size used to enforce spacing between initially selected frames.",
    )
    ap.add_argument(
        "--flow_crop_ratio",
        type=float,
        default=FLOW_CROP_RATIO,
        help="Central crop ratio to apply before optical flow (0<r<=1).",
    )
    ap.add_argument(
        "--prune_motion",
        action="store_true",
        help="Remove the lowest-motion frames based on optical flow (bottom 20%%).",
    )
    ap.add_argument(
        "-f",
        "--csv",
        help="Optional CSV output path relative to the input directory.",
    )

    # NEW: Brightness+Sharpness augmentation controls
    ap.add_argument(
        "--no_augment_bs",
        action="store_false",
        dest="augment_bs",
        help="Disable brightness+sharpness in-group augmentation step.",
    )
    ap.add_argument(
        "--bs_keep_ratio",
        type=fraction_arg,
        default=0.2,
        help="Per-segment ratio for brightness+sharpness augmentation (default: 0.2).",
    )
    ap.add_argument(
        "--bs_min_keep",
        type=non_negative_int,
        default=0,
        help="Minimum number of frames to add per segment in brightness+sharpness augmentation (default: 0).",
    )

    ap.set_defaults(augment_bs=True)

    args = ap.parse_args()

    if not args.apply_csv and args.segment_size is None:
        print("--segment_size is required unless --apply_csv is provided.")
        sys.exit(1)

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except (ValueError, AttributeError):
        pass
    _cancel_listener_thread = start_cancel_listener()

    flow_crop_ratio = args.flow_crop_ratio
    if not (0.0 < flow_crop_ratio <= 1.0):
        raise SystemExit("--flow_crop_ratio must be in (0, 1]")
    if args.min_diff_ratio < 0.0:
        raise SystemExit("--min_diff_ratio must be >= 0")
    min_diff_ratio = args.min_diff_ratio
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

    if not args.apply_csv:
        if args.max_spacing <= 0:
            args.max_spacing = round_half_up(args.segment_size * 0.8)

        min_spacing_raw = round_half_up(args.segment_size * min_diff_ratio)
        if min_spacing_raw <= 1:
            min_diff = 2
        else:
            min_diff = min_spacing_raw + 1
    else:
        min_diff = 1

    fast_window = FAST_SPACING_WINDOW
    if args.segment_size and args.segment_size > 0:
        fast_window = max(1, round_half_up(args.segment_size * FAST_SPACING_MULTIPLIER))

    sorter = SORTERS[args.sort]
    files = sorted(files, key=sorter)

    blur_dir = os.path.join(args.in_dir, "blur")
    ensure_dir(blur_dir)

    # Score every file in parallel
    n = len(files)
    scores = [None] * n
    p0_arr = [0.0] * n
    p255_arr = [0.0] * n
    brightness_arr = [1.0] * n
    brightness_mean_arr = [0.0] * n
    lap_arr = [None] * n
    ten_arr = [None] * n
    fft_arr = [None] * n
    motion_arr = [1.0] * n
    exposure_arr = [1.0] * n
    group_score_arr = [0.0] * n
    flow_mag_arr = [0.0] * n

    total = n

    cancelled = cancel_event.is_set()
    selection_flags = [0] * n
    gap_added_count = 0
    bs_added_count = 0
    motion_added_count = 0
    low_motion_filtered_count = 0
    final_selected = set()
    initial_selected = set()
    group_infos = []
    existing_indices = []
    motion_pruned_indices = set()
    motion_prune_threshold = None

    workers = (
        args.workers
        if (args.workers and args.workers > 0)
        else min(8, (os.cpu_count() or 4))
    )

    if args.apply_csv:
        apply_csv_path = args.apply_csv
        if not os.path.isabs(apply_csv_path):
            apply_csv_path = os.path.join(args.in_dir, apply_csv_path)
        if not os.path.isfile(apply_csv_path):
            print(f"Selection CSV not found: {apply_csv_path}")
            sys.exit(1)
        try:
            selection_flags = load_selection_from_csv(
                apply_csv_path,
                files,
                scores,
                brightness_mean_arr,
                group_score_arr,
                flow_mag_arr,
            )
        except ValueError as exc:
            print(f"Failed to load selection CSV: {exc}")
            sys.exit(1)
        final_selected = {
            idx for idx, flag in enumerate(selection_flags)
            if flag == 1 and os.path.isfile(files[idx])
        }
        initial_selected = set(final_selected)
        existing_indices = [
            idx for idx in range(total) if os.path.isfile(files[idx])
        ]
        cancelled = cancel_event.is_set()
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    score_one_file, files[i],
                    args.metric, args.crop_ratio,
                    args.use_exposure, args.clip_penalty, args.clip_thresh,
                    args.max_long, args.enhance_motion
                ): i for i in range(n)
            }
            completed = 0
            last_pct = -1
            try:
                for fut in as_completed(futs):
                    if cancel_event.is_set():
                        break
                    i = futs[fut]
                    (
                        s,
                        p0,
                        p255,
                        brightness_mean,
                        brightness_weight,
                        lap_feature,
                        ten_feature,
                        fft_feature,
                        motion_factor,
                        exposure_factor,
                    ) = fut.result()
                    scores[i] = s
                    p0_arr[i] = p0
                    p255_arr[i] = p255
                    brightness_mean_arr[i] = brightness_mean
                    brightness_arr[i] = brightness_weight
                    lap_arr[i] = lap_feature
                    ten_arr[i] = ten_feature
                    fft_arr[i] = fft_feature
                    motion_arr[i] = motion_factor
                    exposure_arr[i] = exposure_factor
                    completed += 1
                    last_pct = update_progress("Scoring", completed, n, last_pct)
            except KeyboardInterrupt:
                cancel_event.set()
        cancelled = cancel_event.is_set()

    flow_pairs_total = 0
    if (not cancelled) and n > 1 and (args.prune_motion or args.enhance_motion):
        flow_pairs_total = _compute_flow_magnitudes(
            files,
            flow_mag_arr,
            flow_crop_ratio,
            workers,
            "Optical flow",
        )
        cancelled = cancel_event.is_set()

    if not cancelled and args.metric == "hybrid":
        # Normalize feature channels and recompute scores into [0,1]-ish for stable blending
        lap_values = [v for v in lap_arr if v is not None]
        ten_values = [v for v in ten_arr if v is not None]
        fft_values = [v for v in fft_arr if v is not None]

        def _normalize_feature(values, value):
            if not values or value is None:
                return 0.0
            vmin = min(values)
            vmax = max(values)
            if math.isclose(vmax, vmin):
                return 0.0
            return (value - vmin) / (vmax - vmin)

        for idx in range(n):
            if lap_arr[idx] is None:
                continue
            lap_norm = _normalize_feature(lap_values, lap_arr[idx])
            ten_norm = _normalize_feature(ten_values, ten_arr[idx])
            fft_norm = _normalize_feature(fft_values, fft_arr[idx])

            combined = (
                HYBRID_LAPVAR_WEIGHT * lap_norm
                + HYBRID_TENENGRAD_WEIGHT * ten_norm
                + HYBRID_FFT_WEIGHT * fft_norm
            )

            combined *= motion_arr[idx]
            combined *= exposure_arr[idx]
            scores[idx] = combined
    
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
                "brightness_mean",
                "group_score",
                "flow_motion",
                "selected(1=keep)",
            ]
        )
    if not args.apply_csv and not cancelled:
        # Grouping and initial selection
        group_infos = []
        for grp_start in range(0, total, args.segment_size):
            grp_end = min(total, grp_start + args.segment_size)
            valid_idx = []
            group_sum = 0.0
            for i in range(grp_start, grp_end):
                s = scores[i]
                if s is None:
                    continue
                valid_idx.append(i)
                if s > 0.0:
                    brightness_factor = brightness_arr[i] * (max(brightness_mean_arr[i], 1e-6) ** GROUP_BRIGHTNESS_POWER)
                    group_sum += s * brightness_factor
            for idx_in_group in range(grp_start, grp_end):
                group_score_arr[idx_in_group] = group_sum
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
                    raw_keep = round_half_up(
                        len(sorted_valid) * args.low_group_keep_ratio
                    )
                    desired = max(args.low_group_min_keep, raw_keep)
                    keep_count = min(len(sorted_valid), max(1, desired))

                chosen = []
                if args.allow_initial_adjacent:
                    chosen = sorted_valid[:keep_count]
                else:
                    for idx in sorted_valid:
                        if not chosen:
                            chosen.append(idx)
                        elif min_diff <= 1 or all(abs(idx - sel) >= min_diff for sel in chosen):
                            chosen.append(idx)
                        if len(chosen) >= keep_count:
                            break
                    if len(chosen) < keep_count:
                        for idx in sorted_valid:
                            if idx not in chosen:
                                chosen.append(idx)
                            if len(chosen) >= keep_count:
                                break
                selected_indices = set(chosen)

            initial_selected.update(selected_indices)

        existing_indices = [
            i for i in range(total) if os.path.isfile(files[i])
        ]
        initial_selected &= set(existing_indices)
        final_selected = evenly_distribute_indices(
            existing_indices,
            initial_selected,
            scores,
            min_diff,
            fast_window,
        )
        if not final_selected and initial_selected:
            final_selected = set(initial_selected)

    if (
        args.prune_motion
        and not cancelled
        and final_selected
    ):
        motion_candidates = [
            (idx, flow_mag_arr[idx])
            for idx in final_selected
            if flow_mag_arr[idx] is not None
            and math.isfinite(flow_mag_arr[idx])
        ]
        if motion_candidates:
            motion_values = [mag for _, mag in motion_candidates]
            motion_prune_threshold = float(
                np.percentile(motion_values, FLOW_LOW_MOTION_PERCENTILE)
            )
            motion_pruned_indices = {
                idx for idx, mag in motion_candidates
                if mag <= motion_prune_threshold
            }
            if motion_pruned_indices:
                low_motion_filtered_count = len(motion_pruned_indices)
                if args.apply_csv:
                    for idx in motion_pruned_indices:
                        selection_flags[idx] = 0
                    final_selected = {
                        idx for idx in range(n)
                        if selection_flags[idx] and os.path.isfile(files[idx])
                    }
                    initial_selected = set(final_selected)
                else:
                    initial_selected -= motion_pruned_indices
                    final_selected -= motion_pruned_indices
                    if existing_indices:
                        existing_indices = [
                            idx for idx in existing_indices
                            if idx not in motion_pruned_indices
                        ]
                    initial_selected &= set(existing_indices)
                print(
                    f"Motion prune removed {low_motion_filtered_count} frame(s) below P{FLOW_LOW_MOTION_PERCENTILE:.0f} "
                    f"(threshold {motion_prune_threshold:.4f})."
                )

    # ----- Augmentations after initial selection -----
    if not args.apply_csv and not cancelled:
        # 1) Gap augmentation
        gap_added_count = 0
        before_gap_aug = set(final_selected)
        final_selected = augment_spacing(
            final_selected,
            existing_indices,
            scores,
            initial_selected,
            args.max_spacing,
            min_diff,
            fast_window,
        )
        gap_added_count = len(final_selected - before_gap_aug)

        # 2) Brightness+Sharpness in-group augmentation (NEW, runs after gap fill)
        bs_added_count = 0
        if args.augment_bs:
            before_bs_aug = set(final_selected)
            final_selected = augment_brightness_sharpness_segments(
                final_selected,
                group_infos,
                existing_indices,
                scores,
                brightness_mean_arr,
                min_diff,
                args.bs_keep_ratio,
                args.bs_min_keep,
            )
            bs_added_count = len(final_selected - before_bs_aug)

        # 3) Motion enhancement (if requested)
        motion_added_count = 0
        if args.enhance_motion:
            before_motion_aug = set(final_selected)
            final_selected = augment_motion_segments(
                final_selected,
                group_infos,
                existing_indices,
                scores,
                flow_mag_arr,
                min_diff,
            )
            motion_added_count = len(final_selected - before_motion_aug)


    kept = 0
    moved = 0
    skipped = 0
    processed = 0
    last_group_pct = -1




    for i in range(total):
        if cancel_event.is_set():
            cancelled = True
            break
        s = scores[i]
        if args.apply_csv and s is None:
            s = 0.0
        processed += 1
        file_exists = os.path.isfile(files[i])
        if not file_exists or s is None:
            skipped += 1
            if csv_writer:
                csv_writer.writerow([
                    i,
                    os.path.basename(files[i]),
                    -1.0,
                    0.0,
                    group_score_arr[i],
                    flow_mag_arr[i],
                    0,
                ])
            last_group_pct = update_progress("Grouping", processed, total, last_group_pct)
            continue

        score_val = s if s is not None else 0.0
        mean_val = brightness_mean_arr[i]

        if i in final_selected:
            kept += 1
            if csv_writer:
                csv_writer.writerow([
                    i,
                    os.path.basename(files[i]),
                    score_val,
                    mean_val,
                    group_score_arr[i],
                    flow_mag_arr[i],
                    1,
                ])
        else:
            if args.dry_run:
                moved += 1
            else:
                dst = os.path.join(blur_dir, os.path.basename(files[i]))
                if safe_move(files[i], dst) is None:
                    skipped += 1
                else:
                    moved += 1
            if csv_writer:
                csv_writer.writerow([
                    i,
                    os.path.basename(files[i]),
                    score_val,
                    mean_val,
                    group_score_arr[i],
                    flow_mag_arr[i],
                    0,
                ])
        last_group_pct = update_progress("Grouping", processed, total, last_group_pct)
    cancelled = cancelled or cancel_event.is_set()

    
    if fcsv:
        fcsv.close()

    if cancelled:
        print("Cancelled by user. Partial results may be incomplete.")

    if low_motion_filtered_count:
        print(f"Flow filter removed {low_motion_filtered_count} frame(s).")

    print(f"Gap augmentation added {gap_added_count} frame(s).")
    if args.augment_bs:
        print(f"Brightness+Sharpness augmentation added {bs_added_count} frame(s).")
    if args.enhance_motion:
        print(f"Motion augmentation added {motion_added_count} frame(s).")

    print(f"Done: input {total} / kept {kept} / moved {moved} / skipped {skipped}")
    if args.dry_run:
        print("Blur directory (dry run, no files moved):", blur_dir)
    else:
        print("Blur directory:", blur_dir)
    print(
        f"workers={workers}, opencv_threads={args.opencv_threads}, "
        f"max_long={args.max_long}"
    )


if __name__ == "__main__":
    main()