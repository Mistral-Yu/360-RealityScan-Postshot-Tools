
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ply_xyz_rgb_tool.py

- Load only (x, y, z) and (R, G, B) from PLY files.
- Optionally downsample on a voxel grid.
  - --target-points automatically adjusts the voxel size to approach the target count (prints detailed logs).
  - --voxel-size uses a fixed voxel size.
- After downsampling, --append-ply concatenates additional point clouds.
- Output is a binary little-endian PLY (vertex: x, y, z, red, green, blue).

Note: With a fixed voxel size the representative point is the original point closest to the voxel center.
      With target-based downsampling the representative is the point closest to the centroid of
      the points within each voxel.

Dependencies: numpy, plyfile
    pip install numpy plyfile
"""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from plyfile import PlyData, PlyElement


# ------------------------------ Utilities ------------------------------

def _fmt3(a) -> str:
    return f"({float(a[0]):.6g}, {float(a[1]):.6g}, {float(a[2]):.6g})"


@dataclass
class PointCloudStats:
    count: int
    xyz_min: np.ndarray
    xyz_max: np.ndarray
    extent: np.ndarray
    volume: float


def compute_point_cloud_stats(xyz: np.ndarray) -> PointCloudStats:
    n = int(xyz.shape[0])
    if n == 0:
        zeros = np.zeros(3, dtype=np.float32)
        return PointCloudStats(0, zeros, zeros, zeros, 0.0)

    xyz_min = np.asarray(xyz.min(axis=0), dtype=np.float32)
    xyz_max = np.asarray(xyz.max(axis=0), dtype=np.float32)
    extent = np.maximum(xyz_max - xyz_min, 1e-9)
    volume = float(extent[0] * extent[1] * extent[2])
    return PointCloudStats(n, xyz_min, xyz_max, extent, volume)


def print_point_cloud_stats(
    stats: PointCloudStats,
    include_voxel_reference: bool = True,
) -> None:
    print(f"input_points={stats.count:,}")
    print(
        f"[aabb] min={_fmt3(stats.xyz_min)}  max={_fmt3(stats.xyz_max)}  "
        f"extent={_fmt3(stats.extent)}  volume≈{stats.volume:.6g}"
    )
    if include_voxel_reference:
        if stats.volume > 0.0 and stats.count > 0:
            v0 = (stats.volume / float(stats.count)) ** (1.0 / 3.0)
        else:
            v0 = 1e-3
        lo = max(v0 / 64.0, 1e-9)
        hi = max(v0 * 64.0, lo * 2.0)
        print(f"[init] v0≈{v0:.6g}  lo={lo:.6g}  hi={hi:.6g}")


# ------------------------------ Load / Save ------------------------------

def _find_vertex_data(ply: PlyData):
    """Return the vertex element (an element that exposes x, y, z)."""
    try:
        return ply['vertex'].data
    except Exception:
        pass
    for el in getattr(ply, 'elements', []):
        data = getattr(el, 'data', None)
        names = getattr(data.dtype, 'names', None) if data is not None else None
        if names and all(k in names for k in ('x', 'y', 'z')):
            return data
    return None


def _extract_xyz_rgb_from_structured(v) -> Tuple[np.ndarray, np.ndarray]:
    names = v.dtype.names
    if not all(k in names for k in ('x', 'y', 'z')):
        raise ValueError("Could not find x, y, z fields")

    xyz = np.stack([v['x'].astype(np.float32),
                    v['y'].astype(np.float32),
                    v['z'].astype(np.float32)], axis=1)

    # Candidate color field triplets
    cand = [
        ('red', 'green', 'blue'),
        ('r', 'g', 'b'),
        ('diffuse_red', 'diffuse_green', 'diffuse_blue'),
    ]
    rgb = None
    for r, g, b in cand:
        if r in names and g in names and b in names:
            rr = v[r]; gg = v[g]; bb = v[b]
            if rr.dtype.kind == 'f' or gg.dtype.kind == 'f' or bb.dtype.kind == 'f':
                rr = np.clip(rr, 0.0, 1.0) * 255.0
                gg = np.clip(gg, 0.0, 1.0) * 255.0
                bb = np.clip(bb, 0.0, 1.0) * 255.0
                rgb = np.stack([rr, gg, bb], axis=1).round().astype(np.uint8)
            else:
                rgb = np.stack([rr, gg, bb], axis=1).astype(np.uint8, copy=False)
            break
    if rgb is None:
        rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)

    return xyz, rgb


def load_ply_xyz_rgb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vdata = _find_vertex_data(ply)
    if vdata is None:
        raise ValueError(f"Could not find a vertex element with x, y, z: {path}")
    xyz, rgb = _extract_xyz_rgb_from_structured(vdata)
    return xyz, rgb


def save_ply_binary_little(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must have the same number of rows")
    n = xyz.shape[0]
    arr = np.empty(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr['x'] = xyz[:, 0].astype(np.float32, copy=False)
    arr['y'] = xyz[:, 1].astype(np.float32, copy=False)
    arr['z'] = xyz[:, 2].astype(np.float32, copy=False)
    arr['red'] = rgb[:, 0].astype(np.uint8, copy=False)
    arr['green'] = rgb[:, 1].astype(np.uint8, copy=False)
    arr['blue'] = rgb[:, 2].astype(np.uint8, copy=False)

    el = PlyElement.describe(arr, 'vertex')
    PlyData([el], text=False, byte_order='<').write(path)


# ------------------------------ Downsampling ------------------------------

def _grid_keys(xyz: np.ndarray, voxel: float, xyz_min: Optional[np.ndarray] = None) -> np.ndarray:
    if voxel <= 0:
        raise ValueError("voxel must be > 0")
    if xyz_min is None:
        xyz_min = xyz.min(axis=0, keepdims=True)
    keys = np.floor((xyz - xyz_min) / voxel).astype(np.int64, copy=False)
    return keys


def voxel_downsample_by_size(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel: float,
    *,
    representative: str = "center",
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample with a fixed voxel size.

    `representative` selects how to pick the kept point for each voxel:
      - "center"   -> point closest to the voxel center.
      - "centroid" -> point closest to the centroid of all points in the voxel.
    """
    xyz_min = xyz.min(axis=0, keepdims=True)
    keys = _grid_keys(xyz, voxel, xyz_min)                                # (N,3) int64
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    k = uniq.shape[0]

    xyz32 = xyz.astype(np.float32, copy=False)

    if representative == "center":
        targets = (xyz_min + (uniq.astype(np.float32) + 0.5) * voxel).astype(np.float32)
    elif representative == "centroid":
        sums = np.zeros((k, 3), dtype=np.float64)
        np.add.at(sums, inv, xyz.astype(np.float64, copy=False))
        centroids = sums / counts[:, None]
        targets = centroids.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown representative strategy: {representative}")

    target_per_point = targets[inv]                                        # (N,3)
    diff = xyz32 - target_per_point
    dist2 = (diff * diff).sum(axis=1)                                      # (N,)

    # Sort by voxel id to pick the minimum dist2 per group
    order = np.argsort(inv, kind='mergesort')                              # Stable sort
    inv_sorted = inv[order]
    dist2_sorted = dist2[order]

    # Detect the start/end of each group
    starts = np.flatnonzero(np.r_[True, inv_sorted[1:] != inv_sorted[:-1]])
    ends = np.r_[starts[1:], inv_sorted.size]

    pick_idx = np.empty(k, dtype=np.int64)
    for a, b in zip(starts, ends):
        gid = int(inv_sorted[a])
        # Argmin within the group and map back to the original index
        rel = a + int(np.argmin(dist2_sorted[a:b]))
        orig = int(order[rel])
        pick_idx[gid] = orig

    out_xyz = xyz[pick_idx].astype(np.float32, copy=False)
    out_rgb = rgb[pick_idx].astype(np.uint8, copy=False)
    return out_xyz, out_rgb


def _unique_voxel_count(xyz: np.ndarray, voxel: float, xyz_min: Optional[np.ndarray] = None) -> int:
    keys = _grid_keys(xyz, voxel, xyz_min)
    return int(np.unique(keys, axis=0).shape[0])


def voxel_downsample_to_target(
    xyz: np.ndarray,
    rgb: np.ndarray,
    target_points: int,
    tol_ratio: float = 0.02,
    max_iter: int = 25,
    stats: Optional[PointCloudStats] = None,
    log_bounds: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Automatically adjust the voxel size to approach a target point count."""

    n = xyz.shape[0]
    if stats is None or stats.count != n:
        stats = compute_point_cloud_stats(xyz)

    print(
        f"[target] input_points={n:,}  target={target_points:,}  "
        f"tol=±{tol_ratio * 100:.1f}%  max_iter={max_iter}"
    )
    if target_points <= 0 or target_points >= n:
        print(
            f"[target] skip: target={target_points} is out of range (output = input)"
        )
        return xyz.astype(np.float32, copy=False), rgb.astype(np.uint8, copy=False)

    xyz_min = stats.xyz_min
    vol = stats.volume
    extent = stats.extent
    v0 = (vol / float(target_points)) ** (1.0 / 3.0) if vol > 0 else 1e-3

    lo = max(v0 / 64.0, 1e-9)
    hi = max(v0 * 64.0, lo * 2.0)
    if log_bounds:
        print(
            f"[aabb] min={_fmt3(xyz_min)}  max={_fmt3(stats.xyz_max)}  "
            f"extent={_fmt3(extent)}  volume≈{vol:.6g}"
        )
    print(f"[init] v0≈{v0:.6g}  lo={lo:.6g}  hi={hi:.6g}")

    # Expand hi until the unique voxel count is at most the target.
    for _ in range(10):
        cnt_hi = _unique_voxel_count(xyz, hi, xyz_min)
        print(f"[expand] try hi={hi:.6g} -> unique={cnt_hi:,}")
        if cnt_hi <= target_points:
            break
        hi *= 2.0
    else:
        print("[expand] warning: could not find a sufficient upper bound")

    best_voxel = v0
    best_diff = 10**18
    best_cnt = n

    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        cnt = _unique_voxel_count(xyz, mid, xyz_min)
        diff = abs(cnt - target_points)
        ratio = diff / float(target_points)
        if diff < best_diff:
            best_diff = diff
            best_voxel = mid
            best_cnt = cnt
        decision = (
            "lo = mid (cnt > target -> increase voxel size)"
            if cnt > target_points
            else "hi = mid (cnt < target -> decrease voxel size)"
        )
        print(
            f"[iter {it:02d}] voxel={mid:.6g}  unique={cnt:,}  diff={diff:,} "
            f"({ratio:.2%})  -> {decision}"
        )
        if ratio <= tol_ratio:
            print(f"[stop] within tolerance: voxel={mid:.6g}  unique={cnt:,}")
            best_voxel = mid
            best_cnt = cnt
            break
        if cnt > target_points:
            lo = mid
        else:
            hi = mid

    print(
        f"[best] voxel≈{best_voxel:.6g}  unique≈{best_cnt:,}  (best_diff={best_diff:,})"
    )

    out_xyz, out_rgb = voxel_downsample_by_size(
        xyz, rgb, best_voxel, representative="centroid"
    )
    print(f"[final] voxel={best_voxel:.6g}  out_points={out_xyz.shape[0]:,}")
    return out_xyz, out_rgb


# ------------------------------ CLI ------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="PLY (XYZ+RGB) load/save + voxel grid downsampling + optional append"
    )
    ap.add_argument(
        "-i",
        "--in",
        dest="input",
        required=True,
        help="Input PLY file path",
    )
    ap.add_argument(
        "-o",
        "--out",
        dest="output",
        required=True,
        help="Output PLY (binary little-endian)",
    )
    ap.add_argument(
        "--target-points",
        type=int,
        default=None,
        help=(
            "Target number of points after downsampling (approximate). "
            "Overrides --voxel-size when provided."
        ),
    )
    ap.add_argument(
        "--target-percent",
        type=float,
        default=None,
        help=(
            "Target percentage of the input point count. "
            "The percentage is converted to --target-points after loading."
        ),
    )
    ap.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Fixed voxel size in meters.",
    )
    ap.add_argument(
        "--append-ply",
        action="append",
        default=[],
        help=(
            "Additional PLY files to append after downsampling."
            " Paths are resolved relative to the input file when not absolute."
            " May be specified multiple times."
        ),
    )
    args = ap.parse_args(argv)

    base_path = os.path.expanduser(args.input)
    if not os.path.isabs(base_path):
        base_path = os.path.abspath(base_path)
    base_dir = os.path.dirname(base_path) or "."

    # 1) Load
    xyz, rgb = load_ply_xyz_rgb(base_path)
    print(f"[load] base: {base_path}  points={xyz.shape[0]:,}")

    stats = compute_point_cloud_stats(xyz)

    target_points = args.target_points if args.target_points and args.target_points > 0 else None
    if args.target_percent is not None:
        pct = args.target_percent
        if pct <= 0 or stats.count == 0:
            computed_target = 0
        else:
            computed_target = int(round(stats.count * (pct / 100.0)))
            computed_target = max(1, min(stats.count, computed_target))
        print(
            "[target-percent]",
            f"{pct:.6g}% of {stats.count:,} -> target_points={computed_target:,}",
        )
        if computed_target > 0:
            if (
                target_points is not None
                and target_points != computed_target
            ):
                print(
                    "[target-percent] override",
                    f"replacing explicit target_points={target_points:,}",
                )
            target_points = computed_target

    include_voxel_reference = not (target_points is not None and target_points > 0)
    print_point_cloud_stats(stats, include_voxel_reference=include_voxel_reference
    # 2) Downsample
    if args.voxel_size is not None and args.voxel_size > 0:
        print(f"[downsample] fixed voxel-size={args.voxel_size:.6g}")
        xyz, rgb = voxel_downsample_by_size(xyz, rgb, args.voxel_size)
        print(f"[downsample] -> {xyz.shape[0]:,} points")
    elif target_points is not None and target_points > 0:
        xyz, rgb = voxel_downsample_to_target(
            xyz,
            rgb,
            target_points,
            stats=stats,
            log_bounds=False,
        )
        print(f"[downsample] target_points={target_points:,} -> {xyz.shape[0]:,} points")
    else:
        print("[downsample] skip (no voxel-size/target-points)")

    # 3) Append additional clouds (after downsampling)
    total_added = 0
    for apath in args.append_ply:
        full_path = os.path.expanduser(apath)
        if not os.path.isabs(full_path):
            full_path = os.path.join(base_dir, full_path)
        full_path = os.path.abspath(full_path)
        ax, ac = load_ply_xyz_rgb(full_path)
        xyz = np.concatenate([xyz, ax], axis=0)
        rgb = np.concatenate([rgb, ac], axis=0)
        total_added += ax.shape[0]
        print(f"[append] {full_path} +{ax.shape[0]:,} -> total {xyz.shape[0]:,}")
    if total_added > 0:
        print(f"[append] total added: {total_added:,}")

    # 4) Save
    out_path = os.path.expanduser(args.output)
    if not os.path.isabs(out_path):
        out_path = os.path.abspath(out_path)
    save_ply_binary_little(out_path, xyz, rgb)
    print(f"[save] {out_path}  points={xyz.shape[0]:,}  (binary little-endian)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
