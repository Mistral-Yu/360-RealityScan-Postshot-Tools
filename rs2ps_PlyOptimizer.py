#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for optimizing RealityScan PLY point clouds.

The module loads XYZ/RGB data, can downsample voxels, append extra clouds,
and writes the result as a binary little-endian PLY file.
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
    """Format a coordinate triple for logging.

    Args:
        a: Iterable with three numeric entries.

    Returns:
        Compact string representation of the first three values.
    """
    return f"({float(a[0]):.6g}, {float(a[1]):.6g}, {float(a[2]):.6g})"


@dataclass
class PointCloudStats:
    """Statistics derived from a point cloud bounding volume.

    Attributes:
        count: Number of points in the cloud.
        xyz_min: Minimum coordinates along each axis.
        xyz_max: Maximum coordinates along each axis.
        extent: Length of the bounding box along each axis.
        volume: Volume of the bounding box in cubic units.
    """

    count: int
    xyz_min: np.ndarray
    xyz_max: np.ndarray
    extent: np.ndarray
    volume: float


def compute_point_cloud_stats(xyz: np.ndarray) -> PointCloudStats:
    """Compute bounding box statistics for a point cloud.

    Args:
        xyz: Array of shape (N, 3) containing point coordinates.

    Returns:
        Aggregated statistics for the provided points.
    """
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
    """Display point cloud statistics.

    Args:
        stats: Computed statistics for the point cloud.
        include_voxel_reference: Whether to show heuristic voxel sizing hints.
    """
    print(f"input_points={stats.count:,}")
    print(
        f"[aabb] min={_fmt3(stats.xyz_min)}  max={_fmt3(stats.xyz_max)}  "
        f"extent={_fmt3(stats.extent)}  volume~{stats.volume:.6g}"
    )
    if include_voxel_reference:
        if stats.volume > 0.0 and stats.count > 0:
            v0 = (stats.volume / float(stats.count)) ** (1.0 / 3.0)
        else:
            v0 = 1e-3
        lo = max(v0 / 64.0, 1e-9)
        hi = max(v0 * 64.0, lo * 2.0)
        print(f"[init] v0~{v0:.6g}  lo={lo:.6g}  hi={hi:.6g}")


# ------------------------------ Load / Save ------------------------------


def _find_vertex_data(ply: PlyData):
    """Return vertex data containing XYZ fields from a PLY structure.

    Args:
        ply: Parsed PLY data.

    Returns:
        Structured array with x, y, z columns, or None if unavailable.
    """
    try:
        return ply["vertex"].data
    except Exception:
        pass
    for el in getattr(ply, "elements", []):
        data = getattr(el, "data", None)
        names = (
            getattr(data.dtype, "names", None) if data is not None else None
        )
        if names and all(k in names for k in ("x", "y", "z")):
            return data
    return None


def _extract_xyz_rgb_from_structured(v) -> Tuple[np.ndarray, np.ndarray]:
    """Extract XYZ and RGB arrays from a structured NumPy array.

    Args:
        v: Structured array produced by plyfile.

    Returns:
        Tuple containing XYZ float32 positions and uint8 RGB colors.

    Raises:
        ValueError: If the array lacks x, y, or z fields.
    """
    names = v.dtype.names
    if not all(k in names for k in ("x", "y", "z")):
        raise ValueError("Could not find x, y, z fields")

    xyz = np.stack(
        [
            v["x"].astype(np.float32),
            v["y"].astype(np.float32),
            v["z"].astype(np.float32),
        ],
        axis=1,
    )

    # Candidate color field triplets
    cand = [
        ("red", "green", "blue"),
        ("r", "g", "b"),
        ("diffuse_red", "diffuse_green", "diffuse_blue"),
    ]
    rgb = None
    for r, g, b in cand:
        if r in names and g in names and b in names:
            rr = v[r]
            gg = v[g]
            bb = v[b]
            if (
                rr.dtype.kind == "f"
                or gg.dtype.kind == "f"
                or bb.dtype.kind == "f"
            ):
                rr = np.clip(rr, 0.0, 1.0) * 255.0
                gg = np.clip(gg, 0.0, 1.0) * 255.0
                bb = np.clip(bb, 0.0, 1.0) * 255.0
                rgb = np.stack([rr, gg, bb], axis=1).round().astype(np.uint8)
            else:
                rgb = np.stack([rr, gg, bb], axis=1).astype(
                    np.uint8, copy=False
                )
            break
    if rgb is None:
        rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)

    return xyz, rgb


def load_ply_xyz_rgb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load XYZ and RGB data from a PLY file.

    Args:
        path: Path to the PLY file.

    Returns:
        Tuple of XYZ positions and RGB colors.
    """
    ply = PlyData.read(path)
    vdata = _find_vertex_data(ply)
    if vdata is None:
        raise ValueError(
            f"Could not find a vertex element with x, y, z: {path}"
        )
    xyz, rgb = _extract_xyz_rgb_from_structured(vdata)
    return xyz, rgb


def save_ply_binary_little(
    path: str, xyz: np.ndarray, rgb: np.ndarray
) -> None:
    """Write XYZ and RGB arrays to a binary little-endian PLY file.

    Args:
        path: Destination path for the PLY file.
        xyz: Array of shape (N, 3) with point coordinates.
        rgb: Array of shape (N, 3) with 8-bit color values.

    Raises:
        ValueError: If xyz and rgb have mismatched lengths.
    """
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must have the same number of rows")
    n = xyz.shape[0]
    arr = np.empty(
        n,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    arr["x"] = xyz[:, 0].astype(np.float32, copy=False)
    arr["y"] = xyz[:, 1].astype(np.float32, copy=False)
    arr["z"] = xyz[:, 2].astype(np.float32, copy=False)
    arr["red"] = rgb[:, 0].astype(np.uint8, copy=False)
    arr["green"] = rgb[:, 1].astype(np.uint8, copy=False)
    arr["blue"] = rgb[:, 2].astype(np.uint8, copy=False)

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False, byte_order="<").write(path)


# ------------------------------ Downsampling ------------------------------


def _grid_keys(
    xyz: np.ndarray, voxel: float, xyz_min: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute integer voxel keys for each point.

    Args:
        xyz: Array of shape (N, 3) with point coordinates.
        voxel: Voxel edge length.
        xyz_min: Optional minimum XYZ values used as the grid origin.

    Returns:
        Integer keys identifying the voxel that contains each point.

    Raises:
        ValueError: If voxel is not strictly positive.
    """
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
    representative: str = "centroid",
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample points using a fixed voxel size.

    Args:
        xyz: Array of shape (N, 3) with point coordinates.
        rgb: Array of shape (N, 3) with 8-bit color values.
        voxel: Voxel edge length.
        representative: Strategy used to pick the voxel representative.

    Returns:
        Downsampled XYZ and RGB arrays.

    Raises:
        ValueError: If representative is not recognized.
    """
    if xyz.shape[0] == 0:
        return xyz.astype(np.float32, copy=False), rgb.astype(
            np.uint8, copy=False
        )

    xyz_min = xyz.min(axis=0, keepdims=True)
    keys = _grid_keys(xyz, voxel, xyz_min)  # (N,3) int64
    uniq, inv, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    k = uniq.shape[0]

    xyz32 = xyz.astype(np.float32, copy=False)

    if representative == "first":
        pick_idx = np.full(k, -1, dtype=np.int64)
        for idx, gid in enumerate(inv):
            if pick_idx[gid] == -1:
                pick_idx[gid] = idx
    else:
        if representative == "center":
            targets = (
                xyz_min + (uniq.astype(np.float32) + 0.5) * voxel
            ).astype(np.float32)
        elif representative == "centroid":
            sums = np.zeros((k, 3), dtype=np.float64)
            np.add.at(sums, inv, xyz.astype(np.float64, copy=False))
            centroids = sums / counts[:, None]
            targets = centroids.astype(np.float32, copy=False)
        else:
            raise ValueError(
                f"Unknown representative strategy: {representative}"
            )

        target_per_point = targets[inv]  # (N,3)
        diff = xyz32 - target_per_point
        dist2 = (diff * diff).sum(axis=1)  # (N,)

        # Sort by voxel id to pick the minimum dist2 per group
        order = np.argsort(inv, kind="mergesort")  # Stable sort
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


def _unique_voxel_count(
    xyz: np.ndarray, voxel: float, xyz_min: Optional[np.ndarray] = None
) -> int:
    """Count unique voxels occupied by the point cloud.

    Args:
        xyz: Array of shape (N, 3) with point coordinates.
        voxel: Voxel edge length.
        xyz_min: Optional grid origin.

    Returns:
        Number of occupied voxels.
    """
    if xyz.shape[0] == 0:
        return 0
    keys = _grid_keys(xyz, voxel, xyz_min)
    return int(np.unique(keys, axis=0).shape[0])


def voxel_downsample_to_target(
    xyz: np.ndarray,
    rgb: np.ndarray,
    target_points: int,
    tol_ratio: float = 0.02,
    max_iter: int = 50,
    stats: Optional[PointCloudStats] = None,
    log_bounds: bool = True,
    representative: str = "centroid",
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample points to approach a target count.

    Args:
        xyz: Array of shape (N, 3) with point coordinates.
        rgb: Array of shape (N, 3) with 8-bit color values.
        target_points: Desired number of output points.
        tol_ratio: Acceptable relative deviation from the target count.
        max_iter: Maximum number of binary search iterations.
        stats: Optional precomputed statistics for the point cloud.
        log_bounds: Whether to log the search bounds while iterating.
        representative: Strategy for choosing voxel representatives.

    Returns:
        Downsampled XYZ and RGB arrays.

    Raises:
        ValueError: If target_points is not positive.

    Notes:
        The representative argument accepts 'centroid', 'center', or 'first'.
    """

    n = xyz.shape[0]
    if n == 0:
        return xyz.astype(np.float32, copy=False), rgb.astype(
            np.uint8, copy=False
        )

    if stats is None or stats.count != n:
        stats = compute_point_cloud_stats(xyz)

    print(
        f"[target] input_points={n:,}  target={target_points:,}  "
        f"tol=+/-{tol_ratio * 100:.1f}%  max_iter={max_iter}"
    )
    if target_points <= 0 or target_points >= n:
        print(
            "[target] skip: "
            f"target={target_points} is out of range "
            "(output = input)"
        )
        return xyz.astype(np.float32, copy=False), rgb.astype(
            np.uint8, copy=False
        )

    xyz_min = stats.xyz_min
    vol = stats.volume
    extent = stats.extent
    v0 = (vol / float(target_points)) ** (1.0 / 3.0) if vol > 0 else 1e-3

    min_voxel = 1e-9
    lo = max(v0 / 64.0, min_voxel)
    hi = max(v0 * 64.0, lo * 2.0)

    cnt_lo = _unique_voxel_count(xyz, lo, xyz_min)
    if cnt_lo < target_points:
        print(
            f"[shrink] initial lo={lo:.6g} -> unique={cnt_lo:,} "
            "(below target); shrinking lower bound"
        )
        shrink_iter = 0
        while cnt_lo < target_points and lo > min_voxel:
            prev_lo = lo
            lo = max(lo * 0.5, min_voxel)
            if lo == prev_lo:
                break
            cnt_lo = _unique_voxel_count(xyz, lo, xyz_min)
            shrink_iter += 1
            print(
                f"[shrink {shrink_iter:02d}] lo={lo:.6g} -> unique={cnt_lo:,}"
            )
            if shrink_iter >= 32:
                break
        if cnt_lo < target_points:
            print(
                "[shrink] warning: minimum voxel size still below target; "
                "will use best available count."
            )
    hi = max(hi, lo * 2.0)

    if log_bounds:
        print(
            f"[aabb] min={_fmt3(xyz_min)}  max={_fmt3(stats.xyz_max)}  "
            f"extent={_fmt3(extent)}  volume~{vol:.6g}"
        )
    print(f"[init] v0~{v0:.6g}  lo={lo:.6g}  hi={hi:.6g}")

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
        f"[best] voxel~{best_voxel:.6g}  unique~{best_cnt:,}  "
        f"(best_diff={best_diff:,})"
    )

    out_xyz, out_rgb = voxel_downsample_by_size(
        xyz, rgb, best_voxel, representative=representative
    )
    print(f"[final] voxel={best_voxel:.6g}  out_points={out_xyz.shape[0]:,}")
    return out_xyz, out_rgb


# ------------------------------ CLI ------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the rs2ps-opt CLI.

    Args:
        argv: Optional argument vector to parse instead of sys.argv.

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(
        prog="rs2ps-opt",
        description=(
            "RealityScan to PostShot PLY optimizer (XYZ+RGB load/save, "
            "voxel downsampling, append)"
        ),
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
        default=None,
        help=(
            "Output PLY (binary little-endian). "
            "Omit to only display point cloud statistics."
        ),
    )
    ap.add_argument(
        "-t",
        "--target-points",
        type=int,
        default=None,
        help=(
            "Target number of points after downsampling (approximate). "
            "Overrides --voxel-size when provided."
        ),
    )
    ap.add_argument(
        "-r",
        "--target-percent",
        type=float,
        default=None,
        help=(
            "Target percentage of the input point count. "
            "The percentage is converted to --target-points after loading."
        ),
    )
    ap.add_argument(
        "-v",
        "--voxel-size",
        type=float,
        default=None,
        help="Fixed voxel size in meters.",
    )
    ap.add_argument(
        "-a",
        "--append-ply",
        action="append",
        default=[],
        help=(
            "Additional PLY files to append after downsampling."
            " Paths are resolved relative to the input file when not absolute."
            " May be specified multiple times."
        ),
    )
    ap.add_argument(
        "-k",
        "--keep-strategy",
        choices=("centroid", "center", "first"),
        default="centroid",
        help=(
            "Representative selection per voxel: "
            "centroid=closest to centroid (default), "
            "center=closest to voxel center, "
            "first=first point encountered."
        ),
    )
    args = ap.parse_args(argv)

    if args.target_points is not None and args.target_points <= 0:
        ap.error("--target-points must be greater than 0")

    base_path = os.path.expanduser(args.input)
    if not os.path.isabs(base_path):
        base_path = os.path.abspath(base_path)
    base_dir = os.path.dirname(base_path) or "."

    # 1) Load
    xyz, rgb = load_ply_xyz_rgb(base_path)
    print(f"[load] base: {base_path}  points={xyz.shape[0]:,}")

    stats = compute_point_cloud_stats(xyz)

    target_points = (
        args.target_points
        if args.target_points and args.target_points > 0
        else None
    )
    if args.target_percent is not None:
        pct = args.target_percent
        if pct <= 0 or stats.count == 0:
            computed_target = 0
        else:
            computed_target = int(round(stats.count * (pct / 100.0)))
            computed_target = max(1, min(stats.count, computed_target))
        print(
            "[target-percent]",
            f"{pct:.6g}% of {stats.count:,} -> "
            f"target_points={computed_target:,}",
        )
        if computed_target > 0:
            if target_points is not None and target_points != computed_target:
                print(
                    "[target-percent] override",
                    f"replacing explicit target_points={target_points:,}",
                )
            target_points = computed_target

    include_voxel_reference = not (
        target_points is not None and target_points > 0
    )
    print_point_cloud_stats(
        stats, include_voxel_reference=include_voxel_reference
    )

    if args.output is None:
        if (
            (args.voxel_size is not None and args.voxel_size > 0)
            or (target_points is not None and target_points > 0)
            or args.append_ply
        ):
            print("[warn] --out missing; skipping downsample/append options.")
        else:
            print("[info] --out not provided; statistics only.")
        return 0

    # 2) Downsample
    if args.voxel_size is not None and args.voxel_size > 0:
        print(f"[downsample] fixed voxel-size={args.voxel_size:.6g}")
        xyz, rgb = voxel_downsample_by_size(
            xyz, rgb, args.voxel_size, representative=args.keep_strategy
        )
        print(f"[downsample] -> {xyz.shape[0]:,} points")
    elif target_points is not None and target_points > 0:
        xyz, rgb = voxel_downsample_to_target(
            xyz,
            rgb,
            target_points,
            stats=stats,
            log_bounds=False,
            representative=args.keep_strategy,
        )
        print(
            f"[downsample] target_points={target_points:,} -> "
            f"{xyz.shape[0]:,} points"
        )
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
        print(
            f"[append] {full_path} +{ax.shape[0]:,} -> total {xyz.shape[0]:,}"
        )
    if total_added > 0:
        print(f"[append] total added: {total_added:,}")

    # 4) Save
    out_path = os.path.expanduser(args.output)
    if not os.path.isabs(out_path):
        out_path = os.path.abspath(out_path)
    save_ply_binary_little(out_path, xyz, rgb)
    print(
        f"[save] {out_path}  points={xyz.shape[0]:,}  (binary little-endian)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
