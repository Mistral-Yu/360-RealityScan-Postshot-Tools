#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

from typing import Optional, Tuple

PROGRESS_INTERVAL = 5


def update_progress(
    label: str,
    completed: float,
    total: float,
    last_pct: int,
    previous_len: int,
    extra: str = '',
) -> Tuple[int, int]:
    """Write a progress line when enough progress has been made.

    Args:
        label: Static label prefix for each progress line.
        completed: Work units completed.
        total: Total work units expected.
        last_pct: Last emitted percentage value.
        previous_len: Length of the previously printed line.
        extra: Additional detail appended after the label.

    Returns:
        Tuple with the updated percentage value and rendered line length.
    """
    if total <= 0:
        return last_pct, previous_len

    pct = int((completed * 100) / total)
    if last_pct < 0 or pct >= 100 or (pct - last_pct) >= PROGRESS_INTERVAL:
        line = f"{label}{extra} {pct:3d}%"
        sys.stdout.write('\r' + line.ljust(previous_len))
        sys.stdout.flush()
        return pct, len(line)

    return last_pct, previous_len


def detect_input_bit_depth(in_path: pathlib.Path) -> int:
    """Detect the source video's bit depth using ffprobe.

    Args:
        in_path: Path to the input media file.

    Returns:
        Either 8 or 10 depending on the detected precision.
    """
    if not shutil.which('ffprobe'):
        return 8

    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=bits_per_raw_sample,pix_fmt',
        '-of', 'json',
        str(in_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads(result.stdout)
        streams = info.get('streams') or [{}]
        stream = streams[0]
        bits_per_raw_sample = stream.get('bits_per_raw_sample')
        if (
            isinstance(bits_per_raw_sample, str)
            and bits_per_raw_sample.isdigit()
        ):
            value = int(bits_per_raw_sample)
            return value if value >= 9 else 8

        pix_fmt = stream.get('pix_fmt') or ''
        if any(
            token in pix_fmt
            for token in (
                'p10', 'p12', 'p14', 'p16',
                'yuv420p10', 'yuv422p10', 'yuv444p10',
                'yuv420p12', 'yuv422p12', 'yuv444p12',
                'p010', 'p012', 'p016',
                'gbrp10', 'gbrp12', 'gbrp14', 'gbrp16',
                'rgb48', 'rgba64',
            )
        ):
            return 10

    except Exception:
        pass

    return 8


def probe_media_duration(in_path: pathlib.Path) -> Optional[float]:
    """Measure the media duration with ffprobe.

    Args:
        in_path: Path to the input media file.

    Returns:
        Duration in seconds, or None when the value cannot be determined.
    """
    if not shutil.which('ffprobe'):
        return None

    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(in_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        return float(value) if value else None
    except Exception:
        return None


def parse_ffmpeg_timecode(value: str) -> Optional[float]:
    """Convert an ffmpeg timecode string to seconds.

    Args:
        value: Timecode string in the format HH:MM:SS.mmm.

    Returns:
        Parsed seconds, or None when parsing fails.
    """
    try:
        hours_str, minutes_str, seconds_str = value.split(':')
        hours = int(hours_str)
        minutes = int(minutes_str)
        seconds = float(seconds_str)
        return hours * 3600.0 + minutes * 60.0 + seconds
    except Exception:
        return None


def format_seconds(seconds: float) -> str:
    """Format a duration in seconds into H:MM:SS.ss text.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human readable time string.
    """
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def main() -> None:
    """CLI entry point that orchestrates frame extraction."""
    ap = argparse.ArgumentParser(
        description='Extract frames from a video at N fps using ffmpeg.'
    )
    ap.add_argument(
        '-i',
        '-in',
        dest='video',
        required=True,
        help='Input video file path.',
    )
    ap.add_argument(
        '-o',
        '-out',
        dest='output',
        default=None,
        help='Output directory (defaults next to the input video).',
    )
    ap.add_argument(
        '-f',
        '--fps',
        type=float,
        required=True,
        help='Frame extraction rate in frames per second (e.g. 5, 2.5).',
    )
    ap.add_argument(
        '-e',
        '--ext',
        default='jpg',
        help='Output image extension (e.g. jpg, png). Defaults to jpg.',
    )
    ap.add_argument(
        '--start',
        type=float,
        default=0.0,
        help='Optional start time in seconds.',
    )
    ap.add_argument(
        '--end',
        type=float,
        default=None,
        help='Optional end time in seconds.',
    )
    ap.add_argument(
        '--keep-rec709',
        action='store_true',
        help='Keep Rec.709 transfer characteristics instead of converting to sRGB.',
    )
    ap.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output if it already exists.',
    )
    args = ap.parse_args()

    if not shutil.which('ffmpeg'):
        print(
            'ffmpeg was not found. Please ensure it is on PATH.',
            file=sys.stderr,
        )
        sys.exit(1)

    in_path = pathlib.Path(args.video).expanduser().resolve()
    if not in_path.exists():
        print(f'Input file not found: {in_path}', file=sys.stderr)
        sys.exit(1)

    stem = in_path.stem
    fps_value = f'{args.fps}'
    fps_str = (
        fps_value.rstrip('0').rstrip('.')
        if '.' in fps_value
        else fps_value
    )

    if args.output:
        out_dir = pathlib.Path(args.output).expanduser().resolve()
    else:
        out_dir = in_path.parent / f'{stem}_frames_{fps_str}fps'

    if out_dir.exists() and not out_dir.is_dir():
        print(
            f'Output path exists and is not a directory: {out_dir}',
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    ext = args.ext.lstrip('.').lower()
    pattern = out_dir / f'out_%07d.{ext}'

    inferred_bits = detect_input_bit_depth(in_path)
    out_bit_depth = 8 if inferred_bits <= 8 else 16

    vf_chain = [f'fps={args.fps}']
    colorspace_filter = 'colorspace=iall=bt709:all=smpte170m'
    if not args.keep_rec709:
        colorspace_filter += ':trc=iec61966-2-1'
    if ext in {'jpg', 'jpeg'}:
        # Keep JPEG outputs in 4:4:4 and match the expected color space.
        vf_chain.append(f'{colorspace_filter}:range=jpeg:format=yuv444p')
    else:
        # Normalize other outputs to the expected BT.709 color space.
        vf_chain.append(f'{colorspace_filter}:format=yuv444p')

    cmd = ['ffmpeg', '-hide_banner', '-y' if args.overwrite else '-n']
    if args.start is not None:
        cmd += ['-ss', str(args.start)]

    cmd += ['-i', str(in_path)]

    if args.end is not None:
        cmd += ['-to', str(args.end)]

    cmd += ['-vf', ','.join(vf_chain)]

    if ext in {'jpg', 'jpeg'}:
        cmd += [
            '-c:v',
            'mjpeg',
            '-q:v',
            '1',
            '-qmin',
            '1',
            '-qmax',
            '1',
            '-pix_fmt',
            'yuvj444p',
            '-huffman',
            'optimal',
            '-colorspace',
            'smpte170m',
            '-color_primaries',
            'smpte170m',
            '-color_trc',
            'smpte170m',
        ]
    elif ext in {'png', 'tif', 'tiff'}:
        if ext in {'tif', 'tiff'}:
            # Use lossless deflate compression for TIFF.
            cmd += ['-c:v', 'tiff', '-compression_algo', 'deflate']
        if out_bit_depth <= 8:
            cmd += ['-pix_fmt', 'rgb24']
        else:
            cmd += ['-pix_fmt', 'rgb48le']

    cmd += ['-vsync', 'vfr', '-start_number', '0', str(pattern)]

    media_duration = probe_media_duration(in_path)
    start_offset = max(float(args.start or 0.0), 0.0)
    progress_span = None
    if media_duration is not None:
        natural_end = (
            media_duration
            if args.end is None
            else min(args.end, media_duration)
        )
        progress_span = max(natural_end - start_offset, 0.0)
    elif args.end is not None:
        progress_span = max(float(args.end) - start_offset, 0.0)
    estimated_frames = None
    if progress_span is not None and progress_span > 0 and args.fps > 0:
        estimated_frames = progress_span * float(args.fps)
    print("\n[exec] " + ' '.join(shlex.quote(x) for x in cmd) + "\n")
    creationflags = 0
    if sys.platform.startswith('win'):
        creationflags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    frame_pattern = re.compile(r'frame=\s*(\d+)')
    time_pattern = re.compile(r'time=\s*([0-9:.]+)')
    duration_pattern = re.compile(r'Duration:\s*([0-9:.]+)')
    duration_locked = media_duration is not None
    progress_last_pct = -1
    progress_line_len = 0
    frame_seen = None
    time_seen = None
    start_wall = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        creationflags=creationflags,
    )
    try:
        assert proc.stderr is not None
        for raw_line in proc.stderr:
            line = raw_line.strip()
            if not line:
                continue

            if not duration_locked:
                match_duration = duration_pattern.search(line)
                if match_duration:
                    parsed_duration = parse_ffmpeg_timecode(
                        match_duration.group(1)
                    )
                    if parsed_duration is not None:
                        media_duration = parsed_duration
                        natural_end = (
                            media_duration
                            if args.end is None
                            else min(args.end, media_duration)
                        )
                        progress_span = max(natural_end - start_offset, 0.0)
                        if (
                            progress_span
                            and progress_span > 0
                            and args.fps > 0
                        ):
                            estimated_frames = progress_span * float(args.fps)
                    duration_locked = True

            match_frame = frame_pattern.search(line)
            if match_frame:
                frame_seen = int(match_frame.group(1))
            match_time = time_pattern.search(line)
            if match_time:
                parsed_time = parse_ffmpeg_timecode(match_time.group(1))
                if parsed_time is not None:
                    time_seen = parsed_time
            percent = None
            if progress_span and time_seen is not None and progress_span > 0:
                percent = min(
                    100.0,
                    max(0.0, (time_seen / progress_span) * 100.0),
                )
            elif (
                estimated_frames
                and frame_seen is not None
                and estimated_frames > 0
            ):
                percent = min(
                    100.0,
                    max(0.0, (frame_seen / estimated_frames) * 100.0),
                )

            extra_parts = []
            if frame_seen is not None:
                extra_parts.append(f' frame {frame_seen}')
            if time_seen is not None:
                extra_parts.append(f' time {format_seconds(time_seen)}')
            if percent is not None and 0.0 < percent < 100.0:
                elapsed = time.monotonic() - start_wall
                if elapsed > 0.0:
                    remaining_ratio = (100.0 - percent) / percent
                    eta_seconds = elapsed * remaining_ratio
                    extra_parts.append(f' ETA {format_seconds(eta_seconds)}')
            extra_text = ''.join(extra_parts)

            if percent is not None:
                completed_units = 0.0
                total_units = 0.0
                if (
                    progress_span
                    and time_seen is not None
                    and progress_span > 0
                ):
                    completed_units = min(time_seen, progress_span)
                    total_units = progress_span
                elif (
                    estimated_frames
                    and frame_seen is not None
                    and estimated_frames > 0
                ):
                    completed_units = min(frame_seen, estimated_frames)
                    total_units = estimated_frames
                else:
                    completed_units = percent
                    total_units = 100.0
                progress_last_pct, progress_line_len = update_progress(
                    '[progress]',
                    completed_units,
                    total_units,
                    progress_last_pct,
                    progress_line_len,
                    extra_text,
                )
            elif extra_text:
                line_text = f'[progress]{extra_text}'
                sys.stdout.write('\r' + line_text.ljust(progress_line_len))
                sys.stdout.flush()
                progress_line_len = len(line_text)
        proc.stderr.close()
    except KeyboardInterrupt:
        sys.stdout.write('\n[INFO] Interrupt received, stopping ffmpeg...\n')
        sys.stdout.flush()
        try:
            if (
                sys.platform.startswith('win')
                and hasattr(signal, 'CTRL_BREAK_EVENT')
            ):
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(130)
    retcode = proc.wait()
    if progress_line_len:
        sys.stdout.write('\n')
        sys.stdout.flush()
    if retcode != 0:
        print(
            f'ffmpeg execution failed (returncode={retcode})',
            file=sys.stderr,
        )
        sys.exit(retcode)
    print(f'Completed: {out_dir}')


if __name__ == '__main__':
    main()
