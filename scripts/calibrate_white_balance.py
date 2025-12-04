import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from raw_utils import demosaic, raw_to_mono
from raw_v4l2 import V4L2Capture, fourcc


def capture_frame(
    device: str,
    width: int,
    height: int,
    pixel_format: str,
    bypass_mode: int,
    fps: int,
    buffer_count: int,
    timeout: float,
    skip_configure: bool,
) -> Tuple[bytes, int, int, int]:
    """Capture a single raw frame and return bytes plus resolved geometry."""
    try:
        pixfmt = fourcc(pixel_format)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    with V4L2Capture(device) as cap:
        if not skip_configure:
            cap.configure(
                width=width,
                height=height,
                pixfmt=pixfmt,
                bypass_mode=bypass_mode,
                fps=fps if fps > 0 else None,
            )
        else:
            cap.current_format()

        fmt = cap.format
        cap.start(buffer_count)
        try:
            frame = cap.read(timeout)
        except TimeoutError as exc:
            print(f"Timed out waiting for a frame: {exc}", file=sys.stderr)
            sys.exit(1)
        finally:
            cap.stop()

    return frame, fmt.width, fmt.height, fmt.bytesperline


def compute_roi(frame: np.ndarray, roi: Tuple[int, int, int, int] | None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Slice the ROI, defaulting to the whole frame."""
    h, w, _ = frame.shape
    if roi is None:
        return frame, (0, 0, w, h)
    x, y, rw, rh = roi
    if rw <= 0 or rh <= 0:
        raise ValueError("ROI width/height must be positive.")
    if x < 0 or y < 0 or x + rw > w or y + rh > h:
        raise ValueError("ROI must lie within the frame bounds.")
    return frame[y : y + rh, x : x + rw], (x, y, rw, rh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate white balance gains from a captured RAW frame.")
    parser.add_argument("--device", default="/dev/video0", help="Path to V4L2 device (/dev/videoN).")
    parser.add_argument("--width", type=int, default=2048, help="Frame width (pixels).")
    parser.add_argument("--height", type=int, default=1536, help="Frame height (pixels).")
    parser.add_argument("--pixel-format", default="RG10", help="FourCC pixel format (e.g. RG10).")
    parser.add_argument("--bypass-mode", type=int, choices=[0, 1], default=0, help="bypass_mode control value.")
    parser.add_argument("--fps", type=int, default=0, help="Optional FPS request via V4L2.")
    parser.add_argument("--buffer-count", type=int, default=4, help="Number of MMAP buffers to request.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout waiting for a frame.")
    parser.add_argument("--skip-configure", action="store_true", help="Skip changing controls/format.")
    parser.add_argument("--bit-depth", type=int, default=10, help="Bit depth of the raw frames.")
    parser.add_argument("--align", choices=["auto", "msb", "lsb"], default="msb", help="Bit alignment strategy.")
    parser.add_argument("--bayer", default="rg", help="Bayer pattern for demosaicing (rg, gr, gb, bg).")
    parser.add_argument(
        "--demosaic",
        choices=["bilinear", "edge", "vng"],
        default="edge",
        help="Demosaicing algorithm to evaluate colors.",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Optional ROI in pixels (top-left X,Y, width, height). Defaults to the whole frame.",
    )
    parser.add_argument(
        "--output-preview",
        type=Path,
        help="Optional path to save the demosaiced frame with the ROI overlay (PNG).",
    )
    args = parser.parse_args()

    frame_bytes, width, height, bytesperline = capture_frame(
        args.device,
        args.width,
        args.height,
        args.pixel_format,
        args.bypass_mode,
        args.fps,
        args.buffer_count,
        args.timeout,
        args.skip_configure,
    )

    mono = raw_to_mono(frame_bytes, width, height, args.bit_depth, args.align, bytesperline)
    bgr = demosaic(mono, args.bayer, method=args.demosaic)
    try:
        roi_view, roi_box = compute_roi(bgr, tuple(args.roi) if args.roi else None)
    except ValueError as exc:
        print(f"Invalid ROI: {exc}", file=sys.stderr)
        sys.exit(2)

    roi_float = roi_view.reshape(-1, 3).astype(np.float32)
    channel_means = roi_float.mean(axis=0)
    if np.any(channel_means <= 1e-3):
        print("ROI appears too dark for calibration (channel mean near zero).", file=sys.stderr)
        sys.exit(1)

    target = float(channel_means.mean())
    gains_bgr = target / channel_means
    gains_rgb = gains_bgr[[2, 1, 0]]

    print("ROI channel means (B, G, R):", " ".join(f"{val:.1f}" for val in channel_means))
    print("Recommended --white-balance gains (R G B):", " ".join(f"{gain:.3f}" for gain in gains_rgb))
    print(
        "Example:",
        "python live_view_imx900.py ... --white-balance",
        " ".join(f"{gain:.3f}" for gain in gains_rgb),
    )

    if args.output_preview:
        preview = bgr.copy()
        x, y, rw, rh = roi_box
        cv2.rectangle(preview, (x, y), (x + rw - 1, y + rh - 1), (0, 0, 255), 2)
        args.output_preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output_preview), preview)
        print(f"Saved preview to {args.output_preview}")


if __name__ == "__main__":
    main()
