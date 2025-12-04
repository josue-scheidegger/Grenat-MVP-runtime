import argparse
import sys
import time
from typing import List, Sequence

import cv2
import numpy as np

from raw_utils import apply_white_balance, demosaic, raw_to_mono
from raw_v4l2 import V4L2Capture, fourcc


def compose_display(frames: Sequence[np.ndarray], labels: Sequence[str], scale: float) -> np.ndarray:
    display: List[np.ndarray] = []
    for frame, label in zip(frames, labels):
        if frame is None:
            continue
        view = frame
        if scale != 1.0:
            view = cv2.resize(view, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.putText(view, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        display.append(view)
    if not display:
        return np.zeros((360, 640, 3), dtype=np.uint8)
    return np.hstack(display)


def setup_capture(args, device: str) -> V4L2Capture:
    cap = V4L2Capture(device)
    pixfmt = fourcc(args.pixel_format)
    if not args.no_configure:
        cap.configure(
            width=args.width,
            height=args.height,
            pixfmt=pixfmt,
            bypass_mode=args.bypass_mode,
            fps=args.fps if args.fps > 0 else None,
        )
    else:
        cap.current_format()
    cap.start(args.buffer_count)
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="Live view for IMX900 cameras using ioctl/mmap.")
    parser.add_argument("--device0", default="/dev/video0", help="Path to first V4L2 device.")
    parser.add_argument("--device1", default="/dev/video1", help="Path to second V4L2 device.")
    parser.add_argument("--single-camera", type=int, choices=[0, 1], help="Capture only camera index 0 or 1.")
    parser.add_argument("--width", type=int, default=2048, help="Frame width.")
    parser.add_argument("--height", type=int, default=1536, help="Frame height.")
    parser.add_argument("--bit-depth", type=int, default=10, help="Bit depth of raw frames.")
    parser.add_argument("--align", choices=["auto", "msb", "lsb"], default="msb", help="Bit alignment strategy.")
    parser.add_argument("--bayer", default="bg", help="Bayer pattern (rg, gr, gb, bg).")
    parser.add_argument(
        "--demosaic",
        choices=["bilinear", "edge", "vng"],
        default="edge",
        help="Demosaicing algorithm: bilinear (fast), edge (edge-aware), or vng (slowest).",
    )
    parser.add_argument(
        "--preview-mode",
        choices=["color", "raw", "both"],
        default="color",
        help="Display demosaiced color, grayscale raw, or both panes per camera.",
    )
    parser.add_argument("--fps", type=int, default=0, help="Optional FPS to request.")
    parser.add_argument("--pixel-format", default="RG10", help="FourCC pixel format.")
    parser.add_argument("--bypass-mode", type=int, choices=[0, 1], default=0, help="bypass_mode control value.")
    parser.add_argument("--buffer-count", type=int, default=4, help="Number of MMAP buffers to request.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Frame timeout per camera.")
    parser.add_argument("--display-scale", type=float, default=0.5, help="Scale factor for display window.")
    parser.add_argument("--no-configure", action="store_true", help="Skip configuring the sensor via ioctl.")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV window (FPS logging only).")
    parser.add_argument(
        "--white-balance",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="Manual R G B gains applied after demosaicing (e.g. 2.1 1.0 1.7).",
    )
    args = parser.parse_args()
    wb_gains_bgr: np.ndarray | None = None
    if args.white_balance is not None:
        wb = np.array(args.white_balance, dtype=np.float32)
        if np.any(wb <= 0):
            parser.error("White balance multipliers must be positive.")
        wb_gains_bgr = wb[[2, 1, 0]]

    devices = [args.device0, args.device1]
    labels = ["cam0", "cam1"]
    if args.single_camera is not None:
        devices = [devices[args.single_camera]]
        labels = [labels[args.single_camera]]

    captures: List[V4L2Capture] = []
    capture_info: List[tuple[V4L2Capture, int, int, int]] = []
    try:
        for device in devices:
            cap = setup_capture(args, device)
            captures.append(cap)
            fmt = cap.format
            capture_info.append((cap, fmt.width, fmt.height, fmt.bytesperline))
            print(
                f"{device}: {fmt.width}x{fmt.height} fourcc=0x{fmt.pixelformat:08x} "
                f"bytesperline={fmt.bytesperline} sizeimage={fmt.sizeimage}"
            )
    except Exception as exc:  # noqa: BLE001
        for cap in captures:
            cap.close()
        print(f"Failed to initialize devices: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Press 'q' to exit." if not args.headless else "Headless mode: Ctrl+C to stop.")
    last_report = time.time()
    frame_counter = 0
    try:
        while True:
            display_frames: List[np.ndarray | None] = []
            display_labels: List[str] = []
            for idx, (cap, width, height, bytesperline) in enumerate(capture_info):
                label = labels[idx]
                try:
                    raw = cap.read(args.timeout)
                    mono = raw_to_mono(raw, width, height, args.bit_depth, args.align, bytesperline)
                    if args.preview_mode in {"color", "both"}:
                        rgb = demosaic(mono, args.bayer, method=args.demosaic)
                        if wb_gains_bgr is not None:
                            rgb = apply_white_balance(rgb, wb_gains_bgr)
                        display_frames.append(rgb)
                        display_labels.append(label if args.preview_mode == "color" else f"{label}-color")
                    if args.preview_mode in {"raw", "both"}:
                        gray_bgr = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
                        display_frames.append(gray_bgr)
                        display_labels.append(label if args.preview_mode == "raw" else f"{label}-raw")
                except TimeoutError:
                    print(f"Timeout from {cap.device}", file=sys.stderr)
                    continue
                except ValueError as exc:
                    print(f"Processing error from {cap.device}: {exc}", file=sys.stderr)
                    continue

            frame_counter += 1
            now = time.time()
            if now - last_report >= 2.0:
                fps = frame_counter / (now - last_report)
                print(f"Approx FPS: {fps:.1f}")
                frame_counter = 0
                last_report = now

            if not args.headless:
                display = compose_display(display_frames, display_labels, args.display_scale)
                cv2.imshow("IMX900 Live View", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        for cap in captures:
            cap.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
