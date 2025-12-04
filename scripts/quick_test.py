import argparse
import sys
from pathlib import Path

from raw_v4l2 import V4L2Capture, fourcc


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a single RAW frame using low-level V4L2 ioctls.")
    parser.add_argument("--device", default="/dev/video0", help="Path to V4L2 device (/dev/videoN).")
    parser.add_argument("--width", type=int, default=2048, help="Frame width (VC IMX900 default: 2048).")
    parser.add_argument("--height", type=int, default=1536, help="Frame height (VC IMX900 default: 1536).")
    parser.add_argument("--pixel-format", default="RG10", help="FourCC pixel format (e.g., RG10, RGGB).")
    parser.add_argument("--bypass-mode", type=int, choices=[0, 1], default=0, help="bypass_mode control value.")
    parser.add_argument("--fps", type=int, default=0, help="Optional FPS request via V4L2.")
    parser.add_argument("--output", type=Path, default=Path("frame.raw"), help="Output RAW filename.")
    parser.add_argument("--skip-configure", action="store_true", help="Skip changing controls/format.")
    parser.add_argument("--buffer-count", type=int, default=4, help="Number of MMAP buffers to request.")
    parser.add_argument("--timeout", type=float, default=15.0, help="Timeout waiting for a frame (seconds).")
    args = parser.parse_args()

    try:
        pixfmt = fourcc(args.pixel_format)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    expected_bytes = args.width * args.height * 2

    with V4L2Capture(args.device) as cap:
        if not args.skip_configure:
            print("Configuring sensor...")
            cap.configure(
                width=args.width,
                height=args.height,
                pixfmt=pixfmt,
                bypass_mode=args.bypass_mode,
                fps=args.fps if args.fps > 0 else None,
            )
        else:
            print("Skipping configuration per --skip-configure.")
            cap.current_format()

        cap.start(args.buffer_count)
        print(
            f"Capturing from {args.device}: "
            f"{cap.format.width}x{cap.format.height} fourcc=0x{cap.format.pixelformat:08x}"
        )
        try:
            frame = cap.read(args.timeout)
        except TimeoutError as exc:
            print(f"Capture failed: {exc}", file=sys.stderr)
            sys.exit(1)
        finally:
            cap.stop()

    if len(frame) < expected_bytes:
        print(
            f"Warning: received {len(frame)} bytes (expected {expected_bytes}). Writing available data.",
            file=sys.stderr,
        )
    args.output.write_bytes(frame[:expected_bytes])
    print(f"Saved RAW frame to {args.output} ({min(len(frame), expected_bytes)} bytes).")


if __name__ == "__main__":
    main()
