import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from raw_utils import align_raw, demosaic, to_uint8


def read_raw(path: Path, width: int, height: int, bit_depth: int, align: str = "auto") -> np.ndarray:
    """
    Load a RAW frame stored as little-endian 16-bit words.
    align:
      - auto: if values exceed bit depth range, assume MSB alignment and shift down.
      - msb: shift down from MSB to fit bit_depth.
      - lsb: use as-is.
    """
    data = np.fromfile(str(path), dtype=np.uint16)
    expected = width * height
    if data.size != expected:
        raise ValueError(f"Size mismatch: expected {expected} pixels, got {data.size}")
    frame = data.reshape((height, width))
    return align_raw(frame, bit_depth, align)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RAW (10/12-bit packed as uint16) to 8-bit and display with OpenCV."
    )
    parser.add_argument("input", type=Path, help="Path to RAW file (uint16 little-endian).")
    parser.add_argument("--width", type=int, default=2048, help="Image width in pixels (default: 2048).")
    parser.add_argument("--height", type=int, default=1536, help="Image height in pixels (default: 1536).")
    parser.add_argument("--bayer", type=str, default="rg", help="Bayer pattern: bg, gb, gr, rg. Default: rg.")
    parser.add_argument("--bit-depth", type=int, default=12, help="Bit depth of RAW data (10 or 12). Default: 12.")
    parser.add_argument(
        "--align",
        type=str,
        default="auto",
        help="Bit alignment: auto (default), msb (shift down from MSB), lsb (use as-is).",
    )
    parser.add_argument(
        "--demosaic",
        choices=["bilinear", "edge", "vng"],
        default="edge",
        help="Demosaicing algorithm when --bayer is set.",
    )
    parser.add_argument(
        "--save", type=Path, default=None, help="Optional path to save the 8-bit output (PNG)."
    )
    args = parser.parse_args()

    frame_raw = read_raw(args.input, args.width, args.height, args.bit_depth, align=args.align)
    print(
        f"Loaded RAW: shape={frame_raw.shape}, dtype={frame_raw.dtype}, "
        f"min={frame_raw.min()}, max={frame_raw.max()}, mean={frame_raw.mean():.1f}"
    )
    frame_8bit = to_uint8(frame_raw, args.bit_depth)
    frame_vis = demosaic(frame_8bit, args.bayer, method=args.demosaic) if args.bayer else frame_8bit

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), frame_vis)
        print(f"Saved 8-bit image to {args.save}")

    cv2.imshow("8-bit preview", frame_vis)
    print("Press any key in the OpenCV window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
