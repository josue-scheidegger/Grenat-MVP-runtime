from typing import Sequence

import cv2
import numpy as np


DEMOSAIC_CODES = {
    "bg": {
        "bilinear": cv2.COLOR_BayerBG2BGR,
        "edge": cv2.COLOR_BayerBG2BGR_EA,
        "vng": cv2.COLOR_BayerBG2BGR_VNG,
    },
    "gb": {
        "bilinear": cv2.COLOR_BayerGB2BGR,
        "edge": cv2.COLOR_BayerGB2BGR_EA,
        "vng": cv2.COLOR_BayerGB2BGR_VNG,
    },
    "gr": {
        "bilinear": cv2.COLOR_BayerGR2BGR,
        "edge": cv2.COLOR_BayerGR2BGR_EA,
        "vng": cv2.COLOR_BayerGR2BGR_VNG,
    },
    "rg": {
        "bilinear": cv2.COLOR_BayerRG2BGR,
        "edge": cv2.COLOR_BayerRG2BGR_EA,
        "vng": cv2.COLOR_BayerRG2BGR_VNG,
    },
}


def align_raw(raw: np.ndarray, bit_depth: int, align: str) -> np.ndarray:
    """Align raw sensor data to the requested bit depth."""
    if align not in {"auto", "msb", "lsb"}:
        raise ValueError("align must be one of: auto, msb, lsb")
    shift = 0
    max_val = (1 << bit_depth) - 1
    if align == "msb":
        shift = 16 - bit_depth
    elif align == "auto":
        if raw.max() > max_val:
            shift = 16 - bit_depth
    # lsb keeps shift at 0
    if shift:
        raw = raw >> shift
    return np.clip(raw, 0, max_val)


def to_uint8(raw: np.ndarray, bit_depth: int) -> np.ndarray:
    """Scale bit_depth data to 8-bit."""
    max_val = max(1, (1 << bit_depth) - 1)
    scaled = (raw.astype(np.float32) / max_val) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def demosaic(frame_8bit: np.ndarray, bayer: str, method: str = "bilinear") -> np.ndarray:
    """Convert a single-channel Bayer frame into BGR using the requested algorithm."""
    pattern = bayer.lower()
    method_key = method.lower()
    pattern_map = DEMOSAIC_CODES.get(pattern)
    if pattern_map is None:
        raise ValueError("Unsupported Bayer pattern. Use bg, gb, gr, or rg.")
    code = pattern_map.get(method_key)
    if code is None:
        raise ValueError("Unsupported demosaic method. Use bilinear, edge, or vng.")
    return cv2.cvtColor(frame_8bit, code)


def apply_white_balance(bgr: np.ndarray, gains_bgr: Sequence[float]) -> np.ndarray:
    """Scale BGR channels independently using (B, G, R) multipliers."""
    multipliers = np.asarray(gains_bgr, dtype=np.float32)
    if multipliers.size != 3:
        raise ValueError("White balance requires three multipliers (B, G, R).")
    if np.any(multipliers < 0):
        raise ValueError("White balance multipliers must be non-negative.")
    multipliers = multipliers.reshape((1, 1, 3))
    balanced = bgr.astype(np.float32) * multipliers
    return np.clip(balanced, 0, 255).astype(np.uint8)


def raw_bytes_to_array(
    data: bytes,
    width: int,
    height: int,
    bytesperline: int | None = None,
) -> np.ndarray:
    """Interpret raw bytes as a uint16 image, respecting optional stride padding."""
    arr = np.frombuffer(data, dtype=np.uint16)
    if not bytesperline or bytesperline == width * 2:
        expected = width * height
        if arr.size < expected:
            raise ValueError(f"Received {arr.size} pixels, expected {expected}")
        return arr[:expected].reshape((height, width))

    stride_pixels = bytesperline // 2
    expected = stride_pixels * height
    if arr.size < expected:
        raise ValueError(f"Received {arr.size} pixels, expected {expected}")
    frame = arr[:expected].reshape((height, stride_pixels))
    return frame[:, :width]


def raw_to_mono(
    data: bytes,
    width: int,
    height: int,
    bit_depth: int,
    align: str,
    bytesperline: int | None = None,
) -> np.ndarray:
    """Convert raw bytes to an 8-bit single-channel view."""
    frame16 = raw_bytes_to_array(data, width, height, bytesperline)
    aligned = align_raw(frame16, bit_depth, align)
    return to_uint8(aligned, bit_depth)


def raw_to_rgb(
    data: bytes,
    width: int,
    height: int,
    bit_depth: int,
    align: str,
    bayer: str,
    bytesperline: int | None = None,
    demosaic_mode: str = "bilinear",
) -> np.ndarray:
    """Convert raw bytes to an 8-bit BGR image."""
    frame_8bit = raw_to_mono(data, width, height, bit_depth, align, bytesperline)
    return demosaic(frame_8bit, bayer, method=demosaic_mode)
