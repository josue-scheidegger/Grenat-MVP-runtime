import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def ensure_log_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()


def parse_capture_time(path: Path) -> datetime:
    """
    Parse timestamps like 20251107_114822474.jpeg -> 2025-11-07 11:48:22.474000.
    Falls back to now() if parsing fails.
    """
    stem = path.stem
    try:
        date_part, time_part = stem.split("_", 1)
        year = int(date_part[0:4])
        month = int(date_part[4:6])
        day = int(date_part[6:8])
        hour = int(time_part[0:2])
        minute = int(time_part[2:4])
        second = int(time_part[4:6])
        milliseconds = int(time_part[6:]) if len(time_part) > 6 else 0
        return datetime(year, month, day, hour, minute, second, milliseconds * 1000)
    except Exception:
        return datetime.now()


def compute_delay_seconds(speed_m_per_min: float, distance_m: float) -> float:
    speed_m_per_s = speed_m_per_min / 60.0
    if speed_m_per_s == 0:
        return 0.0
    return distance_m / speed_m_per_s


def map_detection_to_actuator(
    x_center_px: float, image_width_px: float, conveyor_width_cm: float
) -> Tuple[int, int]:
    # 64 actuators across the width -> 4 boards x 16 each.
    total_actuators = 64
    zone_width_cm = conveyor_width_cm / total_actuators
    position_cm = (x_center_px / image_width_px) * conveyor_width_cm
    finger_index = int(math.floor(position_cm / zone_width_cm))
    finger_index = max(0, min(total_actuators - 1, finger_index))
    board_id = (finger_index // 16) + 1
    actuator_id = (finger_index % 16) + 1
    return board_id, actuator_id


def load_images(sample_dir: Path) -> List[Path]:
    images = sorted(sample_dir.glob("*.jp*g"))
    return images


def tile_image(image: np.ndarray, tile_size: int = 640) -> List[Tuple[np.ndarray, int, int]]:
    """
    Slice image into non-overlapping tiles of tile_size x tile_size.
    Returns list of (tile_array, x_offset, y_offset).
    """
    tiles: List[Tuple[np.ndarray, int, int]] = []
    h, w, _ = image.shape
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y : min(y + tile_size, h), x : min(x + tile_size, w)]
            tiles.append((tile, x, y))
    return tiles


def should_reject(
    class_name: str, accept_classes: Optional[Iterable[str]]
) -> bool:
    if not accept_classes:
        return True
    return class_name not in set(accept_classes)


def run_inference_on_images(
    model: YOLO,
    images: List[Path],
    config: Dict[str, Any],
    log_path: Path,
) -> None:
    delay_s = compute_delay_seconds(
        config["conveyor"]["speed_m_per_min"], config["conveyor"]["distance_to_eject_m"]
    )
    conveyor_width_cm = config["conveyor"]["width_cm"]
    accept_classes = config.get("inference", {}).get("accept_classes") or []
    conf = config.get("inference", {}).get("confidence", 0.25)
    iou = config.get("inference", {}).get("iou", 0.45)

    ensure_log_file(log_path)

    with log_path.open("w") as log_file:
        for image_path in images:
            capture_time = parse_capture_time(image_path)
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: failed to load image {image_path}")
                continue
            tiles = tile_image(image, tile_size=640)
            global_img_w = image.shape[1]
            print(f"{image_path.name}: processing {len(tiles)} tiles")

            for tile, x_off, _ in tiles:
                try:
                    result_obj = model(tile, conf=conf, iou=iou, verbose=False)
                except Exception as e:
                    print(f"Error running inference on tile ({x_off}): {e}")
                    continue
                # model() can return a list; normalize to a single result
                result = result_obj[0] if isinstance(result_obj, list) else result_obj
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_idx = int(box.cls[0])
                    class_name = result.names.get(cls_idx, str(cls_idx))
                    if not should_reject(class_name, accept_classes):
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x_center_global = (x1 + x2) / 2.0 + x_off
                    board_id, actuator_id = map_detection_to_actuator(
                        x_center_global, global_img_w, conveyor_width_cm
                    )
                    activation_time = capture_time + timedelta(seconds=delay_s)
                    duration_ms = 50  # placeholder pulse duration
                    log_line = f"{board_id};{actuator_id};{activation_time.isoformat()};{duration_ms}\n"
                    log_file.write(log_line)
                    print(
                        f"{image_path.name}: reject '{class_name}' -> board {board_id}, actuator {actuator_id}, fire at {activation_time.isoformat()}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run runtime on sample images.")
    parser.add_argument(
        "--config",
        default="configs/runtime.yaml",
        type=Path,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = Path(config["paths"]["model_weights"])
    sample_dir = Path(config["paths"]["sample_images_dir"])
    log_path = Path(config["paths"]["can_log"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample dir not found: {sample_dir}")

    images = load_images(sample_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {sample_dir}")

    print(f"Loaded {len(images)} images from {sample_dir}")
    print(f"Using model: {model_path}")
    print(f"Logging CAN emulation to: {log_path}")

    model = YOLO(model_path)
    run_inference_on_images(model, images, config, log_path)
    print("Done.")


if __name__ == "__main__":
    main()
