from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from matcher_hdw import (
    ImageMatcher,
    build_visualization_windows,
    load_builtin_config,
    load_matcher_config,
    show_visualization_windows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Match a pair of images")
    parser.add_argument("image0")
    parser.add_argument("image1")
    parser.add_argument("--config", default="sift_nn")
    parser.add_argument(
        "--show", action="store_true", help="show visualization windows"
    )
    parser.add_argument(
        "--wait-ms", type=int, default=0, help="cv2.waitKey delay for --show"
    )
    parser.add_argument(
        "--save-viz-dir",
        type=Path,
        default=None,
        help="directory to save visualization PNGs",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    image0 = _read_image(args.image0, "image0")
    image1 = _read_image(args.image1, "image1")
    result = ImageMatcher(config).match(image0, image1)
    _print_result(result)

    if args.show or args.save_viz_dir is not None:
        windows = build_visualization_windows(image0, image1, result)
        if args.save_viz_dir is not None:
            saved_paths = _save_visualization_windows(windows, args.save_viz_dir)
            print("saved visualizations:")
            for saved_path in saved_paths:
                print(saved_path)
        if args.show:
            show_visualization_windows(windows, args.wait_ms)


def _load_config(value: str):
    if value.endswith(".yaml") or "/" in value:
        return load_matcher_config(value)
    return load_builtin_config(value)


def _read_image(path: str, name: str):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{name} could not be read: {path}")
    return image


def _print_result(result) -> None:
    print(f"ok: {result.ok}")
    print(f"message: {result.message}")
    print(f"matches: {result.num_matches}")
    print(f"inliers: {result.num_inliers}")
    print(f"inlier_ratio: {result.inlier_ratio:.3f}")
    print("H_0_to_1:")
    print(result.H_0_to_1)


def _save_visualization_windows(windows: dict, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for name, canvas in windows.items():
        output_path = output_dir / f"{name}.png"
        if not cv2.imwrite(str(output_path), canvas):
            raise RuntimeError(f"failed to save visualization: {output_path}")
        saved_paths.append(output_path)
    return saved_paths


if __name__ == "__main__":
    main()
