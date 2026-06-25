from __future__ import annotations

import cv2
import numpy as np

from .schema import MatchResult

WINDOW_KEYPOINTS = "keypoints"
WINDOW_RAW_MATCHES = "raw_matches"
WINDOW_RANSAC_MATCHES = "ransac_matches"
WINDOW_HOMOGRAPHY = "homography"
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)
KEYPOINT_COLOR = (0, 255, 255)
RAW_INLIER_MATCH_COLOR = (0, 220, 0)
RAW_OUTLIER_MATCH_COLOR = (0, 0, 255)
RANSAC_MATCH_COLOR = (255, 180, 40)
FAIL_COLOR = (60, 60, 255)
KEYPOINT_RADIUS = 3
MATCH_POINT_RADIUS = 3
MATCH_LINE_THICKNESS = 1
TEXT_SCALE = 1.0
TEXT_THICKNESS = 2
PANEL_GAP = 8


def build_visualization_windows(
    image0: np.ndarray, image1: np.ndarray, result: MatchResult
) -> dict[str, np.ndarray]:
    image0 = _as_bgr(image0, "image0")
    image1 = _as_bgr(image1, "image1")
    return {
        WINDOW_KEYPOINTS: render_keypoints(image0, image1, result),
        WINDOW_RAW_MATCHES: render_raw_matches(image0, image1, result),
        WINDOW_RANSAC_MATCHES: render_ransac_matches(image0, image1, result),
        WINDOW_HOMOGRAPHY: render_homography(image0, image1, result),
    }


def show_visualization_windows(windows: dict[str, np.ndarray], wait_ms: int = 0) -> int:
    for name, canvas in windows.items():
        cv2.imshow(name, canvas)
    return cv2.waitKey(wait_ms)


def render_keypoints(
    image0: np.ndarray, image1: np.ndarray, result: MatchResult
) -> np.ndarray:
    image0, image1 = _as_bgr_pair(image0, image1)
    left = _draw_points(image0, result.keypoints0, KEYPOINT_COLOR)
    right = _draw_points(image1, result.keypoints1, KEYPOINT_COLOR)
    _put_text(left, f"image0 keypoints: {result.num_keypoints0}", (10, 24))
    _put_text(right, f"image1 keypoints: {result.num_keypoints1}", (10, 24))
    return _side_by_side(left, right)


def render_raw_matches(
    image0: np.ndarray, image1: np.ndarray, result: MatchResult
) -> np.ndarray:
    image0, image1 = _as_bgr_pair(image0, image1)
    pairs = (result.matched_keypoints0, result.matched_keypoints1)
    match_count = min(len(pairs[0]), len(pairs[1]))
    inlier_mask = _raw_inlier_mask(result, match_count)
    canvas = _draw_raw_match_pairs(image0, image1, pairs, inlier_mask)
    kept = int(inlier_mask.sum())
    filtered = match_count - kept
    label = f"raw matches: {match_count} kept: {kept} filtered: {filtered}"
    _put_text(canvas, label, (10, 24))
    return canvas


def render_ransac_matches(
    image0: np.ndarray, image1: np.ndarray, result: MatchResult
) -> np.ndarray:
    image0, image1 = _as_bgr_pair(image0, image1)
    points0 = (
        _empty_points()
        if result.inlier_keypoints0 is None
        else result.inlier_keypoints0
    )
    points1 = (
        _empty_points()
        if result.inlier_keypoints1 is None
        else result.inlier_keypoints1
    )
    canvas = _draw_match_pairs(image0, image1, (points0, points1), RANSAC_MATCH_COLOR)
    _put_text(canvas, f"RANSAC matches: {result.num_inliers}", (10, 24))
    if not result.ok:
        _put_text(canvas, result.message, (10, 50), FAIL_COLOR)
    return canvas


def render_homography(
    image0: np.ndarray, image1: np.ndarray, result: MatchResult
) -> np.ndarray:
    image0, image1 = _as_bgr_pair(image0, image1)
    if result.H_0_to_1 is None:
        canvas = _side_by_side(image1.copy(), image0.copy())
        _put_text(canvas, "image1 reference", (10, 24))
        _put_text(
            canvas,
            "homography unavailable",
            (image1.shape[1] + PANEL_GAP + 10, 24),
            FAIL_COLOR,
        )
        return canvas

    height, width = image1.shape[:2]
    warped0 = cv2.warpPerspective(image0, result.H_0_to_1, (width, height))
    canvas = _side_by_side(image1.copy(), warped0)
    _put_text(canvas, "image1 reference", (10, 24))
    _put_text(canvas, "image0 warped to image1", (width + PANEL_GAP + 10, 24))
    return canvas


def _draw_match_pairs(
    image0: np.ndarray, image1: np.ndarray, pairs: tuple[np.ndarray, np.ndarray], color
) -> np.ndarray:
    canvas = _side_by_side(image0.copy(), image1.copy())
    offset_x = image0.shape[1] + PANEL_GAP
    points0, points1 = pairs
    for point0, point1 in zip(points0, points1):
        _draw_match_pair(canvas, point0, point1, offset_x, color)
    return canvas


def _draw_raw_match_pairs(
    image0: np.ndarray,
    image1: np.ndarray,
    pairs: tuple[np.ndarray, np.ndarray],
    inlier_mask: np.ndarray,
) -> np.ndarray:
    canvas = _side_by_side(image0.copy(), image1.copy())
    offset_x = image0.shape[1] + PANEL_GAP
    points0, points1 = pairs
    for index, (point0, point1) in enumerate(zip(points0, points1)):
        color = (
            RAW_INLIER_MATCH_COLOR if inlier_mask[index] else RAW_OUTLIER_MATCH_COLOR
        )
        _draw_match_pair(canvas, point0, point1, offset_x, color)
    return canvas


def _draw_match_pair(canvas: np.ndarray, point0, point1, offset_x: int, color) -> None:
    start = _point_tuple(point0)
    end = _point_tuple((point1[0] + offset_x, point1[1]))
    cv2.circle(canvas, start, MATCH_POINT_RADIUS, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, end, MATCH_POINT_RADIUS, color, -1, cv2.LINE_AA)
    cv2.line(canvas, start, end, color, MATCH_LINE_THICKNESS, cv2.LINE_AA)


def _raw_inlier_mask(result: MatchResult, match_count: int) -> np.ndarray:
    if result.inlier_mask is None or len(result.inlier_mask) != match_count:
        return np.zeros(match_count, dtype=bool)
    return result.inlier_mask.astype(bool, copy=False)


def _draw_points(image: np.ndarray, points: np.ndarray, color) -> np.ndarray:
    output = image.copy()
    for point in points:
        cv2.circle(output, _point_tuple(point), KEYPOINT_RADIUS, color, -1, cv2.LINE_AA)
    return output


def _side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + right.shape[1] + PANEL_GAP
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    x0 = left.shape[1] + PANEL_GAP
    canvas[: right.shape[0], x0 : x0 + right.shape[1]] = right
    return canvas


def _put_text(
    canvas: np.ndarray, text: str, origin: tuple[int, int], color=TEXT_COLOR
) -> None:
    size, baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS
    )
    x, y = origin
    cv2.rectangle(
        canvas,
        (x - 4, y - size[1] - 6),
        (x + size[0] + 4, y + baseline + 4),
        TEXT_BG_COLOR,
        -1,
    )
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        color,
        TEXT_THICKNESS,
        cv2.LINE_AA,
    )


def _as_bgr_pair(
    image0: np.ndarray, image1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return _as_bgr(image0, "image0"), _as_bgr(image1, "image1")


def _as_bgr(image: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be a BGR image with shape HxWx3")
    return image


def _point_tuple(point) -> tuple[int, int]:
    values = np.asarray(point, dtype=np.float32).reshape(2)
    return int(round(float(values[0]))), int(round(float(values[1])))


def _empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)
