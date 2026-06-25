from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path

import cv2
import numpy as np

from matcher_hdw import ImageMatcher, build_visualization_windows, load_matcher_config

try:
    import dearpygui.dearpygui as dpg
except ModuleNotFoundError:
    dpg = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "matchers" / "sift_nn.yaml"
DEFAULT_VIEWPORT_SIZE = (2560, 1440)
MIN_VIEWPORT_SIZE = (1280, 720)
SIDEBAR_RATIO = 0.21
MIN_SIDEBAR_WIDTH = 340
MAX_SIDEBAR_WIDTH = 460
MIN_PANEL_SIZE = (360, 240)
PANEL_FRAME_EXTRA = (24, 60)
GRID_GAP = 8
VIEWPORT_MARGIN = (20, 24)
ZOOM_MARGIN = (120, 120)
PATH_BUTTON_WIDTH = 46
CONFIG_FILE_EXTENSIONS = (".yaml", ".yml")
IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
CONFIG_EXTENSION_COLOR = (80, 220, 120, 255)
IMAGE_EXTENSION_COLOR = (120, 180, 255, 255)
RANSAC_METHODS = ["opencv", "usac_magsac", "usac_default"]
WINDOW_ORDER = ["keypoints", "raw_matches", "ransac_matches", "homography"]
THRESHOLD_PARAMS = {
    "sift": "contrast_threshold",
    "orb": "fast_threshold",
    "superpoint": "keypoint_threshold",
    "aliked": "detection_threshold",
}
TAGS = {
    "config_path": "input.config_path",
    "image0_path": "input.image0_path",
    "image1_path": "input.image1_path",
    "max_features": "input.max_features",
    "keypoint_threshold": "input.keypoint_threshold",
    "match_threshold": "input.match_threshold",
    "ransac_method": "input.ransac_method",
    "ransac_reproj": "input.ransac_reproj",
    "ransac_confidence": "input.ransac_confidence",
    "ransac_iter": "input.ransac_iter",
    "config_summary": "status.config_summary",
    "status": "status.text",
    "main_window": "window.main",
    "zoom_window": "window.zoom",
    "zoom_texture": "texture.zoom",
}


@dataclass(frozen=True)
class GuiLayout:
    viewport_size: tuple[int, int]
    sidebar_width: int
    panel_size: tuple[int, int]
    panel_window_size: tuple[int, int]
    zoom_size: tuple[int, int]
    file_dialog_size: tuple[int, int]
    path_input_width: int
    status_wrap: int
    font_scale: float


def _build_layout(viewport_size: tuple[int, int]) -> GuiLayout:
    width, height = _normalize_viewport_size(viewport_size)
    sidebar_width = int(
        _clamp(width * SIDEBAR_RATIO, MIN_SIDEBAR_WIDTH, MAX_SIDEBAR_WIDTH)
    )
    available_width = width - sidebar_width - GRID_GAP - VIEWPORT_MARGIN[0]
    available_height = height - VIEWPORT_MARGIN[1]
    panel_width = int((available_width - GRID_GAP - (PANEL_FRAME_EXTRA[0] * 2)) / 2)
    panel_height = int((available_height - GRID_GAP - (PANEL_FRAME_EXTRA[1] * 2)) / 2)
    panel_size = (
        max(MIN_PANEL_SIZE[0], panel_width),
        max(MIN_PANEL_SIZE[1], panel_height),
    )
    panel_window_size = (
        panel_size[0] + PANEL_FRAME_EXTRA[0],
        panel_size[1] + PANEL_FRAME_EXTRA[1],
    )
    zoom_size = (
        max(panel_size[0], width - ZOOM_MARGIN[0]),
        max(panel_size[1], height - ZOOM_MARGIN[1]),
    )
    file_dialog_size = (
        int(_clamp(width * 0.58, 720, width - 80)),
        int(_clamp(height * 0.58, 420, height - 80)),
    )
    path_input_width = max(180, sidebar_width - PATH_BUTTON_WIDTH - 42)
    status_wrap = max(240, sidebar_width - 24)
    font_scale = _clamp(min(width / 1920.0, height / 1080.0) * 1.5, 1.4, 2.0)
    return GuiLayout(
        viewport_size=(width, height),
        sidebar_width=sidebar_width,
        panel_size=panel_size,
        panel_window_size=panel_window_size,
        zoom_size=zoom_size,
        file_dialog_size=file_dialog_size,
        path_input_width=path_input_width,
        status_wrap=status_wrap,
        font_scale=font_scale,
    )


def _normalize_viewport_size(viewport_size: tuple[int, int]) -> tuple[int, int]:
    width, height = viewport_size
    return max(MIN_VIEWPORT_SIZE[0], int(width)), max(MIN_VIEWPORT_SIZE[1], int(height))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class GuiState:
    def __init__(self) -> None:
        self.config = None
        self.layout = _build_layout(DEFAULT_VIEWPORT_SIZE)
        self.windows: dict[str, np.ndarray] = {}


STATE = GuiState()


def main() -> None:
    if dpg is None:
        raise RuntimeError(
            "DearPyGui is not installed. Install it with `pip install dearpygui` "
            "or initialize/build the thirdparty/DearPyGui submodule."
        )
    STATE.layout = _build_layout(_initial_viewport_size())
    dpg.create_context()
    _create_textures()
    _create_file_dialogs()
    _create_main_window()
    _load_config(DEFAULT_CONFIG)
    dpg.create_viewport(
        title="matcher_hdw GUI",
        width=_layout().viewport_size[0],
        height=_layout().viewport_size[1],
        x_pos=0,
        y_pos=0,
    )
    dpg.setup_dearpygui()
    _apply_ui_scale()
    dpg.set_primary_window(TAGS["main_window"], True)
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


def _initial_viewport_size() -> tuple[int, int]:
    env_size = _env_viewport_size()
    if env_size is not None:
        return env_size
    return DEFAULT_VIEWPORT_SIZE


def _env_viewport_size() -> tuple[int, int] | None:
    width = os.getenv("MATCHER_HDW_GUI_WIDTH")
    height = os.getenv("MATCHER_HDW_GUI_HEIGHT")
    if not width or not height:
        return None
    try:
        return int(width), int(height)
    except ValueError:
        return None


def _layout() -> GuiLayout:
    return STATE.layout


def _apply_ui_scale() -> None:
    dpg.set_global_font_scale(_layout().font_scale)


def _create_textures() -> None:
    with dpg.texture_registry():
        for name in WINDOW_ORDER:
            dpg.add_dynamic_texture(
                *_layout().panel_size,
                _blank_texture(*_layout().panel_size),
                tag=_texture_tag(name),
            )
        dpg.add_dynamic_texture(
            *_layout().zoom_size,
            _blank_texture(*_layout().zoom_size),
            tag=TAGS["zoom_texture"],
        )


def _create_file_dialogs() -> None:
    _add_file_dialog(
        "dialog.image0",
        _set_path_callback,
        TAGS["image0_path"],
        IMAGE_FILE_EXTENSIONS,
        IMAGE_EXTENSION_COLOR,
    )
    _add_file_dialog(
        "dialog.image1",
        _set_path_callback,
        TAGS["image1_path"],
        IMAGE_FILE_EXTENSIONS,
        IMAGE_EXTENSION_COLOR,
    )
    _add_file_dialog(
        "dialog.config",
        _config_path_callback,
        TAGS["config_path"],
        CONFIG_FILE_EXTENSIONS,
        CONFIG_EXTENSION_COLOR,
    )


def _add_file_dialog(
    tag: str,
    callback,
    target: str,
    extensions: tuple[str, ...],
    color: tuple[int, int, int, int],
) -> None:
    with dpg.file_dialog(
        show=False,
        callback=callback,
        tag=tag,
        user_data=target,
        width=_layout().file_dialog_size[0],
        height=_layout().file_dialog_size[1],
    ):
        for extension in extensions:
            dpg.add_file_extension(extension, color=color)


def _create_main_window() -> None:
    with dpg.window(tag=TAGS["main_window"], no_title_bar=True):
        with dpg.group(horizontal=True):
            _create_controls()
            dpg.add_spacer(width=GRID_GAP)
            _create_visualization_grid()
    _create_zoom_window()


def _create_controls() -> None:
    with dpg.child_window(width=_layout().sidebar_width, border=True):
        dpg.add_text("Inputs")
        _path_row("Image 0", TAGS["image0_path"], "dialog.image0")
        _path_row("Image 1", TAGS["image1_path"], "dialog.image1")
        _path_row("Config", TAGS["config_path"], "dialog.config")
        dpg.add_button(
            label="Load Config",
            width=-1,
            callback=lambda: _load_config(dpg.get_value(TAGS["config_path"])),
        )
        dpg.add_text(
            _config_summary_message(None),
            tag=TAGS["config_summary"],
            wrap=_layout().status_wrap,
        )
        dpg.add_separator()
        _input_int_row("Max features", TAGS["max_features"], min_value=1)
        _input_float_row("Keypoint threshold", TAGS["keypoint_threshold"], "%.6f")
        _input_float_row("Match threshold", TAGS["match_threshold"], "%.4f")
        dpg.add_separator()
        _combo_row("RANSAC method", RANSAC_METHODS, TAGS["ransac_method"])
        _input_float_row("RANSAC reproj", TAGS["ransac_reproj"], "%.3f")
        _input_float_row("RANSAC confidence", TAGS["ransac_confidence"], "%.5f")
        _input_int_row("RANSAC max iter", TAGS["ransac_iter"], min_value=1)
        dpg.add_separator()
        dpg.add_button(label="Run Matching", width=-1, callback=_run_matching)
        dpg.add_button(label="Reset", width=-1, callback=_reset)
        dpg.add_separator()
        dpg.add_text("Ready", tag=TAGS["status"], wrap=_layout().status_wrap)


def _path_row(label: str, input_tag: str, dialog_tag: str) -> None:
    dpg.add_text(label)
    with dpg.group(horizontal=True):
        dpg.add_input_text(
            tag=input_tag, width=_layout().path_input_width, readonly=False
        )
        dpg.add_button(
            label="...",
            width=PATH_BUTTON_WIDTH,
            callback=lambda: dpg.show_item(dialog_tag),
        )


def _input_int_row(label: str, tag: str, min_value: int) -> None:
    _field_label(label)
    dpg.add_input_int(label="", tag=tag, width=-1, min_value=min_value)


def _input_float_row(label: str, tag: str, number_format: str) -> None:
    _field_label(label)
    dpg.add_input_float(label="", tag=tag, width=-1, format=number_format)


def _combo_row(label: str, values: list[str], tag: str) -> None:
    _field_label(label)
    dpg.add_combo(values, label="", tag=tag, width=-1)


def _field_label(label: str) -> None:
    dpg.add_text(label, wrap=_layout().status_wrap)


def _create_visualization_grid() -> None:
    with dpg.child_window(width=-1, height=-1, border=False):
        for row in (WINDOW_ORDER[:2], WINDOW_ORDER[2:]):
            with dpg.group(horizontal=True):
                for name in row:
                    _visual_panel(name)


def _visual_panel(name: str) -> None:
    with dpg.child_window(
        width=_layout().panel_window_size[0],
        height=_layout().panel_window_size[1],
        border=True,
    ):
        dpg.add_text(name)
        dpg.add_image_button(
            _texture_tag(name),
            width=_layout().panel_size[0],
            height=_layout().panel_size[1],
            callback=_open_zoom,
            user_data=name,
        )


def _create_zoom_window() -> None:
    with dpg.window(
        label="Preview",
        tag=TAGS["zoom_window"],
        width=_layout().zoom_size[0] + 30,
        height=_layout().zoom_size[1] + 70,
        show=False,
    ):
        dpg.add_image(
            TAGS["zoom_texture"],
            width=_layout().zoom_size[0],
            height=_layout().zoom_size[1],
        )


def _set_path_callback(sender, app_data, user_data) -> None:
    del sender
    dpg.set_value(user_data, _selected_file_path(app_data))


def _selected_file_path(app_data: dict) -> str:
    selections = app_data.get("selections") or {}
    for path_value in selections.values():
        if path_value and not str(path_value).endswith(".*"):
            return str(path_value)
    return str(app_data.get("file_path_name", ""))


def _config_path_callback(sender, app_data, user_data) -> None:
    _set_path_callback(sender, app_data, user_data)
    _load_config(dpg.get_value(user_data))


def _load_config(path_value: str | Path) -> None:
    try:
        config_path = Path(path_value).expanduser()
        STATE.config = load_matcher_config(config_path)
        dpg.set_value(TAGS["config_path"], str(config_path))
        _set_config_values(STATE.config)
        _set_config_summary(STATE.config)
        _set_status(f"Loaded config: {config_path}")
    except Exception as exc:
        _set_config_summary(None)
        _set_status(f"Config load failed: {exc}")


def _set_config_values(config) -> None:
    dpg.set_value(TAGS["max_features"], int(config.feature.max_keypoints))
    dpg.set_value(TAGS["keypoint_threshold"], float(_feature_threshold(config)))
    dpg.set_value(TAGS["match_threshold"], float(config.matcher.match_threshold))
    dpg.set_value(TAGS["ransac_method"], config.geometry.ransac_method)
    dpg.set_value(TAGS["ransac_reproj"], float(config.geometry.reproj_threshold))
    dpg.set_value(TAGS["ransac_confidence"], float(config.geometry.confidence))
    dpg.set_value(TAGS["ransac_iter"], int(config.geometry.max_iter))


def _run_matching() -> None:
    if STATE.config is None:
        _load_config(dpg.get_value(TAGS["config_path"]))
    try:
        image0 = _read_image(dpg.get_value(TAGS["image0_path"]), "image0")
        image1 = _read_image(dpg.get_value(TAGS["image1_path"]), "image1")
        config = _config_with_overrides(STATE.config)
        _set_status("Running matching...")
        result = ImageMatcher(config).match(image0, image1)
        STATE.windows = build_visualization_windows(image0, image1, result)
        _update_panel_textures(STATE.windows)
        _set_status(_result_message(result))
    except Exception as exc:
        _set_status(f"Matching failed: {exc}")


def _read_image(path_value: str, name: str) -> np.ndarray:
    path = Path(str(path_value)).expanduser()
    if not str(path_value).strip() or not path.is_file():
        raise ValueError(f"{name} path is not a readable file: {path_value}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{name} could not be read: {path}")
    return image


def _config_with_overrides(config):
    params = dict(config.feature.params)
    threshold_key = THRESHOLD_PARAMS.get(config.feature.name)
    if threshold_key is not None:
        params[threshold_key] = float(dpg.get_value(TAGS["keypoint_threshold"]))
    feature = replace(
        config.feature,
        max_keypoints=int(dpg.get_value(TAGS["max_features"])),
        params=params,
    )
    matcher = replace(
        config.matcher, match_threshold=float(dpg.get_value(TAGS["match_threshold"]))
    )
    geometry = replace(
        config.geometry,
        ransac_method=dpg.get_value(TAGS["ransac_method"]),
        reproj_threshold=float(dpg.get_value(TAGS["ransac_reproj"])),
        confidence=float(dpg.get_value(TAGS["ransac_confidence"])),
        max_iter=int(dpg.get_value(TAGS["ransac_iter"])),
    )
    return replace(config, feature=feature, matcher=matcher, geometry=geometry)


def _feature_threshold(config) -> float:
    threshold_key = THRESHOLD_PARAMS.get(config.feature.name)
    if threshold_key is None:
        return 0.0
    return config.feature.params.get(threshold_key, 0.0)


def _set_config_summary(config) -> None:
    dpg.set_value(TAGS["config_summary"], _config_summary_message(config))


def _config_summary_message(config) -> str:
    if config is None:
        return "feature: -\nmatcher: -"
    return f"feature: {config.feature.name}\nmatcher: {config.matcher.type}"


def _update_panel_textures(windows: dict[str, np.ndarray]) -> None:
    for name in WINDOW_ORDER:
        _set_texture(_texture_tag(name), windows[name], _layout().panel_size)


def _open_zoom(sender, app_data, user_data) -> None:
    del sender, app_data
    image = STATE.windows.get(user_data)
    if image is None:
        return
    _set_texture(TAGS["zoom_texture"], image, _layout().zoom_size)
    dpg.configure_item(TAGS["zoom_window"], label=user_data, show=True)


def _reset() -> None:
    STATE.windows = {}
    dpg.set_value(TAGS["image0_path"], "")
    dpg.set_value(TAGS["image1_path"], "")
    for name in WINDOW_ORDER:
        dpg.set_value(_texture_tag(name), _blank_texture(*_layout().panel_size))
    dpg.set_value(TAGS["zoom_texture"], _blank_texture(*_layout().zoom_size))
    _load_config(DEFAULT_CONFIG)
    _set_status("Reset")


def _set_texture(tag: str, image_bgr: np.ndarray, size: tuple[int, int]) -> None:
    canvas = _fit_bgr(image_bgr, size)
    rgba = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
    dpg.set_value(tag, rgba.ravel())


def _fit_bgr(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if image_bgr.size == 0:
        return canvas
    scale = min(width / image_bgr.shape[1], height / image_bgr.shape[0])
    new_size = (
        max(1, int(image_bgr.shape[1] * scale)),
        max(1, int(image_bgr.shape[0] * scale)),
    )
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, new_size, interpolation=interpolation)
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def _blank_texture(width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 4), dtype=np.float32)
    image[:, :, 3] = 1.0
    return image.ravel()


def _texture_tag(name: str) -> str:
    return f"texture.{name}"


def _result_message(result) -> str:
    return f"ok={result.ok} matches={result.num_matches} inliers={result.num_inliers} ratio={result.inlier_ratio:.3f}"


def _set_status(message: str) -> None:
    dpg.set_value(TAGS["status"], message)


if __name__ == "__main__":
    main()
