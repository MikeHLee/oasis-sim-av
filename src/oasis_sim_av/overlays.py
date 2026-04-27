"""Overlay helpers for multi-view rendering.

Provides:
- reproject_points_to_camera: project world points into camera image space
- rasterise_lidar_bev: render LiDAR points onto a BEV canvas with kind-based colours
- draw_bboxes: draw 2D bounding boxes onto an image
- draw_fusion_strip: draw a running fusion posterior plot
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


KIND_COLORS = {
    0: np.array([80, 95, 75], dtype=np.uint8),
    1: np.array([140, 140, 150], dtype=np.uint8),
    2: np.array([240, 220, 40], dtype=np.uint8),
    3: np.array([0, 200, 220], dtype=np.uint8),
}


def reproject_points_to_camera(
    points_xyz: np.ndarray,
    camera_offset: np.ndarray,
    camera_forward: np.ndarray,
    camera_up: np.ndarray,
    fov_h_deg: float,
    width: int,
    height: int,
    veh_origin: np.ndarray,
    veh_R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into camera image coordinates.

    Parameters
    ----------
    points_xyz : (N, 3) array
        World-space points.
    camera_offset, camera_forward, camera_up : (3,) arrays
        Camera parameters in vehicle body frame.
    fov_h_deg : float
        Horizontal field of view.
    width, height : int
        Image dimensions.
    veh_origin : (3,) array
        Vehicle world position.
    veh_R : (3, 3) array
        Vehicle body-to-world rotation matrix.

    Returns
    -------
    u, v : (N,) int arrays
        Pixel coordinates (may be outside image bounds).
    mask : (N,) bool array
        True for points that project in front of the camera and within bounds.
    """
    if points_xyz.shape[0] == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=bool),
        )

    cam_pos_world = veh_origin + veh_R @ camera_offset

    fwd = camera_forward / (np.linalg.norm(camera_forward) + 1e-12)
    up = camera_up / (np.linalg.norm(camera_up) + 1e-12)
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, fwd)

    R_cam_to_body = np.stack([fwd, right, up], axis=1)

    R_body_to_world = veh_R
    R_cam_to_world = R_body_to_world @ R_cam_to_body
    R_world_to_cam = R_cam_to_world.T

    p_rel = points_xyz - cam_pos_world
    p_cam = p_rel @ R_world_to_cam.T

    z = p_cam[:, 0]
    mask_front = z > 0.1

    fx = 1.0 / np.tan(np.deg2rad(fov_h_deg) * 0.5)
    aspect = width / height

    x_ndc = p_cam[:, 1] / (z + 1e-12) * fx
    y_ndc = p_cam[:, 2] / (z + 1e-12) * fx * aspect

    u = ((x_ndc + 1.0) * 0.5 * width).astype(np.int32)
    v = ((1.0 - y_ndc) * 0.5 * height).astype(np.int32)

    mask_in_bounds = (
        (u >= 0) & (u < width) & (v >= 0) & (v < height)
    )
    mask = mask_front & mask_in_bounds

    return u, v, mask


def rasterise_lidar_bev(
    points: np.ndarray,
    kinds: np.ndarray,
    center: np.ndarray,
    extent_m: float,
    size_px: int,
    ranges: np.ndarray | None = None,
) -> np.ndarray:
    """Render LiDAR points onto a BEV canvas with kind-based colours.

    Parameters
    ----------
    points : (N, 2) or (N, 3) array
        LiDAR points (x, y) or (x, y, z).
    kinds : (N,) int array
        Point kinds: 0=ground, 1=building, 2=tape, 3=rain.
    center : (2,) array
        BEV center (x, y).
    extent_m : float
        Side length of square viewport.
    size_px : int
        Output image size.

    Returns
    -------
    img : (H, W, 3) uint8 array
    """
    img = np.zeros((size_px, size_px, 3), dtype=np.uint8)

    if points.shape[0] == 0:
        return img

    half = extent_m / 2.0
    cx, cy = center

    xy = points[:, :2]
    px = ((xy[:, 0] - (cx - half)) / extent_m * size_px).astype(np.int32)
    py = ((xy[:, 1] - (cy - half)) / extent_m * size_px).astype(np.int32)

    valid = (px >= 0) & (px < size_px) & (py >= 0) & (py < size_px)

    if ranges is not None:
        r_min, r_max = ranges[valid].min(), ranges[valid].max()
        r_range = max(r_max - r_min, 1e-6)

    for i in np.where(valid)[0]:
        k = int(kinds[i])
        color = KIND_COLORS.get(k, np.array([128, 128, 128], dtype=np.uint8))

        if ranges is not None and k != 3:
            t = (ranges[i] - r_min) / r_range
            color = (color.astype(np.float64) * (0.5 + 0.5 * (1 - t))).astype(np.uint8)

        ix, iy = int(px[i]), int(py[i])
        img[iy, ix] = color

    return img


def draw_bboxes(
    img: np.ndarray,
    bboxes: list[dict],
    color: tuple[int, int, int] = (255, 80, 80),
    thickness: int = 2,
) -> np.ndarray:
    """Draw 2D bounding boxes onto an image.

    bboxes: list of {"bbox": [xmin, ymin, xmax, ymax], "score": float}
    """
    H, W = img.shape[:2]
    out = img.copy()

    for bb in bboxes:
        x0, y0, x1, y1 = bb["bbox"]
        x0 = max(0, min(int(x0), W - 1))
        x1 = max(0, min(int(x1), W - 1))
        y0 = max(0, min(int(y0), H - 1))
        y1 = max(0, min(int(y1), H - 1))

        if x1 <= x0 or y1 <= y0:
            continue

        for t in range(thickness):
            if y0 + t < H:
                out[y0 + t, x0:x1 + 1] = color
            if y1 - t >= 0:
                out[y1 - t, x0:x1 + 1] = color
            if x0 + t < W:
                out[y0:y1 + 1, x0 + t] = color
            if x1 - t >= 0:
                out[y0:y1 + 1, x1 - t] = color

    return out


def draw_fusion_strip(
    p_fused_series: list[float],
    width: int,
    height: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Draw a running fusion posterior as a strip chart.

    Returns (height, width, 3) uint8 image.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = [25, 25, 30]

    if not p_fused_series:
        return img

    n = len(p_fused_series)
    col_w = max(1, width // max(n, 1))

    for i, p in enumerate(p_fused_series):
        x0 = i * col_w
        if x0 >= width:
            break

        bar_h = int(height * min(1.0, max(0.0, p)))
        if bar_h > 0:
            color = [240, 220, 40] if p >= threshold else [140, 140, 150]
            img[height - bar_h:height, x0:x0 + col_w - 1] = color

    thresh_h = int(height * threshold)
    img[height - thresh_h, :] = [180, 60, 60]

    return img


def compose_grid5x2(
    panels: list[np.ndarray],
    titles: list[str] | None = None,
    title_height: int = 18,
    footer_text: str = "",
    footer_height: int = 24,
) -> np.ndarray:
    """Compose 10 panels into a 5×2 grid.

    Returns a single (H, W, 3) uint8 image.

    Layout:
        Row 0: panels[0] ... panels[4]
        Row 1: panels[5] ... panels[9]
    """
    if len(panels) != 10:
        raise ValueError(f"Expected 10 panels, got {len(panels)}")

    sizes = [p.shape[:2] for p in panels]
    max_h = max(s[0] for s in sizes)
    max_w = max(s[1] for s in sizes)

    panel_h = max_h + title_height
    panel_w = max_w

    total_h = 2 * panel_h + footer_height
    total_w = 5 * panel_w

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:] = [15, 15, 18]

    for i, p in enumerate(panels):
        row = i // 5
        col = i % 5

        y0 = row * panel_h
        x0 = col * panel_w

        ph, pw = p.shape[:2]
        canvas[y0 + title_height:y0 + title_height + ph, x0:x0 + pw] = p

        if titles and i < len(titles):
            _draw_text_line(canvas, x0 + 4, y0 + 2, titles[i], max_w - 8)

    if footer_text:
        _draw_text_line(
            canvas, 4, total_h - footer_height + 4, footer_text, total_w - 8
        )

    return canvas


def _draw_text_line(
    img: np.ndarray, x: int, y: int, text: str, max_w: int
) -> None:
    """Simple text overlay using PIL if available, else skip."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            import matplotlib
            fp = (
                Path(matplotlib.__file__).parent
                / "mpl-data"
                / "fonts"
                / "ttf"
                / "DejaVuSans.ttf"
            )
            if fp.exists():
                font = ImageFont.truetype(str(fp), size=11)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        draw.text((x, y), text[:max_w // 7], fill=(200, 200, 200), font=font)
        img[:] = np.asarray(pil)
    except ImportError:
        pass
