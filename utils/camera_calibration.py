"""Calibrated projection onto the original sensor image grid.

Rasterize an enlarged undistorted pinhole canvas, then sample it onto the
original distorted sensor grid. The same differentiable warp is used during
training and inference. COLMAP pixel centers are offset by 0.5 relative to
integer image indices, matching the rasterizer's ndc2Pix convention.
"""

import math
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.graphics_utils import getProjectionMatrix


def unpack_calibration(model, parameters):
    p = np.asarray(parameters, dtype=np.float64)
    distortion = np.zeros(4, dtype=np.float64)
    if model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = p[:4]
        if model == "OPENCV":
            distortion = p[4:8].copy()
    elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx, cx, cy = p[:3]
        fy = fx
        if model != "SIMPLE_PINHOLE":
            distortion[0] = p[3]
        if model == "RADIAL":
            distortion[1] = p[4]
    else:
        raise ValueError(f"Calibrated rasterization does not support COLMAP model {model}")
    return np.array([fx, fy, cx, cy]), distortion


def calibration_grid(width, height, intrinsics, distortion):
    fx, fy, cx, cy = intrinsics
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    yy, xx = np.mgrid[:height, :width]
    distorted = np.stack((xx + .5, yy + .5), -1).astype(np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPointsIter(
        distorted, matrix, distortion, None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9),
    ).reshape(height, width, 2)
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite inverse lens-distortion map")
    extent = np.max(np.abs(normalized), axis=(0, 1))
    raster_width = max(width, int(math.ceil(2 * fx * extent[0] + 4)))
    raster_height = max(height, int(math.ceil(2 * fy * extent[1] + 4)))
    if raster_width > width * 4 or raster_height > height * 4:
        raise ValueError("Inverse camera distortion requires an implausibly large canvas")
    grid = normalized * np.array([2 * fx / raster_width, 2 * fy / raster_height])
    return grid.astype(np.float32), raster_width, raster_height


def attach_calibration(camera, camera_info, cache=None):
    intrinsics, distortion = unpack_calibration(camera_info.camera_model, camera_info.camera_params)
    scale = np.array([camera.image_width / camera_info.width, camera.image_height / camera_info.height])
    intrinsics = intrinsics * np.tile(scale, 2)
    if (not np.any(distortion) and abs(intrinsics[2] - camera.image_width / 2) < 1e-7
            and abs(intrinsics[3] - camera.image_height / 2) < 1e-7):
        return
    key = (camera.image_width, camera.image_height, *intrinsics, *distortion,
           str(camera.world_view_transform.device))
    cached = None if cache is None else cache.get(key)
    if cached is None:
        grid, width, height = calibration_grid(camera.image_width, camera.image_height, intrinsics, distortion)
        cached = (torch.as_tensor(grid, device=camera.world_view_transform.device), width, height)
        if cache is not None:
            cache[key] = cached
    camera.calibration_grid, width, height = cached
    camera.calibration_intrinsics = tuple(float(x) for x in intrinsics)
    camera.calibration_distortion = tuple(float(x) for x in distortion)
    fovx = 2 * math.atan(width / (2 * intrinsics[0]))
    fovy = 2 * math.atan(height / (2 * intrinsics[1]))
    projection = getProjectionMatrix(camera.znear, camera.zfar, fovx, fovy).T.to(camera.world_view_transform)
    camera.raster_camera = SimpleNamespace(
        image_width=width, image_height=height, FoVx=fovx, FoVy=fovy,
        world_view_transform=camera.world_view_transform,
        full_proj_transform=camera.world_view_transform @ projection,
        camera_center=camera.camera_center,
    )


def warp_raster_to_sensor(image, camera):
    grid = getattr(camera, "calibration_grid", None)
    if grid is None:
        return image
    return F.grid_sample(image.unsqueeze(0), grid.unsqueeze(0), mode="bilinear",
                         padding_mode="zeros", align_corners=False).squeeze(0)


def project_sensor_pixels(camera_points, intrinsics, distortion):
    """Brown-Conrady projection into zero-based sensor pixel indices."""
    fx, fy, cx, cy = intrinsics
    k1, k2, p1, p2 = distortion
    x, y = (camera_points[..., :2] / camera_points[..., 2:3].clamp_min(1e-6)).unbind(-1)
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return torch.stack((fx * xd + cx - .5, fy * yd + cy - .5), -1)
