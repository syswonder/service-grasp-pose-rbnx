#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Roboarm-style grasp-pose estimator.

This service owns ``robonix/service/perception/grasp_pose/*`` and exposes one
runtime RPC: ``grasp_request``. It does not publish legacy ROS topics and does
not host the old ``/graspnet/grasp_request`` service; downstream execution is
handled explicitly by ``pick_skill -> roboarm_ik``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from typing import Any

from robonix_api import Service, Ok, Err  # noqa: E402

logging.basicConfig(
    level=os.environ.get("GRASP_POSE_LOG_LEVEL", "INFO"),
    format="[grasp_pose] %(message)s",
)
log = logging.getLogger("grasp_pose")

grasp_pose = Service(
    id=os.environ.get("ROBONIX_CAPABILITY_ID", "grasp_pose"),
    namespace="robonix/service/perception/grasp_pose",
)

_DEFAULT_CATCH_OFFSET = 0.01
_DEFAULT_BOX_ROTATION_DEG = 0.0
_DEFAULT_BIAS_X = 0.0
_DEFAULT_BIAS_Y = 0.0
_DEFAULT_BASE_FRAME = "arm/base_link"

# Discrete detector-supplied orientation labels -> gripper yaw (rad).
#
# The convention (matches _gripper_angle_by_longer + _vertical_quaternion):
#   * yaw is the angle of the *gripper closing line* (i.e. the line between
#     the two finger tips) w.r.t. the image x-axis, in (-pi/2, pi/2].
#   * a good vertical grasp puts the closing line PERPENDICULAR to the
#     object's principal axis, so yaw = axis_angle + pi/2  (mod pi).
#
# Image y goes down, so:
#   horizontal (axis 0)     -> yaw = +pi/2
#   vertical   (axis pi/2)  -> yaw = 0
#   diag_tlbr  (axis +pi/4, "\") -> yaw = -pi/4   (folded from 3pi/4)
#   diag_trbl  (axis +3pi/4, "/") -> yaw = +pi/4
_ORIENTATION_YAW_RAD = {
    "horizontal": math.pi / 2.0,
    "vertical":   0.0,
    "diag_tlbr":  -math.pi / 4.0,
    "diag_trbl":   math.pi / 4.0,
}
# Aliases forgiving of upstream drift; anything not in this set folds to
# "unknown" and the bbox-long-edge fallback is used.
_ORIENTATION_ALIASES = {
    "up_down":   "vertical",   "updown":   "vertical",  "portrait":  "vertical",
    "left_right":"horizontal", "leftright":"horizontal","landscape": "horizontal",
    "tl_br":     "diag_tlbr",  "tlbr":     "diag_tlbr", "backslash": "diag_tlbr",
    "\\":        "diag_tlbr",
    "tr_bl":     "diag_trbl",  "trbl":     "diag_trbl", "slash":     "diag_trbl",
    "/":         "diag_trbl",
}


def _yaw_from_orientation(orientation: str) -> float | None:
    """Map a detector orientation label to a canonical gripper yaw (rad).

    Returns None when the label is "unknown", empty, or unrecognized —
    callers should then fall back to bbox-long-edge estimation.
    """
    if not isinstance(orientation, str):
        return None
    v = orientation.strip().lower().replace("-", "_").replace(" ", "_")
    v = _ORIENTATION_ALIASES.get(v, v)
    if v in _ORIENTATION_YAW_RAD:
        return float(_ORIENTATION_YAW_RAD[v])
    return None

_state_lock = threading.Lock()
_initialized = False
_resolved_cfg: dict[str, Any] = {}
_homography_matrix = None


def _vertical_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    """Quaternion for roboarm's vertical-down grasp convention."""
    half_yaw = float(yaw_rad) * 0.5
    return float(math.sin(half_yaw)), float(math.cos(half_yaw)), 0.0, 0.0


def _load_homography_matrix(cfg: dict[str, Any]):
    """Load the mandatory 3x3 pixel -> arm-plane homography."""
    import numpy as np

    inline = cfg.get("homography_matrix")
    if inline is not None:
        mat = np.asarray(inline, dtype=np.float64)
    else:
        raw_path = (
            cfg.get("hand_eye_calibration_file")
            or cfg.get("homography_file")
            or cfg.get("homography_path")
        )
        if not raw_path:
            raise ValueError(
                "missing required roboarm homography config: set "
                "hand_eye_calibration_file to a 3x3 .npy file or provide "
                "homography_matrix inline"
            )
        path = os.path.expandvars(os.path.expanduser(str(raw_path)))
        if not os.path.isabs(path):
            pkg_root = os.environ.get(
                "RBNX_PACKAGE_ROOT",
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            )
            path = os.path.join(pkg_root, path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"hand_eye_calibration_file does not exist: {path}"
            )
        mat = np.load(path)

    if mat.shape != (3, 3):
        raise ValueError(f"homography matrix must be shape (3, 3), got {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("homography matrix contains non-finite values")
    return mat.astype(np.float64)


def _pixel_to_arm_xy(u: float, v: float) -> tuple[float, float]:
    """Project an RGB pixel center to arm-plane XY."""
    import numpy as np

    if _homography_matrix is None:
        raise ValueError("homography matrix is not loaded")
    pixel_coords = np.array([[float(u)], [float(v)], [1.0]], dtype=np.float64)
    world_coords = _homography_matrix @ pixel_coords
    denom = float(world_coords[2, 0])
    if abs(denom) < 1e-12:
        raise ValueError("homography projection has near-zero scale")
    world_coords /= denom
    return float(world_coords[0, 0]), float(world_coords[1, 0])


def _gripper_angle_by_longer(
    u: float, v: float, w: float, h: float, angle_deg: float
) -> float:
    """roboarm Arm.gripper_angle_by_longer() without requiring cv2.

    ``angle_deg`` is the bbox's rotation w.r.t. the image x-axis, in
    degrees. It exists to support rotated / oriented bounding boxes
    (OBB) from a future detector (e.g. YOLO-OBB, ``cv2.minAreaRect``).

    NOTE (OBB not wired in yet): the current LLM detector
    (``service-object-detect-rbnx``) — and any plain axis-aligned
    detector like standard YOLO — only emits 4-element bboxes
    ``[x_min, y_min, x_max, y_max]``, so in practice ``angle_deg`` is
    always ``0.0`` and the rotation matrix below collapses to identity.
    The function then degenerates to a simple "wider ⇒ yaw=pi/2,
    taller ⇒ yaw=0" binary decision, which is exactly why we added the
    ``orientation`` label path in ``_compute_grasp``. Keep this code
    path so a future OBB detector can plug in without changes here.
    """
    import numpy as np

    theta = math.radians(float(angle_deg))
    c, s = math.cos(theta), math.sin(theta)
    half_w, half_h = float(w) / 2.0, float(h) / 2.0
    local = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float64,
    )
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    box_points = local @ rot.T + np.array([float(u), float(v)], dtype=np.float64)

    edge_01 = np.linalg.norm(box_points[0] - box_points[1])
    edge_12 = np.linalg.norm(box_points[1] - box_points[2])
    if edge_01 > edge_12:
        long_edge_points = (
            [box_points[0], box_points[1]]
            if box_points[0][0] < box_points[1][0]
            else [box_points[1], box_points[0]]
        )
    else:
        long_edge_points = (
            [box_points[1], box_points[2]]
            if box_points[1][0] < box_points[2][0]
            else [box_points[2], box_points[1]]
        )

    gripper_rot_rad = math.pi / 2 + math.atan2(
        float(long_edge_points[1][1] - long_edge_points[0][1]),
        float(long_edge_points[1][0] - long_edge_points[0][0]),
    )
    if gripper_rot_rad > math.pi / 2:
        gripper_rot_rad -= math.pi
    return float(gripper_rot_rad)


def _apply_xy_bias(x: float, y: float, cfg: dict[str, Any]) -> tuple[float, float]:
    bias_x = float(cfg.get("bias_x", _DEFAULT_BIAS_X))
    bias_y = float(cfg.get("bias_y", _DEFAULT_BIAS_Y))
    return x + bias_x, y + bias_y


def _failure(message: str, frame_id: str) -> dict[str, Any]:
    return {
        "success": False,
        "message": message,
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "frame_id": frame_id,
        "gripper_width": 0.0,
        "score": 0.0,
    }


def _compute_grasp(
    *,
    bbox_2d: list[float],
    cfg: dict[str, Any],
    object_name: str = "",
    orientation: str = "",
) -> dict[str, Any]:
    """Compute a vertical grasp pose from a detector bbox."""
    output_frame = str(cfg.get("output_frame", _DEFAULT_BASE_FRAME))
    if "default_desktop_height" not in cfg:
        return _failure(
            "missing required roboarm config: default_desktop_height",
            output_frame,
        )

    if not bbox_2d or len(bbox_2d) not in (4, 5):
        return _failure(f"bbox_2d must be length 4 or 5, got {bbox_2d!r}",
                        output_frame)

    desktop_height = float(cfg["default_desktop_height"])
    catch_offset = float(cfg.get("catch_offset", _DEFAULT_CATCH_OFFSET))
    width = float(cfg.get("gripper_width_default", 0.04))
    # OBB not wired in yet: no current detector (LLM or axis-aligned
    # YOLO) supplies a rotation. ``box_rotation_deg`` from config and
    # the 5th bbox element below are reserved for a future OBB detector
    # (e.g. YOLO-OBB, cv2.minAreaRect). In today's deployment both are
    # effectively ``0.0`` and ``_gripper_angle_by_longer`` operates on
    # an axis-aligned bbox.
    box_rotation_deg = float(
        cfg.get("box_rotation_deg", _DEFAULT_BOX_ROTATION_DEG)
    )

    x_min, y_min, x_max, y_max = (float(value) for value in bbox_2d[:4])
    if len(bbox_2d) == 5:
        # Reserved for future OBB detector output — never hit today.
        box_rotation_deg = float(bbox_2d[4])
    u = 0.5 * (x_min + x_max)
    v_pix = 0.5 * (y_min + y_max)
    bbox_w = abs(x_max - x_min)
    bbox_h = abs(y_max - y_min)
    if bbox_w <= 0.0 or bbox_h <= 0.0:
        return _failure(f"bbox has non-positive size: {bbox_2d!r}",
                        output_frame)

    try:
        raw_x, raw_y = _pixel_to_arm_xy(u, v_pix)
        target_x, target_y = _apply_xy_bias(raw_x, raw_y, cfg)
    except ValueError as e:
        return _failure(f"pixel2pos failed: {e}", output_frame)

    # Prefer an explicit detector-supplied orientation label when available;
    # fall back to the bbox-long-edge heuristic otherwise. Both branches
    # produce a yaw in the same convention (gripper closing line vs. image
    # x-axis, in (-pi/2, pi/2]).
    yaw_source: str
    yaw_from_label = _yaw_from_orientation(orientation)
    if yaw_from_label is not None:
        yaw_rad = yaw_from_label
        yaw_source = f"orientation={orientation!r}"
    else:
        yaw_rad = _gripper_angle_by_longer(
            u, v_pix, bbox_w, bbox_h, box_rotation_deg
        )
        yaw_source = (
            f"bbox_long_edge (orientation={orientation!r} not usable)"
            if orientation else "bbox_long_edge"
        )
    catch_dx = catch_offset * math.cos(yaw_rad)
    catch_dy = catch_offset * math.sin(-yaw_rad)
    grasp_x = target_x + catch_dx
    grasp_y = target_y + catch_dy
    qx, qy, qz, qw = _vertical_quaternion(yaw_rad)

    bias_x = float(cfg.get("bias_x", _DEFAULT_BIAS_X))
    bias_y = float(cfg.get("bias_y", _DEFAULT_BIAS_Y))
    log.info(
        "roboarm grasp: object=%r uv=(%.1f,%.1f) bbox=(%.1fx%.1f, rot=%.1f) "
        "raw_xy=(x=%.3f, y=%.3f) bias=(%.3f, %.3f) biased_xy=(x=%.3f, y=%.3f) "
        "catch_offset=(dx=%.3f, dy=%.3f) -> grasp=(x=%.3f, y=%.3f, z=%.3f) "
        "yaw=%.3f (via %s)",
        object_name, u, v_pix, bbox_w, bbox_h, box_rotation_deg,
        raw_x, raw_y, bias_x, bias_y, target_x, target_y,
        catch_dx, catch_dy, grasp_x, grasp_y, desktop_height, yaw_rad,
        yaw_source)

    return {
        "success": True,
        "message": (
            f"ok (object={object_name!r}, u,v=({u:.1f},{v_pix:.1f}), "
            f"arm_xy=({grasp_x:.3f},{grasp_y:.3f}), yaw={yaw_rad:.3f}, "
            f"yaw_via={yaw_source})"
        ),
        "pose": {
            "position": {
                "x": float(grasp_x),
                "y": float(grasp_y),
                "z": float(desktop_height),
            },
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        },
        "frame_id": output_frame,
        "gripper_width": width,
        "score": 0.8,
    }


def _serve_grasp_request(
    *,
    object_name: str,
    bbox_2d: list[float],
    orientation: str = "",
) -> dict[str, Any]:
    if not bbox_2d:
        return _failure(
            "bbox_2d is required; pick_skill should call object_detect first",
            str(_resolved_cfg.get("output_frame", _DEFAULT_BASE_FRAME)),
        )
    return _compute_grasp(
        bbox_2d=bbox_2d,
        cfg=_resolved_cfg,
        object_name=object_name,
        orientation=orientation,
    )


@grasp_pose.on_init
def init(cfg):
    """Driver(CMD_INIT): parse config and load the hand-eye homography."""
    global _initialized, _resolved_cfg, _homography_matrix
    with _state_lock:
        if _initialized:
            return Ok()

    cfg = cfg or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg) if cfg else {}
        except json.JSONDecodeError as e:
            return Err(f"bad config_json: {e}")

    if "default_desktop_height" not in cfg:
        return Err(
            "missing required roboarm config: default_desktop_height "
            "(meters, arm base frame z for the grasp height)"
        )
    try:
        float(cfg["default_desktop_height"])
    except Exception as e:  # noqa: BLE001
        return Err(f"bad default_desktop_height: {e}")

    try:
        _homography_matrix = _load_homography_matrix(cfg)
    except Exception as e:  # noqa: BLE001
        return Err(f"bad roboarm homography config: {e}")

    _resolved_cfg = cfg
    with _state_lock:
        _initialized = True
    log.info("init complete: gRPC grasp_request live (cfg keys=%d)",
             len(cfg))
    return Ok()


@grasp_pose.on_deactivate
def deactivate():
    global _initialized
    with _state_lock:
        _initialized = False
    log.info("CMD_DEACTIVATE ok")
    return Ok()


import grasp_pb2  # noqa: E402  pylint: disable=wrong-import-position
import geometry_msgs_pb2  # noqa: E402
import std_msgs_pb2  # noqa: E402
import builtin_interfaces_pb2  # noqa: E402


@grasp_pose.grpc("robonix/service/perception/grasp_pose/grasp_request")
def grasp_request(req: grasp_pb2.GraspRequest_Request) -> grasp_pb2.GraspRequest_Response:
    """Compute a grasp pose from a caller-supplied RGB bbox."""
    # `orientation` is a new optional string field on GraspRequest.srv.
    # Older callers that haven't been rebuilt against the updated proto
    # will simply lack the attribute; treat that as "" so we fall back
    # to the bbox-long-edge yaw estimation.
    orientation = getattr(req, "orientation", "") or ""
    result = _serve_grasp_request(
        object_name=req.object_name,
        bbox_2d=list(req.bbox_2d) if req.bbox_2d else [],
        orientation=str(orientation),
    )
    p = result["pose"]
    pose_stamped = geometry_msgs_pb2.PoseStamped(
        header=std_msgs_pb2.Header(
            stamp=builtin_interfaces_pb2.Time(sec=0, nanosec=0),
            frame_id=result["frame_id"],
        ),
        pose=geometry_msgs_pb2.Pose(
            position=geometry_msgs_pb2.Point(
                x=float(p["position"]["x"]),
                y=float(p["position"]["y"]),
                z=float(p["position"]["z"])),
            orientation=geometry_msgs_pb2.Quaternion(
                x=float(p["orientation"]["x"]),
                y=float(p["orientation"]["y"]),
                z=float(p["orientation"]["z"]),
                w=float(p["orientation"]["w"])),
        ),
    )
    return grasp_pb2.GraspRequest_Response(
        grasp_pose=pose_stamped,
        gripper_width=float(result["gripper_width"]),
        score=float(result["score"]),
        success=bool(result["success"]),
        message=str(result["message"]),
    )


def main() -> int:
    grasp_pose.run()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
