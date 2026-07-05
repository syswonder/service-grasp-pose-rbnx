#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""yolo_grasp_rbnx — geometric grasp-pose estimator service.

Owns ``robonix/service/perception/grasp_pose/*``. Pure CPU — no ML
model. Implements the geometric / heuristic logic from the live
upstream pipeline (``detect_grasp/yolo_grasp.py``):

  * given a YOLO 2D bounding box on the RGB frame, take the bbox
    center pixel, sample a median depth value from a grid inside
    the bbox, then back-project the (u, v, z) tuple through the
    camera intrinsics K into the camera optical frame.
  * write a pre-calibrated approach quaternion onto that 3D point
    (the orientation is constant — the live pipeline doesn't try
    to estimate orientation from geometry, it reuses a single
    standard top-down approach with a slight tilt).
  * subtract a small "safe height" from z so the gripper finishes
    ABOVE the object (the moveit cartesian descent finishes the
    last 10 cm).
  * gripper width is clamped to [min, max] from config (default
    [0.0, 0.12] m) — upstream literally hardcodes 0.12.

Three surfaces, sharing one ``_compute_grasp()`` math kernel:

  1. **Atlas-routed MCP RPC** (the new path, what Pilot's LLM sees)
       robonix/service/perception/grasp_pose/grasp_request
       — input/output: ``GraspRequest_Request`` / ``_Response``
         (codegen'd from capabilities/lib/grasp/srv/GraspRequest.srv)

  2. **Legacy ROS service** (compat path for pick.py + the C++
     moveit_control subscriber)
       /graspnet/grasp_request   (graspnet_msgs/srv/GraspRequest)
       + /graspnet/grasps topic  (graspnet_msgs/msg/GraspPose)

  3. **Auto-publish stream mode** (matches upstream yolo_grasp.py
     behaviour). When ``cfg.auto_publish_topic`` is true (DEFAULT
     OFF — see safety note below), subscribe to
     ``cfg.detect_objects_topic`` (default
     ``/yolo/detect_objects``, the legacy YOLOE publisher), and
     for every incoming ``DetectedObjects`` message:
       * pick the first detection whose ``object_name`` is in
         ``cfg.candidates`` (a configurable allowlist defaulting to
         the upstream list).
       * compute a grasp via ``_compute_grasp()`` and publish to
         ``/graspnet/grasps`` — fire-and-forget.
     This is what makes the legacy yolo_world → yolo_grasp →
     piper_moveit_control pipeline work without any caller code.

     **Safety note (default OFF)**: the downstream cpp
     ``moveit_control_node_yolo`` triggers a full grasp on the FIRST
     ``/graspnet/grasps`` it sees while idle, then locks itself
     ``is_busy_=true`` until ``/moveit_control/reset`` is called. If
     auto-publish is left ON, the moment the cpp node enters idle
     (e.g. right after a reset), the next 1Hz tick will start a NEW
     grasp without any caller asking for one. That is unsafe and
     surprising. So this mode is OFF by default; flip
     ``auto_publish_topic: true`` in the deploy manifest only when
     you really want the "any detected object → autograsp" demo
     behaviour and you understand who else is publishing /
     subscribing.

Lifecycle (per Robonix developer guide §5):
    on_init         — parse cfg, atlas-resolve detect_object endpoint
                      (informational), spawn rclpy thread (subs +
                      pubs + service host).
    on_deactivate   — stop rclpy thread; publish a zero-pose to
                      /graspnet/grasps so subscribers see "service
                      stopped" instead of stale data.

Notable algorithmic detail removed from upstream:
    upstream yolo_grasp.py blocks the ROS spin thread on
    ``input("Continue? [y/n]")`` every 10 detections. That is a
    debugging hook for an interactive shell; rbnx-spawned providers
    have stdin closed (no tty), so input() raises EOFError and the
    lifecycle thread crashes. We removed it. If you want
    rate-limiting, set cfg.auto_publish_min_interval_s > 0.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Any, Optional

from robonix_api import ATLAS, Service, Ok, Err  # noqa: E402

logging.basicConfig(
    level=os.environ.get("YOLO_GRASP_LOG_LEVEL", "INFO"),
    format="[yolo_grasp] %(message)s",
)
log = logging.getLogger("yolo_grasp")

yolo_grasp = Service(
    id=os.environ.get("ROBONIX_CAPABILITY_ID", "yolo_grasp"),
    namespace="robonix/service/perception/grasp_pose",
)

# ── default candidates allowlist (matches upstream yolo_grasp.py) ───────────
# When auto_publish_topic is on and a DetectedObjects message arrives,
# we pick the FIRST detection whose object_name is in this list. The
# list is configurable per-deploy via cfg.candidates.
_DEFAULT_CANDIDATES: list[str] = [
    "bookmark", "lamp", "test tube", "vacuum", "flyer", "movie ticket",
    "poster page", "paper", "paper bag", "payphone", "placard",
    "sheet music", "document", "monitor", "decorative picture",
]

# Pre-calibrated approach quaternion from upstream yolo_grasp.py.
# This is a single fixed top-down orientation; the live pipeline
# does NOT estimate orientation from geometry. Replace via cfg
# (orientation_xyzw) only if you've recalibrated.
_DEFAULT_QUAT = (-0.1329, 0.1508, -0.6840, -0.7013)

# ── vertical grasp mode constants ───────────────────────────────────────────
# In vertical mode, the grasp pose is computed by intersecting the camera
# ray through the bbox center with the table plane (z = z_table in
# base_link), then writing a fixed vertical-down quaternion. This replaces
# the depth-median back-projection used in the original algorithm.
#
# The pose is output in arm/base_link (not camera_color_optical_frame),
# so piper_moveit_rbnx's TF transform becomes a no-op.
_DEFAULT_Z_TABLE        = 0.02    # table height in base_link (m)
_DEFAULT_Z_OFFSET        = 0.0    # TCP offset below z_table (m)
_DEFAULT_APPROACH_DIST   = 0.10   # pre/post grasp hover height (m)
_DEFAULT_YAW_RAD         = 0.0    # default yaw (rad)
_DEFAULT_RADIAL_YAW      = False  # if True, override yaw with atan2(y,x)
_DEFAULT_RADIAL_YAW_OFFSET = 0.0  # extra yaw added on top of radial (rad)
_DEFAULT_BASE_FRAME      = "arm/base_link"
_DEFAULT_CAMERA_FRAME    = "camera_color_optical_frame"
_DEFAULT_COLOR_INFO_TOPIC = "/camera/color/camera_info"

# ── shared state ────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_initialized = False
_resolved_cfg: dict[str, Any] = {}

_ros_node = None
_ros_thread: Optional[threading.Thread] = None
_ros_stop_evt = threading.Event()
_grasps_pub = None       # /graspnet/grasps publisher

# Synchronization between init() and _ros_thread_main:
#   - _ros_ready_evt is set exactly once after the rclpy node has
#     successfully created all of its publishers / subscribers /
#     services, OR immediately after a setup-time exception is caught.
#   - _ros_thread_error holds the exception (if any) so init() can
#     propagate it to atlas as Err(...). Without this fail-fast path,
#     a thread crash inside create_publisher (e.g. typesupport .so
#     missing → "type_support is null") leaves the package looking
#     ACTIVE on atlas while none of the ROS surfaces are actually up.
_ros_ready_evt = threading.Event()
_ros_thread_error: Optional[BaseException] = None
_ROS_READY_TIMEOUT_S = 15.0

# Latest depth + camera_info (subscribers update these from the rclpy
# thread; readers — service handlers + auto-publish callback — read
# under _state_lock). Both are raw ROS messages, decoded lazily.
_latest_depth_msg = None         # sensor_msgs/Image
_latest_cam_info = None          # sensor_msgs/CameraInfo

# Vertical mode: latest T_base_cam (4×4 numpy array) cached by a periodic
# TF lookup in the rclpy thread. None until the first successful lookup.
_latest_T_base_cam = None       # numpy 4×4 or None
_latest_T_stamp_s: float = 0.0  # monotonic time of last successful TF lookup

# Latest /yolo/detect_objects message — cached unconditionally (i.e.
# even when auto_publish_topic is OFF) so that grasp_request handlers
# can resolve a bbox from object_name without making a blocking ROS
# service call to /yolo/detect_object.
#
# Why cache instead of calling /yolo/detect_object: the ROS service
# call from inside our own ROS service handler self-deadlocks. Both
# handlers run in the same single-threaded rclpy executor, so while
# our handler awaits the future, no spin can happen, and the response
# never gets routed through. Result: 5s timeout. Subscribing to the
# 1Hz broadcast that yolo_world emits anyway side-steps that entirely
# (a 1Hz cache is plenty fresh for anything a caller would do).
_latest_detected_objects = None  # graspnet_msgs/msg/DetectedObjects
_latest_detected_objects_stamp_s: float = 0.0

# auto-publish rate limit
_last_auto_publish_time = 0.0


# ── atlas: resolve upstream detect_object (informational) ───────────────────
def _resolve_detect_object_endpoint() -> Optional[str]:
    """Query atlas for object_detect's MCP endpoint. We don't use the
    MCP path yet (Stage 6 will); logging it is just a sanity check
    that yolo_world_rbnx is up before we proceed."""
    cid = "robonix/service/perception/object_detect/detect_object"
    try:
        caps = ATLAS.find_capability(contract_id=cid, transport="mcp")
    except Exception as e:  # noqa: BLE001
        log.warning("atlas query %s failed: %s", cid, e)
        return None
    if not caps:
        log.warning(
            "atlas has no provider for %s — yolo_world_rbnx not active "
            "or not yet declared its MCP endpoint",
            cid,
        )
        return None
    try:
        ch = yolo_grasp.connect_capability(caps[0], cid, "mcp")
        ep = ch.endpoint
        try:
            ch.close()
        except Exception:  # noqa: BLE001
            pass
        log.info("atlas resolved %s @ %s (provider=%s)",
                 cid, ep, caps[0].provider_id)
        return ep
    except Exception as e:  # noqa: BLE001
        log.warning("atlas connect %s failed: %s", cid, e)
        return None


# ── geometry: depth → metric helpers ────────────────────────────────────────
def _depth_to_numpy(depth_msg) -> Optional["object"]:
    """Convert a sensor_msgs/Image depth frame to numpy (meters,
    float32). Returns None on unsupported encoding."""
    import numpy as np
    from cv_bridge import CvBridge
    bridge = _depth_to_numpy._bridge  # type: ignore[attr-defined]
    try:
        arr = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    except Exception as e:  # noqa: BLE001
        log.error("cv_bridge conversion failed: %s", e)
        return None
    a = np.asarray(arr)
    if depth_msg.encoding in ("16UC1", "mono16"):
        return a.astype(np.float32) * 0.001  # mm → m
    if depth_msg.encoding == "32FC1":
        return a.astype(np.float32)
    log.error("unsupported depth encoding: %s", depth_msg.encoding)
    return None


def _depth_median_in_bbox(depth_m, bbox, *, grid: int,
                          min_depth: float, max_depth: float) -> Optional[float]:
    """Sample a `grid x grid` lattice inside bbox; return median of
    depths within [min, max] meters. Returns None if no valid sample."""
    import numpy as np
    h, w = depth_m.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    x0 = int(np.clip(math.floor(x_min), 0, w - 1))
    x1 = int(np.clip(math.ceil(x_max),  0, w - 1))
    y0 = int(np.clip(math.floor(y_min), 0, h - 1))
    y1 = int(np.clip(math.ceil(y_max),  0, h - 1))
    g = max(3, int(grid))
    xs = np.linspace(x0, x1, g).astype(int)
    ys = np.linspace(y0, y1, g).astype(int)
    samples = []
    for yy in ys:
        for xx in xs:
            z = float(depth_m[yy, xx])
            if np.isfinite(z) and min_depth <= z <= max_depth and z > 0.0:
                samples.append(z)
    if not samples:
        return None
    return float(np.median(np.asarray(samples, dtype=np.float32)))


def _parse_intrinsics(cam_info) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) from a sensor_msgs/CameraInfo."""
    K = cam_info.k  # row-major: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    return float(K[0]), float(K[4]), float(K[2]), float(K[5])


def _pixel_to_table_xy(
    u: float, v: float,
    fx: float, fy: float, cx: float, cy: float,
    T_base_cam, z_table: float,
) -> tuple[float, float]:
    """Ray-plane intersection: cast a ray from the camera through pixel
    (u, v), transform it into base_link, and intersect with the table
    plane z = z_table.

    Returns (x, y) in base_link frame. Raises ValueError on degenerate
    cases (ray parallel to table, intersection behind camera).
    """
    import numpy as np

    # 1) pixel → camera optical frame unit direction
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    d_cam /= np.linalg.norm(d_cam)

    # 2) rotate ray into base_link; origin = camera position in base_link
    R_bc = T_base_cam[:3, :3]
    t_bc = T_base_cam[:3, 3]
    d_base = R_bc @ d_cam
    o_base = t_bc

    # 3) intersect with plane z = z_table
    if abs(d_base[2]) < 1e-6:
        raise ValueError("camera ray is parallel to table plane")
    t_param = (z_table - o_base[2]) / d_base[2]
    if t_param <= 0:
        raise ValueError("intersection is behind camera")
    p_base = o_base + t_param * d_base
    return float(p_base[0]), float(p_base[1])


def _vertical_quaternion(yaw_rad: float):
    """Quaternion for a vertical-down grasp (end-effector z-axis
    pointing at world -z), with an optional yaw rotation about the
    world z-axis.

    Returns (qx, qy, qz, qw).
    """
    import numpy as np
    # Euler roll=pi, pitch=0, yaw=yaw_rad in 'sxyz' convention.
    # roll=pi flips the end-effector so it points down.
    cr = np.cos(np.pi / 2)
    sr = np.sin(np.pi / 2)
    cp = 1.0  # cos(0)
    sp = 0.0  # sin(0)
    cy = np.cos(yaw_rad / 2)
    sy = np.sin(yaw_rad / 2)
    # quaternion_from_euler(pi, 0, yaw, 'sxyz') expanded:
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return float(qx), float(qy), float(qz), float(qw)


def _compute_grasp(
    *,
    bbox_2d: list[float],
    depth_msg,
    cam_info,
    cfg: dict[str, Any],
    object_name: str = "",
) -> dict:
    """Geometric grasp computation — vertical-grasp version.

    Instead of depth-median back-projection, this casts a ray from the
    camera through the bbox center pixel, transforms it into base_link
    via the hand-eye-calibrated TF, and intersects it with the table
    plane (z = z_table). The pose is output in arm/base_link with a
    fixed vertical-down quaternion.

    depth_msg is accepted but **ignored** in vertical mode — kept in
    the signature for backward compatibility with callers that still
    pass it.

    Returns the same dict shape regardless of success/failure:
        {
          "success":       bool,
          "message":       str,
          "pose":          {position: {x, y, z},
                            orientation: {x, y, z, w}},
          "frame_id":      str,
          "gripper_width": float,
          "score":         float,
        }
    """
    import numpy as np

    output_frame = cfg.get("output_frame", _DEFAULT_BASE_FRAME)
    z_table      = float(cfg.get("z_table",       _DEFAULT_Z_TABLE))
    z_offset     = float(cfg.get("z_offset",      _DEFAULT_Z_OFFSET))
    yaw_rad      = float(cfg.get("default_yaw_rad", _DEFAULT_YAW_RAD))
    width        = float(cfg.get("gripper_width_default", 0.04))

    fail = lambda msg: {  # noqa: E731
        "success": False, "message": msg,
        "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        "frame_id": output_frame,
        "gripper_width": 0.0, "score": 0.0,
    }

    if cam_info is None:
        return fail("no camera_info received yet")
    if not bbox_2d or len(bbox_2d) != 4:
        return fail(f"bbox_2d must be length 4, got {bbox_2d!r}")

    # ── read cached TF (base_link ← camera_color_optical_frame) ──
    with _state_lock:
        T_base_cam = _latest_T_base_cam
        T_age_s    = time.monotonic() - _latest_T_stamp_s if _latest_T_stamp_s else 999.0

    if T_base_cam is None:
        return fail(
            "no TF transform cached yet (arm/base_link ← "
            "camera_color_optical_frame). Is easy_handeye2_rbnx "
            "publishing the hand-eye TF?")
    if T_age_s > 5.0:
        log.warning("TF cache is %.1fs old — hand-eye TF may be stale", T_age_s)

    # ── bbox center pixel ──
    x_min, y_min, x_max, y_max = (float(v) for v in bbox_2d)
    u = 0.5 * (x_min + x_max)
    v_pix = 0.5 * (y_min + y_max)

    # ── camera intrinsics ──
    fx, fy, cx, cy = _parse_intrinsics(cam_info)
    if fx <= 0.0 or fy <= 0.0:
        return fail(f"invalid intrinsics fx={fx} fy={fy}")

    # ── ray-plane intersection ──
    try:
        x_base, y_base = _pixel_to_table_xy(
            u, v_pix, fx, fy, cx, cy, T_base_cam, z_table)
    except ValueError as e:
        return fail(f"ray-plane intersection failed: {e}")

    # ── assemble pose ──
    grasp_z = z_table + z_offset  # TCP height (z_offset < 0 → dip below table)

    # Plan-A wrist-flip mitigation: when `radial_yaw` is enabled, override
    # the fixed default yaw with atan2(y_base, x_base) (+ optional offset).
    # Aligning the gripper opening with the radial direction from the arm
    # base to the target sharply narrows joint6's valid range, so MoveIt's
    # IK is far less likely to pick the "wrist-flipped" branch (j6 ± π).
    # `radial_yaw_offset_rad` lets you rotate the opening by e.g. ±π/2
    # if you want the fingers to close *across* the radius instead of
    # *along* it.
    if bool(cfg.get("radial_yaw", _DEFAULT_RADIAL_YAW)):
        offset = float(cfg.get("radial_yaw_offset_rad",
                                _DEFAULT_RADIAL_YAW_OFFSET))
        yaw_rad = float(np.arctan2(y_base, x_base)) + offset

    qx, qy, qz, qw = _vertical_quaternion(yaw_rad)

    # Score: crude quality proxy based on bbox area (same as before).
    bbox_area = max(0.0, (x_max - x_min)) * max(0.0, (y_max - y_min))
    # We don't have image dims here without depth_msg, so just use
    # a flat 0.8 default — the score is advisory only.
    score = 0.8

    log.info(
        "vertical grasp: object=%r uv=(%.1f,%.1f) → base_link "
        "(x=%.3f, y=%.3f, z=%.3f) yaw=%.3f",
        object_name, u, v_pix, x_base, y_base, grasp_z, yaw_rad)

    return {
        "success":       True,
        "message":       f"ok (object={object_name!r}, "
                         f"u,v=({u:.1f},{v_pix:.1f}), "
                         f"base_xy=({x_base:.3f},{y_base:.3f}))",
        "pose": {
            "position":    {"x": float(x_base), "y": float(y_base),
                            "z": float(grasp_z)},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        },
        "frame_id":      output_frame,
        "gripper_width": float(width),
        "score":         float(score),
    }


# ── ROS bring-up (background thread) ────────────────────────────────────────
def _ros_thread_main() -> None:
    """Body of the rclpy background thread.

    Setup phase wrapped in try/except: any failure (typesupport .so
    not loadable, intra-thread import error, rclpy.init failure,
    etc.) is captured into _ros_thread_error so init() can return
    Err(...) instead of falsely reporting Ok and leaving us "ACTIVE
    but mute" on atlas. _ros_ready_evt is set in BOTH the success
    and failure paths so init()'s wait() always returns within
    _ROS_READY_TIMEOUT_S (or sooner).
    """
    global _ros_node, _grasps_pub
    global _latest_depth_msg, _latest_cam_info
    global _latest_T_base_cam, _latest_T_stamp_s
    global _ros_thread_error

    node = None
    try:
        import rclpy                                              # noqa: E402
        from rclpy.node import Node                               # noqa: E402
        from sensor_msgs.msg import Image, CameraInfo             # noqa: E402
        from graspnet_msgs.srv import GraspRequest                # noqa: E402
        from graspnet_msgs.msg import GraspPose as RosGraspPose   # noqa: E402
        from graspnet_msgs.msg import DetectedObjects             # noqa: E402
        from cv_bridge import CvBridge                            # noqa: E402
        import tf2_ros                                            # noqa: E402
        import numpy as np                                        # noqa: E402

        # cv_bridge instance attached to the helper for cheap reuse
        _depth_to_numpy._bridge = CvBridge()  # type: ignore[attr-defined]

        rclpy.init(args=None)
        node = Node("yolo_grasp_node")
        _ros_node = node

        cfg = _resolved_cfg

        depth_topic    = str(cfg.get("depth_topic",          "/camera/depth/image_raw"))
        cam_info_topic = str(cfg.get("camera_info_topic",    _DEFAULT_COLOR_INFO_TOPIC))
        det_topic      = str(cfg.get("detect_objects_topic", "/yolo/detect_objects"))
        grasps_topic   = str(cfg.get("grasps_topic",         "/graspnet/grasps"))

        base_frame   = str(cfg.get("base_frame",   _DEFAULT_BASE_FRAME))
        camera_frame = str(cfg.get("camera_frame", _DEFAULT_CAMERA_FRAME))

        # Subscribers — feed the latest depth / camera_info into shared
        # state. We deliberately keep only the most recent message; older
        # ones are dropped because grasp_request is always "use what's
        # current right now". qos depth=10 mirrors upstream.
        #
        # NOTE: In vertical mode, depth is NOT consumed by _compute_grasp
        # (ray-plane intersection replaces depth back-projection). The
        # subscription is kept for backward compatibility / auto-publish
        # stream mode which still needs it.
        def _on_depth(msg):
            global _latest_depth_msg
            with _state_lock:
                _latest_depth_msg = msg

        def _on_cam_info(msg):
            global _latest_cam_info
            with _state_lock:
                _latest_cam_info = msg

        node.create_subscription(Image,      depth_topic,    _on_depth,    10)
        node.create_subscription(CameraInfo, cam_info_topic, _on_cam_info, 10)
        log.info("subscribing depth=%s camera_info=%s", depth_topic, cam_info_topic)

        # ── TF: cache T_base_cam periodically for the vertical grasp
        # ray-plane intersection. The hand-eye TF (base_link ←
        # camera_color_optical_frame) is published by easy_handeye2_rbnx.
        # We poll it every 0.5s in the spin loop below; _compute_grasp
        # reads the cached 4×4 matrix from shared state. ──
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer, node)
        log.info("TF listener ready: %s ← %s", base_frame, camera_frame)

        def _poll_tf():
            """Look up base_link ← camera TF and cache the 4×4 matrix."""
            global _latest_T_base_cam, _latest_T_stamp_s
            try:
                tfm = tf_buffer.lookup_transform(
                    base_frame, camera_frame,
                    rclpy.time.Time(),  # latest available
                    timeout=rclpy.duration.Duration(seconds=0.05))
            except Exception:  # noqa: BLE001
                return  # transform not yet available — will retry next tick
            t = tfm.transform.translation
            r = tfm.transform.rotation
            # quaternion → rotation matrix
            qx, qy, qz, qw = r.x, r.y, r.z, r.w
            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx*qx + qy*qy)],
            ])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = [t.x, t.y, t.z]
            with _state_lock:
                _latest_T_base_cam = T
                _latest_T_stamp_s  = time.monotonic()

        # Publisher — C++ piper_moveit_control + every other downstream.
        _grasps_pub = node.create_publisher(RosGraspPose, grasps_topic, 10)
        log.info("publishing grasps=%s", grasps_topic)

        # /yolo/detect_objects subscription — ALWAYS ON. Two reasons:
        #
        #   (a) caller-driven path: when /graspnet/grasp_request or
        #       MCP grasp_request comes in WITHOUT a bbox, we used to
        #       call yolo_world's /yolo/detect_object ROS service to
        #       get one. That self-deadlocks: the rclpy executor is
        #       single-threaded, so awaiting the future blocks the
        #       same thread that would route the response, and the
        #       call always times out at 5s. Subscribing to the 1Hz
        #       broadcast yolo_world emits anyway gives us the same
        #       data without any RPC.
        #
        #   (b) optional auto_publish path: same data feeds the
        #       lambda below when auto_publish_topic=true.
        #
        # cb fan-out: the subscription cb writes to the cache
        # unconditionally, then conditionally calls the
        # auto-publish handler.
        candidates   = list(cfg.get("candidates") or _DEFAULT_CANDIDATES)
        auto_publish = bool(cfg.get("auto_publish_topic", False))

        def _on_detect_objects_msg(msg):
            global _latest_detected_objects, _latest_detected_objects_stamp_s
            with _state_lock:
                _latest_detected_objects = msg
                _latest_detected_objects_stamp_s = time.monotonic()
            if auto_publish:
                _on_detected_objects(msg, candidates)

        node.create_subscription(
            DetectedObjects, det_topic, _on_detect_objects_msg, 10)
        log.info("subscribing detect_objects=%s (cache always on; "
                 "auto_publish=%s)", det_topic, auto_publish)

        # /graspnet/grasp_request service host (legacy compat).
        #
        # also_publish=True: a ROS-service grasp_request is the caller
        # explicitly saying "compute AND have the executor act on it"
        # (same as the MCP path). Filling the response gives the caller
        # the pose for their own bookkeeping; publishing the same pose
        # to /graspnet/grasps wakes the cpp moveit_control_node_yolo
        # subscriber. With auto_publish_topic OFF by default, this is
        # the ONLY path that triggers the cpp node, which is exactly
        # what we want — caller-driven, one-shot.
        def _grasp_request_cb(req, resp):
            result = _serve_grasp_request(
                object_name      = req.object_name,
                bbox_2d          = list(req.bbox_2d) if req.bbox_2d else [],
                object_center_3d = list(req.object_center_3d) if req.object_center_3d else [],
                retry            = int(req.retry),
            )
            _pack_response_and_publish(resp, result, also_publish=True)
            return resp
        node.create_service(GraspRequest, "/graspnet/grasp_request", _grasp_request_cb)
        log.info("ROS service up: /graspnet/grasp_request")

        # Auto-publish bookkeeping log only — the actual subscription
        # is the same one we created above.
        # DEFAULT OFF — see file header "Safety note" for why.
        if auto_publish:
            log.warning(
                "auto_publish_topic ON — yolo_grasp will publish a fresh "
                "grasp pose to /graspnet/grasps for every detected object "
                "matching candidates (%d). The cpp moveit_control_node_yolo "
                "will trigger a real arm motion on the FIRST such message "
                "while it is idle. Only safe if you know what you're doing.",
                len(candidates))
        else:
            log.info("auto_publish_topic OFF — service / MCP mode only "
                     "(safe default; flip cfg.auto_publish_topic=true to "
                     "stream grasps from detect_objects)")
    except BaseException as e:  # noqa: BLE001 — must include SystemExit/KeyboardInterrupt etc.
        # Setup-time failure. Most common cause in practice: graspnet_msgs
        # typesupport .so files not on LD_LIBRARY_PATH, so create_publisher
        # raises "type_support is null". Capture and signal init().
        _ros_thread_error = e
        log.error("rclpy thread setup failed: %s: %s",
                  type(e).__name__, e, exc_info=True)
        # Best-effort cleanup so re-init has a clean slate.
        try:
            if node is not None:
                node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            import rclpy as _rclpy_for_shutdown
            if _rclpy_for_shutdown.ok():
                _rclpy_for_shutdown.shutdown()
        except Exception:  # noqa: BLE001
            pass
        _ros_ready_evt.set()
        return

    # Setup OK — let init() proceed.
    _ros_ready_evt.set()

    # Spin until shutdown.
    import rclpy  # noqa: E402  (re-import for the spin loop scope)
    _tf_poll_timer = 0.0
    while not _ros_stop_evt.is_set():
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
            # Poll TF every ~0.5s (10 spin iterations at 0.1s each).
            _tf_poll_timer += 0.1
            if _tf_poll_timer >= 0.5:
                _tf_poll_timer = 0.0
                _poll_tf()
        except Exception as e:  # noqa: BLE001
            # Per-iteration errors should NOT bring the whole thread down
            # — that would silently re-introduce the "alive but mute"
            # failure mode. Log and continue.
            log.warning("rclpy.spin_once raised: %s", e)

    # Graceful shutdown: emit a zero-pose so subscribers know we're
    # gone (upstream does this; piper_moveit_control treats zero as
    # "no grasp pending"). Wrap in try since we may already be in
    # the middle of rclpy teardown.
    try:
        if _grasps_pub is not None:
            from graspnet_msgs.msg import GraspPose as RosGraspPose  # noqa: E402
            zero = RosGraspPose()
            _fill_zero_grasp(zero, frame_id=str(_resolved_cfg.get("output_frame",
                                                       "camera_color_optical_frame")))
            _grasps_pub.publish(zero)
            log.info("published shutdown zero-pose to %s",
                     _resolved_cfg.get("grasps_topic", "/graspnet/grasps"))
    except Exception as e:  # noqa: BLE001
        log.warning("shutdown zero-pose publish failed: %s", e)

    try:
        node.destroy_node()
    except Exception:  # noqa: BLE001
        pass
    try:
        rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass
    log.info("rclpy thread exited")


def _fill_zero_grasp(gp, *, frame_id: str) -> None:
    """Zero-out a GraspPose msg (upstream sentinel)."""
    from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion  # noqa: E402
    if _ros_node is not None:
        gp.target_pose.header.stamp = _ros_node.get_clock().now().to_msg()
    gp.target_pose.header.frame_id = frame_id
    gp.target_pose.pose = Pose(
        position=Point(x=0.0, y=0.0, z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    gp.gripper_width = 0.0


def _on_detected_objects(msg, candidates: list[str]) -> None:
    """Auto-publish handler. Picks the first detection whose
    object_name matches the candidates allowlist, computes a grasp
    using the latest depth + camera_info, and publishes."""
    global _last_auto_publish_time

    if not msg.objects:
        return

    # Optional rate limit (upstream blocked on input(); we just
    # silently drop. Default 0.0 → no limit, every msg triggers.)
    min_interval = float(_resolved_cfg.get("auto_publish_min_interval_s", 0.0))
    if min_interval > 0.0:
        now = time.monotonic()
        if (now - _last_auto_publish_time) < min_interval:
            return

    best = next((o for o in msg.objects if o.object_name in candidates), None)
    if best is None:
        # Log once per "burst" — too noisy otherwise on a busy feed.
        if log.isEnabledFor(logging.DEBUG):
            seen = ",".join(o.object_name for o in msg.objects[:5])
            log.debug("no candidate match in detections [%s]", seen)
        return

    bbox = list(best.bbox_2d)
    with _state_lock:
        depth = _latest_depth_msg
        info  = _latest_cam_info
    result = _compute_grasp(
        bbox_2d=bbox, depth_msg=depth, cam_info=info,
        cfg=_resolved_cfg, object_name=best.object_name,
    )
    if not result["success"]:
        log.warning("auto_publish: %s", result["message"])
        return

    _publish_to_grasps_topic(result)
    _last_auto_publish_time = time.monotonic()
    log.info(
        "auto_publish %s conf=%.3f -> XYZ=[%.3f, %.3f, %.3f] width=%.3fm",
        best.object_name, float(getattr(best, "confidence", 0.0)),
        result["pose"]["position"]["x"], result["pose"]["position"]["y"],
        result["pose"]["position"]["z"], result["gripper_width"],
    )


def _pack_response_and_publish(resp, result: dict, *, also_publish: bool) -> None:
    """Stuff a graspnet_msgs/srv/GraspRequest response from a
    _compute_grasp result, optionally also publishing to /graspnet/grasps."""
    from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion  # noqa: E402
    p = result["pose"]
    ps = PoseStamped()
    if _ros_node is not None:
        ps.header.stamp = _ros_node.get_clock().now().to_msg()
    ps.header.frame_id = result["frame_id"]
    ps.pose = Pose(
        position=Point(
            x=float(p["position"]["x"]),
            y=float(p["position"]["y"]),
            z=float(p["position"]["z"])),
        orientation=Quaternion(
            x=float(p["orientation"]["x"]),
            y=float(p["orientation"]["y"]),
            z=float(p["orientation"]["z"]),
            w=float(p["orientation"]["w"])),
    )
    resp.grasp_pose    = ps
    resp.gripper_width = float(result["gripper_width"])
    resp.score         = float(result["score"])
    resp.success       = bool(result["success"])
    resp.message       = str(result["message"])
    if also_publish and result["success"]:
        _publish_to_grasps_topic(result)


def _publish_to_grasps_topic(result: dict) -> None:
    """Fire-and-forget publish to /graspnet/grasps (legacy topic)."""
    if _grasps_pub is None:
        return
    try:
        from graspnet_msgs.msg import GraspPose as RosGraspPose  # noqa: E402
        from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion  # noqa: E402
        p = result["pose"]
        ps = PoseStamped()
        if _ros_node is not None:
            ps.header.stamp = _ros_node.get_clock().now().to_msg()
        ps.header.frame_id = result["frame_id"]
        ps.pose = Pose(
            position=Point(
                x=float(p["position"]["x"]),
                y=float(p["position"]["y"]),
                z=float(p["position"]["z"])),
            orientation=Quaternion(
                x=float(p["orientation"]["x"]),
                y=float(p["orientation"]["y"]),
                z=float(p["orientation"]["z"]),
                w=float(p["orientation"]["w"])),
        )
        gp = RosGraspPose()
        gp.target_pose = ps
        gp.gripper_width = float(result["gripper_width"])
        _grasps_pub.publish(gp)
    except Exception as e:  # noqa: BLE001
        log.warning("/graspnet/grasps publish failed: %s", e)


def _call_detect_object(object_name: str, *, max_age_s: float = 3.0) -> dict:
    """Resolve a bbox for ``object_name`` from the cached
    ``/yolo/detect_objects`` broadcast.

    Why not call yolo_world's /yolo/detect_object ROS service directly:
    that self-deadlocks when invoked from inside our own ROS service
    handler. Both run in the same single-threaded rclpy executor on
    this node, so awaiting the future blocks the very thread that
    would route the response. Result: 5s timeout every time.

    The cache is fed unconditionally by the /yolo/detect_objects
    subscription created in _ros_thread_main, regardless of whether
    auto_publish_topic is enabled. yolo_world publishes that topic at
    1Hz in its periodic broadcast (see yolo_world.main._periodic_broadcast),
    which is more than fresh enough for any caller that's about to ask
    for a grasp.

    Match rule mirrors yolo_world._detect_object: case-insensitive
    substring on object_name vs detected name (either direction). We
    return the highest-confidence match.
    """
    with _state_lock:
        msg = _latest_detected_objects
        stamp_s = _latest_detected_objects_stamp_s

    if msg is None:
        return {"success": False,
                "message": "no /yolo/detect_objects message received yet "
                           "(is yolo_world up and publishing?)",
                "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}

    age_s = time.monotonic() - stamp_s
    if age_s > max_age_s:
        return {"success": False,
                "message": f"/yolo/detect_objects cache stale ({age_s:.1f}s old)",
                "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}

    if not msg.objects:
        return {"success": False,
                "message": "/yolo/detect_objects has 0 detections in latest frame",
                "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}

    name_lower = object_name.lower().strip()
    best = None  # (confidence, det_obj)
    for det in msg.objects:
        det_name = (det.object_name or "").lower().strip()
        if not det_name:
            continue
        if (name_lower == det_name or name_lower in det_name
                or det_name in name_lower):
            conf = float(getattr(det, "confidence", 0.0))
            if best is None or conf > best[0]:
                best = (conf, det)

    if best is None:
        seen = ", ".join(o.object_name for o in msg.objects[:8])
        return {"success": False,
                "message": (f"object {object_name!r} not in latest "
                            f"/yolo/detect_objects (saw: {seen}…)"),
                "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}

    conf, det = best
    return {
        "success":          True,
        "message":          (f"resolved from /yolo/detect_objects cache "
                             f"(age={age_s:.2f}s, conf={conf:.3f})"),
        "bbox_2d":          list(det.bbox_2d),
        "object_center_3d": list(det.object_center_3d) if det.object_center_3d else [],
        "confidence":       conf,
    }


def _serve_grasp_request(*, object_name, bbox_2d, object_center_3d, retry):
    """Shared handler for both RPC surfaces (MCP + ROS service).

    If bbox_2d is missing, we ask yolo_world's detect_object service
    first. ``object_center_3d`` from the caller is currently advisory
    only — the geometric estimator always re-back-projects from depth
    so that "depth at the time of grasp" matches the published TF.
    """
    if not bbox_2d:
        det = _call_detect_object(object_name)
        if not det["success"]:
            return {
                "success": False,
                "message": f"detect_object pre-call failed: {det['message']}",
                "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0},
                         "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
                "frame_id": _resolved_cfg.get("output_frame",
                                              "camera_color_optical_frame"),
                "gripper_width": 0.0,
                "score": 0.0,
            }
        bbox_2d = det["bbox_2d"]

    with _state_lock:
        depth = _latest_depth_msg
        info  = _latest_cam_info

    return _compute_grasp(
        bbox_2d=bbox_2d, depth_msg=depth, cam_info=info,
        cfg=_resolved_cfg, object_name=object_name,
    )


# ── lifecycle ───────────────────────────────────────────────────────────────
@yolo_grasp.on_init
def init(cfg):
    """Driver(CMD_INIT). Light-medium:
      1. parse cfg
      2. atlas-resolve detect_object endpoint (informational)
      3. spawn rclpy thread
    """
    global _initialized, _resolved_cfg
    with _state_lock:
        if _initialized:
            return Ok()

    cfg = cfg or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg) if cfg else {}
        except json.JSONDecodeError as e:
            return Err(f"bad config_json: {e}")
    _resolved_cfg = cfg

    log.info("cfg: %d keys (auto_publish=%s, candidates=%d)",
             len(cfg), cfg.get("auto_publish_topic", False),
             len(cfg.get("candidates") or _DEFAULT_CANDIDATES))

    # Informational: log which provider owns detect_object on atlas.
    _resolve_detect_object_endpoint()

    # Spawn rclpy thread. We then BLOCK on _ros_ready_evt — the thread
    # signals it after either successful setup of every publisher /
    # subscriber / service, OR a setup-time exception (captured into
    # _ros_thread_error). This makes init() fail-fast and propagate
    # the error to atlas via Err(...), instead of returning Ok and
    # leaving us "ACTIVE but mute" (the typesupport.so / LD_LIBRARY_PATH
    # failure mode that was previously silent).
    global _ros_thread, _ros_thread_error
    _ros_stop_evt.clear()
    _ros_ready_evt.clear()
    _ros_thread_error = None
    _ros_thread = threading.Thread(
        target=_ros_thread_main,
        name="yolo_grasp-ros",
        daemon=True,
    )
    _ros_thread.start()

    if not _ros_ready_evt.wait(timeout=_ROS_READY_TIMEOUT_S):
        # Thread is hung somewhere in setup — try to bring it down so
        # a retry has a clean slate, then surface a clear error.
        _ros_stop_evt.set()
        _ros_thread.join(timeout=2.0)
        return Err(
            f"rclpy thread did not become ready within "
            f"{_ROS_READY_TIMEOUT_S}s (likely blocked in rclpy.init or "
            f"create_publisher; check `ps -T` and the log just above)"
        )

    if _ros_thread_error is not None:
        # Setup-time exception. Most common: typesupport .so missing.
        err = _ros_thread_error
        _ros_stop_evt.set()
        _ros_thread.join(timeout=2.0)
        return Err(
            f"rclpy thread setup failed: {type(err).__name__}: {err} — "
            f"if message mentions 'libgraspnet_msgs__rosidl_typesupport_*.so', "
            f"the vendored graspnet_msgs lib/ is not on LD_LIBRARY_PATH "
            f"(check scripts/start.sh's graspnet_msgs path injection)"
        )

    with _state_lock:
        _initialized = True
    log.info("init complete: grasp_request MCP + /graspnet/grasp_request live")
    return Ok()


@yolo_grasp.on_deactivate
def deactivate():
    log.info("CMD_DEACTIVATE: stopping rclpy thread")
    _ros_stop_evt.set()
    if _ros_thread is not None:
        _ros_thread.join(timeout=5.0)
    with _state_lock:
        global _initialized
        _initialized = False
    return Ok()


# ── atlas-routed MCP handler (Pilot's view) ─────────────────────────────────
# Import top-level Request/Response from the package-local IDL, plus
# the nested geometry_msgs / std_msgs / builtin_interfaces types we
# need to instantiate. The `_mcp` suffix is codegen's convention:
# `{ros_package}_mcp.py` per ROS package.
from grasp_mcp import (  # noqa: E402  pylint: disable=wrong-import-position
    GraspRequest_Request, GraspRequest_Response,
)
from geometry_msgs_mcp import (  # noqa: E402
    PoseStamped, Pose, Point, Quaternion,
)
from std_msgs_mcp import Header  # noqa: E402
from builtin_interfaces_mcp import Time  # noqa: E402


@yolo_grasp.mcp("robonix/service/perception/grasp_pose/grasp_request")
def grasp_request(req: GraspRequest_Request) -> GraspRequest_Response:
    """Compute a grasp pose for ``req.object_name``.

    If the caller supplies ``bbox_2d`` (length 4 = [x_min, y_min,
    x_max, y_max] in pixels on the RGB frame), we use it directly.
    Otherwise we ask yolo_world's /yolo/detect_object service for one
    on the fly. ``object_center_3d`` from the caller is advisory only.
    """
    result = _serve_grasp_request(
        object_name      = req.object_name,
        bbox_2d          = list(req.bbox_2d) if req.bbox_2d else [],
        object_center_3d = list(req.object_center_3d) if req.object_center_3d else [],
        retry            = int(req.retry),
    )
    p = result["pose"]
    pose_stamped = PoseStamped(
        header=Header(
            stamp=Time(sec=0, nanosec=0),
            frame_id=result["frame_id"],
        ),
        pose=Pose(
            position=Point(
                x=float(p["position"]["x"]),
                y=float(p["position"]["y"]),
                z=float(p["position"]["z"])),
            orientation=Quaternion(
                x=float(p["orientation"]["x"]),
                y=float(p["orientation"]["y"]),
                z=float(p["orientation"]["z"]),
                w=float(p["orientation"]["w"])),
        ),
    )
    # Also publish to legacy topic so the C++ piper_moveit_control
    # subscriber kicks in even if the caller is on the MCP path.
    if result["success"]:
        _publish_to_grasps_topic(result)

    return GraspRequest_Response(
        grasp_pose    = pose_stamped,
        gripper_width = float(result["gripper_width"]),
        score         = float(result["score"]),
        success       = bool(result["success"]),
        message       = str(result["message"]),
    )


def main() -> int:
    import signal
    def _on_signal(sig, _frame):
        log.info("signal %d — shutting down", sig)
        _ros_stop_evt.set()
        if _ros_thread is not None:
            _ros_thread.join(timeout=3.0)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)
    try:
        yolo_grasp.run()
    finally:
        _ros_stop_evt.set()
        if _ros_thread is not None:
            _ros_thread.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
