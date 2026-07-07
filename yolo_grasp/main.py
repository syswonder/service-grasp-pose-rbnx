#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""yolo_grasp_rbnx — roboarm-style grasp-pose estimator service.

Owns ``robonix/service/perception/grasp_pose/*``. Pure CPU, no ML model.
The core math intentionally follows ``~/lhw/roboarm`` rather than the
previous vertical-grasp implementation:

  * take the LLM / detector bbox center pixel.
  * project it to arm-plane XY through a required 3x3 2D hand-eye
    homography (same idea as roboarm ``Arm.pixel2pos``).
  * estimate gripper yaw from the bbox long edge (same as roboarm
    ``Arm.gripper_angle_by_longer``).
  * shift the grasp point by ``catch_offset`` along that yaw.
  * output a vertical-down PoseStamped in ``arm/base_link`` at
    ``default_desktop_height``.

There is no fallback to the old camera-intrinsics + TF ray-plane
algorithm. If the homography or tabletop height is missing, init fails
with a clear configuration error.

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
                      (informational), load roboarm homography, spawn
                      rclpy thread (subs + pubs + service host).
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

# ── roboarm-style grasp mode constants ──────────────────────────────────────
# In roboarm mode, the grasp pose is computed exactly like roboarm's
# llm/catch_by_llm.py:
#   bbox center pixel -> 2D homography -> arm XY
#   bbox long edge    -> gripper yaw
#   XY + catch_offset -> final grasp point
#
# There is intentionally no fallback to the old ray-plane / TF algorithm.
_DEFAULT_CATCH_OFFSET   = 0.01    # meters
_DEFAULT_BOX_ROTATION_DEG = 0.0
_DEFAULT_APPROACH_DIST   = 0.10   # pre/post grasp hover height (m)
_DEFAULT_BASE_FRAME      = "arm/base_link"

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

# 3x3 pixel -> arm-plane homography loaded from cfg at init time.
_homography_matrix = None


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


def _load_homography_matrix(cfg: dict[str, Any]):
    """Load the mandatory roboarm 2D hand-eye homography.

    Accepted config:
      * homography_matrix: nested 3x3 list
      * hand_eye_calibration_file / homography_file: path to a .npy
    """
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
    """roboarm Arm.pixel2pos(): pixel center -> arm-plane XY."""
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
    """roboarm Arm.gripper_angle_by_longer() without requiring cv2."""
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


def _compute_grasp(
    *,
    bbox_2d: list[float],
    depth_msg,
    cam_info,
    cfg: dict[str, Any],
    object_name: str = "",
) -> dict:
    """roboarm-style grasp computation.

    The bbox center is projected to arm-plane XY via a required 3x3
    homography. The end-effector yaw follows roboarm's long-edge rule,
    then the grasp point is shifted by catch_offset along that yaw.

    depth_msg and cam_info are accepted for API compatibility but are
    intentionally ignored. There is no fallback to the old TF
    ray-plane algorithm.

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
    output_frame = cfg.get("output_frame", _DEFAULT_BASE_FRAME)
    if "default_desktop_height" not in cfg:
        return {
            "success": False,
            "message": "missing required roboarm config: default_desktop_height",
            "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0},
                     "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
            "frame_id": output_frame,
            "gripper_width": 0.0,
            "score": 0.0,
        }
    desktop_height = float(cfg.get("default_desktop_height"))
    catch_offset = float(cfg.get("catch_offset", _DEFAULT_CATCH_OFFSET))
    width        = float(cfg.get("gripper_width_default", 0.04))
    box_rotation_deg = float(
        cfg.get("box_rotation_deg", _DEFAULT_BOX_ROTATION_DEG)
    )

    fail = lambda msg: {  # noqa: E731
        "success": False, "message": msg,
        "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        "frame_id": output_frame,
        "gripper_width": 0.0, "score": 0.0,
    }

    if not bbox_2d or len(bbox_2d) not in (4, 5):
        return fail(f"bbox_2d must be length 4 or 5, got {bbox_2d!r}")

    # ── bbox center pixel ──
    x_min, y_min, x_max, y_max = (float(value) for value in bbox_2d[:4])
    if len(bbox_2d) == 5:
        box_rotation_deg = float(bbox_2d[4])
    u = 0.5 * (x_min + x_max)
    v_pix = 0.5 * (y_min + y_max)
    bbox_w = abs(x_max - x_min)
    bbox_h = abs(y_max - y_min)
    if bbox_w <= 0.0 or bbox_h <= 0.0:
        return fail(f"bbox has non-positive size: {bbox_2d!r}")

    try:
        target_x, target_y = _pixel_to_arm_xy(u, v_pix)
    except ValueError as e:
        return fail(f"pixel2pos failed: {e}")

    yaw_rad = _gripper_angle_by_longer(
        u, v_pix, bbox_w, bbox_h, box_rotation_deg
    )
    grasp_x = target_x + catch_offset * math.cos(yaw_rad)
    grasp_y = target_y + catch_offset * math.sin(-yaw_rad)
    qx, qy, qz, qw = _vertical_quaternion(yaw_rad)

    # Score: crude quality proxy based on bbox area (same as before).
    bbox_area = max(0.0, (x_max - x_min)) * max(0.0, (y_max - y_min))
    # We don't have image dims here without depth_msg, so just use
    # a flat 0.8 default — the score is advisory only.
    score = 0.8

    log.info(
        "roboarm grasp: object=%r uv=(%.1f,%.1f) bbox=(%.1fx%.1f, rot=%.1f) "
        "pixel2pos=(x=%.3f, y=%.3f) offset=%.3f -> "
        "grasp=(x=%.3f, y=%.3f, z=%.3f) yaw=%.3f",
        object_name, u, v_pix, bbox_w, bbox_h, box_rotation_deg,
        target_x, target_y, catch_offset, grasp_x, grasp_y,
        desktop_height, yaw_rad)

    return {
        "success":       True,
        "message":       f"ok (object={object_name!r}, "
                         f"u,v=({u:.1f},{v_pix:.1f}), "
                         f"arm_xy=({grasp_x:.3f},{grasp_y:.3f}), "
                         f"yaw={yaw_rad:.3f})",
        "pose": {
            "position":    {"x": float(grasp_x), "y": float(grasp_y),
                            "z": float(desktop_height)},
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
    global _ros_thread_error

    node = None
    try:
        import rclpy                                              # noqa: E402
        from rclpy.node import Node                               # noqa: E402
        from graspnet_msgs.srv import GraspRequest                # noqa: E402
        from graspnet_msgs.msg import GraspPose as RosGraspPose   # noqa: E402
        from graspnet_msgs.msg import DetectedObjects             # noqa: E402

        rclpy.init(args=None)
        node = Node("yolo_grasp_node")
        _ros_node = node

        cfg = _resolved_cfg

        det_topic      = str(cfg.get("detect_objects_topic", "/yolo/detect_objects"))
        grasps_topic   = str(cfg.get("grasps_topic",         "/graspnet/grasps"))

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
    while not _ros_stop_evt.is_set():
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
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
                                                       _DEFAULT_BASE_FRAME)))
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
    object_name matches the candidates allowlist, computes a
    roboarm-style grasp from its bbox, and publishes."""
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
    result = _compute_grasp(
        bbox_2d=bbox, depth_msg=None, cam_info=None,
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
    first. ``object_center_3d`` from the caller is advisory only; the
    roboarm-style estimator uses bbox center + 2D homography.
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
                                              _DEFAULT_BASE_FRAME),
                "gripper_width": 0.0,
                "score": 0.0,
            }
        bbox_2d = det["bbox_2d"]

    return _compute_grasp(
        bbox_2d=bbox_2d, depth_msg=None, cam_info=None,
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

    log.info("cfg: %d keys (roboarm homography loaded, auto_publish=%s, candidates=%d)",
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
    on the fly. ``object_center_3d`` from the caller is advisory only;
    the roboarm-style estimator uses bbox center + 2D homography.
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
