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
     behaviour). When ``cfg.auto_publish_topic`` is true (default),
     subscribe to ``cfg.detect_objects_topic`` (default
     ``/yolo/detect_objects``, the legacy YOLOE publisher), and
     for every incoming ``DetectedObjects`` message:
       * pick the first detection whose ``object_name`` is in
         ``cfg.candidates`` (a configurable allowlist defaulting to
         the upstream list).
       * compute a grasp via ``_compute_grasp()`` and publish to
         ``/graspnet/grasps`` — fire-and-forget.
     This is what makes the legacy yolo_world → yolo_grasp →
     piper_moveit_control pipeline work without any caller code.

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

# ── shared state ────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_initialized = False
_resolved_cfg: dict[str, Any] = {}

_ros_node = None
_ros_thread: Optional[threading.Thread] = None
_ros_stop_evt = threading.Event()
_grasps_pub = None       # /graspnet/grasps publisher

# Latest depth + camera_info (subscribers update these from the rclpy
# thread; readers — service handlers + auto-publish callback — read
# under _state_lock). Both are raw ROS messages, decoded lazily.
_latest_depth_msg = None         # sensor_msgs/Image
_latest_cam_info = None          # sensor_msgs/CameraInfo

# auto-publish rate limit
_last_auto_publish_time = 0.0

# /yolo/detect_object client (used when MCP / ROS service caller
# didn't supply a bbox themselves)
_detect_client = None


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


def _compute_grasp(
    *,
    bbox_2d: list[float],
    depth_msg,
    cam_info,
    cfg: dict[str, Any],
    object_name: str = "",
) -> dict:
    """Pure geometric grasp computation. The math kernel shared by
    all three surfaces (MCP RPC, ROS service, auto-publish stream).

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

    output_frame = cfg.get("output_frame", "camera_color_optical_frame")
    safe_height_m = float(cfg.get("safe_height_m", 0.10))
    quat = tuple(cfg.get("orientation_xyzw") or _DEFAULT_QUAT)
    if len(quat) != 4:
        quat = _DEFAULT_QUAT

    fail = lambda msg: {  # noqa: E731
        "success": False, "message": msg,
        "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        "frame_id": output_frame,
        "gripper_width": 0.0, "score": 0.0,
    }

    if depth_msg is None:
        return fail("no depth frame received yet")
    if cam_info is None:
        return fail("no camera_info received yet")
    if not bbox_2d or len(bbox_2d) != 4:
        return fail(f"bbox_2d must be length 4, got {bbox_2d!r}")

    x_min, y_min, x_max, y_max = (float(v) for v in bbox_2d)
    u = 0.5 * (x_min + x_max)
    v_pix = 0.5 * (y_min + y_max)

    depth_m = _depth_to_numpy(depth_msg)
    if depth_m is None:
        return fail(f"unsupported depth encoding: {depth_msg.encoding}")

    z_m = _depth_median_in_bbox(
        depth_m, (x_min, y_min, x_max, y_max),
        grid       = int(cfg.get("median_grid",  7)),
        min_depth  = float(cfg.get("min_depth_m", 0.05)),
        max_depth  = float(cfg.get("max_depth_m", 3.0)),
    )
    if z_m is None:
        return fail(
            f"no valid depth in bbox {bbox_2d} (range "
            f"{cfg.get('min_depth_m', 0.05)}..{cfg.get('max_depth_m', 3.0)} m)")

    fx, fy, cx, cy = _parse_intrinsics(cam_info)
    if fx <= 0.0 or fy <= 0.0:
        return fail(f"invalid intrinsics fx={fx} fy={fy}")

    # Pinhole back-projection in optical frame.
    px = (u     - cx) * z_m / fx
    py = (v_pix - cy) * z_m / fy
    pz = z_m - safe_height_m  # safe height: gripper finishes ABOVE the object

    # Gripper width: upstream literally hardcoded 0.12 m, but the
    # config-clamp shape is preserved so a future deploy can wire
    # "narrower for small bboxes" without changing this code.
    width = float(cfg.get("gripper_width_default", 0.12))
    width = float(np.clip(
        width,
        float(cfg.get("gripper_width_min", 0.0)),
        float(cfg.get("gripper_width_max", 0.12)),
    ))

    # Score: a crude quality proxy = depth validity * bbox-coverage.
    # Used by Stage 6 pick_skill to decide whether to retry.
    bbox_area = max(0.0, (x_max - x_min)) * max(0.0, (y_max - y_min))
    h_img, w_img = depth_m.shape[:2]
    rel_area = bbox_area / max(1.0, float(h_img * w_img))
    score = max(0.0, min(1.0, 0.5 + 0.5 * rel_area))

    return {
        "success":       True,
        "message":       f"ok (object={object_name!r}, "
                         f"u,v=({u:.1f},{v_pix:.1f}), z={z_m:.3f}m)",
        "pose": {
            "position":    {"x": float(px), "y": float(py), "z": float(pz)},
            "orientation": {"x": float(quat[0]), "y": float(quat[1]),
                            "z": float(quat[2]), "w": float(quat[3])},
        },
        "frame_id":      output_frame,
        "gripper_width": float(width),
        "score":         float(score),
    }


# ── ROS bring-up (background thread) ────────────────────────────────────────
def _ros_thread_main() -> None:
    global _ros_node, _grasps_pub, _detect_client
    global _latest_depth_msg, _latest_cam_info

    import rclpy                                              # noqa: E402
    from rclpy.node import Node                               # noqa: E402
    from sensor_msgs.msg import Image, CameraInfo             # noqa: E402
    from graspnet_msgs.srv import GraspRequest                # noqa: E402
    from graspnet_msgs.srv import ObjectDetectionRequest      # noqa: E402
    from graspnet_msgs.msg import GraspPose as RosGraspPose   # noqa: E402
    from graspnet_msgs.msg import DetectedObjects             # noqa: E402
    from cv_bridge import CvBridge                            # noqa: E402

    # cv_bridge instance attached to the helper for cheap reuse
    _depth_to_numpy._bridge = CvBridge()  # type: ignore[attr-defined]

    rclpy.init(args=None)
    node = Node("yolo_grasp_node")
    _ros_node = node

    cfg = _resolved_cfg

    depth_topic    = str(cfg.get("depth_topic",          "/camera/depth/image_raw"))
    cam_info_topic = str(cfg.get("camera_info_topic",    "/camera/depth/camera_info"))
    det_topic      = str(cfg.get("detect_objects_topic", "/yolo/detect_objects"))
    grasps_topic   = str(cfg.get("grasps_topic",         "/graspnet/grasps"))

    # Subscribers — feed the latest depth / camera_info into shared
    # state. We deliberately keep only the most recent message; older
    # ones are dropped because grasp_request is always "use what's
    # current right now". qos depth=10 mirrors upstream.
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

    # Publisher — C++ piper_moveit_control + every other downstream.
    _grasps_pub = node.create_publisher(RosGraspPose, grasps_topic, 10)
    log.info("publishing grasps=%s", grasps_topic)

    # /yolo/detect_object client — only used when caller didn't supply
    # a bbox (so we ask yolo_world for one). Don't block on its
    # availability; if it's not up and someone calls grasp_request
    # without a bbox, _serve_grasp_request will say so.
    _detect_client = node.create_client(
        ObjectDetectionRequest, "/yolo/detect_object")

    # /graspnet/grasp_request service host (legacy compat).
    def _grasp_request_cb(req, resp):
        result = _serve_grasp_request(
            object_name      = req.object_name,
            bbox_2d          = list(req.bbox_2d) if req.bbox_2d else [],
            object_center_3d = list(req.object_center_3d) if req.object_center_3d else [],
            retry            = int(req.retry),
        )
        _pack_response_and_publish(resp, result, also_publish=False)
        return resp
    node.create_service(GraspRequest, "/graspnet/grasp_request", _grasp_request_cb)
    log.info("ROS service up: /graspnet/grasp_request")

    # Optional auto-publish: subscribe to /yolo/detect_objects and
    # publish grasps for matches against `candidates`.
    auto_publish = bool(cfg.get("auto_publish_topic", True))
    candidates   = list(cfg.get("candidates") or _DEFAULT_CANDIDATES)
    if auto_publish:
        log.info("auto_publish_topic ON — subscribing %s, candidates=%d",
                 det_topic, len(candidates))
        node.create_subscription(
            DetectedObjects, det_topic,
            lambda m: _on_detected_objects(m, candidates), 10)
    else:
        log.info("auto_publish_topic OFF — service mode only")

    # Spin until shutdown.
    while not _ros_stop_evt.is_set():
        rclpy.spin_once(node, timeout_sec=0.1)

    # Graceful shutdown: emit a zero-pose so subscribers know we're
    # gone (upstream does this; piper_moveit_control treats zero as
    # "no grasp pending"). Wrap in try since we may already be in
    # the middle of rclpy teardown.
    try:
        if _grasps_pub is not None:
            zero = RosGraspPose()
            _fill_zero_grasp(zero, frame_id=str(cfg.get("output_frame",
                                                       "camera_color_optical_frame")))
            _grasps_pub.publish(zero)
            log.info("published shutdown zero-pose to %s", grasps_topic)
    except Exception as e:  # noqa: BLE001
        log.warning("shutdown zero-pose publish failed: %s", e)

    node.destroy_node()
    rclpy.shutdown()
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


def _call_detect_object(object_name: str, timeout_s: float = 5.0) -> dict:
    """Synchronous client call to /yolo/detect_object (legacy path
    owned by yolo_world_rbnx). Used when caller didn't supply bbox."""
    from graspnet_msgs.srv import ObjectDetectionRequest  # noqa: E402
    if _detect_client is None:
        return {"success": False,
                "message": "ROS thread not initialized",
                "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}
    if not _detect_client.service_is_ready():
        return {"success": False,
                "message": "/yolo/detect_object service not advertised",
                "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}
    req = ObjectDetectionRequest.Request()
    req.object_name = object_name
    fut = _detect_client.call_async(req)
    deadline = time.monotonic() + timeout_s
    while not fut.done() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not fut.done():
        return {"success": False,
                "message": f"/yolo/detect_object call timed out after {timeout_s}s",
                "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}
    resp = fut.result()
    return {
        "success":          bool(resp.success),
        "message":          str(resp.message),
        "bbox_2d":          list(resp.bbox_2d),
        "object_center_3d": list(resp.object_center_3d),
        "confidence":       float(resp.confidence),
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
             len(cfg), cfg.get("auto_publish_topic", True),
             len(cfg.get("candidates") or _DEFAULT_CANDIDATES))

    # Informational: log which provider owns detect_object on atlas.
    _resolve_detect_object_endpoint()

    # Spawn rclpy thread. on_init returning Ok() triggers atlas to
    # mark us ACTIVE; the thread itself is what advertises the
    # service / topic.
    global _ros_thread
    _ros_stop_evt.clear()
    _ros_thread = threading.Thread(
        target=_ros_thread_main,
        name="yolo_grasp-ros",
        daemon=True,
    )
    _ros_thread.start()
    time.sleep(0.5)  # let create_service / create_publisher land

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
