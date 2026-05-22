#!/usr/bin/env python3
# This file is the upstream yolo_grasp.py, kept here as a reference
# of the algorithm yolo_grasp_rbnx implements. The packaged version
# lives in ../main.py and is the version actually run by `rbnx boot`.
#
# Notable differences between this upstream and main.py:
#   1. Upstream is a PURE rclpy node (subscriber-driven, publishes
#      to /yolo/grasps every time a DetectedObjects message arrives).
#      main.py adds an atlas-routed MCP `grasp_request` RPC entry
#      point on top, sharing the same _compute_grasp() math.
#   2. Upstream blocks the ROS spin thread on `input("Continue?")`
#      every 10 detections — REMOVED in main.py because rbnx-spawned
#      providers run with stdin closed (no tty), so input() raises
#      EOFError and the lifecycle thread crashes.
#   3. Upstream uses a `published` flag + interactive prompt to
#      gate publishing — REMOVED in main.py. The new behaviour:
#        - MCP grasp_request: always responds (one grasp per call)
#        - auto_publish_topic mode: emits one grasp message per
#          incoming DetectedObjects (no rate-limiting; if you need
#          dedup, consume only the latest message on the subscriber
#          side — that's what piper_moveit_control does anyway).
#
import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge

try:
    from graspnet_msgs.msg import DetectedObject, DetectedObjects, GraspPose
except Exception as e:
    print("[!] Missing ROS2 service types 'graspnet_msgs/ObjectDetectionRequest'.")
    print("    Please build the graspnet_msgs package before running:")
    print("    1) cd robonix/driver/graspnet && bash build.sh")
    print("    2) source install/setup.bash")
    raise e

candidates = ["bookmark", "lamp", 'test tube', "vacuum", "flyer", "movie ticket",
              "poster page", "paper", "paper bag", "payphone", "placard",
              "sheet music", "document", "monitor", "decorative picture"]


class YoloDepthToGrasp(Node):
    def __init__(self):
        super().__init__("yolo_depth_to_grasp")

        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("det_topic", "/yolo/detect_objects")
        self.declare_parameter("out_topic", "/yolo/grasps")
        self.declare_parameter("output_frame", "camera_color_optical_frame")

        self.declare_parameter("use_bbox_median", True)
        self.declare_parameter("median_grid", 7)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 3.0)

        self.declare_parameter("gripper_width_scale", 1.0)
        self.declare_parameter("gripper_width_min", 0.0)
        self.declare_parameter("gripper_width_max", 0.12)

        self.bridge = CvBridge()
        self.depth_msg: Image | None = None
        self.cam_info: CameraInfo | None = None

        self.sub_depth = self.create_subscription(
            Image,
            self.get_parameter("depth_topic").get_parameter_value().string_value,
            self.on_depth, 10)
        self.sub_info = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").get_parameter_value().string_value,
            self.on_info, 10)
        self.sub_det = self.create_subscription(
            DetectedObjects,
            self.get_parameter("det_topic").get_parameter_value().string_value,
            self.on_det, 10)
        self.pub = self.create_publisher(
            GraspPose,
            self.get_parameter("out_topic").get_parameter_value().string_value,
            10)

        self.published = False
        self.header = None
        self.detect_cnt = 0
        self.get_logger().info("yolo_depth_to_grasp node started.")

    def on_depth(self, msg): self.depth_msg = msg
    def on_info(self, msg): self.cam_info = msg

    def on_det(self, msg):
        if self.depth_msg is None or self.cam_info is None:
            return
        if not msg.objects:
            return
        # NOTE: upstream had an interactive `input("Continue? [y/n]")`
        # block here every 10 detections — see file header for why
        # we removed it in main.py.
        best = next((o for o in msg.objects if o.object_name in candidates), None)
        if best is None:
            return
        bbox = list(best.bbox_2d)
        if len(bbox) != 4:
            return
        x_min, y_min, x_max, y_max = bbox
        u = 0.5 * (x_min + x_max)
        v = 0.5 * (y_min + y_max)

        depth_img = self._depth_to_numpy(self.depth_msg)
        if depth_img is None:
            return
        z_m = self._get_depth_m(depth_img, x_min, y_min, x_max, y_max, u, v)
        if z_m is None:
            return
        fx, fy, cx, cy = self._parse_intrinsics(self.cam_info)
        x = (u - cx) * z_m / fx
        y = (v - cy) * z_m / fy
        z = z_m

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.get_parameter("output_frame").get_parameter_value().string_value
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z) - 0.10  # safe height
        # Pre-calibrated approach quaternion.
        pose.pose.orientation.x = -0.1329
        pose.pose.orientation.y = 0.1508
        pose.pose.orientation.z = -0.6840
        pose.pose.orientation.w = -0.7013

        width_m = float(np.clip(
            0.12,
            float(self.get_parameter("gripper_width_min").value),
            float(self.get_parameter("gripper_width_max").value)))

        out = GraspPose()
        out.target_pose = pose
        out.gripper_width = float(width_m)

        # Upstream had a `published` gate + input() prompt here —
        # see main.py for the rbnx-friendly version.
        self.pub.publish(out)

    def _parse_intrinsics(self, info):
        K = info.k
        return float(K[0]), float(K[4]), float(K[2]), float(K[5])

    def _depth_to_numpy(self, depth_msg):
        try:
            d = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception:
            return None
        a = np.asarray(d)
        if depth_msg.encoding in ("16UC1", "mono16"):
            return a.astype(np.float32) * 0.001
        if depth_msg.encoding == "32FC1":
            return a.astype(np.float32)
        return None

    def _get_depth_m(self, depth_m, x_min, y_min, x_max, y_max, u, v):
        h, w = depth_m.shape[:2]
        min_d = float(self.get_parameter("min_depth_m").value)
        max_d = float(self.get_parameter("max_depth_m").value)
        x0 = int(np.clip(math.floor(x_min), 0, w - 1))
        x1 = int(np.clip(math.ceil(x_max),  0, w - 1))
        y0 = int(np.clip(math.floor(y_min), 0, h - 1))
        y1 = int(np.clip(math.ceil(y_max),  0, h - 1))
        if not bool(self.get_parameter("use_bbox_median").value):
            uu = int(np.clip(round(u), 0, w - 1))
            vv = int(np.clip(round(v), 0, h - 1))
            z = float(depth_m[vv, uu])
            return z if (np.isfinite(z) and min_d <= z <= max_d) else None
        grid = max(3, int(self.get_parameter("median_grid").value))
        xs = np.linspace(x0, x1, grid).astype(int)
        ys = np.linspace(y0, y1, grid).astype(int)
        samples = []
        for yy in ys:
            for xx in xs:
                z = float(depth_m[yy, xx])
                if np.isfinite(z) and min_d <= z <= max_d and z > 0:
                    samples.append(z)
        return float(np.median(samples)) if samples else None
