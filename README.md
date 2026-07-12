# service-grasp-pose-rbnx

Robonix package for **grasp-pose estimation** on the Piper + Orbbec Dabai DCW grasp pipeline. Atlas + ROS dual surface. Owns the `service/perception/grasp_pose/*` namespace.

Catalog name: `robonix.service.grasp_pose`.

## Capability surface

| Contract                                                | Mode      | Transport | Source / handler                                                            |
| ------------------------------------------------------- | --------- | --------- | --------------------------------------------------------------------------- |
| `robonix/service/perception/grasp_pose/driver`          | rpc       | gRPC      | `Driver(CMD_INIT, config_json)` — lifecycle gate                            |
| `robonix/service/perception/grasp_pose/grasp_request`   | rpc       | gRPC/MCP  | `GraspRequest(object_name, bbox_2d?, object_center_3d?, retry?) → GraspPose` |
| `robonix/service/perception/grasp_pose/grasps`          | topic_out | ROS 2     | `/graspnet/grasps` (graspnet_msgs/msg/GraspPose)                            |

The legacy ROS service `/graspnet/grasp_request` (`graspnet_msgs/srv/GraspRequest`) is kept alive as a secondary surface. `_serve_grasp_request()` internally calls `service-object-detect-rbnx`'s `detect_object` when the caller does not provide a bbox.

The C++ `piper_moveit_control` subscriber on `/graspnet/grasps` always gets a fire-and-forget message on success even when the caller used the MCP path — that way `service-piper-moveit-rbnx` does not have to track which surface produced the pose.

## Two surfaces, one estimator

```
       Pilot LLM               skill-pick-rbnx
           │                        │
           ▼                        ▼
    atlas-routed MCP           ROS service
    grasp_request           /graspnet/grasp_request
           │                        │
           └──► _serve_grasp_request() ◄──┘
                          │
                          ▼
                   _estimate_grasp_pose()
                          │
                          ▼              ┌─► topic /graspnet/grasps
                   GraspPose response ───┤   (C++ moveit_control subscriber)
                                         └─► (also returned synchronously)
```

## Boot ordering

Must come **after** `service-object-detect-rbnx` (consumes its detection output). Also requires the TF tree published by `primitive-agilex-piper-description-rbnx` + `primitive-agilex-piper-handeye-rbnx`.

## Two algorithms shipped side-by-side

The package ships two grasp-pose strategies in the same estimator surface. Which one is active depends on the deploy branch / config.

### Vertical-grasp (feature/vertical-grasp branch — the current default)

- **xy**: bbox center pixel from `service-object-detect-rbnx` → back-project a ray using camera intrinsics → intersect with the tabletop plane `z = z_table` → `(x, y)` in `base_link` frame.
- **z**: fixed constant `z_table + z_offset`.
- **orientation**: fixed `roll = π`, `pitch = 0`, `yaw = default_yaw_rad` (or radial-yaw mode).
- **gripper opening**: `default_gripper_width`.
- The three-stage motion sequence (pre-grasp / grasp / post-grasp) is orchestrated by `skill-pick-rbnx`; `approach_dist` here is the z lift used for the pre/post motion.

### Depth-median geometric estimator (main branch — legacy)

Direct port of upstream `detect_grasp/yolo_grasp.py`, preserved at `yolo_grasp/_upstream/yolo_grasp.py` for reference. Per grasp:

1. Take the bbox center pixel `(u, v) = ((x_min+x_max)/2, (y_min+y_max)/2)`.
2. Sample a `median_grid × median_grid` lattice (default 7×7) of depth pixels inside the bbox. Keep only depths in `[min_depth_m, max_depth_m]` (default `[0.05, 3.0]` m). Take the median.
3. Back-project `(u, v, z)` through the pinhole intrinsics from `camera_info` (`fx, fy, cx, cy` from `K`): `x = (u - cx) * z / fx; y = (v - cy) * z / fy`.
4. Subtract `safe_height_m` (default 0.10 m) from z so the gripper ends ABOVE the object. The MoveIt cartesian descent finishes the last 10 cm.
5. Orientation = a pre-calibrated approach quaternion `[-0.1329, 0.1508, -0.6840, -0.7013]` (override via `cfg.orientation_xyzw`).
6. Gripper width = `clamp(0.12, gripper_width_min, gripper_width_max)`.

Output frame is `camera_color_optical_frame`; `primitive-agilex-piper-handeye-rbnx` supplies the `link6 → camera_color_optical_frame` static TF so `service-piper-moveit-rbnx` can transform to `base_link` without extra plumbing.

## Driver-init lifecycle

```
on_init ── parse cfg
        ── atlas resolve object_detect endpoint (informational)
        ── spawn rclpy thread
             (ROS service host
              + /graspnet/grasps publisher
              + /yolo/detect_object client)

on_deactivate ── stop rclpy thread.
```

`on_init` does not depend on `detect_object` being up — we wait up to 30 s in the rclpy thread, log a warning if it never appears, and let `grasp_request` calls without a bbox fail cleanly afterwards. This keeps boot from deadlocking when the detection service is still warming up.

## Auto-publish stream mode (default OFF — safety)

Independent of the `grasp_request` RPC surfaces, the package can run in "stream mode" — exactly what upstream `yolo_grasp.py` did:

```
object_detect publishes /yolo/detect_objects (DetectedObjects)
                  │
                  ▼
grasp_pose subscribes; for each detection whose object_name is in
`cfg.candidates`, compute a grasp and publish to /graspnet/grasps
                  │
                  ▼
piper_moveit_control C++ subscriber receives → executes
```

**OFF by default.** The C++ `moveit_control_node_yolo` triggers a real arm motion on the FIRST `/graspnet/grasps` message it sees while idle. With auto-publish ON, the moment the cpp node returns to idle (e.g. right after `/moveit_control/reset`), the next 1 Hz tick from the auto-publish stream will start an unsolicited grasp — exactly the kind of surprise we don't want for a hardware deploy. Enable explicitly only for the legacy demo:

```yaml
config:
  auto_publish_topic: true
```

The candidates allowlist defaults to the upstream list (15 prompt-free YOLOE class names); override via `cfg.candidates` to scope to your scene.

## Layout

```
service-grasp-pose-rbnx/
├── package_manifest.yaml
├── capabilities/
│   ├── service/perception/grasp_pose/{driver,grasp_request,grasps}.v1.toml
│   └── lib/grasp/
│       ├── srv/GraspRequest.srv    # codegen → GraspRequest_Request/_Response
│       └── msg/GraspPose.msg       # codegen → GraspPose dataclass
├── yolo_grasp/
│   ├── __init__.py
│   ├── main.py                     # robonix Service + rclpy thread (live)
│   └── _upstream/
│       └── yolo_grasp.py           # upstream source kept for reference
├── scripts/
│   ├── build.sh                    # colcon graspnet_msgs + rbnx codegen --mcp
│   └── start.sh                    # source overlays, exec yolo_grasp.main
└── src/
    └── graspnet_msgs/              # vendored (32 KB)
```

## Config (passed via `Driver(CMD_INIT, config_json)`)

All keys are optional. Defaults below.

```yaml
# ── Vertical-grasp parameters (feature/vertical-grasp branch) ──
z_table:                -0.186           # measure on-site! see below
z_offset:               0.0
bias_x:                 0.0
bias_y:                 0.025
approach_dist:          0.10
default_yaw_rad:        0.0
radial_yaw:             true             # ignore default_yaw_rad if true
radial_yaw_offset_rad:  0.0
default_gripper_width:  0.04
base_frame:             arm/base_link
camera_frame:           camera_color_optical_frame
camera_info_topic:      /camera/color/camera_info
output_frame:           arm/base_link

# ── Depth-median estimator (main branch) ──
median_grid:            7
min_depth_m:            0.05
max_depth_m:            3.0
safe_height_m:          0.10
gripper_width_default:  0.12
gripper_width_min:      0.0
gripper_width_max:      0.12
orientation_xyzw:       [-0.1329, 0.1508, -0.6840, -0.7013]

# ── Auto-publish stream (legacy upstream behaviour) ──
auto_publish_topic:          false        # default false — SAFETY
auto_publish_min_interval_s: 0.0          # 0 = no rate limit
candidates:                               # detection allowlist
  - bookmark
  - lamp
  - paper
  - document
  - monitor
  # …see _DEFAULT_CANDIDATES in main.py for full list

# ── ROS topic name overrides (rarely needed) ──
depth_topic:          /camera/depth/image_raw
detect_objects_topic: /yolo/detect_objects
grasps_topic:         /graspnet/grasps
```

- `z_table` **MUST** be measured on-site with `ros2 run tf2_ros tf2_echo arm/base_link <table_marker_link>` or by direct physical measurement. Getting it wrong grabs air or crashes into the table.
- `radial_yaw`: when true, ignore `default_yaw_rad` and use `atan2(y_base, x_base)` — the gripper opening aligns radially from base to target, avoiding wrist flips at j6 ± π. `radial_yaw_offset_rad` rotates the opening direction.

## Build / run standalone

```bash
bash scripts/build.sh                           # colcon graspnet_msgs + rbnx codegen --mcp
ROBONIX_ATLAS=127.0.0.1:50051 \
    bash scripts/start.sh                       # registers, awaits Init
```

### Manual single-package debugging

`scripts/dev_source.sh` is a helper for running this package by hand outside `rbnx boot` — useful for pdb / fast iter. It sources the same overlays / PYTHONPATH that `scripts/start.sh` would, and verifies that the vendored `graspnet_msgs` is the one importable in this shell.

```bash
source scripts/dev_source.sh
python3 -u -m yolo_grasp.main
```

**Do NOT add `source dev_source.sh` to `~/.bashrc`.** The script spawns a child bash; if `.bashrc` re-sources the script, every child bash recurses, which manifests as N copies of "package root: …" in the rbnx-boot log followed by a 60 s registration timeout.

## Verification

```bash
# 1. atlas-side: provider + capabilities visible
rbnx caps | grep grasp_pose
# expect: robonix.service.grasp_pose  ACTIVE  with
#   robonix/service/perception/grasp_pose/{driver, grasp_request, grasps}

# 2. End-to-end via Pilot LLM (MCP path):
rbnx ask "grasp the paper on the desk"

# 3. Legacy ROS service shape (with bbox supplied):
ros2 service call /graspnet/grasp_request \
    graspnet_msgs/srv/GraspRequest \
    "{object_name: 'paper', bbox_2d: [200.0, 100.0, 400.0, 300.0], retry: 0}"

# 4. Auto-publish stream mode (no manual call needed):
ros2 topic echo /graspnet/grasps --once
# expect: a GraspPose with non-trivial target_pose + gripper_width > 0
# whenever object_detect publishes a detection in `cfg.candidates`.
```

## Failure modes

| symptom                                                                       | cause                                                                          | fix                                                                              |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| `grasp_request` returns `success=false, message="no depth frame received yet"` | Camera primitive not active, or `depth_topic` mismatch                         | Check camera ACTIVE; verify topic with `ros2 topic hz $depth_topic`              |
| `grasp_request` returns `success=false, message="no valid depth in bbox …"`   | Object outside `[min_depth_m, max_depth_m]`, or all-zero depth                  | Widen `max_depth_m`, or increase `median_grid`                                   |
| `detect_object pre-call failed: service not advertised`                       | `service-object-detect-rbnx` not active (only matters for RPC path w/o bbox)    | Check `rbnx caps object_detect`; ensure it is up first                           |
| `/graspnet/grasps` silent even on detections                                  | Mismatched topic name, or rclpy thread crashed                                 | Look for "rclpy thread exited" in logs; verify `grasps_topic` matches C++ subscriber |

## What was removed from upstream

The packaged `yolo_grasp/main.py` drops these debugging hooks from upstream `_upstream/yolo_grasp.py`:

| upstream                                                       | rbnx version                                                                                        |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `input("Continue? [y/n]")` block every 10 detections           | **removed** — rbnx-spawned providers run with stdin closed. Use `auto_publish_min_interval_s`.       |
| `published` flag = "publish exactly one grasp ever"            | **removed** — every detection / RPC call gets a fresh grasp.                                        |
| Hardcoded `det_topic = /yolo/detect_objects` etc.              | **kept as defaults**, but configurable.                                                             |
| Shutdown publishes a zero-pose to `/graspnet/grasps`           | **kept** — useful sentinel for downstream subscribers.                                              |

## License

This package: MulanPSL-2.0.
