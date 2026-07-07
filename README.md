# yolo_grasp_rbnx

Robonix package for geometric grasp-pose estimation on the Piper +
Orbbec Dabai DCW grasp pipeline. Stage 4B of the migration.

> **Status:** live algorithm. The geometric estimator is implemented
> and wired up — `grasp_request` calls return real grasps as long as
> a depth frame + camera_info + a valid bbox are available.

## What it does

Geometric / heuristic grasp-pose estimator on top of a YOLO-World 2D
detection. Pure CPU, no ML model. Owns
`robonix/service/perception/grasp_pose/*`. Three roles:

| role | surface | who talks to it |
|---|---|---|
| **Server (atlas MCP)** | `grasp_pose/grasp_request` | Pilot's LLM (new path) |
| **Server (legacy ROS)** | `/graspnet/grasp_request` (graspnet_msgs/srv/GraspRequest) | pick.py (Stage 6 will switch to MCP) |
| **Topic publisher** | `/graspnet/grasps` (graspnet_msgs/msg/GraspPose) | C++ piper_moveit_control subscriber |
| **Client (legacy ROS)** | `/yolo/detect_object` | yolo_world_rbnx (Stage 4A) |

When called without a `bbox_2d` / `object_center_3d` hint, the handler
calls `/yolo/detect_object` itself to localise the target. The C++
piper_moveit_control subscriber on `/graspnet/grasps` always gets a
fire-and-forget message on success even when the caller used the MCP
path — that way Stage 5 doesn't have to track which surface produced
the pose.

## Two surfaces, one estimator

```
       Pilot LLM           pick.py (Stage 6 cutover)
           │                       │
           ▼                       ▼
    atlas-routed MCP          ROS service
    grasp_request           /graspnet/grasp_request
           │                       │
           └──► _serve_grasp_request() ◄──┘
                          │
                          ▼
                   _estimate_grasp_pose()
                          │
                          ▼              ┌─► topic /graspnet/grasps
                   GraspPose response ───┤   (C++ moveit_control subscriber)
                                          └─► (also returned synchronously)
```

`_serve_grasp_request()` may internally call upstream
`/yolo/detect_object` when the caller doesn't provide a bbox.

## Architecture

```
yolo_grasp_rbnx/
├── package_manifest.yaml
├── capabilities/
│   ├── service/perception/grasp_pose/
│   │   ├── driver.v1.toml          # rpc, lifecycle/srv/Driver.srv
│   │   ├── grasp_request.v1.toml   # rpc/MCP, grasp/srv/GraspRequest.srv
│   │   └── grasps.v1.toml          # topic_out/ROS2, grasp/msg/GraspPose.msg
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

## Lifecycle

```
on_init ── parse cfg ──► atlas resolve detect_object endpoint (informational)
                       ──► spawn rclpy thread
                           (ROS service host
                            + /graspnet/grasps publisher
                            + /yolo/detect_object client)

on_deactivate ── stop rclpy thread.
```

Note: this package's `on_init` does not depend on `/yolo/detect_object`
being up — we wait up to 30s for the upstream service in the rclpy
thread, log a warning if it never appears, and let `grasp_request`
calls without a bbox fail cleanly afterwards. This keeps boot from
deadlocking when yolo_world_rbnx is still warming up.

## Algorithm (roboarm estimator)

The math is a direct port of `~/lhw/roboarm`'s LLM grasp path. Per
grasp:

1. Take the bbox center pixel `(u, v) = ((x_min+x_max)/2, (y_min+y_max)/2)`.
2. Project `[u, v, 1]` through the required 3x3 homography matrix to
   get arm-plane `(x, y)`, same as roboarm `Arm.pixel2pos`.
3. Apply a global XY bias (`bias_x`, `bias_y`) after homography.
4. Compute gripper yaw from the bbox long edge, same as roboarm
   `Arm.gripper_angle_by_longer`.
5. Shift the grasp point by `catch_offset`:
   `x += offset * cos(yaw)` and `y += offset * sin(-yaw)`.
6. Use `default_desktop_height` as the final grasp z.
7. Output a vertical-down pose in `arm/base_link`.

Depth, camera intrinsics, and hand-eye TF are not used. If the
homography or desktop height is missing, `on_init` fails.

### Score (heuristic)

`score = clamp(0.5 + 0.5 * bbox_area / image_area, 0, 1)`. Used by
Stage 6 pick_skill to decide whether to retry. Upstream doesn't
expose a per-grasp score; this is a Stage 4B-only addition.

## Auto-publish stream mode (default OFF — safety)

Independent of the `grasp_request` RPC surfaces, the package can also
run in "stream mode" — exactly what upstream `yolo_grasp.py` did:

```
yolo_world publishes /yolo/detect_objects (DetectedObjects)
                  │
                  ▼
yolo_grasp subscribes; for each detection whose object_name is in
`cfg.candidates`, compute a grasp and publish to /graspnet/grasps
                  │
                  ▼
piper_moveit_control C++ subscriber receives → executes
```

This is the legacy yolo_world → yolo_grasp → piper_moveit_control
pipeline running end-to-end without any caller code at all.

**It is OFF by default.** The cpp `moveit_control_node_yolo`
triggers a real arm motion on the FIRST `/graspnet/grasps` message
it sees while idle. With auto-publish ON, the moment the cpp node
returns to idle (e.g. right after `/moveit_control/reset`), the
NEXT 1Hz tick from the auto-publish stream will start an
unsolicited grasp. That is exactly the kind of surprise we don't
want for a hardware deploy. Enable explicitly only when you want
the legacy demo behaviour:

```yaml
config:
  auto_publish_topic: true
```

The candidates allowlist defaults to the upstream list (15 prompt-free
YOLOE class names: `bookmark`, `lamp`, `paper`, `document`, …); replace
via `cfg.candidates` to scope to your scene.

## How callers should trigger a grasp

With `auto_publish_topic` defaulting to false, exactly two paths
trigger the cpp executor:

| Caller surface | What it does |
|---|---|
| ROS service `/graspnet/grasp_request` (`graspnet_msgs/srv/GraspRequest`) | Computes a grasp, fills the response, AND publishes the same pose to `/graspnet/grasps` so the cpp node executes it. |
| MCP `robonix/service/perception/grasp_pose/grasp_request` (Pilot / pick_skill path) | Same: response + topic publish. |

Both are one-shot — one call ⇒ at most one grasp. No background stream.

## Configuration

`hand_eye_calibration_file` (or inline `homography_matrix`) and
`default_desktop_height` are required. The estimator intentionally uses
the same 2D homography + long-edge yaw + catch offset strategy as
`~/lhw/roboarm`; it does not use depth, camera intrinsics, or TF.

```yaml
config:
  # Auto-publish stream (legacy upstream behaviour) ──────────────
  auto_publish_topic: false               # default false; see above
  auto_publish_min_interval_s: 0.0        # 0 = no rate limit
  candidates:                             # detection allowlist
    - bookmark
    - lamp
    - paper
    - document
    - monitor
    # ...etc; see _DEFAULT_CANDIDATES in main.py for full list

  # ROS topic names ─────────────────────────────────────────────
  detect_objects_topic: /yolo/detect_objects
  grasps_topic:         /graspnet/grasps

  # Roboarm-style estimator knobs ───────────────────────────────
  hand_eye_calibration_file: /absolute/path/to/2d_homography.npy
  # or:
  # homography_matrix:
  #   - [1.0, 0.0, 0.0]
  #   - [0.0, 1.0, 0.0]
  #   - [0.0, 0.0, 1.0]
  default_desktop_height: 0.075
  bias_x:                 0.0           # global X correction after homography
  bias_y:                 0.0           # global Y correction after homography
  catch_offset:           0.01          # local offset along gripper yaw
  box_rotation_deg:       0.0
  gripper_width_default:  0.04
  output_frame:           arm/base_link
```

## What was removed from upstream

The packaged version (`yolo_grasp/main.py`) drops these debugging
hooks from upstream `_upstream/yolo_grasp.py`:

| upstream | rbnx version |
|---|---|
| `input("Continue? [y/n]")` block every 10 detections | **removed** — rbnx-spawned providers run with stdin closed, `input()` raises `EOFError`. Use `auto_publish_min_interval_s` for rate limiting instead. |
| `published` flag = "publish exactly one grasp ever" | **removed** — every detection or RPC call gets a fresh grasp. Subscribers that want dedup can keep the latest message themselves (piper_moveit_control already does). |
| Hardcoded `det_topic = /yolo/detect_objects` etc. | **kept as defaults**, but configurable. |
| Shutdown publishes a zero-pose to `/graspnet/grasps` | **kept** — useful sentinel for downstream subscribers when the service stops. |

## Build / run

```bash
cd /Users/howenliu/lab/packages/yolo_grasp_rbnx
bash scripts/build.sh

cd /Users/howenliu/lab/piper_grasp_deploy
rbnx boot
```

## Manual single-package debugging

`scripts/dev_source.sh` is a helper for running this package by hand
outside of `rbnx boot` — useful when you need pdb / fast iter / a
clean stdout. It sources the same overlays / PYTHONPATH that
`scripts/start.sh` would, and verifies that the vendored
`graspnet_msgs` is the one importable in this shell.

```bash
# In a shell that will run python3 -m yolo_grasp.main:
cd /Users/howenliu/lab/packages/yolo_grasp_rbnx
source scripts/dev_source.sh
python3 -u -m yolo_grasp.main
```

**Do NOT add `source dev_source.sh` to `~/.bashrc`.** The script does
`$(rbnx path …)` which spawns a child bash; if `.bashrc` re-sources
the script, every child bash recurses, which manifests as N copies of
`[yolo_grasp-source] package root: …` in the rbnx-boot log followed
by a 60s registration timeout (the real `python3 -m yolo_grasp.main`
never actually runs). The script has a reentrancy guard against this,
but the right place for it is `source` it on demand from a single
debugging shell, not your shell init.

`rbnx boot` itself uses `scripts/start.sh`, which has its own
sourcing chain and doesn't need this helper.

## Verification (in order)

```bash
# 1. atlas-side: provider + capabilities visible
rbnx caps | grep yolo_grasp
# expect:
#   yolo_grasp  com.robonix.piper_grasp.yolo_grasp  ACTIVE
#     robonix/service/perception/grasp_pose/driver         (rpc/grpc)
#     robonix/service/perception/grasp_pose/grasp_request  (rpc/mcp)
#     robonix/service/perception/grasp_pose/grasps         (topic_out/ros2)

# 2. End-to-end via Pilot LLM (MCP path):
rbnx ask "grasp the paper on the desk"
# pilot calls grasp_request → expects success=true with a real pose
# (assumes yolo_world is publishing detections + camera is on)

# 3. Legacy ROS service shape (with bbox supplied):
ros2 service call /graspnet/grasp_request \
    graspnet_msgs/srv/GraspRequest \
    "{object_name: 'paper', bbox_2d: [200.0, 100.0, 400.0, 300.0], retry: 0}"
# expect: success=true, target_pose in camera_color_optical_frame

# 4. Auto-publish stream mode (no manual call needed):
ros2 topic echo /graspnet/grasps --once
# expect: a GraspPose with non-trivial target_pose + gripper_width > 0
# whenever yolo_world publishes a detection in `cfg.candidates`.
```

## Failure modes

| symptom | cause | fix |
|---|---|---|
| `bad roboarm homography config: missing required ...` | `hand_eye_calibration_file` / `homography_matrix` not configured | Add a real 3x3 pixel-to-arm-plane homography to `yolo_grasp.config` |
| `bad roboarm homography config: hand_eye_calibration_file does not exist ...` | Config points at a missing `.npy` | Put the calibrated `.npy` on disk and update the path |
| `missing required roboarm config: default_desktop_height` | Grasp z is not configured | Add the measured tabletop/grasp z in meters |
| `auto_publish: no candidate match in detections` (debug log) | None of the YOLOE detections are in `cfg.candidates` | Add the actual class name to `cfg.candidates`, or change YOLO prompts |
| `detect_object pre-call failed: service not advertised` | yolo_world_rbnx not active (only matters for RPC path without bbox) | Check `rbnx caps yolo_world`; ensure Stage 4A is up first |
| MCP path returns "ROS thread not initialized" | on_init not yet completed | rbnx boot reports the actual blocker; check the package log |
| `/graspnet/grasps` silent even on detections | Mismatched topic name, or rclpy thread crashed | Look for "rclpy thread exited" in logs; verify `grasps_topic` cfg matches the C++ subscriber |

## Coupling with neighbors

* **Upstream** yolo_world_rbnx (Stage 4A) — provides
  `/yolo/detect_object` and the atlas MCP `object_detect/detect_object`.
  yolo_grasp_rbnx calls it as a CLIENT.
* **Downstream** piper_moveit_rbnx (Stage 5) — owns the C++
  `piper_moveit_control` subscriber on `/graspnet/grasps`. Plus pick_skill_rbnx
  (Stage 6) will call `grasp_request` over MCP.

So the deploy ordering is:
```
yolo_world ── yolo_grasp ── piper_moveit ── pick_skill
```
