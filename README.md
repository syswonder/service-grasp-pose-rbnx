# grasp_pose_rbnx

gRPC geometric grasp-pose estimator for the Piper + Orbbec vertical grasp
pipeline.

## What It Does

`grasp_pose_rbnx` owns:

```text
robonix/service/perception/grasp_pose/driver
robonix/service/perception/grasp_pose/grasp_request
```

It accepts a detector-provided RGB bbox and returns a vertical grasp pose in
`arm/base_link`. It does not publish `/graspnet/grasps`, does not host
`/graspnet/grasp_request`, and does not build or import `graspnet_msgs`.

Runtime path:

```text
pick_skill
  -> llm_detect.detect_object
  -> grasp_pose.grasp_request
  -> roboarm_ik.execute_grasp
```

## Algorithm

For each `grasp_request`:

1. Require `bbox_2d = [x_min, y_min, x_max, y_max]`.
2. Compute bbox center pixel `(u, v)`.
3. Project `[u, v, 1]` through the required 3x3 homography to get
   arm-plane `(x, y)`.
4. Apply optional global `bias_x` / `bias_y`.
5. Pick the gripper yaw: if the request carries a usable
   `orientation` label (`vertical` / `horizontal` / `diag_tlbr` /
   `diag_trbl`), the yaw is looked up directly from
   `_ORIENTATION_YAW_RAD`; otherwise (`unknown` / empty / unrecognized)
   fall back to the bbox-long-edge estimator
   `_gripper_angle_by_longer`.
6. Apply `catch_offset` along the yaw direction.
7. Use `default_desktop_height` as the final grasp z.
8. Return a vertical-down `PoseStamped` in `output_frame`, default
   `arm/base_link`.

### Orientation → yaw mapping

The convention is that yaw is the angle of the gripper's *closing line*
(between the two fingertips) w.r.t. the image x-axis, in `(-π/2, π/2]`.
The gripper closing line is placed **perpendicular** to the object's
principal axis, so:

| detector label | object long axis | gripper yaw |
| -------------- | ---------------- | ----------- |
| `horizontal`   | `—`              | `+π/2`      |
| `vertical`     | \|               | `0`         |
| `diag_tlbr`    | `\`              | `-π/4`      |
| `diag_trbl`    | `/`              | `+π/4`      |
| `unknown` / "" / other | —        | fallback: bbox long edge |

The server logs which branch fired: look for
`yaw=<value> (via orientation=…)` vs `yaw=<value> (via bbox_long_edge)`.
The success `message` field also echoes `yaw_via=…` for quick
inspection from the pick_skill side. The detector's own 180°
camera-to-arm remap is already applied upstream (in `llm_detect`), so
`grasp_pose` does not flip the label a second time.

Depth, camera intrinsics, and TF are intentionally not used.

## Required Config

Runtime config comes from `robonix_manifest.yaml`:

```yaml
hand_eye_calibration_file: /absolute/path/to/2d_homography.npy
default_desktop_height: -0.205
```

You may provide an inline `homography_matrix` instead of
`hand_eye_calibration_file`.

Useful optional knobs:

```yaml
bias_x: 0.0
bias_y: 0.0
catch_offset: 0.0
box_rotation_deg: 0.0
gripper_width_default: 0.04
output_frame: arm/base_link
approach_dist: 0.10
```

## Build / Run

```bash
bash scripts/build.sh
rbnx boot
```

`scripts/build.sh` runs `rbnx codegen` for protobuf/gRPC stubs.

## Debugging

Check:

```text
rbnx-boot/logs/grasp_pose.log
```

Common failure causes:

| symptom | likely cause |
|---|---|
| `bbox_2d is required` | caller did not pass the detection bbox |
| `hand_eye_calibration_file does not exist` | missing/stale 2D calibration file |
| `missing required roboarm config: default_desktop_height` | grasp height not configured |
| grasp XY wrong | stale homography or wrong camera/image orientation |
| z too high/low | incorrect `default_desktop_height` |
