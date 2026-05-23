#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Manual sourcing helper for yolo_grasp_rbnx — DEBUG / single-package use.
#
# Sources the colcon overlay built INSIDE this package
# (rbnx-build/ws/install/), which contains the VENDORED graspnet_msgs
# at src/graspnet_msgs/. Does NOT touch the upstream grasp/driver/
# graspnet workspace — Stage 4 vendored its own copy on purpose.
#
# Usage (in the SAME shell that will run `rbnx boot` or
# `python3 -m yolo_grasp.main`):
#     source scripts/dev_source.sh
#
# DO NOT run with `bash` — a child shell loses env on exit. Always:
#     . dev_source.sh
#     source dev_source.sh
#
# WARNING: This is a DEBUG / single-package helper. Do NOT put a
# `source ...` of this file into your ~/.bashrc — see the reentrancy
# guard below for why that previously caused an unbounded recursion
# (60s rbnx-boot registration timeout with hundreds of "package root:"
# lines in the log). `rbnx boot` itself uses scripts/start.sh which
# does its own sourcing chain — you don't need this helper unless
# you're running the package by hand outside of `rbnx boot`.
# -----------------------------------------------------------------------

# Refuse `bash this_file.sh` — only sourcing makes sense here.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERR: this file must be SOURCED, not executed." >&2
    echo "    source $0" >&2
    exit 1
fi

# ── Reentrancy guard ────────────────────────────────────────────────
# This script does $(rbnx path ...) which spawns a child bash. If the
# user accidentally puts `source dev_source.sh` into ~/.bashrc, every
# child bash re-reads .bashrc, which re-sources us, which spawns
# another child bash, ad infinitum (visible as N copies of "package
# root: ..." plus a 60s rbnx-boot registration timeout because the
# real python module never gets exec'd).
#
# (a) once-per-shell `_YOLO_GRASP_SOURCED` flag.
# (b) `_YOLO_GRASP_SOURCING_IN_PROGRESS` — bail out instantly inside
#     a child of an in-progress source so $(rbnx path ...) returns.
if [ -n "${_YOLO_GRASP_SOURCING_IN_PROGRESS:-}" ]; then
    return 0 2>/dev/null || exit 0
fi
if [ "${_YOLO_GRASP_SOURCED:-}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi
export _YOLO_GRASP_SOURCING_IN_PROGRESS=1

# ── Configurable paths ────────────────────────────────────────────────
: "${ROS_DISTRO:=humble}"

# yolo_grasp_rbnx package root. Three ways to find it:
#   1. operator overrides via $YOLO_GRASP_PKG (preferred)
#   2. THIS-script-relative — when sourced from inside the package.
#   3. auto-discover at standard rbnx-boot cache path / lab path / cwd.
_self_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -z "${YOLO_GRASP_PKG:-}" ]; then
    for cand in \
        "$_self_dir/.." \
        "$HOME/lhw/rbnx_piper_packages/rbnx-boot/cache/yolo_grasp" \
        "$HOME/rbnx_piper_packages/rbnx-boot/cache/yolo_grasp" \
        "$HOME/lab/packages/yolo_grasp_rbnx" \
        "$(pwd)"
    do
        cand_abs="$( cd "$cand" 2>/dev/null && pwd )"
        if [ -n "$cand_abs" ] && [ -f "$cand_abs/package_manifest.yaml" ] && \
           grep -q 'name: com.robonix.piper_grasp.yolo_grasp' "$cand_abs/package_manifest.yaml" 2>/dev/null; then
            YOLO_GRASP_PKG="$cand_abs"
            break
        fi
    done
fi
unset _self_dir cand_abs

if [ -z "${YOLO_GRASP_PKG:-}" ] || [ ! -d "$YOLO_GRASP_PKG" ]; then
    echo "[yolo_grasp-source] FATAL: cannot locate yolo_grasp_rbnx package." >&2
    echo "[yolo_grasp-source] Override with:  export YOLO_GRASP_PKG=/path/to/yolo_grasp_rbnx" >&2
    return 1 2>/dev/null || exit 1
fi
echo "[yolo_grasp-source] package root: $YOLO_GRASP_PKG"

OVERLAY="${YOLO_GRASP_PKG}/rbnx-build/ws/install/setup.bash"
if [ ! -f "$OVERLAY" ]; then
    echo "[yolo_grasp-source] FATAL: vendored colcon overlay missing:" >&2
    echo "[yolo_grasp-source]   $OVERLAY" >&2
    echo "[yolo_grasp-source] Build first:  bash $YOLO_GRASP_PKG/scripts/build.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# ── helper: skip set -u inside ROS setup files (they touch unset vars)
_yolo_chain_source() {
    local f="$1"
    if [ -f "$f" ]; then
        if [ -n "${YOLO_SOURCE_TRACE:-}" ]; then
            echo "[yolo_grasp-source] . $f"
        fi
        local _had_u=0
        case "$-" in *u*) _had_u=1; set +u ;; esac
        # shellcheck disable=SC1090
        . "$f"
        [ "$_had_u" = 1 ] && set -u
        return 0
    fi
    echo "[yolo_grasp-source] not found: $f" >&2
    return 1
}

# 1. Base ROS distro — required.
_yolo_chain_source "/opt/ros/${ROS_DISTRO}/setup.bash" || {
    echo "[yolo_grasp-source] FATAL: /opt/ros/${ROS_DISTRO} missing" >&2
    return 1 2>/dev/null || exit 1
}

# 2. Vendored colcon overlay — the WHOLE point of this script. Brings
#    in our own graspnet_msgs build (Python bindings under
#    rbnx-build/ws/install/graspnet_msgs/lib/python3.X/site-packages/).
_yolo_chain_source "$OVERLAY" || {
    echo "[yolo_grasp-source] FATAL: failed to source $OVERLAY" >&2
    return 1 2>/dev/null || exit 1
}

# 3. Codegen output — atlas_pb2 / grasp_mcp / etc. Match start.sh's
#    PYTHONPATH ordering so a manual `python3 -m yolo_grasp.main` from
#    this shell works the same as `rbnx boot` would.
CODEGEN_PROTO="${YOLO_GRASP_PKG}/rbnx-build/codegen/proto_gen"
CODEGEN_MCP="${YOLO_GRASP_PKG}/rbnx-build/codegen/robonix_mcp_types"
if [ -d "$CODEGEN_PROTO" ] && [ -d "$CODEGEN_MCP" ]; then
    export PYTHONPATH="$CODEGEN_PROTO:$CODEGEN_MCP:$YOLO_GRASP_PKG:${PYTHONPATH:-}"
else
    echo "[yolo_grasp-source] WARN: codegen output missing — only colcon" >&2
    echo "[yolo_grasp-source]   overlay sourced. \`rbnx boot\` will be fine" >&2
    echo "[yolo_grasp-source]   (its start.sh re-builds), but a manual" >&2
    echo "[yolo_grasp-source]   \`python3 -m yolo_grasp.main\` will fail." >&2
fi

# robonix-api on PYTHONPATH (same trick start.sh uses)
if ROBONIX_API="$(rbnx path robonix-api 2>/dev/null)"; then
    export PYTHONPATH="$ROBONIX_API:${PYTHONPATH:-}"
fi

# ── verify graspnet_msgs is importable from our overlay specifically ──
if python3 -c "import graspnet_msgs.srv; import graspnet_msgs.msg" 2>/dev/null; then
    PY_PATH=$(python3 -c "import graspnet_msgs.srv as s; print(s.__file__)")
    echo "[yolo_grasp-source] graspnet_msgs OK"
    echo "[yolo_grasp-source]   bindings: ${PY_PATH}"
    # Sanity check: the imported file should live UNDER our overlay,
    # not somewhere else (e.g. /opt/ros/humble/... or upstream graspnet
    # workspace). Warn if it resolved elsewhere.
    case "$PY_PATH" in
        "${YOLO_GRASP_PKG}"/rbnx-build/ws/install/*)
            echo "[yolo_grasp-source]   ✓ resolved to vendored copy"
            ;;
        *)
            echo "[yolo_grasp-source]   ⚠️  WARNING: bindings did NOT resolve to" >&2
            echo "[yolo_grasp-source]      our vendored overlay. Another" >&2
            echo "[yolo_grasp-source]      graspnet_msgs install is shadowing it" >&2
            echo "[yolo_grasp-source]      via AMENT_PREFIX_PATH ordering." >&2
            ;;
    esac
else
    echo "[yolo_grasp-source] FATAL: graspnet_msgs.srv not importable" >&2
    echo "[yolo_grasp-source] AMENT_PREFIX_PATH:" >&2
    printf '  %s\n' ${AMENT_PREFIX_PATH//:/ } >&2
    echo "[yolo_grasp-source] vendored install tree:" >&2
    find "${YOLO_GRASP_PKG}/rbnx-build/ws/install/graspnet_msgs" \
        -name '*.py' 2>&1 | head -10 >&2 || true
    return 1 2>/dev/null || exit 1
fi

# Also export so future rbnx-spawned providers (which inherit env) and
# the package's own start.sh fallback chain see this path. Idempotent —
# repeated sources must not keep growing this colon-list.
case ":${YOLO_GRASP_EXTRA_OVERLAYS:-}:" in
    *":${OVERLAY}:"*) ;;  # already present
    *) export YOLO_GRASP_EXTRA_OVERLAYS="${OVERLAY}:${YOLO_GRASP_EXTRA_OVERLAYS:-}" ;;
esac

unset _yolo_chain_source OVERLAY CODEGEN_PROTO CODEGEN_MCP PY_PATH
unset _YOLO_GRASP_SOURCING_IN_PROGRESS
export _YOLO_GRASP_SOURCED=1
echo "[yolo_grasp-source] done. You can now run \`rbnx boot\` or" \
     "\`python3 -m yolo_grasp.main\` in this shell."
