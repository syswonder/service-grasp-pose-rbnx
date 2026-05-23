#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
#
# Start phase. Source ROS + colcon overlay + codegen PYTHONPATH, then
# exec the python module.
#
# Sourcing chain (each step adds to AMENT_PREFIX_PATH / PYTHONPATH /
# LD_LIBRARY_PATH; later wins on conflict for paths the user pinned):
#   1. /opt/ros/humble/setup.bash        — base ROS 2 distro
#   2. <pkg>/rbnx-build/ws/install/...   — vendored graspnet_msgs that
#                                          we colcon-built ourselves
#   3. $YOLO_GRASP_EXTRA_OVERLAYS        — colon-separated list of
#                                          EXTRA setup.bash files to
#                                          source (escape hatch when
#                                          the operator already has
#                                          a working graspnet_msgs
#                                          install elsewhere)
#   4. Auto-discover legacy upstream installs at standard paths
#      ($HOME/lhw/grasp/driver/graspnet/install, etc.) — only if
#      our own overlay didn't ship graspnet_msgs Python bindings.
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG"

ROS_DISTRO="${ROS_DISTRO:-humble}"
# shellcheck disable=SC1091
set +u; source "/opt/ros/${ROS_DISTRO}/setup.bash"; set -u

if [[ -f "$PKG/rbnx-build/ws/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    set +u; source "$PKG/rbnx-build/ws/install/setup.bash"; set -u
else
    echo "[yolo_grasp/start] ERR: colcon overlay missing — run scripts/build.sh" >&2
    exit 2
fi

# ── Direct PYTHONPATH / AMENT_PREFIX_PATH injection for graspnet_msgs ──
# Why: when this start.sh runs inside an outer shell that already has
# ANOTHER colcon overlay sourced (e.g. operator's ~/.bashrc sources
# /home/.../tracing_ws/install/setup.bash), colcon's idempotent prefix
# markers can cause our `source $PKG/rbnx-build/ws/install/setup.bash`
# above to silently NOT add our overlay's paths to AMENT_PREFIX_PATH /
# PYTHONPATH. The package then `import graspnet_msgs.srv` fails despite
# the build tree being fully populated. We side-step that by computing
# the two paths the colcon overlay would add for graspnet_msgs and
# prepending them ourselves, unconditionally. Idempotent — if colcon
# DID source correctly, the paths are merely duplicated, not corrupted.
GMSGS_PREFIX="$PKG/rbnx-build/ws/install/graspnet_msgs"
if [[ -d "$GMSGS_PREFIX" ]]; then
    case ":${AMENT_PREFIX_PATH:-}:" in
        *":${GMSGS_PREFIX}:"*) ;;
        *) export AMENT_PREFIX_PATH="${GMSGS_PREFIX}:${AMENT_PREFIX_PATH:-}" ;;
    esac
    for _site in \
        "$GMSGS_PREFIX"/local/lib/python*/dist-packages \
        "$GMSGS_PREFIX"/lib/python*/site-packages \
        "$GMSGS_PREFIX"/lib/python*/dist-packages
    do
        if [[ -d "$_site" ]]; then
            case ":${PYTHONPATH:-}:" in
                *":${_site}:"*) ;;
                *) export PYTHONPATH="${_site}:${PYTHONPATH:-}" ;;
            esac
        fi
    done
    unset _site
fi
unset GMSGS_PREFIX

# ── EXTRA OVERLAYS: explicit user override + auto-discovery ─────────
# The operator can prepend extra colon-separated setup.bash paths via
# YOLO_GRASP_EXTRA_OVERLAYS. We try them in order; missing files are
# skipped silently. After explicit overlays we do a small auto-probe
# for the historical `driver/graspnet/install/setup.bash` workspace
# the user mentioned in the migration thread, in case the vendored
# build is incomplete (e.g. a stale rbnx-build that pre-dates a
# graspnet_msgs schema change).
_source_overlay() {
    local f="$1"
    if [[ -f "$f" ]]; then
        echo "[yolo_grasp/start] sourcing extra overlay: $f" >&2
        # shellcheck disable=SC1090
        set +u; source "$f"; set -u
        return 0
    fi
    return 1
}

if [[ -n "${YOLO_GRASP_EXTRA_OVERLAYS:-}" ]]; then
    IFS=':' read -ra _extras <<< "$YOLO_GRASP_EXTRA_OVERLAYS"
    for f in "${_extras[@]}"; do _source_overlay "$f" || true; done
fi

# Auto-discovery fallback — only if our colcon-built graspnet_msgs is
# not actually importable (i.e. our build is broken or stale).
if ! python3 -c "import graspnet_msgs.srv" 2>/dev/null; then
    echo "[yolo_grasp/start] WARN: graspnet_msgs not importable from \
own overlay — trying fallback paths" >&2
    for f in \
        "$HOME/lhw/grasp/driver/graspnet/install/setup.bash" \
        "$HOME/grasp/driver/graspnet/install/setup.bash" \
        "/home/syswonder/lhw/grasp/driver/graspnet/install/setup.bash"
    do
        _source_overlay "$f" && break || true
    done
fi

# Final assertion — if graspnet_msgs is still not importable, fail
# loud BEFORE we hand off to python3 -m, because the rclpy thread will
# silently swallow this in a thread `Exception in thread`.
if ! python3 -c "import graspnet_msgs.srv" 2>&1 >/dev/null; then
    echo "[yolo_grasp/start] FATAL: cannot import graspnet_msgs.srv" >&2
    echo "[yolo_grasp/start] AMENT_PREFIX_PATH:" >&2
    printf '  %s\n' ${AMENT_PREFIX_PATH//:/ } >&2
    echo "[yolo_grasp/start] PYTHONPATH:" >&2
    printf '  %s\n' ${PYTHONPATH//:/ } >&2
    echo "[yolo_grasp/start] vendored install tree:" >&2
    find "$PKG/rbnx-build/ws/install/graspnet_msgs" -name '*.py' 2>&1 | head -10 >&2 || true
    exit 3
fi
echo "[yolo_grasp/start] graspnet_msgs OK: $(python3 -c \
'import graspnet_msgs.srv as s; print(s.__file__)')" >&2

CODEGEN_PROTO="$PKG/rbnx-build/codegen/proto_gen"
CODEGEN_MCP="$PKG/rbnx-build/codegen/robonix_mcp_types"
if [[ ! -d "$CODEGEN_PROTO" || ! -d "$CODEGEN_MCP" ]]; then
    echo "[yolo_grasp/start] ERR: codegen output missing — run scripts/build.sh" >&2
    exit 2
fi
export PYTHONPATH="$CODEGEN_PROTO:$CODEGEN_MCP:$PKG:${PYTHONPATH:-}"
if ROBONIX_API="$(rbnx path robonix-api 2>/dev/null)"; then
    export PYTHONPATH="$ROBONIX_API:$PYTHONPATH"
fi

exec python3 -u -m yolo_grasp.main
