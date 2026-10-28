"""Inject a surgical-sequence animation into an assembled GLB.

Trimesh cannot author glTF animations, so the exported assembly is post-
processed with pygltflib. Every animated node stays at its built (reduced)
placement; the clip starts from the displaced/retracted pose and converges to
the file pose, so an in-place viewer shows the final result at t=end.

The motion is physics-inspired (rigid fragment taken to contact, small
impaction, damped settle) but is authored as keyframes, not solved at runtime:

Reduction (femur_distal, 0.0-1.45 s):
  fragment approaches along its measured displacement vector with an
  ease-in/ease-out profile, crosses the anatomical seat slightly (-8%),
  rebounds a hair, and settles with a damped micro-oscillation. The rotation
  blends a yaw (angulation) with small pitch/roll for a non-planar path.

Instrumentation (implant parts, staggered from 1.5 s):
  each part accelerates toward its seat on its own keyframe grid, seats at
  m=0, recoils ~5% of the approach distance (impaction), and settles. Parts
  remain translated at identity afterwards.

SimView scrubs clip time linearly, so the eased keyframes read correctly both
in autoplay and when the reduction slider is dragged.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pygltflib import (
    Accessor,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    BufferView,
    GLTF2,
)

CLIP_NAME = "surgical_sequence"

# --- reduction profile (femur_distal) ---------------------------------------
# ratio 1.0 = fully displaced, 0.0 = anatomical seat, small negative = the
# fragment momentarily oscillates past the seat before settling.
REDUCTION_TIMES = (0.0, 0.30, 0.60, 0.85, 1.00, 1.20, 1.45, 3.0)
REDUCTION_RATIOS = (1.0, 0.90, 0.52, 0.08, -0.08, 0.05, 0.0, 0.0)

# --- instrumentation profile (per implant node) -----------------------------
IMPLANT_START = 1.5     # starts after reduction settles (~1.45 s)
IMPLANT_RAMP = 0.55     # approach window
RECOIL_WINDOW = 0.5     # impaction rebound window
RECOIL_DEPTH = 0.05     # fraction of the approach distance bounced past seat
APPROACH_T = (0.0, 0.35, 0.65, 0.85, 1.0)
APPROACH_M = (1.0, 0.78, 0.42, 0.10, 0.0)


class _BlobWriter:
    """Append raw data to the GLB binary chunk as aligned buffer views."""

    def __init__(self, gltf: GLTF2) -> None:
        self.gltf = gltf
        blob = bytearray(gltf.binary_blob())
        while len(blob) % 4:
            blob.append(0)
        self.blob = blob

    def add(self, data: bytes) -> int:
        start = len(self.blob)
        self.blob.extend(data)
        self.gltf.bufferViews.append(
            BufferView(buffer=0, byteOffset=start, byteLength=len(data))
        )
        return len(self.gltf.bufferViews) - 1

    def flush(self) -> None:
        self.gltf.set_binary_blob(bytes(self.blob))
        if self.gltf.buffers:
            self.gltf.buffers[0].byteLength = len(self.blob)


def _pack_floats(values: Sequence[float]) -> bytes:
    return b"".join(struct.pack("<f", v) for v in values)


def _pack_vec3(rows: Sequence[Sequence[float]]) -> bytes:
    return b"".join(struct.pack("<3f", *row) for row in rows)


def _pack_quat(rows: Sequence[Sequence[float]]) -> bytes:
    return b"".join(struct.pack("<4f", *row) for row in rows)


def _smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _quat_axis(axis: Tuple[float, float, float], angle_rad: float):
    half = angle_rad / 2.0
    s = math.sin(half)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half))


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _reduction_quat(ratio: float, angulation_deg: float):
    """Non-planar fragment path: dominant yaw plus small pitch/roll coupling."""
    yaw = math.radians(angulation_deg) * ratio
    return _quat_mul(
        _quat_axis((1.0, 0.0, 0.0), yaw * 0.35),
        _quat_mul(
            _quat_axis((0.0, 1.0, 0.0), yaw),
            _quat_axis((0.0, 0.0, 1.0), -yaw * 0.18),
        ),
    )


def _implant_track(start: float, net_end: float = 3.4) -> Tuple[List[float], List[float]]:
    """(times, scalar multipliers) from fully retracted (1) to seated (0).

    Approaches with an ease profile, seats, recoils an impaction fraction of
    the approach distance, and settles to 0 (damped, one hump).
    """
    times = [start + IMPLANT_RAMP * t for t in APPROACH_T]
    values = list(APPROACH_M)
    base = start + IMPLANT_RAMP
    for c in (0.35, 0.7, 1.0):
        times.append(base + RECOIL_WINDOW * c)
        values.append(-RECOIL_DEPTH * math.sin(math.pi * c))
    # Explicit hold so a longer clip never reveals probe state.
    times.append(max(times[-1] + 0.01, net_end))
    values.append(0.0)
    return times, values


def _mesh_bounds(gltf: GLTF2, node_index: int) -> Optional[tuple]:
    node = gltf.nodes[node_index]
    if node.mesh is None:
        return None
    mesh = gltf.meshes[node.mesh]
    prim = mesh.primitives[0]
    pos = prim.attributes.POSITION
    acc = gltf.accessors[pos]
    if acc.min is None or acc.max is None:
        return None
    lo, hi = tuple(acc.min), tuple(acc.max)
    size = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
    return size


def _node_index_by_name(gltf: GLTF2, name: str) -> Optional[int]:
    for i, node in enumerate(gltf.nodes):
        if node.name == name:
            return i
    return None


def _retract_for(name: str, size: tuple[float, float, float]) -> tuple[float, float, float]:
    """Start offset a part should sit at before inserting."""
    lowered = name.lower()
    if "nail" in lowered and "screw" not in lowered:
        # axial device: pulled out toward the proximal entry (+z in the
        # canonical femur frame, distal at z=0, tip at z>0)
        reach = max(size[2] * 1.1, 180.0)
        return (0.0, 0.0, reach)
    # lateral family (plates, screws, lag screws) enters from the -x side
    reach = max(size[0] * 2.5, 70.0)
    reach = min(reach, 160.0)
    return (-reach, 0.0, 0.0)


def bake_sequence(
    glb_path: str | Path,
    *,
    device_id: str,
    insertion: str,
    bone_nodes: List[str],
    implant_nodes: List[str],
    displacement_mm: float = 6.0,
    angulation_deg: float = 3.5,
) -> bool:
    """Add the sequence clip. Returns True on success; failures never raise."""
    path = Path(glb_path)
    try:
        gltf = GLTF2().load(str(path))
    except Exception:
        return False

    distal_name = next((n for n in bone_nodes if "distal" in n.lower()), None)

    distal_idx = _node_index_by_name(gltf, distal_name) if distal_name else None
    implant_indices = []
    for name in implant_nodes:
        idx = _node_index_by_name(gltf, name)
        if idx is not None:
            implant_indices.append((idx, name, _mesh_bounds(gltf, idx)))
    if not implant_indices and distal_idx is None:
        return False

    writer = _BlobWriter(gltf)
    try:  # does not exist until first save
        gltf.set_binary_blob(bytes(writer.blob))
    except Exception:
        pass

    anim = Animation(name=CLIP_NAME, channels=[], samplers=[])

    def add_channel(node: int, path_key: str, times, values, vtype: str) -> None:
        t_bv = writer.add(_pack_floats(times))
        t_acc = Accessor(
            bufferView=t_bv,
            componentType=5126,  # FLOAT
            count=len(times),
            type="SCALAR",
            min=[min(times)],
            max=[max(times)],
        )
        gltf.accessors.append(t_acc)
        packed = _pack_quat(values) if vtype == "VEC4" else _pack_vec3(values)
        v_bv = writer.add(packed)
        width = 4 if vtype == "VEC4" else 3
        v_acc = Accessor(
            bufferView=v_bv,
            componentType=5126,
            count=len(values),
            type=vtype,
            min=[min(row[i] for row in values) for i in range(width)],
            max=[max(row[i] for row in values) for i in range(width)],
        )
        gltf.accessors.append(v_acc)
        anim.samplers.append(
            AnimationSampler(input=len(gltf.accessors) - 2, output=len(gltf.accessors) - 1)
        )
        anim.channels.append(
            AnimationChannel(
                sampler=len(anim.samplers) - 1,
                target=AnimationChannelTarget(node=node, path=path_key),
            )
        )

    # ---- distal reduction track: displaced pose -> contact -> settle ----
    if distal_idx is not None:
        ax, ay, az = float(displacement_mm), 1.5, -abs(0.3 * float(displacement_mm))
        translations = [(ax * r, ay * r, az * r) for r in REDUCTION_RATIOS]
        quats = [_reduction_quat(r, angulation_deg) for r in REDUCTION_RATIOS]
        add_channel(distal_idx, "translation", list(REDUCTION_TIMES), translations, "VEC3")
        add_channel(distal_idx, "rotation", list(REDUCTION_TIMES), quats, "VEC4")

    # ---- implant tracks: staggered approach + impaction recoil ----
    # Stagger the parts so the nail leads and screws follow on their own grids
    # even when the part count would overflow a hard 3.0 s end.
    n = len(implant_indices)
    budget = 3.0 - IMPLANT_START - IMPLANT_RAMP - RECOIL_WINDOW
    gap = min(0.2, max(0.06, budget / max(1, n - 1))) if n > 1 else 0.2
    for order, (idx, name, size) in enumerate(implant_indices):
        retract = _retract_for(name, size)
        times, scalars = _implant_track(IMPLANT_START + order * gap)
        values = [[retract[0] * m, retract[1] * m, retract[2] * m] for m in scalars]
        try:
            add_channel(idx, "translation", times, values, "VEC3")
        except Exception:
            continue

    if not anim.channels:
        return False

    writer.flush()
    gltf.animations.append(anim)

    try:
        gltf.save(str(path))
        return True
    except Exception:
        return False
