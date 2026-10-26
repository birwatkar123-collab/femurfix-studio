"""Inject a surgical-sequence keyframe animation into an assembled GLB.

Trimesh cannot author glTF animations, so the exported assembly is post-
processed with pygltflib. Every animated node stays at its built (reduced)
placement; the clip starts from the displaced/retracted pose and converges to
the file pose, so an in-place viewer shows the final result at t=duration.

Clip "surgical_sequence" (LINEAR interpolation, TRS channels only):
  0.0-1.0 s   femur_distal translates/rotates from the displaced pose back
              into place (numbers mirror SimView's displacement slider).
  1.0-3.0 s   implant parts insert in order: nail first (axial), then
              lateral parts (screws/plate) staggered along -x.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
INTERP = "LINEAR"
TIMES = (0.0, 0.5, 1.0, 1.4, 1.9, 2.4, 3.0)  # shared keyframe grid


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


def _quat_y(angle_rad: float):
    half = angle_rad / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


def _smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


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

    n_t = len(TIMES)
    writer = _BlobWriter(gltf)
    try:  # does not exist until first save
        gltf.set_binary_blob(bytes(writer.blob))
    except Exception:
        pass

    # ---- times accessor (shared by every sampler) ----
    times_bv = writer.add(_pack_floats(TIMES))
    times_acc = Accessor(
        bufferView=times_bv,
        componentType=5126,  # FLOAT
        count=n_t,
        type="SCALAR",
        min=[min(TIMES)],
        max=[max(TIMES)],
    )
    gltf.accessors.append(times_acc)
    times_id = len(gltf.accessors) - 1

    # ---- distal reduction track: displaced pose -> identity ----
    ax, ay, az = float(displacement_mm), 1.5, -abs(0.3 * float(displacement_mm))
    ratios = [1.0, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0][:n_t]
    translations = [(ax * r, ay * r, az * r) for r in ratios]
    angles = [math.radians(angulation_deg) * r for r in ratios]
    quats = [_quat_y(a) for a in angles]

    anim = Animation(name=CLIP_NAME, channels=[], samplers=[])

    def add_channel(node: int, path: str, out_bv: int, count: int, comp_min, comp_max, vtype: str):
        out_acc = Accessor(
            bufferView=out_bv,
            componentType=5126,
            count=count,
            type=vtype,
            min=comp_min,
            max=comp_max,
        )
        gltf.accessors.append(out_acc)
        out_id = len(gltf.accessors) - 1
        anim.samplers.append(
            AnimationSampler(input=times_id, output=out_id, interpolation=INTERP)
        )
        anim.channels.append(
            AnimationChannel(
                sampler=len(anim.samplers) - 1,
                target=AnimationChannelTarget(node=node, path=path),
            )
        )

    if distal_idx is not None:
        tr_bv = writer.add(_pack_vec3(translations))
        rot_bv = writer.add(_pack_quat(quats))
        add_channel(
            distal_idx, "translation", tr_bv, n_t,
            [min(c[i] for c in translations) for i in range(3)],
            [max(c[i] for c in translations) for i in range(3)],
            "VEC3",
        )
        flat_min = [min(c[i] for c in quats) for i in range(4)]
        flat_max = [max(c[i] for c in quats) for i in range(4)]
        add_channel(
            distal_idx, "rotation", rot_bv, n_t,
            flat_min,
            flat_max,
            "VEC4",
        )

    # ---- implant insertion tracks: retracted -> identity, staggered ----
    for order, (idx, name, size) in enumerate(implant_indices):
        retract = _retract_for(name, size)
        start_t = 1.0 + 0.15 * order
        ramp = 0.5
        values = []
        for t in TIMES:
            alpha = _smoothstep((t - start_t) / ramp)
            values.append((retract[0] * (1 - alpha), retract[1] * (1 - alpha), retract[2] * (1 - alpha)))
        tr_bv = writer.add(_pack_vec3(values))
        add_channel(
            idx, "translation", tr_bv, n_t,
            [min(min(vals[i] for vals in values), 0.0) for i in range(3)],
            [max(max(vals[i] for vals in values), 0.0) for i in range(3)],
            "VEC3",
        )

    if not anim.channels:
        return False

    writer.flush()
    gltf.animations.append(anim)

    try:
        gltf.save(str(path))
        return True
    except Exception:
        return False
