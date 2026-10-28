import struct
from pathlib import Path
from pygltflib import GLTF2

path = Path(__file__).resolve().parents[2] / "backend" / "outputs" / "e2e_assembly.glb"
g = GLTF2().load(str(path))
a = g.animations[-1]
print("animations:", len(g.animations), "name:", a.name, "channels:", len(a.channels))


def read_acc(i):
    acc = g.accessors[i]
    bv = g.bufferViews[acc.bufferView]
    blob = g.binary_blob()
    off = bv.byteOffset
    n = acc.count * {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc.type]
    return list(struct.unpack("<%df" % n, blob[off : off + 4 * n]))


for ch in a.channels:
    s = a.samplers[ch.sampler]
    node = g.nodes[ch.target.node]
    t = [round(x, 2) for x in read_acc(s.input)]
    v = read_acc(s.output)
    if ch.target.path == "translation":
        rows = [tuple(round(x, 3) for x in v[k : k + 3]) for k in range(0, len(v), 3)]
        print(f"node={node.name!r} path=translation times={t}")
        print("   vals:", rows)
    else:
        print(f"node={node.name!r} path=rotation times={t}")
