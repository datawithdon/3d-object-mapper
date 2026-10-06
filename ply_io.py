"""Tiny, dependency-free ASCII PLY writer for colored point clouds."""


def write_ply(path: str, points: list) -> None:
    """Write `points` (list of [x, y, z, r, g, b]) to an ASCII PLY file."""
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in points:
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
