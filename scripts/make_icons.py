"""Génère les icônes PWA (PNG) sans dépendance image : python scripts/make_icons.py"""

import struct
import zlib
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "app" / "static" / "icons"
BG = np.array([15, 23, 42], dtype=float)
LINE = np.array([34, 197, 94], dtype=float)
DOT = np.array([56, 189, 248], dtype=float)


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def render(size: int, margin: float) -> np.ndarray:
    ss = 3  # supersampling pour l'anti-crénelage
    n = size * ss
    ys, xs = np.mgrid[0:n, 0:n] / n
    img = np.tile(BG, (n, n, 1))
    pts = [(0.22, 0.68), (0.38, 0.52), (0.50, 0.60), (0.78, 0.30)]
    pts = [(0.5 + (x - 0.5) * (1 - 2 * margin), 0.5 + (y - 0.5) * (1 - 2 * margin)) for x, y in pts]
    width = 0.045 * (1 - 2 * margin)
    d = np.min([_seg_dist(xs, ys, *a, *b) for a, b in zip(pts[:-1], pts[1:], strict=True)], axis=0)
    img = np.where((d < width)[..., None], LINE, img)
    dot = (np.hypot(xs - pts[-1][0], ys - pts[-1][1]) < width * 1.7)[..., None]
    img = np.where(dot, DOT, img)
    return img.reshape(size, ss, size, ss, 3).mean(axis=(1, 3)).astype(np.uint8)


def write_png(path: Path, arr: np.ndarray) -> None:
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    write_png(OUT / "icon-192.png", render(192, 0.0))
    write_png(OUT / "icon-512.png", render(512, 0.0))
    write_png(OUT / "icon-maskable-512.png", render(512, 0.12))  # zone de sécurité Android
    write_png(OUT / "apple-touch-icon.png", render(180, 0.0))  # iOS applique son propre arrondi
    (OUT / "icon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="#0f172a"/>'
        '<polyline points="22,68 38,52 50,60 78,30" fill="none" stroke="#22c55e" stroke-width="4.5" '
        'stroke-linecap="round" stroke-linejoin="round"/><circle cx="78" cy="30" r="7" fill="#38bdf8"/></svg>\n'
    )
    print("icônes générées dans", OUT)
