"""Generate a 512x512 'blue-and-white china' albedo texture for the
teapot demo. Cobalt-blue floral spots on a warm cream background.

Reproducible: deterministic seed.
"""

import numpy as np
from PIL import Image
from pathlib import Path


SIZE = 512
SEED = 20260609
OUT_PATH = Path("examples/assets/teapot/teapot_albedo.png")


def main():
    rng = np.random.default_rng(SEED)

    # Background: warm cream (#f4ead8).
    img = np.full((SIZE, SIZE, 4), [244, 234, 216, 255], dtype=np.uint8)

    # Cobalt-blue floral spots.
    n_flowers = 28
    flower_color = np.array([29, 58, 138], dtype=np.float32)
    xs = rng.uniform(0, SIZE, size=n_flowers)
    ys = rng.uniform(0, SIZE, size=n_flowers)
    radii = rng.uniform(20, 55, size=n_flowers)

    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    for cx, cy, r in zip(xs, ys, radii):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        falloff = np.clip(1.0 - d2 / (r * r), 0.0, 1.0) ** 1.5
        ang = np.arctan2(yy - cy, xx - cx)
        petals = 0.6 + 0.4 * np.cos(6 * ang)
        mask = (falloff * petals)[..., None]
        img_f = img[..., :3].astype(np.float32)
        img_f = img_f * (1 - mask) + flower_color * mask
        img[..., :3] = np.clip(img_f, 0, 255).astype(np.uint8)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGBA").save(OUT_PATH)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
