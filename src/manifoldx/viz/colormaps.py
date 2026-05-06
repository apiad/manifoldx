"""1D colormap LUTs for sci-viz materials.

Each colormap is a 256-texel RGBA8 array (shape (256, 4), dtype uint8).
LUTs are precomputed from matplotlib at package-build time and frozen
here as hex strings to avoid a matplotlib runtime dependency.
"""
import numpy as np

# Precomputed from matplotlib 3.x: cm.get_cmap("viridis", 256)
# Run scripts/regenerate_colormaps.py to regenerate.
_VIRIDIS_HEX = (
    "440154ff440255ff440357ff450558ff45065aff45085bff46095cff460b5eff460c5fff460e61ff470f62ff471163ff471265ff471466ff471567ff471669ff47186aff48196bff481a6cff481c6eff481d6fff481e70ff482071ff482172ff482273ff482374ff472575ff472676ff472777ff472878ff472a79ff472b7aff472c7bff462d7cff462f7cff46307dff46317eff45327fff45347fff453580ff453681ff443781ff443982ff433a83ff433b83ff433c84ff423d84ff423e85ff424085ff414186ff414286ff404387ff404487ff3f4587ff3f4788ff3e4888ff3e4989ff3d4a89ff3d4b89ff3d4c89ff3c4d8aff3c4e8aff3b508aff3b518aff3a528bff3a538bff39548bff39558bff38568bff38578cff37588cff37598cff365a8cff365b8cff355c8cff355d8cff345e8dff345f8dff33608dff33618dff32628dff32638dff31648dff31658dff31668dff30678dff30688dff2f698dff2f6a8dff2e6b8eff2e6c8eff2e6d8eff2d6e8eff2d6f8eff2c708eff2c718eff2c728eff2b738eff2b748eff2a758eff2a768eff2a778eff29788eff29798eff287a8eff287a8eff287b8eff277c8eff277d8eff277e8eff267f8eff26808eff26818eff25828eff25838dff24848dff24858dff24868dff23878dff23888dff23898dff22898dff228a8dff228b8dff218c8dff218d8cff218e8cff208f8cff20908cff20918cff1f928cff1f938bff1f948bff1f958bff1f968bff1e978aff1e988aff1e998aff1e998aff1e9a89ff1e9b89ff1e9c89ff1e9d88ff1e9e88ff1e9f88ff1ea087ff1fa187ff1fa286ff1fa386ff20a485ff20a585ff21a685ff21a784ff22a784ff23a883ff23a982ff24aa82ff25ab81ff26ac81ff27ad80ff28ae7fff29af7fff2ab07eff2bb17dff2cb17dff2eb27cff2fb37bff30b47aff32b57aff33b679ff35b778ff36b877ff38b976ff39b976ff3bba75ff3dbb74ff3ebc73ff40bd72ff42be71ff44be70ff45bf6fff47c06eff49c16dff4bc26cff4dc26bff4fc369ff51c468ff53c567ff55c666ff57c665ff59c764ff5bc862ff5ec961ff60c960ff62ca5fff64cb5dff67cc5cff69cc5bff6bcd59ff6dce58ff70ce56ff72cf55ff74d054ff77d052ff79d151ff7cd24fff7ed24eff81d34cff83d34bff86d449ff88d547ff8bd546ff8dd644ff90d643ff92d741ff95d73fff97d83eff9ad83cff9dd93aff9fd938ffa2da37ffa5da35ffa7db33ffaadb32ffaddc30ffafdc2effb2dd2cffb5dd2bffb7dd29ffbade27ffbdde26ffbfdf24ffc2df22ffc5df21ffc7e01fffcae01effcde01dffcfe11cffd2e11bffd4e11affd7e219ffdae218ffdce218ffdfe318ffe1e318ffe4e318ffe7e419ffe9e419ffece41affeee51bfff1e51cfff3e51efff6e61ffff8e621fffae622fffde724ff"
)


def _decode_lut(hex_str: str) -> np.ndarray:
    """Decode a hex-encoded RGBA8 LUT into a (256, 4) uint8 array."""
    raw = bytes.fromhex(hex_str)
    if len(raw) != 256 * 4:
        raise ValueError(f"LUT must be 1024 bytes; got {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(256, 4).copy()


_LUTS = {
    "viridis": _decode_lut(_VIRIDIS_HEX),
}


def get_colormap(name: str) -> np.ndarray:
    """Return the (256, 4) uint8 LUT for the named colormap."""
    if name not in _LUTS:
        raise KeyError(f"unknown colormap: {name!r}; available: {sorted(_LUTS)}")
    return _LUTS[name]


def lookup(name: str, value: float) -> np.ndarray:
    """Sample the colormap at normalized value in [0, 1]; returns (4,) uint8."""
    lut = get_colormap(name)
    idx = int(np.clip(value, 0.0, 1.0) * 255 + 0.5)
    return lut[idx]
