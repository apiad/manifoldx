"""The swapchain prefers the sRGB variant and falls back to the plain format."""
from manifoldx.engine import _configure_swapchain


class _Ctx:
    def __init__(self, preferred, reject_srgb=False):
        self.preferred, self.reject_srgb, self.configured = preferred, reject_srgb, None

    def get_preferred_format(self, adapter):
        return self.preferred

    def configure(self, device, format):
        if self.reject_srgb and str(format).endswith("-srgb"):
            raise RuntimeError("format not supported by surface")
        self.configured = format


def test_srgb_preferred_is_kept():
    ctx = _Ctx("bgra8unorm-srgb")
    assert _configure_swapchain(ctx, None, None) == "bgra8unorm-srgb" == ctx.configured


def test_plain_preferred_is_upgraded_to_srgb():
    ctx = _Ctx("rgba8unorm")
    assert _configure_swapchain(ctx, None, None) == "rgba8unorm-srgb" == ctx.configured


def test_surface_without_srgb_falls_back_to_plain():
    ctx = _Ctx("bgra8unorm", reject_srgb=True)
    assert _configure_swapchain(ctx, None, None) == "bgra8unorm" == ctx.configured
