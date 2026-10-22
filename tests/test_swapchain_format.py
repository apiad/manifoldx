"""The swapchain prefers the sRGB variant and falls back to the plain format."""
import wgpu

from manifoldx.engine import _configure_swapchain

COPY_SRC = wgpu.TextureUsage.COPY_SRC


class _Ctx:
    def __init__(self, preferred, reject_srgb=False, reject_copy=False):
        self.preferred, self.reject_srgb, self.reject_copy = preferred, reject_srgb, reject_copy
        self.configured, self.usage = None, None

    def get_preferred_format(self, adapter):
        return self.preferred

    def configure(self, device, format, usage):
        if self.reject_srgb and str(format).endswith("-srgb"):
            raise RuntimeError("format not supported by surface")
        if self.reject_copy and usage & COPY_SRC:
            raise RuntimeError("usage not supported by surface")
        self.configured, self.usage = format, usage


def test_srgb_preferred_is_kept():
    ctx = _Ctx("bgra8unorm-srgb")
    assert _configure_swapchain(ctx, None, None) == "bgra8unorm-srgb" == ctx.configured


def test_plain_preferred_is_upgraded_to_srgb():
    ctx = _Ctx("rgba8unorm")
    assert _configure_swapchain(ctx, None, None) == "rgba8unorm-srgb" == ctx.configured


def test_surface_without_srgb_falls_back_to_plain():
    ctx = _Ctx("bgra8unorm", reject_srgb=True)
    assert _configure_swapchain(ctx, None, None) == "bgra8unorm" == ctx.configured


def test_the_frame_can_be_read_back_when_the_surface_allows_it():
    ctx = _Ctx("bgra8unorm")
    _configure_swapchain(ctx, None, None)
    assert ctx.usage & COPY_SRC


def test_surface_without_copy_src_still_draws_in_srgb():
    ctx = _Ctx("bgra8unorm", reject_copy=True)
    assert _configure_swapchain(ctx, None, None) == "bgra8unorm-srgb" == ctx.configured
    assert not ctx.usage & COPY_SRC
