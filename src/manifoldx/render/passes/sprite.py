"""Sprite render pass — camera-facing point sprites with sphere imposters."""

import numpy as np
import wgpu

from manifoldx.viz.materials import ColormapMaterial


def render_sprite_batches(
    rp, engine, render_pass, sprite_batches, model_matrices, material_data
):
    """Render all sprite batches (PointCloud entities).

    Each batch is one (mat_id) group; geometry is always SPRITE_QUAD.
    Per-instance buffers: transforms, scalar_values, radii.
    """
    # Flatten all sprite local indices for single buffer upload
    all_local_indices = []  # ordered list across sprite batches
    batch_draw_info = {}  # mat_id -> (first_instance, instance_count)
    instance_offset = 0

    for mat_id, local_indices in sprite_batches.items():
        count = len(local_indices)
        batch_draw_info[mat_id] = (instance_offset, count)
        all_local_indices.extend(local_indices)
        instance_offset += count

    if not all_local_indices:
        return

    # Collect per-instance arrays
    ent_arr = np.asarray(all_local_indices, dtype=np.int64)

    # Transform matrices
    all_matrices = model_matrices[ent_arr]  # (total_instances, 16)
    all_matrices_t = all_matrices.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
    rp._sprite_batch_buffers.upload_transforms(all_matrices_t.astype(np.float32))

    # Scalar values (must be registered if we're rendering sprites)
    if "ScalarValue" in rp._store._components:
        # Get entity indices (not local indices) for component lookup
        alive_indices = np.where(rp._store._alive)[0]
        entity_indices = alive_indices[ent_arr]
        scalar_data = rp._store.get_component_data("ScalarValue", entity_indices)
        rp._sprite_batch_buffers.upload_scalar_values(scalar_data[:, 0].astype(np.float32))
    else:
        scalar_data = np.zeros((len(ent_arr),), dtype=np.float32)
        rp._sprite_batch_buffers.upload_scalar_values(scalar_data)

    # Radii
    if "Radius" in rp._store._components:
        alive_indices = np.where(rp._store._alive)[0]
        entity_indices = alive_indices[ent_arr]
        radius_data = rp._store.get_component_data("Radius", entity_indices)
        rp._sprite_batch_buffers.upload_radii(radius_data[:, 0].astype(np.float32))
    else:
        radius_data = np.ones((len(ent_arr),), dtype=np.float32)
        rp._sprite_batch_buffers.upload_radii(radius_data)

    # Get sprite quad geometry and create buffers if needed
    sprite_geom_id = engine._geometry_registry.get_id("sprite_quad")
    gpu_buffers = engine._geometry_registry.get_gpu_buffers(sprite_geom_id)
    if gpu_buffers is None:
        geom_obj = engine._geometry_registry.get(sprite_geom_id)
        if geom_obj is not None:
            gpu_buffers = engine._geometry_registry.create_buffers(
                sprite_geom_id, geom_obj, rp._device.queue
            )
    if gpu_buffers is None:
        return

    # ---------------------------------------------------------------
    # Draw each sprite batch
    # ---------------------------------------------------------------
    for mat_id, local_indices in sprite_batches.items():
        mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None
        if mat_obj is None:
            continue

        if not isinstance(mat_obj, ColormapMaterial):
            raise TypeError(
                f"sprite batch material must be ColormapMaterial; got {type(mat_obj).__name__}"
            )

        first_instance, instance_count = batch_draw_info[mat_id]

        # Create or fetch sprite pipeline
        pipeline, bind_group_layout = rp._get_or_create_pipeline(
            rp._device,
            engine._texture_format,
            sprite_geom_id,
            mat_obj,
            engine._material_registry,
            sprite=True,
        )

        # Upload material uniforms
        mat_data = mat_obj.get_data(instance_count, engine._material_registry)
        material_type = type(mat_obj).__name__
        material_subtype = getattr(mat_obj, "pipeline_subtype", None)
        bkey = (sprite_geom_id, material_type, material_subtype, True)
        mat_buffer = rp._material_buffers.get(bkey)
        if mat_buffer is not None:
            first_row = mat_data[0] if mat_data.ndim > 1 else mat_data
            rp._device.queue.write_buffer(
                mat_buffer, 0, first_row.astype(np.float32).tobytes()
            )

        # Build sprite bind group
        bind_group = make_sprite_bind_group(rp, bind_group_layout, mat_obj, mat_buffer)

        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group, [], 0, 0)
        render_pass.set_vertex_buffer(0, gpu_buffers["vertex_buffer"])
        render_pass.set_index_buffer(gpu_buffers["index_buffer"], wgpu.IndexFormat.uint32)
        render_pass.draw_indexed(
            gpu_buffers["index_count"],
            instance_count,
            first_index=0,
            base_vertex=0,
            first_instance=first_instance,
        )


def get_or_create_lut_texture(rp, device, material):
    """Create or retrieve a cached 1D LUT texture + sampler for a colormap."""
    cmap_name = material.cmap
    if cmap_name in rp._lut_textures:
        return rp._lut_textures[cmap_name]

    lut = material.get_lut()  # (256, 4) uint8 — matplotlib-encoded sRGB
    # Use rgba8unorm-srgb so the GPU sRGB-decodes on sample. The framebuffer
    # is also sRGB-encoded on write, so the round trip preserves the
    # author-intended display colors. Without -srgb the gamma curve gets
    # applied twice and colors come out brighter than matplotlib's swatch.
    texture = device.create_texture(
        size=(256, 1, 1),
        format=wgpu.TextureFormat.rgba8unorm_srgb,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        dimension=wgpu.TextureDimension.d1,
    )
    device.queue.write_texture(
        {"texture": texture},
        lut.tobytes(),
        {"bytes_per_row": 256 * 4, "rows_per_image": 1},
        (256, 1, 1),
    )
    sampler = device.create_sampler(
        address_mode_u=wgpu.AddressMode.clamp_to_edge,
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
    )
    rp._lut_textures[cmap_name] = (texture, sampler)
    return rp._lut_textures[cmap_name]


def make_sprite_bind_group(rp, bind_group_layout, material, mat_buffer):
    """Create bind group with bindings 0-6 for sprite rendering.

    Bindings:
        0: globals uniform (vp, view, camera_pos)
        1: transforms storage
        2: material uniform (vmin, vmax, lit_flag, _pad)
        3: scalar_values storage
        4: radii storage
        5: lut_texture (1D RGBA8)
        6: lut_sampler
    """
    lut_texture, lut_sampler = get_or_create_lut_texture(rp, rp._device, material)

    mat_buffer_size = 16  # 4 floats: vmin, vmax, lit_flag, _pad

    bind_group_entries = [
        {
            "binding": 0,
            "resource": {
                "buffer": rp._globals_buffer,
                "offset": 0,
                "size": 224,
            },
        },
        {
            "binding": 1,
            "resource": {
                "buffer": rp._sprite_batch_buffers.transforms_buf,
                "offset": 0,
                "size": rp._sprite_batch_buffers.transforms_capacity,
            },
        },
        {
            "binding": 2,
            "resource": {
                "buffer": mat_buffer,
                "offset": 0,
                "size": mat_buffer_size,
            },
        },
        {
            "binding": 3,
            "resource": {
                "buffer": rp._sprite_batch_buffers.scalar_values_buf,
                "offset": 0,
                "size": rp._sprite_batch_buffers.scalar_values_capacity,
            },
        },
        {
            "binding": 4,
            "resource": {
                "buffer": rp._sprite_batch_buffers.radii_buf,
                "offset": 0,
                "size": rp._sprite_batch_buffers.radii_capacity,
            },
        },
        {
            "binding": 5,
            "resource": lut_texture.create_view(),
        },
        {
            "binding": 6,
            "resource": lut_sampler,
        },
    ]

    return rp._device.create_bind_group(
        layout=bind_group_layout,
        entries=bind_group_entries,
    )
