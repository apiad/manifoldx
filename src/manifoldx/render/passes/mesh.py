"""Mesh render pass — instanced indexed draws batched by (geom_id, material_type)."""

import numpy as np
import wgpu


def render_mesh_batches(
    rp, engine, render_pass, mesh_batches, model_matrices, material_data
):
    """Render all mesh batches using instanced draw with shared transform buffer."""
    # ---------------------------------------------------------------
    # Upload ALL transforms at once (queue.write_buffer happens
    # before GPU processes the command buffer, so per-batch writes
    # would be overwritten by the last batch).
    # ---------------------------------------------------------------

    # Flatten batch order and record (first_instance, instance_count) per batch
    all_local_indices = []  # ordered list of local indices across all batches
    batch_draw_info = {}  # key -> (first_instance, instance_count)
    instance_offset = 0

    for key, local_indices in mesh_batches.items():
        count = len(local_indices)
        batch_draw_info[key] = (instance_offset, count)
        all_local_indices.extend(local_indices)
        instance_offset += count

    if not all_local_indices:
        return

    # Transpose all matrices for WGSL column-major layout and upload once
    all_matrices = model_matrices[all_local_indices]  # (total_instances, 16)
    all_matrices_t = all_matrices.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
    rp._batch_buffers.upload_transforms(all_matrices_t.astype(np.float32))

    # ---------------------------------------------------------------
    # Draw each batch using first_instance to index into the
    # shared transform buffer.
    # ---------------------------------------------------------------
    for (geom_id, mat_type, mat_subtype), local_indices in mesh_batches.items():
        first_instance, instance_count = batch_draw_info[(geom_id, mat_type, mat_subtype)]

        # Get GPU buffers for geometry
        gpu_buffers = engine._geometry_registry.get_gpu_buffers(geom_id)
        if gpu_buffers is None:
            geom_obj = engine._geometry_registry.get(geom_id)
            if geom_obj is None:
                continue
            gpu_buffers = engine._geometry_registry.create_buffers(
                geom_id, geom_obj, rp._device.queue
            )
            if gpu_buffers is None:
                continue

        # Get material and create/fetch pipeline
        mat_id = int(material_data[local_indices[0], 0]) if material_data is not None else 0
        mat_obj = engine._material_registry.get(mat_id) if mat_id > 0 else None

        if mat_obj is None:
            continue

        pipeline, bind_group_layout = rp._get_or_create_pipeline(
            rp._device,
            engine._texture_format,
            geom_id,
            mat_obj,
            engine._material_registry,
            geometry_buffers=gpu_buffers,
        )

        # Upload material uniforms for this batch
        mat_data = mat_obj.get_data(instance_count, engine._material_registry)
        bkey = (geom_id, mat_type, mat_subtype)
        mat_buffer = rp._material_buffers.get(bkey)
        if mat_buffer is not None:
            first_row = mat_data[0] if mat_data.ndim > 1 else mat_data
            rp._device.queue.write_buffer(
                mat_buffer, 0, first_row.astype(np.float32).tobytes()
            )

        # Build bind group
        needs_lights = "@binding(3)" in type(mat_obj)._compile()
        mat_buffer_size = 32 if needs_lights else 16
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
                    "buffer": rp._batch_buffers.transforms_buf,
                    "offset": 0,
                    "size": rp._batch_buffers.transforms_capacity,
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
        ]

        if needs_lights:
            bind_group_entries.append(
                {
                    "binding": 3,
                    "resource": {
                        "buffer": rp._lights_buffer,
                        "offset": 0,
                        "size": 128,
                    },
                }
            )

        # Append texture bindings declared by the material
        # (sampler at N, view at N+1 for each entry).
        texture_bindings = mat_obj.get_texture_bindings()
        for binding, handle in texture_bindings.items():
            bind_group_entries.append({
                "binding": binding,
                "resource": handle.sampler,
            })
            bind_group_entries.append({
                "binding": binding + 1,
                "resource": handle.view,
            })

        bind_group = rp._device.create_bind_group(
            layout=bind_group_layout,
            entries=bind_group_entries,
        )

        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.set_vertex_buffer(0, gpu_buffers["vertex_buffer"])
        render_pass.set_index_buffer(gpu_buffers["index_buffer"], wgpu.IndexFormat.uint32)

        # first_instance offsets into the shared transform buffer
        render_pass.draw_indexed(
            gpu_buffers["index_count"],
            instance_count,
            first_index=0,
            base_vertex=0,
            first_instance=first_instance,
        )
