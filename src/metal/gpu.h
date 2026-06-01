/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

/* Metal GPU backend for libplacebo — internal header.
 * Shared types + internal entry points for the Metal pl_gpu backend.
 */
#pragma once

#import <Metal/Metal.h>

// common.h MUSS vor ../gpu.h kommen: es definiert PL_DEPRECATED_IN leer, bevor es
// <libplacebo/config.h> zieht (damit interner Code keine deprecation-Warnings
// bekommt). Zieht ../gpu.h zuerst config.h mit der echten __attribute__-Variante,
// gibt's eine Makro-Redefinition (unter --werror fatal, z.B. im macOS-CI-Job).
#include "../common.h"
#include "../gpu.h"
#include "../glsl/spirv.h"
#include <libplacebo/swapchain.h>

// Backend-private object. First member MUST be the pl_gpu_fns vtable so that
// PL_PRIV(gpu) can be cast to it (same convention as struct pl_gl).
struct pl_gpu_metal {
    struct pl_gpu_fns impl;
    pl_log log;
    void *device;   // id<MTLDevice>        (bridged, +1 retained)
    void *queue;    // id<MTLCommandQueue>  (bridged, +1 retained)
    pl_spirv spirv; // GLSL → SPIR-V (shaderc/glslang); SPIR-V → MSL is ours
    // Lazily-built MTLSamplerState cache, [sample_mode][address_mode].
    void *samplers[PL_TEX_SAMPLE_MODE_COUNT][PL_TEX_ADDRESS_MODE_COUNT];
    bool failed;
};

static inline id<MTLDevice> mtl_device(pl_gpu gpu)
{
    struct pl_gpu_metal *p = PL_PRIV(gpu);
    return (__bridge id<MTLDevice>) p->device;
}

static inline id<MTLCommandQueue> mtl_queue(pl_gpu gpu)
{
    struct pl_gpu_metal *p = PL_PRIV(gpu);
    return (__bridge id<MTLCommandQueue>) p->queue;
}

// Per-format private: the MTLPixelFormat. Attached to each pl_fmt via
// PL_PRIV(fmt). gpu_tex.m reads it to build MTLTextureDescriptors.
struct mtl_format_priv {
    MTLPixelFormat mtl;
};

static inline MTLPixelFormat mtl_fmt(pl_fmt fmt)
{
    const struct mtl_format_priv *fp = PL_PRIV(fmt);
    return fp->mtl;
}

// Per-texture private. Holds the MTLTexture (+1) and the last command buffer
// that touched it (for pl_tex_poll-style waits, +1 while pending).
struct pl_tex_metal {
    void *tex;          // id<MTLTexture>
    void *pending_cb;   // id<MTLCommandBuffer> or NULL
};

// Per-buffer private. MTLBuffer is Shared-storage (host-coherent on Apple
// unified memory), so pl_buf.data points straight at [buf contents].
struct pl_buf_metal {
    void *buf;          // id<MTLBuffer>
    void *pending_cb;   // id<MTLCommandBuffer> or NULL
};

// Per-pass private. One of render_pso / compute_pso is set.
struct pl_pass_metal {
    void *render_pso;       // id<MTLRenderPipelineState>
    void *compute_pso;      // id<MTLComputePipelineState>
    void *library;          // id<MTLLibrary> kept alive for the pass
    int push_constant_slot; // MSL [[buffer(N)]] for push consts, -1 if none
    int compute_threads[3]; // threadgroup size from GLSL local_size
};

// Internal creation entry point. `mtl_device` is an `id<MTLDevice>` opaque
// pointer (bridged from ObjC). Returns a finalized pl_gpu or NULL on failure.
pl_gpu pl_gpu_create_metal(pl_log log, void *mtl_device, void *mtl_queue);

// Create a swapchain over a CAMetalLayer (`layer` is a bridged CAMetalLayer*).
pl_swapchain pl_metal_swapchain_create(pl_gpu gpu, void *layer);

// Format table setup (formats.m). Takes the mutable in-construction gpu.
void    mtl_setup_formats(struct pl_gpu_t *gpu);

// Retain `cb` into *slot, releasing whatever was there (gpu_tex.m). Used to
// mark an object as having in-flight GPU work that readback must wait on.
void    mtl_set_pending(void **slot, id<MTLCommandBuffer> cb);

// pl_gpu_fns slot implementations (defined across gpu.m / gpu_tex.m / gpu_pass.m).
void    mtl_gpu_destroy(pl_gpu);
bool    mtl_gpu_is_failed(pl_gpu);
pl_tex  mtl_tex_create(pl_gpu, const struct pl_tex_params *);
void    mtl_tex_destroy(pl_gpu, pl_tex);
void    mtl_tex_invalidate(pl_gpu, pl_tex);
void    mtl_tex_clear_ex(pl_gpu, pl_tex, const union pl_clear_color);
void    mtl_tex_blit(pl_gpu, const struct pl_tex_blit_params *);
bool    mtl_tex_upload(pl_gpu, const struct pl_tex_transfer_params *);
bool    mtl_tex_download(pl_gpu, const struct pl_tex_transfer_params *);
pl_buf  mtl_buf_create(pl_gpu, const struct pl_buf_params *);
void    mtl_buf_destroy(pl_gpu, pl_buf);
void    mtl_buf_write(pl_gpu, pl_buf, size_t offset, const void *src, size_t size);
bool    mtl_buf_read(pl_gpu, pl_buf, size_t offset, void *dst, size_t size);
void    mtl_buf_copy(pl_gpu, pl_buf dst, size_t dst_offset,
                       pl_buf src, size_t src_offset, size_t size);
bool    mtl_buf_poll(pl_gpu, pl_buf, uint64_t timeout);
int     mtl_desc_namespace(pl_gpu, enum pl_desc_type type);
pl_pass mtl_pass_create(pl_gpu, const struct pl_pass_params *);
void    mtl_pass_destroy(pl_gpu, pl_pass);
void    mtl_pass_run(pl_gpu, const struct pl_pass_run_params *);
void    mtl_gpu_finish(pl_gpu);
void    mtl_gpu_flush(pl_gpu);

// Attach GPU timestamping for `timer` to `cb` before commit (no-op if NULL).
// Defined in gpu.m alongside the timer ring; called from mtl_pass_run.
void    mtl_timer_attach(pl_timer timer, id<MTLCommandBuffer> cb);
