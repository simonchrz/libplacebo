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

/* Metal GPU backend for libplacebo — gpu object + pl_gpu_fns vtable.
 * pl_gpu_create_metal() owns the MTLDevice/queue and wires the pl_gpu_fns
 * vtable; the per-area implementations live in gpu_tex.m, gpu_pass.m,
 * formats.m and swapchain.m.
 */

#import <Metal/Metal.h>

#include <stdatomic.h>
#include <sched.h>

#include "gpu.h"

static const struct pl_gpu_fns pl_fns_metal;

void mtl_gpu_destroy(pl_gpu gpu)
{
    struct pl_gpu_metal *p = PL_PRIV(gpu);
    mtl_gpu_finish(gpu);   // flush + drain a possibly-open frame CB
    pl_spirv_destroy(&p->spirv);
    for (int s = 0; s < PL_TEX_SAMPLE_MODE_COUNT; s++)
        for (int a = 0; a < PL_TEX_ADDRESS_MODE_COUNT; a++)
            if (p->samplers[s][a])
                [(__bridge id<MTLSamplerState>) p->samplers[s][a] release];
    if (p->queue)  CFRelease(p->queue);
    if (p->device) CFRelease(p->device);
    pl_free((void *) gpu);
}

bool mtl_gpu_is_failed(pl_gpu gpu)
{
    struct pl_gpu_metal *p = PL_PRIV(gpu);
    return p->failed;
}

// ---- Shared per-frame command buffer --------------------------------------
//
// All GPU work of a frame (passes, clears, blits) is encoded into one shared
// command buffer instead of one CB per pass: one queue submit per frame, and
// Metal can pipeline the encoders back-to-back. The CB is committed at
// flush/finish, or earlier when host access must wait on in-frame work
// (mtl_wait_pending) — waiting on an uncommitted CB would deadlock.

id<MTLCommandBuffer> mtl_frame_cb(pl_gpu gpu)
{
    struct pl_gpu_metal *p = PL_PRIV(gpu);
    if (!p->frame_cb) {
        id<MTLCommandBuffer> cb = [mtl_queue(gpu) commandBuffer];
        [cb retain];
        p->frame_cb = (__bridge void *) cb;
    }
    return (__bridge id<MTLCommandBuffer>) p->frame_cb;
}

void mtl_frame_commit(pl_gpu gpu)
{
    struct pl_gpu_metal *p = PL_PRIV(gpu);
    if (!p->frame_cb)
        return;
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) p->frame_cb;
    [cb commit];
    [cb release];
    p->frame_cb = NULL;
}

void mtl_wait_pending(pl_gpu gpu, void **slot)
{
    if (!*slot)
        return;
    struct pl_gpu_metal *p = PL_PRIV(gpu);
    if (*slot == p->frame_cb)
        mtl_frame_commit(gpu);
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) *slot;
    [cb waitUntilCompleted];
    [cb release];
    *slot = NULL;
}

// Wait for all submitted GPU work to complete. Command buffers on our single
// queue execute in commit order, so an empty cb that we wait on is done only
// after everything committed before it.
void mtl_gpu_finish(pl_gpu gpu)
{
    mtl_frame_commit(gpu);
    id<MTLCommandBuffer> cb = [mtl_queue(gpu) commandBuffer];
    [cb commit];
    [cb waitUntilCompleted];
}

// Kick off all recorded work without blocking: commit the open frame CB.
void mtl_gpu_flush(pl_gpu gpu)
{
    mtl_frame_commit(gpu);
}

// ---- GPU timers ----------------------------------------------------------
//
// Metal exposes per-command-buffer GPU timestamps (GPUStartTime/GPUEndTime, in
// seconds) once the buffer has completed — no query pool needed. We attach a
// completion handler per timed pass that publishes the elapsed nanoseconds into
// a small lock-free ring; pl_timer_query drains it. Completion handlers run on
// an arbitrary thread, so the ring is a single-consumer / multi-producer SPSC-
// style buffer keyed on the slot value (0 == empty), and `pending` keeps the
// timer alive until every in-flight handler that captured it has run.

#define MTL_TIMER_RING 8

struct pl_timer_t {
    _Atomic(uint64_t) ring[MTL_TIMER_RING]; // elapsed ns; 0 == empty slot
    _Atomic(uint_fast32_t) widx;            // producer slot allocator
    uint_fast32_t ridx;                     // consumer cursor (single-threaded)
    _Atomic(int) pending;                   // handlers still referencing `timer`
};

static pl_timer mtl_timer_create(pl_gpu gpu)
{
    (void) gpu;
    pl_timer t = pl_zalloc_ptr(NULL, t);
    return t;
}

static void mtl_timer_destroy(pl_gpu gpu, pl_timer t)
{
    // A completion handler may still hold `t` — and seit dem Frame-CB-Batching
    // kann der Handler am noch UNCOMMITTETEN frame_cb hängen (committen, sonst
    // feuert er nie → Endlos-Spin). Danach drained der Loop wie gehabt.
    mtl_frame_commit(gpu);
    while (atomic_load_explicit(&t->pending, memory_order_acquire))
        sched_yield();
    pl_free(t);
}

static uint64_t mtl_timer_query(pl_gpu gpu, pl_timer t)
{
    (void) gpu;
    uint_fast32_t i = t->ridx % MTL_TIMER_RING;
    uint64_t ns = atomic_load_explicit(&t->ring[i], memory_order_acquire);
    if (!ns)
        return 0; // nothing ready at the read cursor
    atomic_store_explicit(&t->ring[i], 0, memory_order_relaxed); // mark consumed
    t->ridx++;
    return ns;
}

// Attach GPU timestamping for `timer` to `cb` (no-op if timer is NULL). Called
// from mtl_pass_run just before commit.
void mtl_timer_attach(pl_timer t, id<MTLCommandBuffer> cb)
{
    if (!t)
        return;
    atomic_fetch_add_explicit(&t->pending, 1, memory_order_relaxed);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull c) {
        double dt = c.GPUEndTime - c.GPUStartTime;
        uint64_t ns = dt > 0 ? (uint64_t) (dt * 1e9) : 1; // never publish 0
        uint_fast32_t i = atomic_fetch_add_explicit(&t->widx, 1,
                                                    memory_order_relaxed)
                          % MTL_TIMER_RING;
        atomic_store_explicit(&t->ring[i], ns, memory_order_release);
        atomic_fetch_sub_explicit(&t->pending, 1, memory_order_release);
    }];
}

// ---- texture / buffer impls live in gpu_tex.m, pass impls in gpu_pass.m ----

static const struct pl_gpu_fns pl_fns_metal = {
    .destroy        = mtl_gpu_destroy,
    .tex_create     = mtl_tex_create,
    .tex_destroy    = mtl_tex_destroy,
    .tex_invalidate = mtl_tex_invalidate,
    .tex_clear_ex   = mtl_tex_clear_ex,
    .tex_blit       = mtl_tex_blit,
    .tex_upload     = mtl_tex_upload,
    .tex_download   = mtl_tex_download,
    .buf_create     = mtl_buf_create,
    .buf_destroy    = mtl_buf_destroy,
    .buf_write      = mtl_buf_write,
    .buf_read       = mtl_buf_read,
    .buf_copy       = mtl_buf_copy,
    .buf_poll       = mtl_buf_poll,
    .desc_namespace = mtl_desc_namespace,
    .pass_create    = mtl_pass_create,
    .pass_destroy   = mtl_pass_destroy,
    .pass_run       = mtl_pass_run,
    .timer_create   = mtl_timer_create,
    .timer_destroy  = mtl_timer_destroy,
    .timer_query    = mtl_timer_query,
    .gpu_flush      = mtl_gpu_flush,
    .gpu_finish     = mtl_gpu_finish,
    .gpu_is_failed  = mtl_gpu_is_failed,
};

pl_gpu pl_gpu_create_metal(pl_log log, void *mtl_device, void *mtl_queue)
{
    struct pl_gpu_t *gpu = pl_zalloc_obj(NULL, gpu, struct pl_gpu_metal);
    gpu->log = log;

    struct pl_gpu_metal *p = PL_PRIV(gpu);
    p->impl   = pl_fns_metal;
    p->log    = log;
    p->device = mtl_device ? (void *) CFRetain(mtl_device) : NULL;
    p->queue  = mtl_queue  ? (void *) CFRetain(mtl_queue)  : NULL;

    // Minimal GLSL caps — Metal goes GLSL → (libplacebo spirv) → SPIRV-Cross MSL.
    // We emit Vulkan-flavoured GLSL, same as the vulkan backend.
    struct pl_glsl_version *glsl = &gpu->glsl;
    id<MTLDevice> dev = (__bridge id<MTLDevice>) mtl_device;

    glsl->version = 450;
    glsl->vulkan  = true;
    glsl->compute = true;
    // Compute limits. Apple GPUs: 1024 threads/threadgroup, 32 KiB threadgroup
    // memory (queried from the device).
    glsl->max_group_threads = 1024;
    glsl->max_group_size[0] = 1024;
    glsl->max_group_size[1] = 1024;
    glsl->max_group_size[2] = 1024;
    glsl->max_shmem_size = dev ? (size_t) dev.maxThreadgroupMemoryLength : (32 << 10);

    // Minimal device limits so pl_gpu_finalize() validates. Apple-silicon is
    // unified memory, so the whole buffer budget is host-mappable. Texture
    // dimension is 16384 on GPU family Apple4+ (A11/2017+), our deployment
    // floor. Refined against real MTLDevice queries in the caps phase.
    size_t max_buf = dev ? (size_t) dev.maxBufferLength : (256 << 20);
    struct pl_gpu_limits *lim = &gpu->limits;
    lim->max_tex_1d_dim   = 16384;
    lim->max_tex_2d_dim   = 16384;
    lim->max_tex_3d_dim   = 2048;
    lim->max_buf_size     = max_buf;
    lim->max_ubo_size     = max_buf;
    lim->max_ssbo_size    = max_buf;
    lim->max_vbo_size     = max_buf;
    lim->max_mapped_size  = max_buf;
    lim->max_mapped_vram  = max_buf;
    lim->align_tex_xfer_pitch   = 1;
    lim->align_tex_xfer_offset  = 1;
    lim->align_vertex_stride    = 4;
    lim->max_pushc_size         = 4096;   // Metal setBytes inline limit
    lim->max_variable_comps     = 0;      // no loose uniforms → pl uses pushc/UBO
    lim->compute_queues         = 1;
    lim->max_dispatch[0]        = 65535;  // threadgroups per grid dimension
    lim->max_dispatch[1]        = 65535;
    lim->max_dispatch[2]        = 65535;

    // GLSL → SPIR-V compiler (shaderc/glslang). We cross-compile the SPIR-V to
    // MSL ourselves via SPIRV-Cross in gpu_pass.m. SPIR-V 1.3 is broadly
    // supported by SPIRV-Cross's MSL backend.
    uint32_t spv_ver = PL_SPV_VERSION(1, 3);
    p->spirv = pl_spirv_create(log, (struct pl_spirv_version) {
        .env_version = pl_spirv_version_to_vulkan(spv_ver),
        .spv_version = spv_ver,
    });
    if (!p->spirv) {
        PL_ERR(gpu, "Failed creating SPIR-V compiler (no shaderc/glslang?)");
        mtl_gpu_destroy(gpu);
        return NULL;
    }

    mtl_setup_formats(gpu);

    PL_INFO(gpu, "Initialized Metal GPU backend");
    return pl_gpu_finalize(gpu);
}
