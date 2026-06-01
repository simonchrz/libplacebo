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

/* Metal GPU backend — swapchain (CAMetalLayer present).
 *
 * start_frame vends the next CAMetalDrawable and wraps its texture as a
 * renderable pl_tex; the caller renders into it via the normal pass path;
 * swap_buffers presents the drawable on the queue.
 *
 * MRC (no -fobjc-arc): drawable/layer are explicitly retained/released.
 */

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <libplacebo/metal.h>

#include "gpu.h"
#include "../swapchain.h"

struct pl_sw_metal {
    struct pl_sw_fns impl;
    void *layer;      // CAMetalLayer            (+1 retained)
    void *drawable;   // id<CAMetalDrawable>     current frame, or NULL
    pl_tex fbo;       // wrapped drawable texture, or NULL
};

// Wrap an externally-owned MTLTexture as a renderable pl_tex. We retain it so
// the standard mtl_tex_destroy (which releases) stays balanced.
static pl_tex mtl_wrap_tex(pl_gpu gpu, id<MTLTexture> mtex, pl_fmt fmt)
{
    struct pl_tex_t *tex = pl_zalloc_obj(NULL, tex, struct pl_tex_metal);
    tex->params = (struct pl_tex_params) {
        .w = (int) mtex.width,
        .h = (int) mtex.height,
        .format = fmt,
        .renderable = true,
        .blit_dst = true,   // allow pl_tex_clear / blits onto the drawable
    };
    tex->sampler_type = PL_SAMPLER_NORMAL;
    struct pl_tex_metal *tp = PL_PRIV(tex);
    tp->tex = (__bridge void *) [mtex retain];
    (void) gpu;
    return tex;
}

static void mtl_sw_destroy(pl_swapchain sw)
{
    struct pl_sw_metal *p = PL_PRIV(sw);
    if (p->fbo)
        pl_tex_destroy(sw->gpu, &p->fbo);
    if (p->drawable)
        [(__bridge id<CAMetalDrawable>) p->drawable release];
    if (p->layer)
        [(__bridge CAMetalLayer *) p->layer release];
    pl_free((void *) sw);
}

static int mtl_sw_latency(pl_swapchain sw)
{
    struct pl_sw_metal *p = PL_PRIV(sw);
    CAMetalLayer *layer = (__bridge CAMetalLayer *) p->layer;
    return (int) layer.maximumDrawableCount;   // typically 3 (triple-buffered)
}

static bool mtl_sw_resize(pl_swapchain sw, int *width, int *height)
{
    struct pl_sw_metal *p = PL_PRIV(sw);
    CAMetalLayer *layer = (__bridge CAMetalLayer *) p->layer;
    CGSize sz = layer.drawableSize;
    int w = PL_DEF(*width, (int) sz.width);
    int h = PL_DEF(*height, (int) sz.height);
    if (w != (int) sz.width || h != (int) sz.height)
        layer.drawableSize = CGSizeMake(w, h);
    *width = w;
    *height = h;
    return true;
}

static bool mtl_sw_start_frame(pl_swapchain sw, struct pl_swapchain_frame *out)
{
    struct pl_sw_metal *p = PL_PRIV(sw);
    if (p->drawable) {
        PL_ERR(sw, "start_frame called with a frame already in progress");
        return false;
    }
    CAMetalLayer *layer = (__bridge CAMetalLayer *) p->layer;
    id<CAMetalDrawable> drawable = [layer nextDrawable];
    if (!drawable) {
        PL_ERR(sw, "CAMetalLayer vended no drawable");
        return false;
    }
    p->drawable = (__bridge void *) [drawable retain];

    pl_fmt fmt = pl_find_named_fmt(sw->gpu, "bgra8");
    p->fbo = mtl_wrap_tex(sw->gpu, drawable.texture, fmt);

    int bits = 8;
    *out = (struct pl_swapchain_frame) {
        .fbo = p->fbo,
        .flipped = false,
        .color_repr = {
            .sys = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_FULL,
            .alpha = PL_ALPHA_UNKNOWN,
            .bits = { .sample_depth = bits, .color_depth = bits },
        },
        .color_space = pl_color_space_srgb,
    };
    return true;
}

static bool mtl_sw_submit_frame(pl_swapchain sw)
{
    // Rendering went onto its own (already-committed) command buffers. We keep
    // the drawable + fbo until swap_buffers presents them.
    (void) sw;
    return true;
}

static void mtl_sw_swap_buffers(pl_swapchain sw)
{
    struct pl_sw_metal *p = PL_PRIV(sw);
    if (!p->drawable)
        return;
    id<CAMetalDrawable> drawable = (__bridge id<CAMetalDrawable>) p->drawable;

    // Present on the queue, after the render command buffers (committed earlier
    // on the same queue) complete.
    id<MTLCommandBuffer> cb = [mtl_queue(sw->gpu) commandBuffer];
    [cb presentDrawable:drawable];
    [cb commit];

    pl_tex_destroy(sw->gpu, &p->fbo);
    [drawable release];
    p->drawable = NULL;
}

static const struct pl_sw_fns pl_fns_metal_sw = {
    .destroy      = mtl_sw_destroy,
    .latency      = mtl_sw_latency,
    .resize       = mtl_sw_resize,
    .start_frame  = mtl_sw_start_frame,
    .submit_frame = mtl_sw_submit_frame,
    .swap_buffers = mtl_sw_swap_buffers,
};

pl_tex pl_metal_wrap_tex(pl_metal metal, void *mtl_texture)
{
    pl_gpu gpu = metal->gpu;
    id<MTLTexture> mtex = (__bridge id<MTLTexture>) mtl_texture;
    if (!mtex) {
        PL_ERR(gpu, "pl_metal_wrap_tex: NULL texture");
        return NULL;
    }
    MTLPixelFormat pf = mtex.pixelFormat;
    pl_fmt fmt = NULL;
    for (int i = 0; i < gpu->num_formats; i++) {
        if (mtl_fmt(gpu->formats[i]) == pf) {
            fmt = gpu->formats[i];
            break;
        }
    }
    if (!fmt) {
        PL_ERR(gpu, "pl_metal_wrap_tex: no pl_fmt for MTLPixelFormat %lu",
               (unsigned long) pf);
        return NULL;
    }
    return mtl_wrap_tex(gpu, mtex, fmt);
}

pl_swapchain pl_metal_swapchain_create(pl_gpu gpu, void *layer_ptr)
{
    CAMetalLayer *layer = (__bridge CAMetalLayer *) layer_ptr;
    if (!layer) {
        PL_ERR(gpu, "pl_metal_swapchain_create: NULL layer");
        return NULL;
    }
    layer.device = mtl_device(gpu);
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = NO;   // allow blit/readback off the drawable

    struct pl_swapchain_t *sw = pl_zalloc_obj(NULL, sw, struct pl_sw_metal);
    *sw = (struct pl_swapchain_t) { .log = gpu->log, .gpu = gpu };
    struct pl_sw_metal *p = PL_PRIV(sw);
    p->impl  = pl_fns_metal_sw;
    p->layer = (__bridge void *) [layer retain];
    return sw;
}
