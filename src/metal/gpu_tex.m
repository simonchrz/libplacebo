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

/* Metal GPU backend — textures + buffers.
 *
 * Storage model: Apple unified memory. Host-accessed textures (host_writable/
 * host_readable/initial_data) and all buffers use MTLStorageModeShared, so
 * host transfers are direct CPU memcpy / replaceRegion / getBytes — no staging
 * buffer, no blit for uploads. GPU-only textures (renderer intermediates) use
 * MTLStorageModePrivate so they stay eligible for Apple's lossless texture
 * compression (Shared disables it → uncompressed bandwidth on every pass).
 * Command buffers only enter the picture once render passes write textures
 * (Phase 5); a per-object pending command buffer is waited on before readback.
 *
 * MRC (no -fobjc-arc): created objects are +1 and explicitly released.
 */

#import <Metal/Metal.h>

#include "gpu.h"

// Retain `cb` into *slot, releasing whatever was there. cb may be nil.
// (The commit-aware wait counterpart, mtl_wait_pending, lives in gpu.m — it
// needs the gpu handle to commit the shared frame CB before waiting on it.)
void mtl_set_pending(void **slot, id<MTLCommandBuffer> cb)
{
    if (*slot)
        [(__bridge id<MTLCommandBuffer>) *slot release];
    *slot = (__bridge void *) cb;
    if (cb)
        [cb retain];
}

static int tex_dimensions(const struct pl_tex_params *p)
{
    return p->d ? 3 : (p->h ? 2 : 1);
}

// ---------------------------------------------------------------- buffers

void mtl_buf_destroy(pl_gpu gpu, pl_buf buf)
{
    struct pl_buf_metal *bp = PL_PRIV(buf);
    mtl_wait_pending(gpu, &bp->pending_cb);
    if (bp->buf)
        [(__bridge id<MTLBuffer>) bp->buf release];
    pl_free((void *) buf);
    (void) gpu;
}

pl_buf mtl_buf_create(pl_gpu gpu, const struct pl_buf_params *params)
{
    struct pl_buf_t *buf = pl_zalloc_obj(NULL, buf, struct pl_buf_metal);
    buf->params = *params;
    buf->params.initial_data = NULL;
    struct pl_buf_metal *bp = PL_PRIV(buf);

    NSUInteger len = PL_MAX(params->size, 1);
    MTLResourceOptions opts = MTLResourceStorageModeShared;
    id<MTLBuffer> mbuf;
    if (params->initial_data) {
        mbuf = [mtl_device(gpu) newBufferWithBytes:params->initial_data
                                              length:len
                                             options:opts];
    } else {
        mbuf = [mtl_device(gpu) newBufferWithLength:len options:opts];
    }
    if (!mbuf) {
        PL_ERR(gpu, "metal: newBuffer(%zu) failed", (size_t) len);
        pl_free((void *) buf);
        return NULL;
    }
    bp->buf = (__bridge void *) mbuf;   // owns the +1 from newBuffer

    if (params->host_mapped)
        buf->data = [mbuf contents];

    return buf;
}

void mtl_buf_write(pl_gpu gpu, pl_buf buf, size_t offset,
                     const void *data, size_t size)
{
    struct pl_buf_metal *bp = PL_PRIV(buf);
    id<MTLBuffer> mbuf = (__bridge id<MTLBuffer>) bp->buf;
    // Shared storage is host-coherent on Apple silicon — direct write.
    memcpy((uint8_t *) [mbuf contents] + offset, data, size);
    (void) gpu;
}

bool mtl_buf_read(pl_gpu gpu, pl_buf buf, size_t offset,
                    void *dest, size_t size)
{
    struct pl_buf_metal *bp = PL_PRIV(buf);
    mtl_wait_pending(gpu, &bp->pending_cb);
    id<MTLBuffer> mbuf = (__bridge id<MTLBuffer>) bp->buf;
    memcpy(dest, (uint8_t *) [mbuf contents] + offset, size);
    (void) gpu;
    return true;
}

void mtl_buf_copy(pl_gpu gpu, pl_buf dst, size_t dst_offset,
                    pl_buf src, size_t src_offset, size_t size)
{
    struct pl_buf_metal *sp = PL_PRIV(src);
    struct pl_buf_metal *dp = PL_PRIV(dst);
    mtl_wait_pending(gpu, &sp->pending_cb);
    id<MTLBuffer> s = (__bridge id<MTLBuffer>) sp->buf;
    id<MTLBuffer> d = (__bridge id<MTLBuffer>) dp->buf;
    memcpy((uint8_t *) [d contents] + dst_offset,
           (uint8_t *) [s contents] + src_offset, size);
    (void) gpu;
}

bool mtl_buf_poll(pl_gpu gpu, pl_buf buf, uint64_t timeout)
{
    struct pl_buf_metal *bp = PL_PRIV(buf);
    if (!bp->pending_cb)
        return false;
    if (timeout > 0) {
        // Blocking poll — commit-aware (waiting on the still-open frame CB
        // would deadlock).
        mtl_wait_pending(gpu, &bp->pending_cb);
        return false;
    }
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>) bp->pending_cb;
    MTLCommandBufferStatus st = cb.status;
    if (st == MTLCommandBufferStatusCompleted || st == MTLCommandBufferStatusError) {
        [cb release];
        bp->pending_cb = NULL;
        return false;
    }
    return true;   // still in flight (incl. recorded-but-uncommitted frame CB)
}

// --------------------------------------------------------------- textures

void mtl_tex_destroy(pl_gpu gpu, pl_tex tex)
{
    struct pl_tex_metal *tp = PL_PRIV(tex);
    mtl_wait_pending(gpu, &tp->pending_cb);
    if (tp->tex)
        [(__bridge id<MTLTexture>) tp->tex release];
    pl_free((void *) tex);
    (void) gpu;
}

pl_tex mtl_tex_create(pl_gpu gpu, const struct pl_tex_params *params)
{
    struct pl_tex_t *tex = pl_zalloc_obj(NULL, tex, struct pl_tex_metal);
    tex->params = *params;
    tex->params.initial_data = NULL;
    tex->sampler_type = PL_SAMPLER_NORMAL;
    struct pl_tex_metal *tp = PL_PRIV(tex);

    int dims = tex_dimensions(params);

    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    desc.pixelFormat = mtl_fmt(params->format);
    desc.textureType = dims == 1 ? MTLTextureType1D :
                       dims == 3 ? MTLTextureType3D : MTLTextureType2D;
    desc.width  = PL_MAX(params->w, 1);
    desc.height = PL_MAX(params->h, 1);
    desc.depth  = PL_MAX(params->d, 1);
    desc.mipmapLevelCount = 1;
    // GPU-only textures use Private storage: on Apple GPUs, host-visible
    // (Shared) textures are ineligible for lossless texture compression, so
    // every intermediate FBO the renderer creates would be written and
    // re-sampled uncompressed — pure bandwidth waste. Only textures the host
    // actually touches (uploads via replaceRegion, downloads via getBytes,
    // creation-time initial_data) need Shared.
    bool host_access = params->host_writable || params->host_readable ||
                       params->initial_data;
    desc.storageMode = host_access ? MTLStorageModeShared
                                   : MTLStorageModePrivate;

    MTLTextureUsage usage = 0;
    if (params->sampleable)
        usage |= MTLTextureUsageShaderRead;
    if (params->renderable)
        usage |= MTLTextureUsageRenderTarget;
    if (params->storable)
        usage |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.usage = usage ? usage : MTLTextureUsageShaderRead;

    id<MTLTexture> mtex = [mtl_device(gpu) newTextureWithDescriptor:desc];
    [desc release];
    if (!mtex) {
        PL_ERR(gpu, "metal: newTextureWithDescriptor (%dx%dx%d) failed",
               params->w, params->h, params->d);
        pl_free((void *) tex);
        return NULL;
    }
    tp->tex = (__bridge void *) mtex;   // owns the +1

    if (params->initial_data) {
        struct pl_tex_transfer_params ul = {
            .tex = tex,
            .ptr = (void *) params->initial_data,
            .rc  = { 0, 0, 0, params->w, params->h, params->d },
        };
        if (!mtl_tex_upload(gpu, &ul)) {
            mtl_tex_destroy(gpu, tex);
            return NULL;
        }
    }

    return tex;
}

void mtl_tex_invalidate(pl_gpu gpu, pl_tex tex)
{
    // Shared-storage contents are CPU-backed; nothing to discard. No-op until
    // we add transient/memoryless render targets in the swapchain phase.
    (void) gpu; (void) tex;
}

// Resolve row/image pitch defaults (pl normally fills these, but be safe).
static void xfer_pitches(const struct pl_tex_transfer_params *params,
                         int dims, size_t *row_pitch, size_t *img_pitch)
{
    pl_tex tex = params->tex;
    size_t texel = tex->params.format->texel_size;
    int w = pl_rect_w(params->rc);
    int h = dims >= 2 ? pl_rect_h(params->rc) : 1;
    *row_pitch = params->row_pitch   ? params->row_pitch   : (size_t) w * texel;
    *img_pitch = params->depth_pitch ? params->depth_pitch : (size_t) h * *row_pitch;
}

bool mtl_tex_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    pl_tex tex = params->tex;
    struct pl_tex_metal *tp = PL_PRIV(tex);
    id<MTLTexture> mtex = (__bridge id<MTLTexture>) tp->tex;
    int dims = tex_dimensions(&tex->params);

    // Make sure any in-flight GPU writes to this texture are visible before we
    // overwrite its backing store from the CPU.
    mtl_wait_pending(gpu, &tp->pending_cb);

    size_t row_pitch, img_pitch;
    xfer_pitches(params, dims, &row_pitch, &img_pitch);

    const uint8_t *src;
    if (params->buf) {
        struct pl_buf_metal *bp = PL_PRIV(params->buf);
        mtl_wait_pending(gpu, &bp->pending_cb);
        src = (uint8_t *) [(__bridge id<MTLBuffer>) bp->buf contents] + params->buf_offset;
    } else {
        src = params->ptr;
    }

    pl_rect3d rc = params->rc;
    MTLRegion region = {
        .origin = { rc.x0, dims >= 2 ? rc.y0 : 0, dims >= 3 ? rc.z0 : 0 },
        .size   = { pl_rect_w(rc), dims >= 2 ? pl_rect_h(rc) : 1,
                                   dims >= 3 ? pl_rect_d(rc) : 1 },
    };

    [mtex replaceRegion:region
            mipmapLevel:0
                  slice:0
              withBytes:src
            bytesPerRow:(dims == 1 ? 0 : row_pitch)
          bytesPerImage:(dims == 3 ? img_pitch : 0)];

    if (params->callback)
        params->callback(params->priv);   // transfer is synchronous (CPU)
    (void) gpu;
    return true;
}

bool mtl_tex_download(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    pl_tex tex = params->tex;
    struct pl_tex_metal *tp = PL_PRIV(tex);
    id<MTLTexture> mtex = (__bridge id<MTLTexture>) tp->tex;
    int dims = tex_dimensions(&tex->params);

    mtl_wait_pending(gpu, &tp->pending_cb);   // ensure GPU writes landed

    size_t row_pitch, img_pitch;
    xfer_pitches(params, dims, &row_pitch, &img_pitch);

    uint8_t *dst;
    if (params->buf) {
        struct pl_buf_metal *bp = PL_PRIV(params->buf);
        mtl_wait_pending(gpu, &bp->pending_cb);
        dst = (uint8_t *) [(__bridge id<MTLBuffer>) bp->buf contents] + params->buf_offset;
    } else {
        dst = params->ptr;
    }

    pl_rect3d rc = params->rc;
    MTLRegion region = {
        .origin = { rc.x0, dims >= 2 ? rc.y0 : 0, dims >= 3 ? rc.z0 : 0 },
        .size   = { pl_rect_w(rc), dims >= 2 ? pl_rect_h(rc) : 1,
                                   dims >= 3 ? pl_rect_d(rc) : 1 },
    };

    [mtex getBytes:dst
       bytesPerRow:(dims == 1 ? 0 : row_pitch)
     bytesPerImage:(dims == 3 ? img_pitch : 0)
        fromRegion:region
       mipmapLevel:0
             slice:0];

    if (params->callback)
        params->callback(params->priv);
    (void) gpu;
    return true;
}

void mtl_tex_clear_ex(pl_gpu gpu, pl_tex tex, const union pl_clear_color color)
{
    struct pl_tex_metal *tp = PL_PRIV(tex);
    id<MTLTexture> mtex = (__bridge id<MTLTexture>) tp->tex;

    // Pure clear via a render pass with loadAction=Clear and no draws — needs
    // no pipeline state. Works for any renderable format.
    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = mtex;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[0].clearColor =
        MTLClearColorMake(color.f[0], color.f[1], color.f[2], color.f[3]);

    id<MTLCommandBuffer> cb = mtl_frame_cb(gpu);
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
    [enc endEncoding];
    mtl_set_pending(&tp->pending_cb, cb);
}

void mtl_tex_blit(pl_gpu gpu, const struct pl_tex_blit_params *params)
{
    struct pl_tex_metal *sp = PL_PRIV(params->src);
    struct pl_tex_metal *dp = PL_PRIV(params->dst);
    id<MTLTexture> s = (__bridge id<MTLTexture>) sp->tex;
    id<MTLTexture> d = (__bridge id<MTLTexture>) dp->tex;

    pl_rect3d src_rc = params->src_rc, dst_rc = params->dst_rc;
    int sw = abs(pl_rect_w(src_rc)), dw = abs(pl_rect_w(dst_rc));
    int sh = abs(pl_rect_h(src_rc)), dh = abs(pl_rect_h(dst_rc));
    int sd = abs(pl_rect_d(src_rc)), dd = abs(pl_rect_d(dst_rc));

    // Scaling blits need a sampler+shader pass — deferred to the render phase.
    if (sw != dw || sh != dh || sd != dd) {
        PL_ERR(gpu, "metal: scaling pl_tex_blit not yet implemented "
                    "(src %dx%dx%d → dst %dx%dx%d)", sw, sh, sd, dw, dh, dd);
        return;
    }

    // Quelle muss fertig geschrieben sein. Innerhalb des offenen Frame-CB ist
    // die Reihenfolge Encoder-seitig garantiert (gleicher CB) — nur fremde,
    // bereits committete CBs brauchen den Wait.
    struct pl_gpu_metal *pg = PL_PRIV(gpu);
    if (sp->pending_cb && sp->pending_cb != pg->frame_cb)
        mtl_wait_pending(gpu, &sp->pending_cb);

    id<MTLCommandBuffer> cb = mtl_frame_cb(gpu);
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    [enc copyFromTexture:s
             sourceSlice:0
             sourceLevel:0
            sourceOrigin:MTLOriginMake(PL_MIN(src_rc.x0, src_rc.x1),
                                       PL_MIN(src_rc.y0, src_rc.y1),
                                       PL_MIN(src_rc.z0, src_rc.z1))
              sourceSize:MTLSizeMake(PL_MAX(sw, 1), PL_MAX(sh, 1), PL_MAX(sd, 1))
               toTexture:d
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(PL_MIN(dst_rc.x0, dst_rc.x1),
                                       PL_MIN(dst_rc.y0, dst_rc.y1),
                                       PL_MIN(dst_rc.z0, dst_rc.z1))];
    [enc endEncoding];
    mtl_set_pending(&dp->pending_cb, cb);
}
