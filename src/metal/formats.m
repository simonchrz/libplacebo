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

/* Metal GPU backend — format table.
 * Maps MTLPixelFormat → pl_fmt. Per-format backend-private holds the
 * MTLPixelFormat.
 *
 * Most entries are component-uniform (every component the same byte size) so
 * texel_size / host_bits derive from comp_size. Packed formats (rgb10a2,
 * bgr565) carry an explicit per-component bit layout in `packed_bits`.
 */

#import <Metal/Metal.h>
#import <TargetConditionals.h>

#include "gpu.h"

struct mtl_fmt_desc {
    const char       *name;
    MTLPixelFormat    mtl;
    enum pl_fmt_type  type;
    int               num_comp;
    int               comp_size;    // bytes per component (uniform formats; 0 if packed)
    bool              bgr;          // host component order is B,G,R(,A)
    bool              linear;       // filterable
    bool              renderable;
    bool              storable;     // usable as storage image
    bool              vertex_only;  // no MTL texture format; vertex attrib only
    int8_t            packed_bits[4]; // per-component bit depths for packed
                                      // formats (component order); {0} = uniform
};

static const struct mtl_fmt_desc mtl_fmts[] = {
    // 8-bit UNORM
    { "r8",       MTLPixelFormatR8Unorm,     PL_FMT_UNORM, 1, 1, false, true,  true,  true  },
    { "rg8",      MTLPixelFormatRG8Unorm,    PL_FMT_UNORM, 2, 1, false, true,  true,  true  },
    { "rgba8",    MTLPixelFormatRGBA8Unorm,  PL_FMT_UNORM, 4, 1, false, true,  true,  true  },
    { "bgra8",    MTLPixelFormatBGRA8Unorm,  PL_FMT_UNORM, 4, 1, true,  true,  true,  false },
    // 16-bit UNORM
    { "r16",      MTLPixelFormatR16Unorm,    PL_FMT_UNORM, 1, 2, false, true,  true,  true  },
    { "rg16",     MTLPixelFormatRG16Unorm,   PL_FMT_UNORM, 2, 2, false, true,  true,  true  },
    { "rgba16",   MTLPixelFormatRGBA16Unorm, PL_FMT_UNORM, 4, 2, false, true,  true,  true  },
    // 16-bit float
    { "r16f",     MTLPixelFormatR16Float,    PL_FMT_FLOAT, 1, 2, false, true,  true,  true  },
    { "rg16f",    MTLPixelFormatRG16Float,   PL_FMT_FLOAT, 2, 2, false, true,  true,  true  },
    { "rgba16f",  MTLPixelFormatRGBA16Float, PL_FMT_FLOAT, 4, 2, false, true,  true,  true  },
    // 32-bit float
    { "r32f",     MTLPixelFormatR32Float,    PL_FMT_FLOAT, 1, 4, false, true,  true,  true  },
    { "rg32f",    MTLPixelFormatRG32Float,   PL_FMT_FLOAT, 2, 4, false, true,  true,  true  },
    { "rgba32f",  MTLPixelFormatRGBA32Float, PL_FMT_FLOAT, 4, 4, false, true,  true,  true  },
    // 8-bit UINT (non-filterable)
    { "r8ui",     MTLPixelFormatR8Uint,      PL_FMT_UINT,  1, 1, false, false, true,  true  },
    { "rg8ui",    MTLPixelFormatRG8Uint,     PL_FMT_UINT,  2, 1, false, false, true,  true  },
    { "rgba8ui",  MTLPixelFormatRGBA8Uint,   PL_FMT_UINT,  4, 1, false, false, true,  true  },
    // 32-bit UINT / SINT (LUTs, integer storage; non-filterable)
    { "r32ui",    MTLPixelFormatR32Uint,     PL_FMT_UINT,  1, 4, false, false, true,  true  },
    { "rg32ui",   MTLPixelFormatRG32Uint,    PL_FMT_UINT,  2, 4, false, false, true,  true  },
    { "rgba32ui", MTLPixelFormatRGBA32Uint,  PL_FMT_UINT,  4, 4, false, false, true,  true  },
    { "r32i",     MTLPixelFormatR32Sint,     PL_FMT_SINT,  1, 4, false, false, true,  true  },
    { "rg32i",    MTLPixelFormatRG32Sint,    PL_FMT_SINT,  2, 4, false, false, true,  true  },
    { "rgba32i",  MTLPixelFormatRGBA32Sint,  PL_FMT_SINT,  4, 4, false, false, true,  true  },
    // 3-component formats: no Metal texture equivalent (RGB textures don't
    // exist), but valid as vertex attributes. vertex_only → VERTEX cap only.
    { "rgb32f",   MTLPixelFormatInvalid,     PL_FMT_FLOAT, 3, 4, false, false, false, false, true },
    { "rgb16f",   MTLPixelFormatInvalid,     PL_FMT_FLOAT, 3, 2, false, false, false, false, true },
    // Packed: components share one machine word, non-uniform widths. comp_size
    // is 0 — the layout comes from packed_bits (LSB-first, component order).
    // rgb10a2 (10/10/10/2) is the HDR intermediate; matches the natural R,G,B,A
    // order so no bgr swap. Renderable + filterable on all supported families,
    // but not a valid storage-image format → storable stays false.
    { "rgb10a2",  MTLPixelFormatRGB10A2Unorm, PL_FMT_UNORM, 4, 0, false, true, true, false, false, {10, 10, 10, 2} },
#if TARGET_OS_IPHONE || TARGET_OS_TV
    // bgr565 (5/6/5) only exists as an MTLPixelFormat on iOS/tvOS — the enum is
    // unavailable on the macOS SDK, so guard it out there. B5G6R5 stores R in
    // the low bits → bgr component order (sample_order 2,1,0), same layout the
    // d3d11 backend uses for its "bgr565".
    { "bgr565",   MTLPixelFormatB5G6R5Unorm,  PL_FMT_UNORM, 3, 0, true,  true, true, false, false, {5, 6, 5} },
#endif
};

#define METAL_NUM_FORMATS (sizeof(mtl_fmts) / sizeof(mtl_fmts[0]))

// Minimum MTLReadWriteTextureTier that supports read-write access for this
// format, or 0 if Metal never allows read-write on it. Tier 1 covers R32
// (float/uint/sint); tier 2 additionally covers R8/RGBA8 (any), R16/RGBA16
// (float/uint/sint, NOT unorm) and RGBA32. Only R and RGBA layouts qualify —
// RG, BGRA, packed and vertex-only formats are never read-write.
static int mtl_readwrite_tier(const struct mtl_fmt_desc *d)
{
    if (d->packed_bits[0] != 0 || d->vertex_only || d->bgr)
        return 0;
    if (d->num_comp != 1 && d->num_comp != 4)
        return 0;
    switch (d->comp_size) {
    case 1:  return 2;                                   // r8 / rgba8 (unorm/uint/sint)
    case 2:  return d->type == PL_FMT_UNORM ? 0 : 2;     // r16f/rgba16f…, not 16-unorm
    case 4:  return d->num_comp == 1 ? 1 : 2;            // r32 = tier1, rgba32 = tier2
    default: return 0;
    }
}

void mtl_setup_formats(struct pl_gpu_t *gpu)
{
    id<MTLDevice> dev = mtl_device(gpu);
    MTLReadWriteTextureTier rw_tier =
        dev ? dev.readWriteTextureSupport : MTLReadWriteTextureTierNone;

    for (size_t i = 0; i < METAL_NUM_FORMATS; i++) {
        const struct mtl_fmt_desc *d = &mtl_fmts[i];
        struct pl_fmt_t *fmt = pl_alloc_obj(gpu, fmt, struct mtl_format_priv);
        struct mtl_format_priv *fp = PL_PRIV(fmt);
        fp->mtl = d->mtl;

        bool packed = d->packed_bits[0] != 0;

        fmt->name           = d->name;
        fmt->signature      = (uint64_t) d->mtl;   // MTLPixelFormat is stable+unique
        fmt->type           = d->type;
        fmt->num_components  = d->num_comp;
        int total_bits = 0;
        for (int c = 0; c < d->num_comp; c++)
            total_bits += packed ? d->packed_bits[c] : d->comp_size * 8;
        fmt->internal_size  = (total_bits + 7) / 8;
        fmt->texel_size     = fmt->internal_size;
        fmt->texel_align    = 1;
        for (int c = 0; c < d->num_comp; c++) {
            int bits = packed ? d->packed_bits[c] : d->comp_size * 8;
            fmt->component_depth[c] = bits;
            fmt->host_bits[c]       = bits;
            // bgr* swaps the R and B component slots (B,G,R,A)
            fmt->sample_order[c]    = (d->bgr && c < 3) ? (2 - c) : c;
        }

        enum pl_fmt_caps caps = 0;
        if (d->vertex_only) {
            // No texture format — vertex attribute use only.
            caps = PL_FMT_CAP_VERTEX;
        } else {
            caps = PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_BLITTABLE |
                   PL_FMT_CAP_HOST_READABLE;
            if (d->linear)
                caps |= PL_FMT_CAP_LINEAR;
            if (d->renderable) {
                caps |= PL_FMT_CAP_RENDERABLE;
                if (d->type != PL_FMT_UINT && d->type != PL_FMT_SINT)
                    caps |= PL_FMT_CAP_BLENDABLE;
            }
            if (d->storable && gpu->glsl.compute) {
                caps |= PL_FMT_CAP_STORABLE;
                // Read-write access (imageLoad+imageStore on the same image) is
                // tiered in Metal; only advertise it where the device supports it.
                int rw = mtl_readwrite_tier(d);
                if (rw > 0 && (int) rw_tier >= rw)
                    caps |= PL_FMT_CAP_READWRITE;
            }
            // Vertex use needs a plain (non-packed) layout + glsl_type; non-bgr
            // numeric only. Packed formats have no matching MTLVertexFormat.
            if (!d->bgr && d->type != PL_FMT_UINT && !packed)
                caps |= PL_FMT_CAP_VERTEX;
        }
        fmt->caps = caps;

        // Storage images need a GLSL format qualifier (e.g. `layout(rgba8)`).
        if (caps & PL_FMT_CAP_STORABLE)
            fmt->glsl_format = pl_fmt_glsl_format(fmt, fmt->num_components);
        fmt->glsl_type = pl_var_glsl_type_name(pl_var_from_fmt(fmt, ""));
        fmt->fourcc    = pl_fmt_fourcc(fmt);
        pl_assert(fmt->glsl_type);

        PL_ARRAY_APPEND_RAW(gpu, gpu->formats, gpu->num_formats, fmt);
    }

    PL_INFO(gpu, "Registered %d Metal formats", gpu->num_formats);
}
