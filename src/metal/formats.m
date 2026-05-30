/* Metal GPU backend — format table.
 * Maps MTLPixelFormat → pl_fmt. Per-format backend-private holds the
 * MTLPixelFormat.
 *
 * Scope: component-uniform formats (every component the same byte size) so
 * texel_size / host_bits derive cleanly. Packed formats (rgb10_a2, rgb565)
 * need bespoke host_bits and land with the texture-upload phase.
 */

#import <Metal/Metal.h>

#include "gpu.h"

struct mtl_fmt_desc {
    const char       *name;
    MTLPixelFormat    mtl;
    enum pl_fmt_type  type;
    int               num_comp;
    int               comp_size;    // bytes per component (uniform)
    bool              bgr;          // host byte order is B,G,R(,A)
    bool              linear;       // filterable
    bool              renderable;
    bool              storable;     // usable as storage image
    bool              vertex_only;  // no MTL texture format; vertex attrib only
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
};

#define METAL_NUM_FORMATS (sizeof(mtl_fmts) / sizeof(mtl_fmts[0]))

void mtl_setup_formats(struct pl_gpu_t *gpu)
{
    for (size_t i = 0; i < METAL_NUM_FORMATS; i++) {
        const struct mtl_fmt_desc *d = &mtl_fmts[i];
        struct pl_fmt_t *fmt = pl_alloc_obj(gpu, fmt, struct mtl_format_priv);
        struct mtl_format_priv *fp = PL_PRIV(fmt);
        fp->mtl = d->mtl;

        fmt->name           = d->name;
        fmt->signature      = (uint64_t) d->mtl;   // MTLPixelFormat is stable+unique
        fmt->type           = d->type;
        fmt->num_components  = d->num_comp;
        fmt->internal_size  = d->num_comp * d->comp_size;
        fmt->texel_size     = fmt->internal_size;
        fmt->texel_align    = 1;
        for (int c = 0; c < d->num_comp; c++) {
            fmt->component_depth[c] = d->comp_size * 8;
            fmt->host_bits[c]       = d->comp_size * 8;
            // bgr* swaps the R and B host byte slots (B,G,R,A)
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
            if (d->storable && gpu->glsl.compute)
                caps |= PL_FMT_CAP_STORABLE;
            // Vertex use needs a glsl_type (set below); non-bgr numeric only.
            if (!d->bgr && d->type != PL_FMT_UINT)
                caps |= PL_FMT_CAP_VERTEX;
        }
        fmt->caps = caps;

        fmt->glsl_type = pl_var_glsl_type_name(pl_var_from_fmt(fmt, ""));
        fmt->fourcc    = pl_fmt_fourcc(fmt);
        pl_assert(fmt->glsl_type);

        PL_ARRAY_APPEND_RAW(gpu, gpu->formats, gpu->num_formats, fmt);
    }

    PL_INFO(gpu, "Registered %d Metal formats", gpu->num_formats);
}
