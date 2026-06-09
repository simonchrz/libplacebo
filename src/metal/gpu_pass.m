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

/* Metal GPU backend — passes (shaders + pipelines).
 *
 * Pipeline: GLSL → SPIR-V (libplacebo's pl_spirv / shaderc) → MSL (SPIRV-Cross)
 *           → MTLLibrary → MTLRenderPipelineState / MTLComputePipelineState.
 *
 * Binding model: libplacebo hands out globally-unique descriptor bindings
 * (single namespace, see mtl_desc_namespace), and we pin each MSL resource
 * slot == its SPIR-V binding via spvc_compiler_msl_add_resource_binding so the
 * Metal-side bind index matches what the emitted MSL actually reads. Push
 * constants are pinned to a fixed high buffer slot; the vertex stream sits
 * one above that.
 *
 * Loose GLSL uniforms are NOT supported (max_variable_comps stays 0), so
 * libplacebo packs everything into push constants / uniform buffers, which
 * SPIRV-Cross turns into plain `constant T&` MSL buffer arguments.
 *
 * MRC (no -fobjc-arc): created objects are +1 and explicitly released.
 */

#import <Metal/Metal.h>

#include <spirv_cross_c.h>

#include "gpu.h"
#include "../cache.h"
#include "../pl_clock.h"   // pl_clock_now/diff für Compile-Time-Tracking

#define METAL_VERTEX_BUF_SLOT     30   // MSL [[buffer(30)]] for vertex stream
#define METAL_PUSH_CONST_SLOT     28   // MSL [[buffer(28)]] for push constants
#define METAL_STORAGE_IMG_OFFSET  16   // keep storage images clear of sampled

// libplacebo numbers all descriptors in one shared space → unique bindings.
int mtl_desc_namespace(pl_gpu gpu, enum pl_desc_type type)
{
    (void) gpu; (void) type;
    return 0;
}

// ----------------------------------------------------------- small mappings

static MTLVertexFormat mtl_vertex_format(pl_fmt fmt)
{
    int n = fmt->num_components;
    if (fmt->type == PL_FMT_FLOAT && fmt->component_depth[0] == 32) {
        switch (n) {
        case 1: return MTLVertexFormatFloat;
        case 2: return MTLVertexFormatFloat2;
        case 3: return MTLVertexFormatFloat3;
        case 4: return MTLVertexFormatFloat4;
        }
    }
    if (fmt->type == PL_FMT_FLOAT && fmt->component_depth[0] == 16) {
        switch (n) {
        case 2: return MTLVertexFormatHalf2;
        case 3: return MTLVertexFormatHalf3;
        case 4: return MTLVertexFormatHalf4;
        }
    }
    if (fmt->type == PL_FMT_UNORM && fmt->component_depth[0] == 8) {
        switch (n) {
        case 1: return MTLVertexFormatUCharNormalized;
        case 2: return MTLVertexFormatUChar2Normalized;
        case 4: return MTLVertexFormatUChar4Normalized;
        }
    }
    return MTLVertexFormatInvalid;
}

static MTLBlendFactor mtl_blend_factor(enum pl_blend_mode m)
{
    switch (m) {
    case PL_BLEND_ONE:                 return MTLBlendFactorOne;
    case PL_BLEND_SRC_ALPHA:           return MTLBlendFactorSourceAlpha;
    case PL_BLEND_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
    case PL_BLEND_ZERO:
    default:                           return MTLBlendFactorZero;
    }
}

static id<MTLSamplerState> mtl_get_sampler(pl_gpu gpu,
                                             enum pl_tex_sample_mode sample,
                                             enum pl_tex_address_mode addr)
{
    struct pl_gpu_metal *p = PL_PRIV(gpu);
    if (p->samplers[sample][addr])
        return (__bridge id<MTLSamplerState>) p->samplers[sample][addr];

    MTLSamplerAddressMode am =
        addr == PL_TEX_ADDRESS_REPEAT ? MTLSamplerAddressModeRepeat :
        addr == PL_TEX_ADDRESS_MIRROR ? MTLSamplerAddressModeMirrorRepeat :
                                        MTLSamplerAddressModeClampToEdge;
    MTLSamplerMinMagFilter mf =
        sample == PL_TEX_SAMPLE_LINEAR ? MTLSamplerMinMagFilterLinear
                                       : MTLSamplerMinMagFilterNearest;

    MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = mf;
    sd.magFilter = mf;
    sd.sAddressMode = am;
    sd.tAddressMode = am;
    sd.rAddressMode = am;
    id<MTLSamplerState> s = [mtl_device(gpu) newSamplerStateWithDescriptor:sd];
    [sd release];
    p->samplers[sample][addr] = (__bridge void *) s;   // +1, owned by gpu
    return s;
}

// ------------------------------------------------------- shader compilation

struct mtl_shader_stage {
    pl_str spv;                 // SPIR-V (child of the tmp alloc)
    spvc_context sc_ctx;        // owns parsed_ir + compiler
    spvc_compiler sc_compiler;
    const char *msl;            // lifetime tied to sc_ctx
    unsigned tg[3];             // compute LocalSize (threadgroup dims); from the
                                // SPIR-V on a fresh compile, or the cache on a hit
                                // (sc_compiler is NULL on a cache hit, so pass
                                // creation must read these, not query the compiler)
    id<MTLLibrary> library;
    id<MTLFunction> function;
};

static void mtl_stage_uninit(struct mtl_shader_stage *st)
{
    if (st->function) [st->function release];
    if (st->library)  [st->library release];
    st->function = nil;
    st->library  = nil;
    if (st->sc_ctx) {                       // also frees sc_compiler + msl
        spvc_context_destroy(st->sc_ctx);
        st->sc_ctx = NULL;
        st->sc_compiler = NULL;
        st->msl = NULL;
    }
}

static SpvExecutionModel spv_stage(enum glsl_shader_stage s)
{
    return s == GLSL_SHADER_VERTEX   ? SpvExecutionModelVertex
         : s == GLSL_SHADER_FRAGMENT ? SpvExecutionModelFragment
                                     : SpvExecutionModelGLCompute;
}

// Highest Metal Shading Language version the running OS supports, encoded the
// way SPIRV-Cross expects (major*10000 + minor*100). We target the OS maximum:
// the emitted MSL always compiles (it never exceeds device support) and the
// transpiler gets the newest constructs (e.g. the threadgroup atomics/barriers
// the compute path relies on). Goes into the shader-cache key so an OS upgrade
// transparently regenerates the cache.
static unsigned mtl_msl_version(void)
{
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) return 30200;
    if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) return 30100;
    if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, *)) return 30000;
    if (@available(macOS 12.0, iOS 15.0, tvOS 15.0, *)) return 20400;
    if (@available(macOS 11.0, iOS 14.0, tvOS 14.0, *)) return 20300;
    return 20200; // floor: MSL 2.2 (macOS 10.15 / iOS 13)
}

static bool mtl_compile_stage(pl_gpu gpu, void *tmp,
                                enum glsl_shader_stage stage_type,
                                const char *glsl,
                                struct mtl_shader_stage *st)
{
    struct pl_gpu_metal *p = PL_PRIV(gpu);
    unsigned msl_version = mtl_msl_version();

    // Compile-Time-Tracking: per-Sub-Step-Timings (0 = übersprungen, z.B. bei
    // Cache-Hit). Eine PL_INFO-Summary-Zeile pro Stage vor dem return.
    pl_clock_t _t;
    double spirv_ms = 0, cross_ms = 0, mtllib_ms = 0;

    // The expensive part below is GLSL → SPIR-V (shaderc) → MSL (SPIRV-Cross),
    // ~100-300ms per stage and ~15 stages on a cold renderer. Cache the emitted
    // MSL keyed by the GLSL source, so a pl_cache loaded from disk by the
    // embedder skips the whole transpile on later launches. (The MSL → MTLLibrary
    // / PSO step is cached by the Metal driver itself.)
    pl_cache cache = pl_gpu_cache(gpu);
    pl_cache_obj obj = {0};
    const char *msl = NULL;
    bool from_cache = false;
    if (cache) {
        uint64_t key = CACHE_KEY_METAL_MSL;
        pl_hash_merge(&key, p->spirv->signature);
        pl_hash_merge(&key, pl_str0_hash(glsl));
        pl_hash_merge(&key, msl_version);   // OS-dependent MSL target
        obj.key = key;
        // Blob layout: [3 × uint (threadgroup dims)][NUL-terminated MSL].
        if (pl_cache_get(cache, &obj) && obj.size > sizeof(st->tg)) {
            memcpy(st->tg, obj.data, sizeof(st->tg));
            msl = (const char *) obj.data + sizeof(st->tg);
            from_cache = true;
        }
    }

    if (!msl) {
    // GLSL → SPIR-V (libplacebo)
    _t = pl_clock_now();
    st->spv = pl_spirv_compile_glsl(p->spirv, tmp, gpu->glsl, stage_type, glsl);
    if (!st->spv.len) {
        PL_ERR(gpu, "metal: GLSL → SPIR-V failed");
        return false;
    }
    spirv_ms = pl_clock_diff(pl_clock_now(), _t) * 1e3;

    // SPIR-V → MSL (SPIRV-Cross)
    _t = pl_clock_now();
    if (spvc_context_create(&st->sc_ctx) != SPVC_SUCCESS) {
        PL_ERR(gpu, "metal: spvc_context_create failed");
        return false;
    }

    spvc_parsed_ir ir = NULL;
    if (spvc_context_parse_spirv(st->sc_ctx, (const SpvId *) st->spv.buf,
                                 st->spv.len / sizeof(SpvId), &ir) != SPVC_SUCCESS) {
        PL_ERR(gpu, "metal: parse_spirv: %s",
               spvc_context_get_last_error_string(st->sc_ctx));
        return false;
    }
    if (spvc_context_create_compiler(st->sc_ctx, SPVC_BACKEND_MSL, ir,
                                     SPVC_CAPTURE_MODE_TAKE_OWNERSHIP,
                                     &st->sc_compiler) != SPVC_SUCCESS) {
        PL_ERR(gpu, "metal: create_compiler: %s",
               spvc_context_get_last_error_string(st->sc_ctx));
        return false;
    }

    spvc_compiler_options opts = NULL;
    spvc_compiler_create_compiler_options(st->sc_compiler, &opts);
    spvc_compiler_options_set_uint(opts, SPVC_COMPILER_OPTION_MSL_VERSION, msl_version);
    spvc_compiler_options_set_bool(opts,
        SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, SPVC_FALSE);
    // Metal clip-space is y-flipped vs Vulkan/GL (like D3D11): a vertex at
    // (0,-1) lands where you'd sample (0,1). Inject gl_Position.y = -y in the
    // vertex stage so framebuffer-space matches what libplacebo expects.
    if (stage_type == GLSL_SHADER_VERTEX) {
        spvc_compiler_options_set_bool(opts,
            SPVC_COMPILER_OPTION_FLIP_VERTEX_Y, SPVC_TRUE);
    }
    spvc_compiler_install_compiler_options(st->sc_compiler, opts);

    // Pin MSL slots == SPIR-V bindings so our setFragment*/setCompute* indices
    // line up with the emitted MSL. (Auto-renumbering otherwise samples the
    // wrong texture.) Storage images are bumped clear of sampled textures.
    spvc_resources res = NULL;
    if (spvc_compiler_create_shader_resources(st->sc_compiler, &res) == SPVC_SUCCESS) {
        static const spvc_resource_type kinds[] = {
            SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
            SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
            SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
            SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        };
        for (size_t k = 0; k < sizeof(kinds) / sizeof(kinds[0]); k++) {
            const spvc_reflected_resource *list = NULL;
            size_t count = 0;
            if (spvc_resources_get_resource_list_for_type(res, kinds[k],
                                                          &list, &count) != SPVC_SUCCESS)
                continue;
            for (size_t i = 0; i < count; i++) {
                unsigned set = spvc_compiler_get_decoration(st->sc_compiler,
                                  list[i].id, SpvDecorationDescriptorSet);
                unsigned bind = spvc_compiler_get_decoration(st->sc_compiler,
                                  list[i].id, SpvDecorationBinding);
                if (kinds[k] == SPVC_RESOURCE_TYPE_STORAGE_IMAGE) {
                    bind += METAL_STORAGE_IMG_OFFSET;
                    spvc_compiler_set_decoration(st->sc_compiler, list[i].id,
                                                 SpvDecorationBinding, bind);
                }
                spvc_msl_resource_binding rb = {0};
                rb.stage       = spv_stage(stage_type);
                rb.desc_set    = set;
                rb.binding     = bind;
                rb.msl_buffer  = bind;
                rb.msl_texture = bind;
                rb.msl_sampler = bind;
                spvc_compiler_msl_add_resource_binding(st->sc_compiler, &rb);
            }
        }
        // Push constants have no binding decoration; pin to a fixed slot so
        // SPIRV-Cross doesn't collide them with buffer(0).
        spvc_msl_resource_binding pc = {0};
        pc.stage      = spv_stage(stage_type);
        pc.desc_set   = SPVC_MSL_PUSH_CONSTANT_DESC_SET;
        pc.binding    = SPVC_MSL_PUSH_CONSTANT_BINDING;
        pc.msl_buffer = METAL_PUSH_CONST_SLOT;
        spvc_compiler_msl_add_resource_binding(st->sc_compiler, &pc);
    }

    if (spvc_compiler_compile(st->sc_compiler, &st->msl) != SPVC_SUCCESS) {
        PL_ERR(gpu, "metal: MSL emit: %s",
               spvc_context_get_last_error_string(st->sc_ctx));
        return false;
    }
    // Capture the compute threadgroup dims now, while the compiler exists — a
    // cache hit later won't have it. (LocalSize is unset for vertex/fragment;
    // SPIRV-Cross returns 0 there, which pass creation ignores.)
    for (int i = 0; i < 3; i++)
        st->tg[i] = spvc_compiler_get_execution_mode_argument_by_index(
            st->sc_compiler, SpvExecutionModeLocalSize, i);
    msl = st->msl;
    cross_ms = pl_clock_diff(pl_clock_now(), _t) * 1e3;
    }  // end cache-miss transpile

    // MSL → MTLLibrary → MTLFunction("main0")
    NSError *err = nil;
    NSString *src = [[NSString alloc] initWithUTF8String:msl];
    MTLCompileOptions *co = [[MTLCompileOptions alloc] init];
    _t = pl_clock_now();
    st->library = [mtl_device(gpu) newLibraryWithSource:src options:co error:&err];
    mtllib_ms = pl_clock_diff(pl_clock_now(), _t) * 1e3;
    [co release];
    [src release];
    if (!st->library) {
        PL_ERR(gpu, "metal: newLibraryWithSource failed: %s",
               err.localizedDescription ? err.localizedDescription.UTF8String : "?");
        PL_ERR(gpu, "metal: MSL was:\n%s", msl);
        pl_cache_obj_free(&obj);
        return false;
    }
    st->function = [st->library newFunctionWithName:@"main0"];
    if (!st->function) {
        PL_ERR(gpu, "metal: no 'main0' entry point");
        pl_cache_obj_free(&obj);
        return false;
    }

    // Persist [threadgroup dims][NUL-terminated MSL] for the next launch.
    if (cache && !from_cache) {
        size_t msl_size = strlen(msl) + 1;
        pl_cache_obj_resize(NULL, &obj, sizeof(st->tg) + msl_size);
        if (obj.data) {
            memcpy(obj.data, st->tg, sizeof(st->tg));
            memcpy((uint8_t *) obj.data + sizeof(st->tg), msl, msl_size);
            pl_cache_set(cache, &obj);   // takes ownership of obj.data
        }
    }
    pl_cache_obj_free(&obj);             // no-op after set; frees the loaded blob on hit

    const char *sname = stage_type == GLSL_SHADER_VERTEX   ? "vert"
                      : stage_type == GLSL_SHADER_FRAGMENT ? "frag" : "comp";
    PL_INFO(gpu, "metal-compile: stage=%s cache=%s spirv=%.1fms cross=%.1fms mtllib=%.1fms",
            sname, from_cache ? "hit" : "miss", spirv_ms, cross_ms, mtllib_ms);
    return true;
}

// ------------------------------------------------------------ pass lifecycle

void mtl_pass_destroy(pl_gpu gpu, pl_pass pass)
{
    struct pl_pass_metal *pm = PL_PRIV(pass);
    if (pm->render_pso)  [(__bridge id<MTLRenderPipelineState>) pm->render_pso release];
    if (pm->compute_pso) [(__bridge id<MTLComputePipelineState>) pm->compute_pso release];
    if (pm->library)     [(__bridge id<MTLLibrary>) pm->library release];
    pl_free((void *) pass);
    (void) gpu;
}

pl_pass mtl_pass_create(pl_gpu gpu, const struct pl_pass_params *params)
{
    struct pl_pass_t *pass = pl_zalloc_obj(NULL, pass, struct pl_pass_metal);
    pass->params = pl_pass_params_copy(pass, params);
    struct pl_pass_metal *pm = PL_PRIV(pass);
    pm->push_constant_slot = params->push_constants_size > 0 ? METAL_PUSH_CONST_SLOT : -1;

    void *tmp = pl_tmp(NULL);
    struct mtl_shader_stage vert = {0}, frag = {0}, comp = {0};
    id<MTLDevice> dev = mtl_device(gpu);

    if (params->type == PL_PASS_RASTER) {
        if (!mtl_compile_stage(gpu, tmp, GLSL_SHADER_VERTEX, params->vertex_shader, &vert))
            goto error;
        if (!mtl_compile_stage(gpu, tmp, GLSL_SHADER_FRAGMENT, params->glsl_shader, &frag))
            goto error;

        MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
        for (int i = 0; i < params->num_vertex_attribs; i++) {
            const struct pl_vertex_attrib *va = &params->vertex_attribs[i];
            MTLVertexFormat vf = mtl_vertex_format(va->fmt);
            if (vf == MTLVertexFormatInvalid) {
                PL_ERR(gpu, "metal: unsupported vertex attrib '%s'", va->name);
                goto error;
            }
            vd.attributes[va->location].format      = vf;
            vd.attributes[va->location].offset      = va->offset;
            vd.attributes[va->location].bufferIndex = METAL_VERTEX_BUF_SLOT;
        }
        vd.layouts[METAL_VERTEX_BUF_SLOT].stride = params->vertex_stride;
        vd.layouts[METAL_VERTEX_BUF_SLOT].stepFunction = MTLVertexStepFunctionPerVertex;

        MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
        pd.vertexFunction   = vert.function;
        pd.fragmentFunction = frag.function;
        pd.vertexDescriptor = vd;
        pd.colorAttachments[0].pixelFormat = mtl_fmt(params->target_format);
        if (params->blend_params) {
            const struct pl_blend_params *b = params->blend_params;
            pd.colorAttachments[0].blendingEnabled = YES;
            pd.colorAttachments[0].sourceRGBBlendFactor        = mtl_blend_factor(b->src_rgb);
            pd.colorAttachments[0].destinationRGBBlendFactor   = mtl_blend_factor(b->dst_rgb);
            pd.colorAttachments[0].sourceAlphaBlendFactor      = mtl_blend_factor(b->src_alpha);
            pd.colorAttachments[0].destinationAlphaBlendFactor = mtl_blend_factor(b->dst_alpha);
            pd.colorAttachments[0].rgbBlendOperation   = MTLBlendOperationAdd;
            pd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        }

        NSError *err = nil;
        pl_clock_t _tp = pl_clock_now();
        id<MTLRenderPipelineState> pso =
            [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        PL_INFO(gpu, "metal-compile: pso=render %.1fms",
                pl_clock_diff(pl_clock_now(), _tp) * 1e3);
        [pd release];
        if (!pso) {
            PL_ERR(gpu, "metal: newRenderPipelineState failed: %s",
                   err.localizedDescription ? err.localizedDescription.UTF8String : "?");
            goto error;
        }
        pm->render_pso = (__bridge void *) pso;
        pm->library = (__bridge void *) [frag.library retain];
    } else {
        if (!mtl_compile_stage(gpu, tmp, GLSL_SHADER_COMPUTE, params->glsl_shader, &comp))
            goto error;

        MTLComputePipelineDescriptor *cd = [[MTLComputePipelineDescriptor alloc] init];
        cd.computeFunction = comp.function;
        NSError *err = nil;
        pl_clock_t _tp = pl_clock_now();
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithDescriptor:cd
                                               options:MTLPipelineOptionNone
                                            reflection:nil error:&err];
        PL_INFO(gpu, "metal-compile: pso=compute %.1fms",
                pl_clock_diff(pl_clock_now(), _tp) * 1e3);
        [cd release];
        if (!pso) {
            PL_ERR(gpu, "metal: newComputePipelineState failed: %s",
                   err.localizedDescription ? err.localizedDescription.UTF8String : "?");
            goto error;
        }
        pm->compute_pso = (__bridge void *) pso;
        pm->library = (__bridge void *) [comp.library retain];

        // From the stage (captured at compile time or restored from cache) —
        // comp.sc_compiler is NULL on a cache hit, so don't query it here.
        pm->compute_threads[0] = comp.tg[0] ? comp.tg[0] : 1;
        pm->compute_threads[1] = comp.tg[1] ? comp.tg[1] : 1;
        pm->compute_threads[2] = comp.tg[2] ? comp.tg[2] : 1;
    }

    mtl_stage_uninit(&vert);
    mtl_stage_uninit(&frag);
    mtl_stage_uninit(&comp);
    pl_free(tmp);
    return pass;

error:
    mtl_stage_uninit(&vert);
    mtl_stage_uninit(&frag);
    mtl_stage_uninit(&comp);
    pl_free(tmp);
    pl_free((void *) pass);
    return NULL;
}

// ------------------------------------------------------------------ pass run

// Bind one descriptor onto whichever encoder is active. Exactly one of renc /
// cenc is non-nil.
static void mtl_bind_desc(pl_gpu gpu, id<MTLRenderCommandEncoder> renc,
                            id<MTLComputeCommandEncoder> cenc,
                            const struct pl_desc *desc,
                            const struct pl_desc_binding *db,
                            id<MTLCommandBuffer> cb)
{
    int slot = desc->binding;
    switch (desc->type) {
    case PL_DESC_SAMPLED_TEX: {
        pl_tex tex = db->object;
        struct pl_tex_metal *tp = PL_PRIV(tex);
        id<MTLTexture> mt = (__bridge id<MTLTexture>) tp->tex;
        id<MTLSamplerState> samp = mtl_get_sampler(gpu, db->sample_mode, db->address_mode);
        if (renc) {
            [renc setFragmentTexture:mt atIndex:slot];
            [renc setFragmentSamplerState:samp atIndex:slot];
        } else {
            [cenc setTexture:mt atIndex:slot];
            [cenc setSamplerState:samp atIndex:slot];
        }
        // Auch LESE-Bindings tracken: replaceRegion/Host-Writes auf eine noch
        // in-flight gesampelte Textur wären ein Hazard. Der Wait ist nach dem
        // Frame-Commit praktisch gratis (CB längst fertig).
        mtl_set_pending(&tp->pending_cb, cb);
        break;
    }
    case PL_DESC_STORAGE_IMG: {
        pl_tex tex = db->object;
        struct pl_tex_metal *tp = PL_PRIV(tex);
        id<MTLTexture> mt = (__bridge id<MTLTexture>) tp->tex;
        int s = slot + METAL_STORAGE_IMG_OFFSET;
        if (renc) [renc setFragmentTexture:mt atIndex:s];
        else      [cenc setTexture:mt atIndex:s];
        mtl_set_pending(&tp->pending_cb, cb);
        break;
    }
    case PL_DESC_BUF_UNIFORM:
    case PL_DESC_BUF_STORAGE: {
        pl_buf buf = db->object;
        struct pl_buf_metal *bp = PL_PRIV(buf);
        id<MTLBuffer> mb = (__bridge id<MTLBuffer>) bp->buf;
        if (renc) [renc setFragmentBuffer:mb offset:0 atIndex:slot];
        else      [cenc setBuffer:mb offset:0 atIndex:slot];
        // Uniform-Buffer ebenfalls tracken (Read-Binding): pl_dispatch pollt
        // vor dem Recycling — ohne Tracking würde ein noch referenzierter
        // Buffer per CPU-memcpy überschrieben (Shared-Storage liest zur
        // AUSFÜHRUNGS-Zeit, nicht zur Encode-Zeit).
        mtl_set_pending(&bp->pending_cb, cb);
        break;
    }
    default:
        PL_WARN(gpu, "metal: unhandled descriptor type %d", desc->type);
        break;
    }
}

void mtl_pass_run(pl_gpu gpu, const struct pl_pass_run_params *params)
{
    pl_pass pass = params->pass;
    struct pl_pass_metal *pm = PL_PRIV(pass);
    // Alle Passes eines Frames encoden in den geteilten Frame-CB (ein Submit
    // pro Frame statt pro Pass); committed wird in flush/finish.
    id<MTLCommandBuffer> cb = mtl_frame_cb(gpu);

    if (pass->params.type == PL_PASS_RASTER) {
        struct pl_tex_metal *tt = PL_PRIV(params->target);
        id<MTLTexture> target = (__bridge id<MTLTexture>) tt->tex;

        MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
        rpd.colorAttachments[0].texture = target;
        rpd.colorAttachments[0].loadAction =
            pass->params.load_target ? MTLLoadActionLoad : MTLLoadActionDontCare;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
        [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>) pm->render_pso];

        if (pm->push_constant_slot >= 0 && params->push_constants) {
            [enc setVertexBytes:params->push_constants
                         length:pass->params.push_constants_size
                        atIndex:pm->push_constant_slot];
            [enc setFragmentBytes:params->push_constants
                           length:pass->params.push_constants_size
                          atIndex:pm->push_constant_slot];
        }

        MTLViewport vp = {
            .originX = params->viewport.x0, .originY = params->viewport.y0,
            .width  = pl_rect_w(params->viewport),
            .height = pl_rect_h(params->viewport),
            .znear = 0, .zfar = 1,
        };
        [enc setViewport:vp];

        NSUInteger tw = target.width, th = target.height;
        NSInteger sx = PL_CLAMP(params->scissors.x0, 0, (int) tw);
        NSInteger sy = PL_CLAMP(params->scissors.y0, 0, (int) th);
        NSInteger sw = PL_CLAMP(params->scissors.x1, sx, (int) tw) - sx;
        NSInteger sh = PL_CLAMP(params->scissors.y1, sy, (int) th) - sy;
        [enc setScissorRect:(MTLScissorRect){ sx, sy, PL_MAX(sw, 0), PL_MAX(sh, 0) }];

        for (int i = 0; i < pass->params.num_descriptors; i++)
            mtl_bind_desc(gpu, enc, nil, &pass->params.descriptors[i],
                            &params->desc_bindings[i], cb);

        // Vertex stream: from a pl_buf, or inline host data.
        size_t vbytes = (size_t) params->vertex_count * pass->params.vertex_stride;
        if (params->vertex_buf) {
            struct pl_buf_metal *vb = PL_PRIV(params->vertex_buf);
            [enc setVertexBuffer:(__bridge id<MTLBuffer>) vb->buf
                          offset:params->buf_offset
                         atIndex:METAL_VERTEX_BUF_SLOT];
        } else if (vbytes <= 4096) {
            [enc setVertexBytes:params->vertex_data length:vbytes
                        atIndex:METAL_VERTEX_BUF_SLOT];
        } else {
            id<MTLBuffer> vbuf = [mtl_device(gpu) newBufferWithBytes:params->vertex_data
                                  length:vbytes options:MTLResourceStorageModeShared];
            [enc setVertexBuffer:vbuf offset:0 atIndex:METAL_VERTEX_BUF_SLOT];
            [cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull c) { (void) c; [vbuf release]; }];
        }

        MTLPrimitiveType prim = pass->params.vertex_type == PL_PRIM_TRIANGLE_STRIP
                              ? MTLPrimitiveTypeTriangleStrip : MTLPrimitiveTypeTriangle;
        if (params->index_data || params->index_buf) {
            // Index buffer path: stage indices into a Shared buffer.
            size_t isz = params->index_fmt == PL_INDEX_UINT16 ? 2 : 4;
            MTLIndexType it = params->index_fmt == PL_INDEX_UINT16
                            ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
            id<MTLBuffer> ibuf;
            size_t ioff = 0;
            if (params->index_buf) {
                ibuf = (__bridge id<MTLBuffer>) ((struct pl_buf_metal *) PL_PRIV(params->index_buf))->buf;
                ioff = params->index_offset;
            } else {
                ibuf = [mtl_device(gpu) newBufferWithBytes:params->index_data
                        length:params->vertex_count * isz
                        options:MTLResourceStorageModeShared];
                [cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull c) { (void) c; [ibuf release]; }];
            }
            [enc drawIndexedPrimitives:prim
                            indexCount:params->vertex_count
                             indexType:it
                           indexBuffer:ibuf
                     indexBufferOffset:ioff];
        } else {
            [enc drawPrimitives:prim vertexStart:0 vertexCount:params->vertex_count];
        }
        [enc endEncoding];
        mtl_set_pending(&tt->pending_cb, cb);
    } else {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>) pm->compute_pso];

        if (pm->push_constant_slot >= 0 && params->push_constants) {
            [enc setBytes:params->push_constants
                   length:pass->params.push_constants_size
                  atIndex:pm->push_constant_slot];
        }
        for (int i = 0; i < pass->params.num_descriptors; i++)
            mtl_bind_desc(gpu, nil, enc, &pass->params.descriptors[i],
                            &params->desc_bindings[i], cb);

        MTLSize tg = MTLSizeMake(pm->compute_threads[0], pm->compute_threads[1],
                                 pm->compute_threads[2]);
        MTLSize grid = MTLSizeMake(params->compute_groups[0], params->compute_groups[1],
                                   params->compute_groups[2]);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    // NB: mit dem geteilten Frame-CB liefern GPU-Timer Frame- statt Pass-
    // Granularität (GPUStartTime/GPUEndTime gelten pro CB).
    mtl_timer_attach(params->timer, cb);
}
