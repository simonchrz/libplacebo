/* Metal backend smoke test.
 *
 * Skeleton/Phase-3 scope: create a real pl_gpu over the system MTLDevice and
 * verify the format table registers and survives pl_gpu_finalize() validation.
 * Texture/buffer/pass coverage (the full gpu_tests suite) lands once those
 * slots are implemented.
 */

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "gpu_tests.h"
#include "metal/gpu.h"

int main()
{
    pl_log log = pl_test_logger();

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        printf("no Metal device available — skipping\n");
        return SKIP;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    pl_gpu gpu = pl_gpu_create_metal(log, (__bridge void *) dev,
                                          (__bridge void *) queue);
    REQUIRE(gpu);

    // Format table registered + survived pl_gpu_finalize() validation.
    REQUIRE(gpu->num_formats > 0);
    printf("metal: %d formats registered\n", gpu->num_formats);

    // Named lookups the renderer relies on must resolve.
    REQUIRE(pl_find_named_fmt(gpu, "r8"));
    REQUIRE(pl_find_named_fmt(gpu, "rgba8"));
    REQUIRE(pl_find_named_fmt(gpu, "rgba16f"));

    // Sampleable+renderable rgba8 is the renderer's workhorse format.
    pl_fmt rgba8 = pl_find_named_fmt(gpu, "rgba8");
    REQUIRE(rgba8->caps & PL_FMT_CAP_SAMPLEABLE);
    REQUIRE(rgba8->caps & PL_FMT_CAP_RENDERABLE);
    REQUIRE(rgba8->caps & PL_FMT_CAP_LINEAR);

    // ---- buffer round-trip (gpu_tex.m: buf_create/write/read) ----
    {
        uint8_t src[256], dst[256];
        for (int i = 0; i < 256; i++)
            src[i] = (uint8_t) (i ^ 0x5a);
        pl_buf buf = pl_buf_create(gpu, pl_buf_params(
            .size = sizeof(src),
            .host_writable = true,
            .host_readable = true,
        ));
        REQUIRE(buf);
        pl_buf_write(gpu, buf, 0, src, sizeof(src));
        memset(dst, 0, sizeof(dst));
        REQUIRE(pl_buf_read(gpu, buf, 0, dst, sizeof(dst)));
        REQUIRE_MEMEQ(src, dst, sizeof(src));
        pl_buf_destroy(gpu, &buf);
        printf("metal: buffer round-trip ok\n");
    }

    // ---- texture round-trip (gpu_tex.m: tex_create/upload/download) ----
    {
        const int W = 4, H = 4;
        uint8_t src[W * H * 4], dst[W * H * 4];
        for (int i = 0; i < W * H * 4; i++)
            src[i] = (uint8_t) (i * 7 + 3);
        pl_tex tex = pl_tex_create(gpu, pl_tex_params(
            .w = W, .h = H,
            .format = rgba8,
            .sampleable = true,
            .host_writable = true,
            .host_readable = true,
        ));
        REQUIRE(tex);
        REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(
            .tex = tex,
            .ptr = src,
        )));
        memset(dst, 0, sizeof(dst));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
            .tex = tex,
            .ptr = dst,
        )));
        REQUIRE_MEMEQ(src, dst, sizeof(src));
        pl_tex_destroy(gpu, &tex);
        printf("metal: texture round-trip ok\n");
    }

    // ---- render round-trip: GLSL → SPIR-V → MSL → PSO → draw → readback ----
    {
        const int W = 8, H = 8;
        pl_tex target = pl_tex_create(gpu, pl_tex_params(
            .w = W, .h = H,
            .format = rgba8,
            .renderable = true,
            .host_readable = true,
        ));
        REQUIRE(target);

        static const char *vert =
            "#version 450\n"
            "layout(location = 0) in vec2 pos;\n"
            "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
        static const char *frag =
            "#version 450\n"
            "layout(location = 0) out vec4 color;\n"
            "void main() { color = vec4(0.25, 0.5, 0.75, 1.0); }\n";

        struct pl_vertex_attrib attr = {
            .name = "pos",
            .fmt = pl_find_named_fmt(gpu, "rg32f"),
            .offset = 0,
            .location = 0,
        };
        pl_pass pass = pl_pass_create(gpu, pl_pass_params(
            .type = PL_PASS_RASTER,
            .glsl_shader = frag,
            .vertex_shader = vert,
            .vertex_type = PL_PRIM_TRIANGLE_LIST,
            .vertex_attribs = &attr,
            .num_vertex_attribs = 1,
            .vertex_stride = 2 * sizeof(float),
            .target_format = rgba8,
        ));
        REQUIRE(pass);
        printf("metal: render pass created (GLSL→MSL→PSO ok)\n");

        // Time the pass with a pl_timer (GPU timestamps via completion handler).
        pl_timer timer = pl_timer_create(gpu);
        REQUIRE(timer);

        // Oversized triangle covering the whole [-1,1] clip square.
        static const float verts[6] = { -1, -1,  3, -1,  -1, 3 };
        pl_pass_run(gpu, pl_pass_run_params(
            .pass = pass,
            .target = target,
            .vertex_count = 3,
            .vertex_data = verts,
            .viewport = {0, 0, W, H},
            .scissors = {0, 0, W, H},
            .timer = timer,
        ));

        uint8_t px[W * H * 4];
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
            .tex = target, .ptr = px,
        )));

        // The completion handler that publishes the timing fires asynchronously
        // shortly after the command buffer completes — drain with a bounded poll.
        pl_gpu_finish(gpu);
        uint64_t gpu_ns = 0;
        for (int i = 0; i < 10000 && !gpu_ns; i++)
            gpu_ns = pl_timer_query(gpu, timer);
        printf("metal: render pass GPU time = %llu ns\n",
               (unsigned long long) gpu_ns);
        REQUIRE(gpu_ns > 0);
        // Ring is drained: the next query yields no further result.
        REQUIRE(pl_timer_query(gpu, timer) == 0);
        pl_timer_destroy(gpu, &timer);
        // Expect ~ (0.25, 0.5, 0.75, 1.0) * 255 = (64, 128, 191, 255).
        printf("metal: rendered pixel[0] = (%d,%d,%d,%d) expect ~(64,128,191,255)\n",
               px[0], px[1], px[2], px[3]);
        REQUIRE(abs((int) px[0] - 64)  <= 2);
        REQUIRE(abs((int) px[1] - 128) <= 2);
        REQUIRE(abs((int) px[2] - 191) <= 2);
        REQUIRE(px[3] == 255);

        pl_pass_destroy(gpu, &pass);
        pl_tex_destroy(gpu, &target);
        printf("metal: render round-trip ok\n");
    }

    // ---- swapchain lifecycle over a headless CAMetalLayer ----
    {
        CAMetalLayer *layer = [CAMetalLayer layer];
        layer.drawableSize = CGSizeMake(16, 16);
        pl_swapchain sw = pl_metal_swapchain_create(gpu, (__bridge void *) layer);
        REQUIRE(sw);

        struct pl_swapchain_frame frame;
        if (pl_swapchain_start_frame(sw, &frame)) {
            REQUIRE(frame.fbo);
            REQUIRE(frame.fbo->params.w == 16);
            REQUIRE(frame.fbo->params.h == 16);
            REQUIRE(frame.fbo->params.renderable);
            printf("metal: swapchain frame %dx%d fmt=%s\n",
                   frame.fbo->params.w, frame.fbo->params.h,
                   frame.fbo->params.format->name);
            // The drawable's texture is a valid render target: clear it.
            pl_tex_clear(gpu, frame.fbo, (float[4]){0.1f, 0.2f, 0.3f, 1.0f});
            REQUIRE(pl_swapchain_submit_frame(sw));
            pl_swapchain_swap_buffers(sw);
            printf("metal: swapchain start→clear→submit→swap ok\n");
        } else {
            printf("metal: no drawable in headless mode — swapchain plumbing built\n");
        }
        pl_swapchain_destroy(&sw);
    }

    // ---- full libplacebo GPU test suite against the Metal backend ----
    printf("metal: --- running libplacebo gpu test suite ---\n");
    pl_buffer_tests(gpu);
    printf("metal: pl_buffer_tests passed\n");
    pl_texture_tests(gpu);
    printf("metal: pl_texture_tests passed\n");
    gpu_shader_tests(gpu);
    printf("metal: gpu_shader_tests passed\n");

    pl_gpu_destroy(gpu);
    pl_log_destroy(&log);
    return 0;
}
