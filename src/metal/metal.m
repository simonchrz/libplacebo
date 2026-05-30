/* Metal GPU backend — public API entry points (libplacebo/metal.h).
 *
 * Thin wrappers around the internal pl_gpu_create_metal / swapchain code that
 * own the MTLDevice + MTLCommandQueue lifetime. MRC (no -fobjc-arc).
 */

#import <Metal/Metal.h>

#include <libplacebo/metal.h>

#include "gpu.h"
#include "common.h"

pl_metal pl_metal_create(pl_log log, const struct pl_metal_params *params)
{
    struct pl_metal_params defaults = {0};
    params = params ? params : &defaults;

    id<MTLDevice> dev;
    if (params->device) {
        dev = (__bridge id<MTLDevice>) params->device;
        [dev retain];                       // we hand a +1 to the wrapper
    } else {
        dev = MTLCreateSystemDefaultDevice(); // +1
    }
    if (!dev) {
        pl_err(log, "Metal: no device available");
        return NULL;
    }

    id<MTLCommandQueue> queue = [dev newCommandQueue]; // +1
    if (!queue) {
        pl_err(log, "Metal: failed creating command queue");
        [dev release];
        return NULL;
    }

    pl_gpu gpu = pl_gpu_create_metal(log, (__bridge void *) dev,
                                          (__bridge void *) queue);
    if (!gpu) {
        [queue release];
        [dev release];
        return NULL;
    }

    struct pl_metal_t *m = pl_zalloc_ptr(NULL, m);
    m->gpu    = gpu;
    m->device = (__bridge void *) dev;   // wrapper owns the +1 created above
    m->queue  = (__bridge void *) queue; // (the gpu holds its own retains)
    return m;
}

void pl_metal_destroy(pl_metal *pmetal)
{
    pl_metal metal = *pmetal;
    if (!metal)
        return;
    pl_gpu_destroy(metal->gpu);
    [(__bridge id<MTLCommandQueue>) metal->queue release];
    [(__bridge id<MTLDevice>) metal->device release];
    pl_free((void *) metal);
    *pmetal = NULL;
}

pl_swapchain pl_metal_create_swapchain(pl_metal metal,
    const struct pl_metal_swapchain_params *params)
{
    if (!params->layer) {
        pl_err(metal->gpu->log, "pl_metal_create_swapchain: NULL layer");
        return NULL;
    }
    return pl_metal_swapchain_create(metal->gpu, params->layer);
}
