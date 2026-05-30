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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LIBPLACEBO_METAL_H_
#define LIBPLACEBO_METAL_H_

#include <libplacebo/gpu.h>
#include <libplacebo/swapchain.h>

PL_API_BEGIN

// Note: Objective-C objects are passed across this API as opaque `void *`
// (bridged from `id<MTL...>` / `CAMetalLayer *`) so the header stays usable
// from plain C. The backend retains them for its own lifetime.

// Structure representing a Metal device and associated GPU instance.
typedef const struct pl_metal_t {
    pl_gpu gpu;
    void *device;   // id<MTLDevice>       used by this instance
    void *queue;    // id<MTLCommandQueue> used by this instance
} *pl_metal;

struct pl_metal_params {
    // An existing id<MTLDevice> to use (bridged to void *). If NULL, the
    // system default device (MTLCreateSystemDefaultDevice) is used.
    void *device;
};

#define pl_metal_params(...) (&(struct pl_metal_params) { __VA_ARGS__ })

// Creates a new Metal device based on the given parameters, or NULL on failure.
PL_API pl_metal pl_metal_create(pl_log log, const struct pl_metal_params *params);

// Release the Metal device. The `pl_gpu` becomes invalid after this call.
PL_API void pl_metal_destroy(pl_metal *metal);

struct pl_metal_swapchain_params {
    // The CAMetalLayer to present onto (bridged to void *). Required. Its
    // `device` / `pixelFormat` / `framebufferOnly` are configured by this call.
    void *layer;
};

#define pl_metal_swapchain_params(...) \
    (&(struct pl_metal_swapchain_params) { __VA_ARGS__ })

// Creates a `pl_swapchain` that presents onto the given CAMetalLayer.
PL_API pl_swapchain pl_metal_create_swapchain(pl_metal metal,
    const struct pl_metal_swapchain_params *params);

// Wrap an externally-owned MTLTexture (bridged to void *) as a renderable
// pl_tex, for callers that own their own drawable/render target (e.g. the
// libmpv render API, which hands a destination texture per frame). The
// returned pl_tex does not own the texture's lifetime beyond a balanced
// retain/release; destroy it with pl_tex_destroy before the frame is presented.
// Returns NULL if the texture's MTLPixelFormat has no matching pl_fmt.
PL_API pl_tex pl_metal_wrap_tex(pl_metal metal, void *mtl_texture);

PL_API_END

#endif // LIBPLACEBO_METAL_H_
