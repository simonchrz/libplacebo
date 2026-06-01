metal_opt = get_option('metal')
metal_build = metal_opt.require(host_machine.system() == 'darwin',
    error_message: 'the Metal backend requires an Apple (darwin) host')

# The Metal backend needs SPIRV-Cross (SPIR-V → MSL). Its homebrew pkg-config
# (spirv-cross-c) only advertises -lspirv-cross-c and omits the transitive
# static C++ libs, so pull those in explicitly.
spvc = dependency('spirv-cross-c', required: false)
metal_build = metal_build.require(spvc.found(),
    error_message: 'the Metal backend requires SPIRV-Cross (spirv-cross-c)')
components.set('metal', metal_build.allowed())

if metal_build.allowed()
  add_languages('objc', native: false, required: true)
  # The backend is written in manual-retain-release Objective-C; never ARC.
  add_project_arguments('-fno-objc-arc', language: 'objc')
  # The shared C warning policy (top-level meson.build `c_opts`) is only
  # registered for language 'c', so the .m files would otherwise warn on things
  # the rest of the tree mutes (e.g. -Wno-sign-compare for PL_ARRAY_APPEND_RAW,
  # -Wno-missing-field-initializers for the positional format table). Apply the
  # same policy to Objective-C so the backend builds as cleanly as the C sources.
  add_project_arguments(c_opts, language: 'objc')
  sources += [
    'metal/formats.m',
    'metal/gpu_tex.m',
    'metal/gpu_pass.m',
    'metal/gpu.m',
    'metal/metal.m',
    'metal/swapchain.m',
  ]
  build_deps += dependency('appleframeworks',
    modules: ['Metal', 'Foundation', 'QuartzCore'])
  build_deps += spvc
  spvc_libdir = spvc.get_variable(pkgconfig: 'libdir',
                                  default_value: '/opt/homebrew/lib')
  # core/glsl/msl are always required; hlsl/cpp/reflect/util may be absent when
  # SPIRV-Cross was built without those backends. Link only the ones present —
  # the C API lib only references the backends it was compiled with.
  foreach l : ['spirv-cross-core', 'spirv-cross-glsl', 'spirv-cross-msl']
    build_deps += cc.find_library(l, dirs: [spvc_libdir], required: true)
  endforeach
  foreach l : ['spirv-cross-hlsl', 'spirv-cross-cpp', 'spirv-cross-reflect',
               'spirv-cross-util']
    optlib = cc.find_library(l, dirs: [spvc_libdir], required: false)
    if optlib.found()
      build_deps += optlib
    endif
  endforeach
endif
