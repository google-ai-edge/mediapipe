package(default_visibility = ["//visibility:public"])

def apple_framework(name):
    native.cc_library(
        name = name,
        linkopts = ["-framework {}".format(name)],
    )

# ===== Danh sách frameworks =====
APPLE_FRAMEWORKS = [
    "CoreGraphics",
    "CoreMedia",
    "UIKit",
    "Accelerate",
    "CoreVideo",
    "Metal",
    "MetalKit",
    "MetalPerformanceShaders",
    "AVFoundation",
    "AVFAudio",
    "Foundation",
    "CoreImage",
    "XCTest",
    "GLKit",
    "OpenGLES",
    "QuartzCore",
    "CoreAudio",
    "MediaToolbox",
]

# ===== Auto generate rules =====
[apple_framework(fw) for fw in APPLE_FRAMEWORKS]
