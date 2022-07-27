load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework", "ios_static_framework", "ios_application")
# bazel build --copt=-fembed-bitcode --apple_bitcode=embedded --config=ios_arm64
# 用上面这条指令build


ios_framework(
    name = "OlaFaceUnityFramework",
    hdrs = [
        "OlaFaceUnity.h",
    ],
    infoplists = ["Info.plist"],
    bundle_id = "com.ola.olarender.develop",
    families = ["iphone", "ipad"],
    minimum_os_version = "11.0",
    deps = [
         ":OlaFaceUnityLibrary",
         "@ios_opencv//:OpencvFramework",
    ],
)

objc_library(
    name = "OlaFaceUnityLibrary",
    hdrs = ["OlaFaceUnity.h"],
    srcs = ["OlaFaceUnity.mm"],
    visibility = ["//visibility:public"],
    deps = [
         "//mediapipe/render/module/beauty:FaceMeshGPULibrary",
         "@ios_opencv//:OpencvFramework",
    ],
    data = [
        "//mediapipe/graphs/face_mesh:face_mesh_mobile_gpu.binarypb",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
        "//mediapipe/render/module/beauty:whiten.png",
    ],
     copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    # alwayslink = True,
    sdk_frameworks = [
        "AVFoundation",
        "CoreGraphics",
        "CoreMedia",
        "UIKit",
        "OpenGLES",
        "MetalKit",
        "GLKit",
        "Metal",
        "CoreVideo",
        "IOSurface",
        "Foundation",
        "AssetsLibrary",
        "QuartzCore",
    ],
)
