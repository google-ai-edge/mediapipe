package(default_visibility = ["//visibility:public"])
load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_application", "ios_framework", "ios_static_framework"
)


ios_framework(
    name = "FaceMeshGPUFramework",
    bundle_id = "com.ola.framework.facemesh",
    infoplists = [
        "Info.plist",
    ],
    minimum_os_version = "11.0",
    families = [
        "iphone",
        "ipad",
    ],
    deps = [
        ":FaceMeshGPULibrary",
        "@ios_opencv//:OpencvFramework",
    ],
)

FACEUNITY_SRCS = [
        "face_mesh_module.cc",
        "face_mesh_beauty_render.cc",
        "face_mesh_module_imp.cc",
]

FACEUNITY_HDRS = [
        "face_mesh_module.h",
        "face_mesh_common.h",
        "face_mesh_beauty_render.h",
        "face_mesh_module_imp.h",
]

cc_library(
    name = "FaceMeshGPULibrary",
    srcs = FACEUNITY_SRCS,
    hdrs = FACEUNITY_HDRS,
    visibility = ["//visibility:public"],
    alwayslink = True,
    linkstatic = True,
    deps = [
        "//mediapipe/render/module/common:olamodule_common_library",
       
        
    ] + select({
        "//mediapipe:apple": [
            "//mediapipe/render/core:core-ios",
            "//mediapipe/graphs/face_mesh:mobile_calculators",
            "//mediapipe/framework/formats:landmark_cc_proto",
            "//mediapipe/render/module/beauty/filters:BeautyFilters",
         ],
        "//conditions:default": [
            "//mediapipe/render/core:core",
            "//mediapipe/graphs/face_mesh:mobile_calculators",
            "//mediapipe/framework/formats:landmark_cc_proto",
            "//mediapipe/render/module/beauty/filters:BeautyFilters",
        ],
    }),
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            # "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": ["-std=c++17"],
    }),
)


exports_files(
    srcs = [
        "whiten.png",
    ],
)