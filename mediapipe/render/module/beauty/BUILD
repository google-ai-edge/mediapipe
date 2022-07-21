package(default_visibility = ["//visibility:public"])
cc_library(
    name = "FaceMeshGPULibrary",
    srcs = [
        "face_mesh_module.cc",
        "face_mesh_beauty_render.cc",
        "face_mesh_module_imp.cc",
    ],
    hdrs = [
        "face_mesh_module.h",
        "face_mesh_beauty_render.h",
        "face_mesh_module_imp.h",
    ],
    data = [
        "//mediapipe/graphs/face_mesh:face_mesh_mobile_gpu.binarypb",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/render/module/common:olamodule_common_library",
        
    ] + select({
        "//mediapipe:ios_i386": [],
        "//mediapipe:ios_x86_64": [],
        "//mediapipe:apple": [
            "//mediapipe/render/core:core-ios",
         ],
        "//conditions:default": [
            # "//mediapipe/render/core:core",
            "//mediapipe/graphs/face_mesh:mobile_calculators",
            "//mediapipe/framework/formats:landmark_cc_proto",
        ],
    }),
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": ["-std=c++17"],
    }),
)
