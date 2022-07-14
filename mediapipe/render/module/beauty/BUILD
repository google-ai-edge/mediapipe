cc_library(
    name = "FaceMeshGPULibrary",
    copts = ["-std=c++17"],
    srcs = [
        "face_mesh_module.hpp",
    
    ],
    hdrs = ["face_mesh_module.cpp"],
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
        "//conditions:default": [
            "//mediapipe/graphs/face_mesh:mobile_calculators",
            "//mediapipe/framework/formats:landmark_cc_proto",
        ],
    }),
)
