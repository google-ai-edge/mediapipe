load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mobile_calculators",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/graphs/face_mesh/subgraphs:face_renderer_gpu",
        "//mediapipe/modules/face_landmark:face_landmark_front_gpu",
        "//mediapipe/calculators/image:recolor_calculator",
        "//mediapipe/modules/selfie_segmentation:selfie_segmentation_gpu",
    ],
)

mediapipe_binary_graph(
    name = "beauty_graph",
    graph = "beauty_graph.pbtxt",
    output_name = "beauty_graph.binarypb",
    deps = [":mobile_calculators"],
)


# cc_library(
#     name = "selfie_segmentation_gpu_deps",
#     deps = [
#         "//mediapipe/calculators/core:flow_limiter_calculator",
        
#     ],
# )

# mediapipe_binary_graph(
#     name = "selfie_segmentation_gpu_binary_graph",
#     graph = "selfie_segmentation_gpu.pbtxt",
#     output_name = "selfie_segmentation_gpu.binarypb",
#     deps = [":selfie_segmentation_gpu_deps"],
# )