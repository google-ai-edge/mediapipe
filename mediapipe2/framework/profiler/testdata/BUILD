load("//mediapipe/framework:encode_binary_proto.bzl", "encode_binary_proto")

package(default_visibility = ["//mediapipe/framework/profiler:__subpackages__"])

licenses(["notice"])

# Compile any proto data into wire format for use in our tests.
[encode_binary_proto(
    name = graph.split("/")[-1].rsplit(".", 1)[0],
    input = graph,
    message_type = "mediapipe.GraphProfile",
    output = graph.rsplit(".", 1)[0] + ".binarypb",
    deps = ["//mediapipe/framework:calculator_profile_proto"],
) for graph in glob(["profile_*.pbtxt"])]

filegroup(
    name = "mediapipe_profile_graphs",
    srcs = [binarypb.rsplit(".", 1)[0] + ".binarypb" for binarypb in glob(["profile_*.pbtxt"])],
)

filegroup(
    name = "pbtxt_files",
    srcs = glob(["*.pbtxt"]),
)
