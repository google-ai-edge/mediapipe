load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application", "ios_framework", "ios_static_framework", "ios_unit_test")

POSETRACKING_DEPS = [
    "//mediapipe/objc/solutions/posetracking_gpu:posetracking_gpu_solution",
    "//mediapipe/objc:mediapipe_framework_ios",
    "//mediapipe/objc:mediapipe_input_sources_ios",
    "//mediapipe/objc:mediapipe_layer_renderer",
] + select({
    "//mediapipe:ios_i386": [],
    "//mediapipe:ios_x86_64": [],
    "//conditions:default": [
        "//mediapipe/graphs/pose_tracking:pose_tracking_gpu_deps",
        "//mediapipe/framework/formats:landmark_cc_proto",
    ],
})

genrule(
    name = "podgen",
    srcs = [
        "lindera-framework",
        "LinderaDetection.podspec",
    ],
    outs = ["LinderaDetection.zip"],
    cmd = """
    mkdir mediapipe/swift/solutions/lindera/frameworks
    unzip $(location lindera-framework) -d mediapipe/swift/solutions/lindera/frameworks
    cd mediapipe/swift/solutions/lindera/

    zip -r LinderaDetection.zip frameworks LinderaDetection.podspec
    cd ../../../../
    cp mediapipe/swift/solutions/lindera/LinderaDetection.zip $@
    """,
)

# bazel build //mediapipe/swift/solutions/lindera:lindera-framework -c opt --config=ios_fat --cxxopt=--std=c++17 --copt=-fembed-bitcode --linkopt="-s"
ios_static_framework(
    name = "lindera-framework",
    #    avoid_deps = POSETRACKING_DEPS,
    bundle_name = "LinderaDetection",
    minimum_os_version = "12.0",
    visibility = ["//visibility:public"],
    deps = [
        ":lindera",
        #        "//third_party:opencv",
        #        "@ios_opencv//:OpencvFramework",
    ] + POSETRACKING_DEPS,
)

swift_library(
    name = "lindera",
    srcs = [
        "Asensei3D.swift",
        "Lindera.swift",
        "utils.swift",
    ],
    linkopts = [
        "-lc++",
        "-std=c++17",
        "-lstdc++",
    ],
    module_name = "LinderaDetection",
    visibility = ["//visibility:public"],
    deps = POSETRACKING_DEPS,
)
