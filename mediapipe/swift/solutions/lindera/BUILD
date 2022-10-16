load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "Lindera",
    srcs = glob(["*.swift"]),
    deps = [
     "//mediapipe/objc/solutions/posetracking_gpu:posetracking_gpu_solution",
     ]
)
