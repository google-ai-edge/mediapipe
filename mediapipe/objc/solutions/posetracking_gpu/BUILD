objc_library(
    name = "posetracking_gpu_solution",
    srcs = glob([
        "*.h",
        "*.mm",
    ]),
    hdrs = ["PoseTrackingOptions.h"],
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    sdk_frameworks = ["Accelerate"],
    # This build rule is public to allow external customers to build their own iOS apps.
    visibility = ["//visibility:public"],
)
