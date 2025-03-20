load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

mediapipe_files(srcs = [
    "libopencv_core.3.4.darwin.a",
    "libopencv_core.3.4.darwin_arm64.a",
    "libopencv_imgproc.3.4.darwin.a",
    "libopencv_imgproc.3.4.darwin_arm64.a",
    "opencv2.xcframework.zip",
])

cc_import(
    name = "opencv_core_darwin",
    static_library = "libopencv_core.3.4.darwin.a",
)

cc_import(
    name = "opencv_core_darwin_arm64",
    static_library = "libopencv_core.3.4.darwin_arm64.a",
)

cc_import(
    name = "opencv_imgproc_darwin",
    static_library = "libopencv_imgproc.3.4.darwin.a",
)

cc_import(
    name = "opencv_imgproc_darwin_arm64",
    static_library = "libopencv_imgproc.3.4.darwin_arm64.a",
)

cc_library(
    name = "opencv_darwin",
    linkstatic = 1,
    deps = [
        ":opencv_core_darwin",
        ":opencv_imgproc_darwin",
        "@macos_opencv//:opencv",  # For headers
    ],
)

cc_library(
    name = "opencv_darwin_arm64",
    linkstatic = 1,
    deps = [
        ":opencv_core_darwin_arm64",
        ":opencv_imgproc_darwin_arm64",
        "@macos_opencv//:opencv",  # For headers
    ],
)
