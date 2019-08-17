# Description:
#   OpenCV libraries for video/image processing on MacOS

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# The following build rule assumes that OpenCV is installed by
# 'brew install opencv' command on macos.
# If you install OpenCV separately, please modify the build rule accordingly.
cc_library(
    name = "opencv",
    srcs = glob(
        [
            "local/opt/opencv/lib/libopencv_core.dylib",
            "local/opt/opencv/lib/libopencv_highgui.dylib",
            "local/opt/opencv/lib/libopencv_imgcodecs.dylib",
            "local/opt/opencv/lib/libopencv_imgproc.dylib",
            "local/opt/opencv/lib/libopencv_optflow.dylib",
            "local/opt/opencv/lib/libopencv_video.dylib",
            "local/opt/opencv/lib/libopencv_videoio.dylib",
        ],
    ),
    hdrs = glob(["local/opt/opencv/include/opencv4/**/*.h*"]),
    includes = ["local/opt/opencv/include/opencv4/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
