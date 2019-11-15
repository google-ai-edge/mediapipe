# Description:
#   OpenCV libraries for video/image processing on MacOS

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# The following build rule assumes that OpenCV is installed by
# 'brew install opencv@3' command on macos.
# If you install OpenCV separately, please modify the build rule accordingly.
cc_library(
    name = "opencv",
    srcs = glob(
        [
            "local/opt/opencv@3/lib/libopencv_core.dylib",
            "local/opt/opencv@3/lib/libopencv_calib3d.dylib",
            "local/opt/opencv@3/lib/libopencv_features2d.dylib",
            "local/opt/opencv@3/lib/libopencv_highgui.dylib",
            "local/opt/opencv@3/lib/libopencv_imgcodecs.dylib",
            "local/opt/opencv@3/lib/libopencv_imgproc.dylib",
            "local/opt/opencv@3/lib/libopencv_video.dylib",
            "local/opt/opencv@3/lib/libopencv_videoio.dylib",
        ],
    ),
    hdrs = glob(["local/opt/opencv@3/include/opencv2/**/*.h*"]),
    includes = ["local/opt/opencv@3/include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
