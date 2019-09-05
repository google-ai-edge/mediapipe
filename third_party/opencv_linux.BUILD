# Description:
#   OpenCV libraries for video/image processing on Linux

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# The following build rule assumes that OpenCV is installed by
# 'apt-get install libopencv-core-dev libopencv-highgui-dev \'
# '                libopencv-imgproc-dev libopencv-video-dev' on Debian/Ubuntu.
# If you install OpenCV separately, please modify the build rule accordingly.
cc_library(
    name = "opencv",
    srcs = glob(
        [
            "lib/x86_64-linux-gnu/libopencv_core.so",
            "lib/x86_64-linux-gnu/libopencv_highgui.so",
            "lib/x86_64-linux-gnu/libopencv_imgcodecs.so",
            "lib/x86_64-linux-gnu/libopencv_imgproc.so",
            "lib/x86_64-linux-gnu/libopencv_video.so",
            "lib/x86_64-linux-gnu/libopencv_videoio.so",
        ],
    ),
    hdrs = glob(["include/opencv2/**/*.h*"]),
    includes = ["include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
