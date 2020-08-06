# Description:
#   OpenCV libraries for video/image processing on Linux

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# The following build rule assumes that OpenCV is installed by
# 'apt-get install libopencv-core-dev libopencv-highgui-dev \'
# '                libopencv-calib3d-dev libopencv-features2d-dev \'
# '                libopencv-imgproc-dev libopencv-video-dev'
# on Debian buster/Ubuntu 18.04.
# If you install OpenCV separately, please modify the build rule accordingly.
cc_library(
    name = "opencv",
    srcs = glob(
        [
            "lib/x86_64-linux-gnu/libopencv_core.so",
            "lib/x86_64-linux-gnu/libopencv_calib3d.so",
            "lib/x86_64-linux-gnu/libopencv_features2d.so",
            "lib/x86_64-linux-gnu/libopencv_highgui.so",
            "lib/x86_64-linux-gnu/libopencv_imgcodecs.so",
            "lib/x86_64-linux-gnu/libopencv_imgproc.so",
            "lib/x86_64-linux-gnu/libopencv_video.so",
            "lib/x86_64-linux-gnu/libopencv_videoio.so",
        ],
    ),
    hdrs = glob([
        # For OpenCV 3.x
        "include/opencv2/**/*.h*",
        # For OpenCV 4.x
        # "include/opencv4/opencv2/**/*.h*",
    ]),
    includes = [
        # For OpenCV 3.x
        "include/",
        # For OpenCV 4.x
        # "include/opencv4/",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
