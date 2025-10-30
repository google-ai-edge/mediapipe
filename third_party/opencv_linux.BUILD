# Description:
#   OpenCV libraries for video/image processing on Linux

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# The following build rule assumes that OpenCV is installed by
# 'apt-get install libopencv-core-dev libopencv-highgui-dev \'
# '                libopencv-calib3d-dev libopencv-features2d-dev \'
# '                libopencv-imgproc-dev libopencv-video-dev'
# on Debian Buster/Ubuntu 18.04.
# If you install OpenCV separately, please modify the build rule accordingly.
cc_library(
    name = "opencv",
    hdrs = glob([
        # For OpenCV 4.x
        "include/opencv4/opencv2/**/*.h*",
        "include/x86_64-linux-gnu/opencv4/opencv2/**/*.h*",
        "include/aarch64-linux-gnu/opencv4/opencv2/**/*.h*",
        "include/arm-linux-gnueabihf/opencv4/opencv2/**/*.h*",
    ]),
    includes = [
        # For OpenCV 4.x
        "include/opencv4",
        "include/x86_64-linux-gnu/opencv4",
        "include/aarch64-linux-gnu/opencv4",
        "include/arm-linux-gnueabihf/opencv4",
    ],
    linkopts = [
        "-l:libopencv_core.so",
        "-l:libopencv_calib3d.so",
        "-l:libopencv_features2d.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_video.so",
        "-l:libopencv_videoio.so",
        "-l:libopencv_optflow.so",
    ],
    visibility = ["//visibility:public"],
)
