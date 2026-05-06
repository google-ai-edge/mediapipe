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
        #"include/aarch64-linux-gnu/opencv4/opencv2/cvconfig.h",
        #"include/arm-linux-gnueabihf/opencv4/opencv2/cvconfig.h",
        #"include/x86_64-linux-gnu/opencv4/opencv2/cvconfig.h",
        #"include/opencv4/opencv2/**/*.h*",
    ]),
    includes = [
        # For OpenCV 4.x
        #"include/aarch64-linux-gnu/opencv4/",
        #"include/arm-linux-gnueabihf/opencv4/",
        #"include/x86_64-linux-gnu/opencv4/",
        #"include/opencv4/",
    ],
    linkopts = [
        "-L/usr/local/lib",
        "-L/usr/local/lib",
        "-L/usr/local/lib",
        "-l:libopencv_core.so",
        "-l:libopencv_calib3d.so",
        "-l:libopencv_features2d.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_video.so",
        "-l:libopencv_videoio.so",
    ],
    visibility = ["//visibility:public"],
)
