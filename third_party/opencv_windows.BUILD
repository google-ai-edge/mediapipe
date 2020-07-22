# Description:
#   OpenCV libraries for video/image processing on Windows

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

OPENCV_VERSION = "3410"  # 3.4.10

# The following build rule assumes that the executable "opencv-3.4.10-vc14_vc15.exe"
# is downloaded and the files are extracted to local.
# If you install OpenCV separately, please modify the build rule accordingly.
cc_library(
    name = "opencv",
    srcs = [
        "x64/vc15/lib/opencv_world" + OPENCV_VERSION + ".lib",
        "x64/vc15/bin/opencv_world" + OPENCV_VERSION + ".dll",
    ],
    hdrs = glob(["include/opencv2/**/*.h*"]),
    includes = ["include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
