# Description:
#   OpenCV libraries for video/image processing on Windows ARM64

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

OPENCV_VERSION = "3410"  # 3.4.10

config_setting(
    name = "opt_build",
    values = {"compilation_mode": "opt"},
)

config_setting(
    name = "dbg_build",
    values = {"compilation_mode": "dbg"},
)

# OpenCV ARM64 build configuration for Windows 
# Uses the prebuilt OpenCV libraries from C:\opencv\install
cc_library(
    name = "opencv",
    srcs = select({
        ":opt_build": [
            "ARM64/vc17/lib/opencv_world" + OPENCV_VERSION + ".lib",
            "ARM64/vc17/bin/opencv_world" + OPENCV_VERSION + ".dll",
        ],
        ":dbg_build": [
            "ARM64/vc17/lib/opencv_world" + OPENCV_VERSION + "d.lib",
            "ARM64/vc17/bin/opencv_world" + OPENCV_VERSION + "d.dll",
        ],
    }),
    hdrs = glob(["include/opencv2/**/*.h*"]),
    includes = ["include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)