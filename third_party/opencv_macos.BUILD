# Description:
#   OpenCV libraries for video/image processing on MacOS

load("@bazel_skylib//lib:paths.bzl", "paths")

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# Example configurations:
#
# # OpenCV 3
# To configure OpenCV 3, obtain the path of OpenCV 3 from Homebrew. The
# following commands show the output of the command with version 3.4.16_10:
#
# $ brew ls opencv@3 | grep version.hpp
# $ /opt/homebrew/Cellar/opencv@3/3.4.16_10/include/opencv2/core/version.hpp
#
# Then set path in "macos_opencv" rule in the WORKSPACE file to
# "/opt/homebrew/Cellar" and the PREFIX below to "opencv/<version>" (e.g.
# "opencv/3.4.16_10" for the example above).
#
# # OpenCV 4
# To configure OpenCV 4, obtain the path of OpenCV 4 from Homebrew. The
# following commands show the output of the command with version 4.10.0_12:
#
# $ brew ls opencv | grep version.hpp
# $ /opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4/opencv2/core/version.hpp
# $ /opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4/opencv2/dnn/version.hpp
#
# Then set path in "macos_opencv" rule in the WORKSPACE file to
# "/opt/homebrew/Cellar" and the PREFIX below to "opencv/<version>" (e.g.
# "opencv/4.10.0_12" for the example above). For OpenCV 4, you will also need to
# adjust the include paths. The header search path should be
# "include/opencv4/opencv2/**/*.h*" and the include prefix needs to be set to
# "include/opencv4".

PREFIX = "opt/opencv@3"

cc_library(
    name = "opencv",
    srcs = glob(
        [
            paths.join(PREFIX, "lib/libopencv_core.dylib"),
            paths.join(PREFIX, "lib/libopencv_calib3d.dylib"),
            paths.join(PREFIX, "lib/libopencv_features2d.dylib"),
            paths.join(PREFIX, "lib/libopencv_highgui.dylib"),
            paths.join(PREFIX, "lib/libopencv_imgcodecs.dylib"),
            paths.join(PREFIX, "lib/libopencv_imgproc.dylib"),
            paths.join(PREFIX, "lib/libopencv_video.dylib"),
            paths.join(PREFIX, "lib/libopencv_videoio.dylib"),
        ],
    ),
    hdrs = glob([paths.join(PREFIX, "include/opencv2/**/*.h*")]),
    includes = [paths.join(PREFIX, "include/")],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
