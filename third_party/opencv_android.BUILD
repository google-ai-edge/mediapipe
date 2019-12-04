# Description:
#   OpenCV libraries for video/image processing on Android

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

OPENCV_LIBRARY_NAME = "libopencv_java3.so"

OPENCVANDROIDSDK_NATIVELIBS_PATH = "sdk/native/libs/"

OPENCVANDROIDSDK_JNI_PATH = "sdk/native/jni/"

[cc_library(
    name = "libopencv_" + arch,
    srcs = [OPENCVANDROIDSDK_NATIVELIBS_PATH + arch + "/" + OPENCV_LIBRARY_NAME],
    hdrs = glob([
        OPENCVANDROIDSDK_JNI_PATH + "include/**/*.h",
        OPENCVANDROIDSDK_JNI_PATH + "include/**/*.hpp",
    ]),
    includes = [
        OPENCVANDROIDSDK_JNI_PATH + "include",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
) for arch in [
    "arm64-v8a",
    "armeabi-v7a",
    "x86",
    "x86_64",
]]
