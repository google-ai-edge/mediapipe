# Description:
#   OpenCV xcframework for video/image processing on iOS.

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

load(
    "@build_bazel_rules_apple//apple:apple.bzl",
    "apple_static_xcframework_import",
)

load(
    "@//third_party:opencv_ios_source.bzl",
    "select_headers",
    "unzip_opencv_xcframework",
)

# Build opencv2.xcframework from source using a convenience script provided in
# OPENCV sources and zip the xcframework. We only build the modules required by MediaPipe by specifying
# the modules to be ignored as command line arguments. 
# We also specify the simulator and device architectures we are building for.
# Currently we only support iOS arm64 (M1 Macs) and x86_64(Intel Macs) simulators
# and arm64 iOS devices.
# Bitcode and Swift support. Swift support will be added in when the final binary
# for MediaPipe iOS Task libraries are built. Shipping with OPENCV built with
# Swift support throws linker errors when the MediaPipe framework is used from
# an iOS project.
genrule(
    name = "build_opencv_xcframework",
    srcs = glob(["opencv-4.5.1/**"]),
    outs = ["opencv2.xcframework.zip"],
    cmd = "&&".join([
        "$(location opencv-4.5.1/platforms/apple/build_xcframework.py) \
        --iphonesimulator_archs arm64,x86_64 \
        --iphoneos_archs arm64 \
        --without dnn \
        --without ml \
        --without stitching \
        --without photo \
        --without objdetect \
        --without gapi \
        --without flann \
        --disable PROTOBUF \
        --disable-bitcode \
        --disable-swift \
        --build_only_specified_archs \
        --out $(@D)",
        "cd $(@D)",
        "zip --symlinks -r opencv2.xcframework.zip opencv2.xcframework",
    ]),
)

# Unzips `opencv2.xcframework.zip` built from source by `build_opencv_xcframework`
# genrule and returns an exhaustive list of all its files including symlinks.
unzip_opencv_xcframework (
    name = "opencv2_unzipped_xcframework_files",
    zip_file = "opencv2.xcframework.zip",
)

# Imports the files of the unzipped `opencv2.xcframework` as an apple static
# framework which can be linked to iOS targets.
apple_static_xcframework_import(
    name = "opencv_xcframework",
    xcframework_imports = [":opencv2_unzipped_xcframework_files"],
    visibility = ["//visibility:public"],
)

# Filters the headers for each platform in `opencv2.xcframework` which will be
# used as headers in a `cc_library` that can be linked to C++ targets.
select_headers(
    name = "opencv_xcframework_device_headers",
    srcs = [":opencv_xcframework"],
    platform = "ios-arm64",
)

select_headers(
    name = "opencv_xcframework_simulator_headers",
    srcs = [":opencv_xcframework"],
    platform = "ios-arm64_x86_64-simulator",
)

# `cc_library` that can be linked to C++ targets to import opencv headers.
cc_library(
    name = "opencv",
    hdrs = select({
        "@//mediapipe:ios_x86_64" : [
            ":opencv_xcframework_simulator_headers"
        ],
        "@//mediapipe:ios_sim_arm64" : [
            ":opencv_xcframework_simulator_headers"
        ],
        "@//mediapipe:ios_arm64" : [
            ":opencv_xcframework_simulator_headers"
        ],
        # A value from above is chosen arbitarily.
        "//conditions:default": [
            ":opencv_xcframework_simulator_headers"
        ],
    }),
    copts = [
        "-std=c++11",
        "-x objective-c++",
    ],
    include_prefix = "opencv2",
    linkopts = [
        "-framework AssetsLibrary",
        "-framework CoreFoundation",
        "-framework CoreGraphics",
        "-framework CoreMedia",
        "-framework Accelerate",
        "-framework CoreImage",
        "-framework AVFoundation",
        "-framework CoreVideo",
        "-framework QuartzCore",
    ],
    strip_include_prefix = select({
        "@//mediapipe:ios_x86_64" : "opencv2.xcframework/ios-arm64_x86_64-simulator/opencv2.framework/Versions/A/Headers",
        "@//mediapipe:ios_sim_arm64" :"opencv2.xcframework/ios-arm64_x86_64-simulator/opencv2.framework/Versions/A/Headers",
        "@//mediapipe:ios_arm64" : "opencv2.xcframework/ios-arm64/opencv2.framework/Versions/A/Headers",
        # Random value is selected for default cases.
        "//conditions:default": "opencv2.xcframework/ios-arm64_x86_64-simulator/opencv2.framework/Versions/A/Headers",
    }),
    visibility = ["//visibility:public"],
    deps = [":opencv_xcframework"],
    # data = [":opencv_xcframework"]
)