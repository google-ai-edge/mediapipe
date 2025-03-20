# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")
load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)

licenses(["notice"])  # Apache License 2.0

package(default_visibility = ["//visibility:public"])

exports_files([
    "LICENSE",
    "requirements_lock.txt",
    "requirements_lock_3_10.txt",
    "requirements_lock_3_11.txt",
    "requirements_lock_3_12.txt",
])

mediapipe_files(srcs = [
    "libc++_shared.so",
])

cc_library(
    name = "glog",
    visibility = ["//visibility:public"],
    deps = select({
        "//mediapipe:android_x86": [
            "@com_github_glog_glog_no_gflags//:glog",
        ],
        "//mediapipe:android_x86_64": [
            "@com_github_glog_glog_no_gflags//:glog",
        ],
        "//mediapipe:android_arm": [
            "@com_github_glog_glog_no_gflags//:glog",
        ],
        "//mediapipe:android_arm64": [
            "@com_github_glog_glog_no_gflags//:glog",
        ],
        "//mediapipe:ios": [
            "@com_github_glog_glog_no_gflags//:glog",
        ],
        "//mediapipe:macos": [
            "@com_github_glog_glog//:glog",
        ],
        "//mediapipe:windows": [
            "@com_github_glog_glog_windows//:glog",
        ],
        "//conditions:default": [
            "@com_github_glog_glog//:glog",
        ],
    }),
)

config_setting(
    name = "opencv_source_build",
    define_values = {
        "OPENCV": "source",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "opencv_prebuilt_build",
    define_values = {
        "OPENCV": "prebuilt",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "opencv_ios_arm64_source_build",
    define_values = {
        "OPENCV": "source",
    },
    values = {
        "apple_platform_type": "ios",
        "cpu": "ios_arm64",
    },
)

config_setting(
    name = "opencv_ios_sim_arm64_source_build",
    define_values = {
        "OPENCV": "source",
    },
    values = {
        "apple_platform_type": "ios",
        "cpu": "ios_sim_arm64",
    },
)

config_setting(
    name = "opencv_ios_x86_64_source_build",
    define_values = {
        "OPENCV": "source",
    },
    values = {
        "apple_platform_type": "ios",
        "cpu": "ios_x86_64",
    },
)

config_setting(
    name = "opencv_ios_sim_fat_source_build",
    define_values = {
        "OPENCV": "source",
    },
    values = {
        "apple_platform_type": "ios",
        "ios_multi_cpus": "sim_arm64,x86_64",
    },
)

config_setting(
    name = "opencv_darwin_prebuild",
    define_values = {
        "OPENCV": "prebuilt",
    },
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
)

config_setting(
    name = "opencv_darwin_arm64_prebuild",
    define_values = {
        "OPENCV": "prebuilt",
    },
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64",
    },
)

alias(
    name = "opencv_prebuilt",
    actual = select({
        "//mediapipe:ios": "@ios_opencv//:opencv",
        ":opencv_darwin_prebuild": "//third_party/prebuilts:opencv_darwin",
        ":opencv_darwin_arm64_prebuild": "//third_party/prebuilts:opencv_darwin_arm64",
        "//conditions:default": ":opencv_cmake",
    }),
)

alias(
    name = "opencv",
    actual = select({
        ":opencv_source_build": ":opencv_cmake",
        ":opencv_prebuilt_build": ":opencv_prebuilt",
        ":opencv_ios_sim_arm64_source_build": "@ios_opencv_source//:opencv",
        ":opencv_ios_arm64_source_build": "@ios_opencv_source//:opencv",
        ":opencv_ios_x86_64_source_build": "@ios_opencv_source//:opencv",
        "//conditions:default": ":opencv_binary",
    }),
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "opencv_ios_xcframework_files_bzl",
    srcs = ["opencv_ios_xcframework_files.bzl"],
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "opencv_ios_source_bzl",
    srcs = ["opencv_ios_source.bzl"],
    visibility = ["//visibility:private"],
)

# Note: this determines the order in which the libraries are passed to the
# linker, so if library A depends on library B, library B must come _after_.
# Hence core is at the bottom.
OPENCV_MODULES = [
    "calib3d",
    "features2d",
    "highgui",
    "video",
    "videoio",
    "imgcodecs",
    "imgproc",
    "core",
]

# Note: passing both BUILD_SHARED_LIBS=ON and BUILD_STATIC_LIBS=ON to cmake
# still only builds the shared libraries, so we have to choose one or the
# other. We build shared libraries by default, but this variable can be used
# to switch to static libraries.
OPENCV_SHARED_LIBS = True

OPENCV_SO_VERSION = "3.4"

cmake(
    name = "opencv_cmake",
    # Values to be passed as -Dkey=value on the CMake command line;
    # here are serving to provide some CMake script configuration options
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        # The module list is always sorted alphabetically so that we do not
        # cause a rebuild when changing the link order.
        "BUILD_LIST": ",".join(sorted(OPENCV_MODULES)),
        "BUILD_TESTS": "OFF",
        "BUILD_PERF_TESTS": "OFF",
        "BUILD_EXAMPLES": "OFF",
        "BUILD_SHARED_LIBS": "ON" if OPENCV_SHARED_LIBS else "OFF",
        "WITH_ITT": "OFF",
        "WITH_JASPER": "OFF",
        "WITH_JPEG": "ON",
        "WITH_PNG": "ON",
        "WITH_TIFF": "ON",
        "WITH_OPENCL": "OFF",
        "WITH_WEBP": "OFF",
        # Optimization flags
        "CV_ENABLE_INTRINSICS": "ON",
        "WITH_EIGEN": "ON",
        "WITH_PTHREADS": "ON",
        "WITH_PTHREADS_PF": "ON",
        # When building tests, by default Bazel builds them in dynamic mode.
        # See https://docs.bazel.build/versions/master/be/c-cpp.html#cc_binary.linkstatic
        # For example, when building //mediapipe/calculators/video:opencv_video_encoder_calculator_test,
        # the dependency //mediapipe/framework/formats:image_frame_opencv will
        # be built as a shared library, which depends on a cv::Mat constructor,
        # and expects it to be provided by the main executable. The main
        # executable depends on libimage_frame_opencv.so and links in
        # libopencv_core.a, which contains cv::Mat. However, if
        # libopencv_core.a marks its symbols as hidden, then cv::Mat is in
        # opencv_video_encoder_calculator_test but it is not exported, so
        # libimage_frame_opencv.so fails to find it.
        "OPENCV_SKIP_VISIBILITY_HIDDEN": "ON" if not OPENCV_SHARED_LIBS else "OFF",
        # The COPY actions in modules/python/python_loader.cmake have issues with symlinks.
        # In any case, we don't use this.
        "OPENCV_SKIP_PYTHON_LOADER": "ON",
        # Need to set this too, for the same reason.
        "BUILD_opencv_python": "OFF",
        # Ccache causes issues in some of our CI setups. It's not clear that
        # ccache would be able to work across sandboxed Bazel builds, either.
        # In any case, Bazel does its own caching of the rule's outputs.
        "ENABLE_CCACHE": "OFF",
    },
    lib_source = "@opencv//:all",
    linkopts = [] if OPENCV_SHARED_LIBS else [
        # When using static libraries, the binary that eventually depends on the
        # libraries also needs to link in their dependencies, which therefore
        # have to be listed here.
        # This list depends on which dependencies CMake finds when it configures
        # the build, and so depends on what is installed on the local system.
        # After building, the linkopts for the current setup can be extracted
        # from lib/pkgconfig/opencv.pc in bazel-out
        "-ljpeg",
        "-lpng",
        "-lz",
        "-ltiff",
        "-lImath",
        "-lIlmImf",
        "-lIex",
        "-lHalf",
        "-lIlmThread",
        "-ldc1394",
        "-lavcodec",
        "-lavformat",
        "-lavutil",
        "-lswscale",
        "-lavresample",
        "-ldl",
        "-lm",
        "-lpthread",
        "-lrt",
    ],
    out_shared_libs = select({
        "@bazel_tools//src/conditions:darwin": ["libopencv_%s.%s.dylib" % (module, OPENCV_SO_VERSION) for module in OPENCV_MODULES],
        # Only the shared objects listed here will be linked in the directory
        # that Bazel adds to the RUNPATH of dependent executables. You cannot
        # list both the versioned and unversioned name of the .so, and the
        # versioned name is the one that the executables actually reference.
        "//conditions:default": ["libopencv_%s.so.%s" % (module, OPENCV_SO_VERSION) for module in OPENCV_MODULES],
    }) if OPENCV_SHARED_LIBS else None,
    out_static_libs = [
        "libopencv_%s.a" % module
        for module in OPENCV_MODULES
    ] if not OPENCV_SHARED_LIBS else None,
)

alias(
    name = "opencv_binary",
    actual = select({
        "//mediapipe:android_x86": "@android_opencv//:libopencv_x86",
        "//mediapipe:android_x86_64": "@android_opencv//:libopencv_x86_64",
        "//mediapipe:android_arm": "@android_opencv//:libopencv_armeabi-v7a",
        "//mediapipe:android_arm64": "@android_opencv//:libopencv_arm64-v8a",
        "//mediapipe:ios": "@ios_opencv//:opencv",
        "//mediapipe:macos": "@macos_opencv//:opencv",
        "//mediapipe:windows": "@windows_opencv//:opencv",
        "//conditions:default": "@linux_opencv//:opencv",
    }),
)

# When "--define EXCLUDE_OPENCV_SO_LIB=1" is set in the build command,
# the OpenCV so libraries will be excluded from the AAR package to
# save the package size.
config_setting(
    name = "exclude_opencv_so_lib",
    define_values = {
        "EXCLUDE_OPENCV_SO_LIB": "1",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "android_jni_opencv_cc_lib",
    srcs = select({
        "//mediapipe:android_arm64": ["@android_opencv//:libopencv_java4_so_arm64-v8a"],
        "//mediapipe:android_arm": ["@android_opencv//:libopencv_java4_so_armeabi-v7a"],
        "//mediapipe:android_x86": ["@android_opencv//:libopencv_java4_so_x86"],
        "//mediapipe:android_x86_64": ["@android_opencv//:libopencv_java4_so_x86_64"],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "libffmpeg",
    visibility = ["//visibility:public"],
    deps = select({
        "//mediapipe:android_x86": [],
        "//mediapipe:android_x86_64": [],
        "//mediapipe:android_arm": [],
        "//mediapipe:android_arm64": [],
        "//mediapipe:ios": [],
        "//mediapipe:macos": [
            "@macos_ffmpeg//:libffmpeg",
        ],
        "//conditions:default": [
            "@linux_ffmpeg//:libffmpeg",
        ],
    }),
)

java_plugin(
    name = "autovalue_plugin",
    processor_class = "com.google.auto.value.processor.AutoValueProcessor",
    deps = [
        "@maven//:com_google_auto_value_auto_value",
        "@maven//:com_google_auto_value_auto_value_annotations",
    ],
)

java_plugin(
    name = "autobuilder_plugin",
    processor_class = "com.google.auto.value.processor.AutoBuilderProcessor",
    deps = [
        "@maven//:com_google_auto_value_auto_value",
        "@maven//:com_google_auto_value_auto_value_annotations",
    ],
)

java_library(
    name = "autovalue",
    exported_plugins = [
        ":autovalue_plugin",
        ":autobuilder_plugin",
    ],
    neverlink = 1,
    exports = [
        "@maven//:com_google_auto_value_auto_value",
        "@maven//:com_google_auto_value_auto_value_annotations",
    ],
)

java_import(
    name = "any_java_proto",
    jars = [
        "@com_google_protobuf//java/core:libcore.jar",
    ],
)

cc_binary(
    name = "libc++_shared",
    srcs = ["libc++_shared.so"],
    linkshared = 1,
    linkstatic = 1,
)

cc_library(
    name = "libc++_shared_lib",
    srcs = [":libc++_shared.so"],
    alwayslink = 1,
)
