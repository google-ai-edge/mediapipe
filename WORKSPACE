workspace(name = "mediapipe")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

########################################################### Python support start

http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-b162c7c88a253e3f6b673df0c621aca27596ce6b",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/b162c7c88a253e3f6b673df0c621aca27596ce6b.zip"],
)

# We still require the pybind library.
http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.11.1",
  urls = ["https://github.com/pybind/pybind11/archive/v2.11.1.tar.gz"],
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

http_archive(
    name = "rules_python",
    sha256 = "29a801171f7ca190c543406f9894abf2d483c206e14d6acbd695623662320097",
    strip_prefix = "rules_python-0.18.1",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.18.1/rules_python-0.18.1.tar.gz",
)

########################################################### Python support end

# Protobuf expects an //external:python_headers target
bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)

http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check(minimum_bazel_version = "3.7.2")

# ABSL cpp library lts_2023_01_25.
http_archive(
    name = "com_google_absl",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/refs/tags/20230125.0.tar.gz",
    ],
    patches = [
        "@//third_party:com_google_absl_windows_patch.diff"
    ],
    patch_args = [
        "-p1",
    ],
    strip_prefix = "abseil-cpp-20230125.0",
    sha256 = "3ea49a7d97421b88a8c48a0de16c16048e17725c7ec0f1d3ea2683a2a75adc21"
)

http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-2f8c04c04462ab83c545ab14c0da68c3b4c96191",
# The commit can be updated if the build passes. Last updated 6/23/22.
    urls = ["https://github.com/bazelbuild/rules_cc/archive/2f8c04c04462ab83c545ab14c0da68c3b4c96191.zip"],
)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository( # Using commit past 0.9.0 that adds cmake 3.26.2 for model api. Be sure to update to 0.10.0 when available.
    name = "rules_foreign_cc",
    remote = "https://github.com/bazelbuild/rules_foreign_cc.git",
    commit = "1fb8a1e",
 #   strip_prefix = "rules_foreign_cc-0.9.0",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies(cmake_version="3.26.2")

http_archive(
    name = "com_google_protobuf",
    sha256 = "87407cd28e7a9c95d9f61a098a53cf031109d451a7763e7dd1253abf8b4df422",
    strip_prefix = "protobuf-3.19.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.19.1.tar.gz"],
    patches = [
        "@//third_party:com_google_protobuf_fixes.diff"
    ],
    patch_args = [
        "-p1",
    ],
)

# Load Zlib before initializing TensorFlow and the iOS build rules to guarantee
# that the target @zlib//:mini_zlib is available
http_archive(
    name = "zlib",
    build_file = "@//third_party:zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "http://mirror.bazel.build/zlib.net/fossils/zlib-1.2.11.tar.gz",
        "http://zlib.net/fossils/zlib-1.2.11.tar.gz",  # 2017-01-15
    ],
    patches = [
        "@//third_party:zlib.diff",
    ],
    patch_args = [
        "-p1",
    ],
)

# iOS basic build deps.
http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "3e2c7ae0ddd181c4053b6491dad1d01ae29011bc322ca87eea45957c76d3a0c3",
    url = "https://github.com/bazelbuild/rules_apple/releases/download/2.1.0/rules_apple.2.1.0.tar.gz",
    patches = [
        # Bypass checking ios unit test runner when building MP ios applications.
        "@//third_party:build_bazel_rules_apple_bypass_test_runner_check.diff"
    ],
    patch_args = [
        "-p1",
    ],
)

load(
    "@build_bazel_rules_apple//apple:repositories.bzl",
    "apple_rules_dependencies",
)
apple_rules_dependencies()

load(
    "@build_bazel_rules_swift//swift:repositories.bzl",
    "swift_rules_dependencies",
)
swift_rules_dependencies()

load(
    "@build_bazel_rules_swift//swift:extras.bzl",
    "swift_rules_extra_dependencies",
)
swift_rules_extra_dependencies()

load(
    "@build_bazel_apple_support//lib:repositories.bzl",
    "apple_support_dependencies",
)
apple_support_dependencies()

# This is used to select all contents of the archives for CMake-based packages to give CMake access to them.
all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""

# GoogleTest/GoogleMock framework. Used by most unit-tests.
# Last updated 2021-07-02.
http_archive(
    name = "com_google_googletest",
    urls = ["https://github.com/google/googletest/archive/4ec4cd23f486bf70efcc5d2caa40f24368f752e3.zip"],
    strip_prefix = "googletest-4ec4cd23f486bf70efcc5d2caa40f24368f752e3",
    sha256 = "de682ea824bfffba05b4e33b67431c247397d6175962534305136aa06f92e049",
)

# Google Benchmark library v1.6.1 released on 2022-01-10.
http_archive(
    name = "com_google_benchmark",
    urls = ["https://github.com/google/benchmark/archive/refs/tags/v1.6.1.tar.gz"],
    strip_prefix = "benchmark-1.6.1",
    sha256 = "6132883bc8c9b0df5375b16ab520fac1a85dc9e4cf5be59480448ece74b278d4",
    build_file = "@//third_party:benchmark.BUILD",
)

# gflags needed by glog
http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-2.2.2",
    sha256 = "19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5",
    url = "https://github.com/gflags/gflags/archive/v2.2.2.zip",
)

# 2020-08-21
http_archive(
    name = "com_github_glog_glog",
    strip_prefix = "glog-3a0d4d22c5ae0b9a2216988411cfa6bf860cc372",
    sha256 = "170d08f80210b82d95563f4723a15095eff1aad1863000e8eeb569c96a98fefb",
    urls = [
        "https://github.com/google/glog/archive/3a0d4d22c5ae0b9a2216988411cfa6bf860cc372.zip",
    ],
)
http_archive(
    name = "com_github_glog_glog_no_gflags",
    strip_prefix = "glog-3a0d4d22c5ae0b9a2216988411cfa6bf860cc372",
    sha256 = "170d08f80210b82d95563f4723a15095eff1aad1863000e8eeb569c96a98fefb",
    build_file = "@//third_party:glog_no_gflags.BUILD",
    urls = [
        "https://github.com/google/glog/archive/3a0d4d22c5ae0b9a2216988411cfa6bf860cc372.zip",
    ],
    patches = [
        "@//third_party:com_github_glog_glog.diff",
    ],
    patch_args = [
        "-p1",
    ],
)

# easyexif
http_archive(
    name = "easyexif",
    url = "https://github.com/mayanklahiri/easyexif/archive/master.zip",
    strip_prefix = "easyexif-master",
    build_file = "@//third_party:easyexif.BUILD",
)

# libyuv
http_archive(
    name = "libyuv",
    # Error: operand type mismatch for `vbroadcastss' caused by commit 8a13626e42f7fdcf3a6acbb0316760ee54cda7d8.
    urls = ["https://chromium.googlesource.com/libyuv/libyuv/+archive/2525698acba9bf9b701ba6b4d9584291a1f62257.tar.gz"],
    build_file = "@//third_party:libyuv.BUILD",
)

# Note: protobuf-javalite is no longer released as a separate download, it's included in the main Java download.
# ...but the Java download is currently broken, so we use the "source" download.
http_archive(
    name = "com_google_protobuf_javalite",
    sha256 = "87407cd28e7a9c95d9f61a098a53cf031109d451a7763e7dd1253abf8b4df422",
    strip_prefix = "protobuf-3.19.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.19.1.tar.gz"],
)

http_archive(
    name = "com_google_protobuf",
    sha256 = "87407cd28e7a9c95d9f61a098a53cf031109d451a7763e7dd1253abf8b4df422",
    strip_prefix = "protobuf-3.19.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.19.1.tar.gz"],
    patches = [
        "@//third_party:com_google_protobuf_fixes.diff"
    ],
    patch_args = [
        "-p1",
    ],
)

load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()

http_archive(
    name = "com_google_audio_tools",
    strip_prefix = "multichannel-audio-tools-1f6b1319f13282eda6ff1317be13de67f4723860",
    urls = ["https://github.com/google/multichannel-audio-tools/archive/1f6b1319f13282eda6ff1317be13de67f4723860.zip"],
    sha256 = "fe346e1aee4f5069c4cbccb88706a9a2b2b4cf98aeb91ec1319be77e07dd7435",
    repo_mapping = {"@com_github_glog_glog" : "@com_github_glog_glog_no_gflags"},
    # TODO: Fix this in AudioTools directly
    patches = ["@//third_party:com_google_audio_tools_fixes.diff"],
    patch_args = ["-p1"]
)

http_archive(
    name = "pffft",
    strip_prefix = "jpommier-pffft-7c3b5a7dc510",
    urls = ["https://bitbucket.org/jpommier/pffft/get/7c3b5a7dc510.zip"],
    build_file = "@//third_party:pffft.BUILD",
)

# sentencepiece
http_archive(
    name = "com_google_sentencepiece",
    strip_prefix = "sentencepiece-1.0.0",
    sha256 = "c05901f30a1d0ed64cbcf40eba08e48894e1b0e985777217b7c9036cac631346",
    urls = [
        "https://github.com/google/sentencepiece/archive/1.0.0.zip",
    ],
    patches = [
        "@//third_party:com_google_sentencepiece_no_gflag_no_gtest.diff",
    ],
    patch_args = ["-p1"],
    repo_mapping = {"@com_google_glog" : "@com_github_glog_glog_no_gflags"},
)

http_archive(
    name = "darts_clone",
    build_file = "@//third_party:darts_clone.BUILD",
    sha256 = "c97f55d05c98da6fcaf7f9ecc6a6dc6bc5b18b8564465f77abff8879d446491c",
    strip_prefix = "darts-clone-e40ce4627526985a7767444b6ed6893ab6ff8983",
    urls = [
        "https://github.com/s-yata/darts-clone/archive/e40ce4627526985a7767444b6ed6893ab6ff8983.zip",
    ],
)

http_archive(
    name = "org_tensorflow_text",
    sha256 = "f64647276f7288d1b1fe4c89581d51404d0ce4ae97f2bcc4c19bd667549adca8",
    strip_prefix = "text-2.2.0",
    urls = [
        "https://github.com/tensorflow/text/archive/v2.2.0.zip",
    ],
    patches = [
        "@//third_party:tensorflow_text_remove_tf_deps.diff",
        "@//third_party:tensorflow_text_a0f49e63.diff",
    ],
    patch_args = ["-p1"],
    repo_mapping = {"@com_google_re2": "@com_googlesource_code_re2"},
)

http_archive(
    name = "com_googlesource_code_re2",
    sha256 = "ef516fb84824a597c4d5d0d6d330daedb18363b5a99eda87d027e6bdd9cba299",
    strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
    urls = [
        "https://github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz",
    ],
)

http_archive(
    name = "pybind11_protobuf",
    sha256 = "baa1f53568283630a5055c85f0898b8810f7a6431bd01bbaedd32b4c1defbcb1",
    strip_prefix = "pybind11_protobuf-3594106f2df3d725e65015ffb4c7886d6eeee683",
    urls = [
        "https://github.com/pybind/pybind11_protobuf/archive/3594106f2df3d725e65015ffb4c7886d6eeee683.tar.gz",
    ],
)

# Point to the commit that deprecates the usage of Eigen::MappedSparseMatrix.
http_archive(
    name = "ceres_solver",
    url = "https://github.com/ceres-solver/ceres-solver/archive/123fba61cf2611a3c8bddc9d91416db26b10b558.zip",
    patches = [
        "@//third_party:ceres_solver_compatibility_fixes.diff"
    ],
    patch_args = [
        "-p1",
    ],
    strip_prefix = "ceres-solver-123fba61cf2611a3c8bddc9d91416db26b10b558",
    sha256 = "8b7b16ceb363420e0fd499576daf73fa338adb0b1449f58bea7862766baa1ac7"
)

http_archive(
    name = "opencv",
    build_file_content = all_content,
    strip_prefix = "opencv-3.4.10",
    urls = ["https://github.com/opencv/opencv/archive/3.4.10.tar.gz"],
)

new_local_repository(
    name = "linux_opencv",
    build_file = "@//third_party:opencv_linux.BUILD",
    path = "/usr/local",
)

new_local_repository(
    name = "linux_ffmpeg",
    build_file = "@//third_party:ffmpeg_linux.BUILD",
    path = "/usr"
)

new_local_repository(
    name = "macos_opencv",
    build_file = "@//third_party:opencv_macos.BUILD",
    # For local MacOS builds, the path should point to an opencv@3 installation.
    # If you edit the path here, you will also need to update the corresponding
    # prefix in "opencv_macos.BUILD".
    path = "/usr/local",  # e.g. /usr/local/Cellar for HomeBrew
)

new_local_repository(
    name = "macos_ffmpeg",
    build_file = "@//third_party:ffmpeg_macos.BUILD",
    path = "/usr/local/opt/ffmpeg",
)

new_local_repository(
    name = "windows_opencv",
    build_file = "@//third_party:opencv_windows.BUILD",
    path = "C:\\opencv\\build",
)

http_archive(
    name = "android_opencv",
    build_file = "@//third_party:opencv_android.BUILD",
    strip_prefix = "OpenCV-android-sdk",
    type = "zip",
    url = "https://github.com/opencv/opencv/releases/download/3.4.3/opencv-3.4.3-android-sdk.zip",
)

# After OpenCV 3.2.0, the pre-compiled opencv2.framework has google protobuf symbols, which will
# trigger duplicate symbol errors in the linking stage of building a mediapipe ios app.
# To get a higher version of OpenCV for iOS, opencv2.framework needs to be built from source with
# '-DBUILD_PROTOBUF=OFF -DBUILD_opencv_dnn=OFF'.
http_archive(
    name = "ios_opencv",
    sha256 = "7dd536d06f59e6e1156b546bd581523d8df92ce83440002885ec5abc06558de2",
    build_file = "@//third_party:opencv_ios.BUILD",
    type = "zip",
    url = "https://github.com/opencv/opencv/releases/download/3.2.0/opencv-3.2.0-ios-framework.zip",
)

# Building an opencv.xcframework from the OpenCV 4.5.3 sources is necessary for
# MediaPipe iOS Task Libraries to be supported on arm64(M1) Macs. An
# `opencv.xcframework` archive has not been released and it is recommended to
# build the same from source using a script provided in OpenCV 4.5.0 upwards.
# OpenCV is fixed to version to 4.5.3 since swift support can only be disabled
# from 4.5.3 upwards. This is needed to avoid errors when the library is linked
# in Xcode. Swift support will be added in when the final binary MediaPipe iOS
# Task libraries are built.
http_archive(
    name = "ios_opencv_source",
    sha256 = "a61e7a4618d353140c857f25843f39b2abe5f451b018aab1604ef0bc34cd23d5",
    build_file = "@//third_party:opencv_ios_source.BUILD",
    type = "zip",
    url = "https://github.com/opencv/opencv/archive/refs/tags/4.5.3.zip",
)

http_archive(
    name = "stblib",
    strip_prefix = "stb-b42009b3b9d4ca35bc703f5310eedc74f584be58",
    sha256 = "13a99ad430e930907f5611325ec384168a958bf7610e63e60e2fd8e7b7379610",
    urls = ["https://github.com/nothings/stb/archive/b42009b3b9d4ca35bc703f5310eedc74f584be58.tar.gz"],
    build_file = "@//third_party:stblib.BUILD",
    patches = [
        "@//third_party:stb_image_impl.diff"
    ],
    patch_args = [
        "-p1",
    ],
)

# More iOS deps.

http_archive(
    name = "google_toolbox_for_mac",
    url = "https://github.com/google/google-toolbox-for-mac/archive/v2.2.1.zip",
    sha256 = "e3ac053813c989a88703556df4dc4466e424e30d32108433ed6beaec76ba4fdc",
    strip_prefix = "google-toolbox-for-mac-2.2.1",
    build_file = "@//third_party:google_toolbox_for_mac.BUILD",
)

# Maven dependencies.

RULES_JVM_EXTERNAL_TAG = "4.0"
RULES_JVM_EXTERNAL_SHA = "31701ad93dbfe544d597dbe62c9a1fdd76d81d8a9150c2bf1ecf928ecdf97169"

http_archive(
    name = "rules_jvm_external",
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    sha256 = RULES_JVM_EXTERNAL_SHA,
    url = "https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG,
)

load("@rules_jvm_external//:defs.bzl", "maven_install")

# Important: there can only be one maven_install rule. Add new maven deps here.
maven_install(
    artifacts = [
        "androidx.concurrent:concurrent-futures:1.0.0-alpha03",
        "androidx.lifecycle:lifecycle-common:2.3.1",
        "androidx.activity:activity:1.2.2",
        "androidx.exifinterface:exifinterface:1.3.3",
        "androidx.fragment:fragment:1.3.4",
        "androidx.annotation:annotation:aar:1.1.0",
        "androidx.appcompat:appcompat:aar:1.1.0-rc01",
        "androidx.camera:camera-core:1.0.0-beta10",
        "androidx.camera:camera-camera2:1.0.0-beta10",
        "androidx.camera:camera-lifecycle:1.0.0-beta10",
        "androidx.constraintlayout:constraintlayout:aar:1.1.3",
        "androidx.core:core:aar:1.1.0-rc03",
        "androidx.legacy:legacy-support-v4:aar:1.0.0",
        "androidx.recyclerview:recyclerview:aar:1.1.0-beta02",
        "androidx.test.espresso:espresso-core:3.1.1",
        "com.github.bumptech.glide:glide:4.11.0",
        "com.google.android.material:material:aar:1.0.0-rc01",
        "com.google.auto.value:auto-value:1.8.1",
        "com.google.auto.value:auto-value-annotations:1.8.1",
        "com.google.code.findbugs:jsr305:latest.release",
        "com.google.android.datatransport:transport-api:3.0.0",
        "com.google.android.datatransport:transport-backend-cct:3.1.0",
        "com.google.android.datatransport:transport-runtime:3.1.0",
        "com.google.flogger:flogger-system-backend:0.6",
        "com.google.flogger:flogger:0.6",
        "com.google.guava:guava:27.0.1-android",
        "com.google.guava:listenablefuture:1.0",
        "junit:junit:4.12",
        "org.hamcrest:hamcrest-library:1.3",
    ],
    repositories = [
        "https://maven.google.com",
        "https://dl.google.com/dl/android/maven2",
        "https://repo1.maven.org/maven2",
        "https://jcenter.bintray.com",
    ],
    fetch_sources = True,
    version_conflict_policy = "pinned",
)

# Needed by TensorFlow
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "e0a111000aeed2051f29fcc7a3f83be3ad8c6c93c186e64beb1ad313f0c7f9f9",
    strip_prefix = "rules_closure-cf1e44edb908e9616030cc83d085989b8e6cd6df",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",  # 2019-04-04
    ],
)

# TensorFlow repo should always go after the other external dependencies.
# TF on 2023-06-13.
#_TENSORFLOW_GIT_COMMIT = "491681a5620e41bf079a582ac39c585cc86878b9"
_TENSORFLOW_GIT_COMMIT = "491681a5620e41bf079a582ac39c585cc86878b9"
# curl -L https://github.com/tensorflow/tensorflow/archive/<TENSORFLOW_GIT_COMMIT>.tar.gz | shasum -a 256
_TENSORFLOW_SHA256 = "9f76389af7a2835e68413322c1eaabfadc912f02a76d71dc16be507f9ca3d3ac"
http_archive(
    name = "org_tensorflow",
    urls = [
      "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
    patches = [
        "@//third_party:org_tensorflow_compatibility_fixes.diff",
        # Diff is generated with a script, don't update it manually.
        "@//third_party:org_tensorflow_custom_ops.diff",
    ],
    patch_args = [
        "-p1",
    ],
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
    sha256 = _TENSORFLOW_SHA256,
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()
load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

# Edge TPU
http_archive(
  name = "libedgetpu",
  sha256 = "14d5527a943a25bc648c28a9961f954f70ba4d79c0a9ca5ae226e1831d72fe80",
  strip_prefix = "libedgetpu-3164995622300286ef2bb14d7fdc2792dae045b7",
  urls = [
    "https://github.com/google-coral/libedgetpu/archive/3164995622300286ef2bb14d7fdc2792dae045b7.tar.gz"
  ],
)
load("@libedgetpu//:workspace.bzl", "libedgetpu_dependencies")
libedgetpu_dependencies()

load("@coral_crosstool//:configure.bzl", "cc_crosstool")
cc_crosstool(name = "crosstool")

# Node dependencies
http_archive(
    name = "build_bazel_rules_nodejs",
    sha256 = "94070eff79305be05b7699207fbac5d2608054dd53e6109f7d00d923919ff45a",
    urls = ["https://github.com/bazelbuild/rules_nodejs/releases/download/5.8.2/rules_nodejs-5.8.2.tar.gz"],
)

load("@build_bazel_rules_nodejs//:repositories.bzl", "build_bazel_rules_nodejs_dependencies")
build_bazel_rules_nodejs_dependencies()

# fetches nodejs, npm, and yarn
load("@build_bazel_rules_nodejs//:index.bzl", "node_repositories", "yarn_install")
node_repositories()
yarn_install(
    name = "npm",
    package_json = "@//:package.json",
    yarn_lock = "@//:yarn.lock",
)

# Protobuf for Node dependencies
http_archive(
    name = "rules_proto_grpc",
    sha256 = "bbe4db93499f5c9414926e46f9e35016999a4e9f6e3522482d3760dc61011070",
    strip_prefix = "rules_proto_grpc-4.2.0",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/4.2.0.tar.gz"],
)

http_archive(
    name = "com_google_protobuf_javascript",
    sha256 = "35bca1729532b0a77280bf28ab5937438e3dcccd6b31a282d9ae84c896b6f6e3",
    strip_prefix = "protobuf-javascript-3.21.2",
    urls = ["https://github.com/protocolbuffers/protobuf-javascript/archive/refs/tags/v3.21.2.tar.gz"],
)

load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_toolchains", "rules_proto_grpc_repos")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

load("@//third_party:external_files.bzl", "external_files")
external_files()

load("@//third_party:wasm_files.bzl", "wasm_files")
wasm_files()

# Halide

new_local_repository(
    name = "halide",
    build_file = "@//third_party/halide:BUILD.bazel",
    path = "third_party/halide"
)

http_archive(
    name = "linux_halide",
    sha256 = "d290fadf3f358c94aacf43c883de6468bb98883e26116920afd491ec0e440cd2",
    strip_prefix = "Halide-15.0.1-x86-64-linux",
    urls = ["https://github.com/halide/Halide/releases/download/v15.0.1/Halide-15.0.1-x86-64-linux-4c63f1befa1063184c5982b11b6a2cc17d4e5815.tar.gz"],
    build_file = "@//third_party:halide.BUILD",
)

http_archive(
    name = "macos_x86_64_halide",
    sha256 = "48ff073ac1aee5c4aca941a4f043cac64b38ba236cdca12567e09d803594a61c",
    strip_prefix = "Halide-15.0.1-x86-64-osx",
    urls = ["https://github.com/halide/Halide/releases/download/v15.0.1/Halide-15.0.1-x86-64-osx-4c63f1befa1063184c5982b11b6a2cc17d4e5815.tar.gz"],
    build_file = "@//third_party:halide.BUILD",
)

http_archive(
    name = "macos_arm_64_halide",
    sha256 = "db5d20d75fa7463490fcbc79c89f0abec9c23991f787c8e3e831fff411d5395c",
    strip_prefix = "Halide-15.0.1-arm-64-osx",
    urls = ["https://github.com/halide/Halide/releases/download/v15.0.1/Halide-15.0.1-arm-64-osx-4c63f1befa1063184c5982b11b6a2cc17d4e5815.tar.gz"],
    build_file = "@//third_party:halide.BUILD",
)

http_archive(
    name = "windows_halide",
    sha256 = "61fd049bd75ee918ac6c30d0693aac6048f63f8d1fc4db31001573e58eae8dae",
    strip_prefix = "Halide-15.0.1-x86-64-windows",
    urls = ["https://github.com/halide/Halide/releases/download/v15.0.1/Halide-15.0.1-x86-64-windows-4c63f1befa1063184c5982b11b6a2cc17d4e5815.zip"],
    build_file = "@//third_party:halide.BUILD",
)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# Update SHA in two places Dockerfile.openvino - ARG OVMS_COMMIT and here
git_repository(
    name = "ovms",
    remote = "https://github.com/openvinotoolkit/model_server",
    commit = "e2267e6b1b5715dc77aec6d89e299968287364fa", # ov 2024.2RC2 (#2484)
)

# DEV ovms - adjust local repository path for build
#new_local_repository(
#    name = "ovms",
#    path = "/ovms/",
#    build_file = "/ovms/BUILD.bazel",
#)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

########################################################### Python support start

load("@ovms//third_party/python:python_repo.bzl", "python_repository")
python_repository(name = "_python3-linux")

new_local_repository(
    name = "python3_linux",
    path = "/usr",
    build_file = "@_python3-linux//:BUILD"
)

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip_deps",
    requirements_lock = "@ovms//src/python/binding:tests/requirements.txt",
)

########################################################### Python support end

# minitrace
new_git_repository(
    name = "minitrace",
    remote = "https://github.com/hrydgard/minitrace.git",
    commit = "020f42b189e8d6ad50e4d8f45d69edee0a6b3f23",
    build_file_content = """
cc_library(
    name = "trace",
    hdrs = ["minitrace.h"],
    srcs = ["minitrace.c"],
    visibility = ["//visibility:public"],
    local_defines = [
    ],
)
""",
)

# Tensorflow serving
git_repository(
    name = "tensorflow_serving",
    remote = "https://github.com/tensorflow/serving.git",
    tag = "2.13.0",
    patch_args = ["-p1"],
    patches = ["@ovms//external:net_http.patch", "@ovms//external:listen.patch", "@ovms//external:partial.patch"]
    #                             ^^^^^^^^^^^^
    #                       make bind address configurable
    #          ^^^^^^^^^^^^
    #        allow all http methods
)

load("@ovms//third_party/aws-sdk-cpp:aws-sdk-cpp.bzl", "aws_sdk_cpp")
aws_sdk_cpp()

# Azure Storage SDK
new_local_repository(
    name = "azure",
    build_file = "@ovms//third_party/azure:BUILD",
    path = "/azure/azure-storage-cpp",
)

# Azure Storage SDK dependency - cpprest
new_local_repository(
    name = "cpprest",
    build_file = "@ovms//third_party/cpprest:BUILD",
    path = "/azure/cpprestsdk",
)

# Boost (needed for Azure Storage SDK)

new_local_repository(
    name = "boost",
    path = "/usr/local/lib/",
    build_file = "@ovms//third_party/boost:BUILD"
)

# Google Cloud SDK
http_archive(
    name = "com_github_googleapis_google_cloud_cpp",
    sha256 = "a370bcf2913717c674a7250c4a310250448ffeb751b930be559a6f1887155f3b",
    strip_prefix = "google-cloud-cpp-0.21.0",
    url = "https://github.com/googleapis/google-cloud-cpp/archive/v0.21.0.tar.gz",
    repo_mapping = {"@com_github_curl_curl" : "@curl"}
)

load("@com_github_googleapis_google_cloud_cpp//bazel:google_cloud_cpp_deps.bzl", "google_cloud_cpp_deps")
google_cloud_cpp_deps()

load("@com_google_googleapis//:repository_rules.bzl", "switched_rules_by_language")
switched_rules_by_language(
    name = "com_google_googleapis_imports",
    cc = True,  # C++ support is only "Partially implemented", roll our own.
    grpc = True,
)

load("@com_github_googleapis_google_cloud_cpp_common//bazel:google_cloud_cpp_common_deps.bzl", "google_cloud_cpp_common_deps")
google_cloud_cpp_common_deps()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

# cxxopts
http_archive(
    name = "com_github_jarro2783_cxxopts",
    url = "https://github.com/jarro2783/cxxopts/archive/v3.1.1.zip",
    sha256 = "25b644a2bfa9c6704d723be51b026bc02420dfdee1277a49bfe5df3f19b0eaa4",
    strip_prefix = "cxxopts-3.1.1",
    build_file = "@ovms//third_party/cxxopts:BUILD",
)

# RapidJSON
http_archive(
    name = "com_github_tencent_rapidjson",
    url = "https://github.com/Tencent/rapidjson/archive/v1.1.0.zip",
    sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
    strip_prefix = "rapidjson-1.1.0",
    build_file = "@ovms//third_party/rapidjson:BUILD"
)

# spdlog
http_archive(
    name = "com_github_gabime_spdlog",
    url = "https://github.com/gabime/spdlog/archive/v1.4.0.tar.gz",
    sha256 = "afd18f62d1bc466c60bef088e6b637b0284be88c515cedc59ad4554150af6043",
    strip_prefix = "spdlog-1.4.0",
    build_file = "@ovms//third_party/spdlog:BUILD"
)

# fmtlib
http_archive(
    name = "fmtlib",
    url = "https://github.com/fmtlib/fmt/archive/6.0.0.tar.gz",
    sha256 = "f1907a58d5e86e6c382e51441d92ad9e23aea63827ba47fd647eacc0d3a16c78",
    strip_prefix = "fmt-6.0.0",
    build_file = "@ovms//third_party/fmtlib:BUILD"
)

# libevent
http_archive(
    name = "com_github_libevent_libevent",
    url = "https://github.com/libevent/libevent/archive/release-2.1.8-stable.zip",
    sha256 = "70158101eab7ed44fd9cc34e7f247b3cae91a8e4490745d9d6eb7edc184e4d96",
    strip_prefix = "libevent-release-2.1.8-stable",
    build_file = "@ovms//third_party/libevent:BUILD",
)

# prometheus-cpp
http_archive(
    name = "com_github_jupp0r_prometheus_cpp",
    strip_prefix = "prometheus-cpp-1.0.1",
    urls = ["https://github.com/jupp0r/prometheus-cpp/archive/refs/tags/v1.0.1.zip"],
)
load("@com_github_jupp0r_prometheus_cpp//bazel:repositories.bzl", "prometheus_cpp_repositories")
prometheus_cpp_repositories()

new_local_repository(
    name = "mediapipe_calculators",
    build_file = "@ovms//third_party/mediapipe_calculators:BUILD",
    path = "/opt/ovms/",
)

new_local_repository(
    name = "linux_openvino",
    build_file = "@ovms//third_party/openvino:BUILD",
    path = "/opt/intel/openvino/runtime",
)

git_repository(
    name = "oneTBB",
    branch = "v2021.10.0",
    remote = "https://github.com/oneapi-src/oneTBB/",
    patch_args = ["-p1"],
    patches = ["@ovms//external:mwaitpkg.patch",]
)

load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")
load("@//third_party/model_api:model_api.bzl", "model_api_repository")
model_api_repository(name="_model-api")
new_git_repository(
    name = "model_api",
    remote = "https:///github.com/openvinotoolkit/model_api/",
    build_file = "@_model-api//:BUILD",
    commit = "eb9fcfb1e1eebc047ff144707f76203b132e1aa6" # master Jun 24 15:02:17 2024 [cpp] Fix num classes check
)

git_repository(
    name = "nlohmann_json",
    remote = "https://github.com/nlohmann/json/",
    tag = "v3.11.3",
)
