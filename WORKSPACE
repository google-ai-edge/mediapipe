workspace(name = "mediapipe")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Protobuf expects an //external:python_headers target
bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)

load("@com_google_protobuf//bazel/private:proto_bazel_features.bzl", "proto_bazel_features")  # buildifier: disable=bzl-visibility

proto_bazel_features(name = "proto_bazel_features")

http_archive(
    name = "rules_android_ndk",
    sha256 = "89bf5012567a5bade4c78eac5ac56c336695c3bfd281a9b0894ff6605328d2d5",
    strip_prefix = "rules_android_ndk-0.1.3",
    url = "https://github.com/bazelbuild/rules_android_ndk/releases/download/v0.1.3/rules_android_ndk-v0.1.3.tar.gz",
)

load("@rules_android_ndk//:rules.bzl", "android_ndk_repository")  # @unused

# GoogleTest/GoogleMock framework. Used by most unit-tests.
# Last updated 2021-07-02.
http_archive(
    name = "com_google_googletest",
    sha256 = "de682ea824bfffba05b4e33b67431c247397d6175962534305136aa06f92e049",
    strip_prefix = "googletest-4ec4cd23f486bf70efcc5d2caa40f24368f752e3",
    urls = ["https://github.com/google/googletest/archive/4ec4cd23f486bf70efcc5d2caa40f24368f752e3.zip"],
)

# Load Zlib before initializing TensorFlow and the iOS build rules to guarantee
# that the target @zlib//:mini_zlib is available
http_archive(
    name = "zlib",
    build_file = "@//third_party:zlib.BUILD",
    sha256 = "17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c",
    strip_prefix = "zlib-1.3.1",
    url = "https://github.com/madler/zlib/archive/refs/tags/v1.3.1.tar.gz",
)

# gflags needed by glog
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5",
    strip_prefix = "gflags-2.2.2",
    url = "https://github.com/gflags/gflags/archive/v2.2.2.zip",
)

# 2020-08-21
http_archive(
    name = "com_github_glog_glog",
    sha256 = "8a83bf982f37bb70825df71a9709fa90ea9f4447fb3c099e1d720a439d88bad6",
    strip_prefix = "glog-0.6.0",
    urls = [
        "https://github.com/google/glog/archive/v0.6.0.tar.gz",
    ],
)

http_archive(
    name = "com_github_glog_glog_no_gflags",
    build_file = "@//third_party:glog_no_gflags.BUILD",
    patch_args = [
        "-p1",
    ],
    patches = [
        "@//third_party:com_github_glog_glog.diff",
    ],
    sha256 = "8a83bf982f37bb70825df71a9709fa90ea9f4447fb3c099e1d720a439d88bad6",
    strip_prefix = "glog-0.6.0",
    urls = [
        "https://github.com/google/glog/archive/v0.6.0.tar.gz",
    ],
)

# 2023-06-05
# This version of Glog is required for Windows support, but currently causes
# crashes on some Android devices.
http_archive(
    name = "com_github_glog_glog_windows",
    patch_args = [
        "-p1",
    ],
    patches = [
        "@//third_party:com_github_glog_glog.diff",
        "@//third_party:com_github_glog_glog_windows_patch.diff",
    ],
    sha256 = "170d08f80210b82d95563f4723a15095eff1aad1863000e8eeb569c96a98fefb",
    strip_prefix = "glog-3a0d4d22c5ae0b9a2216988411cfa6bf860cc372",
    urls = [
        "https://github.com/google/glog/archive/3a0d4d22c5ae0b9a2216988411cfa6bf860cc372.zip",
    ],
)

# Maven dependencies.
RULES_JVM_EXTERNAL_TAG = "6.1"

RULES_JVM_EXTERNAL_SHA = "08ea921df02ffe9924123b0686dc04fd0ff875710bfadb7ad42badb931b0fd50"

http_archive(
    name = "rules_jvm_external",
    sha256 = RULES_JVM_EXTERNAL_SHA,
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    url = "https://github.com/bazel-contrib/rules_jvm_external/releases/download/%s/rules_jvm_external-%s.tar.gz" % (RULES_JVM_EXTERNAL_TAG, RULES_JVM_EXTERNAL_TAG),
)

load("@rules_jvm_external//:defs.bzl", "maven_install")

# Important: there can only be one maven_install rule. Add new maven deps here.
maven_install(
    artifacts = [
        "androidx.activity:activity:aar:1.2.2",
        "androidx.annotation:annotation:1.1.0",
        "androidx.appcompat:appcompat:aar:1.1.0-rc01",
        "androidx.camera:camera-camera2:aar:1.0.0-beta10",
        "androidx.camera:camera-core:aar:1.0.0-beta10",
        "androidx.camera:camera-lifecycle:aar:1.0.0-beta10",
        "androidx.constraintlayout:constraintlayout:aar:1.1.3",
        "androidx.concurrent:concurrent-futures:1.0.0-alpha03",
        "androidx.core:core:aar:1.1.0-rc03",
        "androidx.exifinterface:exifinterface:aar:1.3.3",
        "androidx.fragment:fragment:aar:1.3.4",
        "androidx.legacy:legacy-support-v4:aar:1.0.0",
        "androidx.lifecycle:lifecycle-common:2.3.1",
        "androidx.recyclerview:recyclerview:aar:1.1.0-beta02",
        "androidx.test.espresso:espresso-core:aar:3.1.1",
        "com.android.tools.build:gradle-api:8.12.0",
        "com.github.bumptech.glide:glide:4.11.0",
        "com.google.android.datatransport:transport-api:3.0.0",
        "com.google.android.datatransport:transport-backend-cct:3.1.0",
        "com.google.android.datatransport:transport-runtime:3.1.0",
        "com.google.android.material:material:aar:1.0.0-rc01",
        "com.google.android.play:ai-delivery:aar:0.1.1-alpha01",
        "com.google.android.play:asset-delivery:aar:2.3.0",
        "com.google.android.play:feature-delivery:aar:2.1.0",
        "com.google.auto.value:auto-value-annotations:1.8.1",
        "com.google.auto.value:auto-value:1.8.1",
        "com.google.code.findbugs:jsr305:latest.release",
        "com.google.code.gson:gson:2.13.2",
        "com.google.flogger:flogger-system-backend:0.6",
        "com.google.flogger:flogger:0.6",
        "com.google.guava:guava:27.0.1-android",
        "com.google.guava:listenablefuture:1.0",
        "com.squareup:kotlinpoet-jvm:2.2.0",
        "junit:junit:4.12",
        "org.hamcrest:hamcrest-library:1.3",
        "org.jetbrains.kotlin:kotlin-gradle-plugin:2.2.21",
    ],
    fetch_sources = True,
    repositories = [
        "https://maven.google.com",
        "https://dl.google.com/dl/android/maven2",
        "https://repo1.maven.org/maven2",
        "https://jcenter.bintray.com",
    ],
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

# XNNPACK
# org_tensorflow and @litert depend on XNNPACK. If updating tensorflow
# or LiteRT version, make sure to bump XNNPACK version as well and vice versa.
# Bumped to match what @litert (LiteRT v2.1.6's own pinned org_tensorflow,
# commit bcdab1a62e138c8f8784a7477c0be8af6dd0bd0a) expects - its
# tflite/delegates/xnnpack code uses newer XNNPACK API (qint2/qint4,
# xnn_define_static_constant_pad_v2) than mediapipe's org_tensorflow
# v2.21.0 pin's own XNNPACK version had.
http_archive(
    name = "XNNPACK",
    # `curl -L <url> | shasum -a 256`
    sha256 = "13ae01126b6d4a8b6769433c2a942d6204a3f97157d9c83d79cbfeec1041398c",
    strip_prefix = "XNNPACK-53a1797ba4360cbde068f2a984652be0f0b7b6fe",
    url = "https://github.com/google/XNNPACK/archive/53a1797ba4360cbde068f2a984652be0f0b7b6fe.zip",
)

http_archive(
    name = "pybind11_bazel",
    sha256 = "9df284330336958c837fb70dc34c0a6254dac52a5c983b3373a8c2bbb79ac35e",
    strip_prefix = "pybind11_bazel-2.13.6",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/v2.13.6.zip"],
)

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11-BUILD.bazel",
    sha256 = "d0a116e91f64a4a2d8fb7590c34242df92258a61ec644b79127951e821b47be6",
    strip_prefix = "pybind11-2.13.6",
    urls = [
        "https://github.com/pybind/pybind11/archive/v2.13.6.zip",
    ],
)

# 2025-02-10
# org_tensorflow depends on pybind11_protobuf. If updating tensorflow version,
# make sure to bump pybind11_protobuf version as well and vice versa.
http_archive(
    name = "pybind11_protobuf",
    sha256 = "3cf7bf0f23954c5ce6c37f0a215f506efa3035ca06e3b390d67f4cbe684dce23",
    strip_prefix = "pybind11_protobuf-f02a2b7653bc50eb5119d125842a3870db95d251",
    urls = [
        "https://github.com/pybind/pybind11_protobuf/archive/f02a2b7653bc50eb5119d125842a3870db95d251.zip",
    ],
)

# KleidiAI is needed to get the best possible performance out of XNNPack.
# Kept in sync with the KleidiAI version XNNPACK itself pins - see
# cmake/DownloadKleidiAI.cmake at the XNNPACK commit above.
http_archive(
    name = "KleidiAI",
    sha256 = "b147799b94c51f5e57492930bfd9e5294fb7ffe44fee1dbcd3f8048adeedd5e3",
    strip_prefix = "kleidiai-b87ef9c94f45f11c81a6b1fdaed1b2b45ea58c0c",
    urls = [
        "https://gitlab.arm.com/kleidi/kleidiai/-/archive/b87ef9c94f45f11c81a6b1fdaed1b2b45ea58c0c/kleidiai-b87ef9c94f45f11c81a6b1fdaed1b2b45ea58c0c.zip",
    ],
)

http_archive(
    name = "cpuinfo",
    sha256 = "9213f6f81784eb8679f0621ad1c20eac711e063cb9c7712738720609cbdf1c33",
    strip_prefix = "cpuinfo-ea6b9f1bb6e1001d8b21574d5bc78ddef62e499d",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/ea6b9f1bb6e1001d8b21574d5bc78ddef62e499d.zip",
    ],
)

# pthreadpool is a dependency of XNNPACK.
http_archive(
    name = "pthreadpool",
    # `curl -L <url> | shasum -a 256`
    sha256 = "5ab4e8f63e3dcf62048360c216532bdf62f00dc204883a52d91230402f0feb6a",
    strip_prefix = "pthreadpool-02460584c6092e527c8b89f7df4de143d70e801f",
    urls = ["https://github.com/google/pthreadpool/archive/02460584c6092e527c8b89f7df4de143d70e801f.zip"],
)

# TF v2.21.0
# org_tensorflow depends on Eigen, XNNPACK, and pybind11_protobuf, which also have explicit
# repository definitions in this WORKSPACE. If updating the tensorflow version, make sure to check
# and bump those dependent versions as well and vice versa.
_TENSORFLOW_GIT_COMMIT = "a481b10260dfdf833a1b16007eead49c1d7febf3"

# curl -L https://github.com/tensorflow/tensorflow/archive/<COMMIT>.tar.gz | shasum -a 256
_TENSORFLOW_SHA256 = "6438396f3b19af5d7ad787cf041f857af7505916dc08092e20b07d1b1f8df492"

http_archive(
    name = "org_tensorflow",
    patch_args = [
        "-p1",
    ],
    patches = [
        # Fixes experimental C API headers/exports needed by MediaPipe C++ bindings.
        "@//third_party:org_tensorflow_c_api_experimental.diff",
        # Works around a Bzlmod repository canonical-name issue in tflite_combine_cc_tests
        # (tensorflow/lite/build_def.bzl) where link_extra_lib is duplicated when rules_cc has
        # a version-suffixed canonical name (e.g., under single_version_override or complex dependency graphs).
        "@//third_party:org_tensorflow_combine_cc_tests_link_extra_lib.diff",
    ],
    sha256 = _TENSORFLOW_SHA256,
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

# Initialize hermetic Python
load("@org_tensorflow//third_party/xla/third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@org_tensorflow//third_party/xla/third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = ["mediapipe*"],
    local_wheel_workspaces = ["//:WORKSPACE"],
    requirements = {
        "3.9": "//:requirements_lock.txt",
        "3.10": "//:requirements_lock_3_10.txt",
        "3.11": "//:requirements_lock_3_11.txt",
        "3.12": "//:requirements_lock_3_12.txt",
    },
)

load("@org_tensorflow//third_party/xla/third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@org_tensorflow//third_party/xla/third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()
# End hermetic Python initialization

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

# LLVM/MLIR, needed by tensorflow/compiler/mlir/lite (e.g. metadata tooling
# pulled in by mediapipe/tasks/c:libmediapipe). This is normally set up by
# tf_workspace1(), but that macro also unconditionally calls grpc_deps(),
# benchmark_deps(), and closure_repositories(), which collide with
# repositories this WORKSPACE already defines explicitly above. @xla and its
# @llvm-raw/@local_config_python prerequisites are already established by
# tf_workspace2() above, so just call the one macro we actually need.
load("@xla//third_party/llvm:setup.bzl", "llvm_setup")

llvm_setup(name = "llvm-project")

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "mediapipe_pip_deps",
    requirements_lock = "@//:requirements_lock.txt",
)

load("@mediapipe_pip_deps//:requirements.bzl", mp_install_deps = "install_deps")

mp_install_deps()

pip_parse(
    name = "model_maker_pip_deps",
    requirements_lock = "@//mediapipe/model_maker:requirements_lock.txt",
)

load("@model_maker_pip_deps//:requirements.bzl", mm_install_deps = "install_deps")

mm_install_deps()

http_archive(
    name = "rules_foreign_cc",
    sha256 = "a2e6fb56e649c1ee79703e99aa0c9d13c6cc53c8d7a0cbb8797ab2888bbc99a3",
    strip_prefix = "rules_foreign_cc-0.12.0",
    url = "https://github.com/bazelbuild/rules_foreign_cc/releases/download/0.12.0/rules_foreign_cc-0.12.0.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

# This is used to select all contents of the archives for CMake-based packages to give CMake access to them.
all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""

# Google Benchmark library v1.6.1 released on 2022-01-10.
http_archive(
    name = "com_google_benchmark",
    build_file = "@//third_party:benchmark.BUILD",
    sha256 = "6132883bc8c9b0df5375b16ab520fac1a85dc9e4cf5be59480448ece74b278d4",
    strip_prefix = "benchmark-1.6.1",
    urls = ["https://github.com/google/benchmark/archive/refs/tags/v1.6.1.tar.gz"],
)

# easyexif
http_archive(
    name = "easyexif",
    build_file = "@//third_party:easyexif.BUILD",
    strip_prefix = "easyexif-master",
    url = "https://github.com/mayanklahiri/easyexif/archive/master.zip",
)

# libyuv
http_archive(
    name = "libyuv",
    build_file = "@//third_party:libyuv.BUILD",
    # Error: operand type mismatch for `vbroadcastss' caused by commit 8a13626e42f7fdcf3a6acbb0316760ee54cda7d8.
    urls = [
        "https://storage.googleapis.com/tensorstore-bazel-mirror/chromium.googlesource.com/libyuv/libyuv/+archive/2525698acba9bf9b701ba6b4d9584291a1f62257.tar.gz",
        "https://chromium.googlesource.com/libyuv/libyuv/+archive/2525698acba9bf9b701ba6b4d9584291a1f62257.tar.gz",
    ],
)

# Note: protobuf-javalite is no longer released as a separate download, it's included in the main Java download.
# ...but the Java download is currently broken, so we use the "source" download.
http_archive(
    name = "com_google_protobuf_javalite",
    repo_mapping = {"@abseil-cpp": "@com_google_absl"},
    sha256 = "6e09bbc950ba60c3a7b30280210cd285af8d7d8ed5e0a6ed101c72aff22e8d88",
    strip_prefix = "protobuf-6.31.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.zip"],
)

load("@//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")

flatbuffers()

http_archive(
    name = "com_google_audio_tools",
    patch_args = ["-p1"],
    # TODO: Fix this in AudioTools directly
    patches = ["@//third_party:com_google_audio_tools_fixes.diff"],
    repo_mapping = {"@com_github_glog_glog": "@com_github_glog_glog_no_gflags"},
    sha256 = "7d7227cc6bb1f8917a9c9013e8f3578ec681c49e20fe2fc38ba90965394de60c",
    strip_prefix = "multichannel-audio-tools-bbf15de4b7cd825d650296d21917afc07e8fe18b",
    urls = ["https://github.com/google/multichannel-audio-tools/archive/bbf15de4b7cd825d650296d21917afc07e8fe18b.tar.gz"],
)

http_archive(
    name = "pffft",
    build_file = "@//third_party:pffft.BUILD",
    strip_prefix = "jpommier-pffft-7c3b5a7dc510",
    urls = ["https://bitbucket.org/jpommier/pffft/get/7c3b5a7dc510.zip"],
)

# Sentencepiece
http_archive(
    name = "com_google_sentencepiece",
    add_prefix = "sentencepiece",
    build_file = "@//third_party:sentencepiece.BUILD",
    patch_args = [
        "-d",
        "sentencepiece",
        "-p1",
    ],
    # Fixes build compatibility and removes conflicting protobuf dependencies in sentencepiece.
    patches = ["@//third_party:com_google_sentencepiece.diff"],
    sha256 = "8409b0126ebd62b256c685d5757150cf7fcb2b92a2f2b98efb3f38fc36719754",
    strip_prefix = "sentencepiece-0.1.96",
    urls = [
        "https://github.com/google/sentencepiece/archive/refs/tags/v0.1.96.zip",
    ],
)

http_archive(
    name = "darts_clone",
    build_file = "@//third_party:darts_clone.BUILD",
    sha256 = "96946b2c1ec2a6e171665c5b5b3ec52fc27c325c80e0c957a415bb4c5145e7df",
    strip_prefix = "darts-clone-87b71afd6cf784953e3c08f24c64203397f3b724",
    urls = [
        "https://github.com/s-yata/darts-clone/archive/87b71afd6cf784953e3c08f24c64203397f3b724.zip",
    ],
)

http_archive(
    name = "org_tensorflow_text",
    patch_args = ["-p1"],
    patches = [
        # Replaces tf_cc_library with standard cc_library for core tokenizer kernels (regex_split,
        # wordpiece_tokenizer, etc.) so they can be compiled as lightweight standalone C++ libraries
        # without pulling in TensorFlow op libraries or Python headers.
        "@//third_party:tensorflow_text_remove_tf_deps.diff",
        # tftext.bzl unconditionally loads pybind_extension/pywrap_binaries/
        # pywrap_library from "@local_xla//third_party/py/rules_pywrap", a
        # repo name org_tensorflow's own build wires up internally for its
        # Python-wheel packaging but that this WORKSPACE never defines (and
        # that pulls in a large, unrelated dependency chain). None of the
        # cc_library targets this WORKSPACE actually uses from
        # org_tensorflow_text (regex_split, wordpiece_tokenizer, ...) call
        # py_tf_text_library, so the load is dead weight - stub it out
        # instead of standing up a "local_xla" repo just for this.
        "@//third_party:tensorflow_text_stub_pywrap.diff",
    ],
    repo_mapping = {"@com_google_re2": "@com_googlesource_code_re2"},
    sha256 = "e08834bed6f544be9cc0315895898bf48d94b8090bca993ab45526329df291c8",
    strip_prefix = "text-2.20.0",
    urls = [
        "https://github.com/tensorflow/text/archive/refs/tags/v2.20.0.zip",
    ],
)

# Point to the commit that deprecates the usage of Eigen::MappedSparseMatrix.
http_archive(
    name = "ceres_solver",
    patch_args = [
        "-p1",
    ],
    patches = [
        "@//third_party:ceres_solver_compatibility_fixes.diff",
    ],
    sha256 = "8b7b16ceb363420e0fd499576daf73fa338adb0b1449f58bea7862766baa1ac7",
    strip_prefix = "ceres-solver-123fba61cf2611a3c8bddc9d91416db26b10b558",
    url = "https://github.com/ceres-solver/ceres-solver/archive/123fba61cf2611a3c8bddc9d91416db26b10b558.zip",
)

http_archive(
    name = "opencv",
    build_file_content = all_content,
    strip_prefix = "opencv-3.4.11",
    urls = ["https://github.com/opencv/opencv/archive/3.4.11.tar.gz"],
)

new_local_repository(
    name = "linux_opencv",
    build_file = "@//third_party:opencv_linux.BUILD",
    path = "/usr",
)

new_local_repository(
    name = "linux_ffmpeg",
    build_file = "@//third_party:ffmpeg_linux.BUILD",
    path = "/usr",
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

# protobuf requires @system_python in WORKSPACE
new_local_repository(
    name = "system_python",
    build_file = "@//third_party:python_runtime.BUILD",
    path = ".",
)

http_archive(
    name = "android_opencv",
    build_file = "@//third_party:opencv_android.BUILD",
    strip_prefix = "OpenCV-android-sdk",
    type = "zip",
    url = "https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-android-sdk.zip",
)

# After OpenCV 3.2.0, the pre-compiled opencv2.framework has google protobuf symbols, which will
# trigger duplicate symbol errors in the linking stage of building a mediapipe ios app.
# To get a higher version of OpenCV for iOS, opencv2.framework needs to be built from source with
# '-DBUILD_PROTOBUF=OFF -DBUILD_opencv_dnn=OFF'.
http_archive(
    name = "ios_opencv",
    build_file = "@//third_party:opencv_ios.BUILD",
    sha256 = "7dd536d06f59e6e1156b546bd581523d8df92ce83440002885ec5abc06558de2",
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
    build_file = "@//third_party:opencv_ios_source.BUILD",
    sha256 = "a61e7a4618d353140c857f25843f39b2abe5f451b018aab1604ef0bc34cd23d5",
    type = "zip",
    url = "https://github.com/opencv/opencv/archive/refs/tags/4.5.3.zip",
)

http_archive(
    name = "stblib",
    build_file = "@//third_party:stblib.BUILD",
    patch_args = [
        "-p1",
    ],
    patches = [
        # Fixes image implementation definitions and warnings in stblib header-only libraries.
        "@//third_party:stb_image_impl.diff",
    ],
    sha256 = "13a99ad430e930907f5611325ec384168a958bf7610e63e60e2fd8e7b7379610",
    strip_prefix = "stb-b42009b3b9d4ca35bc703f5310eedc74f584be58",
    urls = ["https://github.com/nothings/stb/archive/b42009b3b9d4ca35bc703f5310eedc74f584be58.tar.gz"],
)

http_archive(
    name = "google_toolbox_for_mac",
    build_file = "@//third_party:google_toolbox_for_mac.BUILD",
    sha256 = "e3ac053813c989a88703556df4dc4466e424e30d32108433ed6beaec76ba4fdc",
    strip_prefix = "google-toolbox-for-mac-2.2.1",
    url = "https://github.com/google/google-toolbox-for-mac/archive/v2.2.1.zip",
)

# Pin kept identical to the rules_ml_toolchain version org_tensorflow itself
# vendors in third_party/xla/workspace0.bzl - see that file's fully-qualified
# http_archive definition when bumping the org_tensorflow version above.
http_archive(
    name = "rules_ml_toolchain",
    sha256 = "54c1a357f71f611efdb4891ebd4bcbe4aeb6dfa7e473f14fd7ecad5062096616",
    strip_prefix = "rules_ml_toolchain-d8cb9c2c168cd64000eaa6eda0781a9615a26ffe",
    urls = [
        "https://github.com/google-ml-infra/rules_ml_toolchain/archive/d8cb9c2c168cd64000eaa6eda0781a9615a26ffe.tar.gz",
    ],
)

load(
    "@org_tensorflow//third_party/xla/third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

# Hermetic C++
# Must be initialized before any CUDA/SYCL initialization below - see
# https://github.com/google-ml-infra/rules_ml_toolchain/blob/main/README.md
load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

# register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")

# Hermetic CUDA
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

# Edge TPU
http_archive(
    name = "libedgetpu",
    sha256 = "14d5527a943a25bc648c28a9961f954f70ba4d79c0a9ca5ae226e1831d72fe80",
    strip_prefix = "libedgetpu-3164995622300286ef2bb14d7fdc2792dae045b7",
    urls = [
        "https://github.com/google-coral/libedgetpu/archive/3164995622300286ef2bb14d7fdc2792dae045b7.tar.gz",
    ],
)

load("@libedgetpu//:workspace.bzl", "libedgetpu_dependencies")

libedgetpu_dependencies()

load("@coral_crosstool//:configure.bzl", "cc_crosstool")

cc_crosstool(name = "crosstool")

load("@//third_party:external_files.bzl", "external_files")

external_files()

load("@//third_party:wasm_files.bzl", "wasm_files")

wasm_files()

# Eigen
# org_tensorflow depends on Eigen. If updating tensorflow version,
# make sure to bump Eigen version as well and vice versa.
EIGEN_COMMIT = "ea13a98decd497a8c5588fb5de71b57bcf10d864"

EIGEN_SHA256 = "35c6126e246585d9cf6600b65471582c2701aae64b784a6fd19168a90cfc841e"

http_archive(
    name = "eigen",
    build_file = "@//third_party:eigen.BUILD",
    sha256 = EIGEN_SHA256,
    strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
    urls = ["https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT)],
)

# Halide

new_local_repository(
    name = "halide",
    build_file = "@//third_party/halide:BUILD.bazel",
    path = "third_party/halide",
)

http_archive(
    name = "linux_halide",
    build_file = "@//third_party:halide.BUILD",
    sha256 = "d290fadf3f358c94aacf43c883de6468bb98883e26116920afd491ec0e440cd2",
    strip_prefix = "Halide-15.0.1-x86-64-linux",
    urls = ["https://github.com/halide/Halide/releases/download/v15.0.1/Halide-15.0.1-x86-64-linux-4c63f1befa1063184c5982b11b6a2cc17d4e5815.tar.gz"],
)

http_archive(
    name = "macos_x86_64_halide",
    build_file = "@//third_party:halide.BUILD",
    sha256 = "48ff073ac1aee5c4aca941a4f043cac64b38ba236cdca12567e09d803594a61c",
    strip_prefix = "Halide-15.0.1-x86-64-osx",
    urls = ["https://github.com/halide/Halide/releases/download/v15.0.1/Halide-15.0.1-x86-64-osx-4c63f1befa1063184c5982b11b6a2cc17d4e5815.tar.gz"],
)

http_archive(
    name = "macos_arm_64_halide",
    build_file = "@//third_party:halide.BUILD",
    sha256 = "db5d20d75fa7463490fcbc79c89f0abec9c23991f787c8e3e831fff411d5395c",
    strip_prefix = "Halide-15.0.1-arm-64-osx",
    urls = ["https://github.com/halide/Halide/releases/download/v15.0.1/Halide-15.0.1-arm-64-osx-4c63f1befa1063184c5982b11b6a2cc17d4e5815.tar.gz"],
)

http_archive(
    name = "windows_halide",
    build_file = "@//third_party:halide.BUILD",
    sha256 = "61fd049bd75ee918ac6c30d0693aac6048f63f8d1fc4db31001573e58eae8dae",
    strip_prefix = "Halide-15.0.1-x86-64-windows",
    urls = ["https://github.com/halide/Halide/releases/download/v15.0.1/Halide-15.0.1-x86-64-windows-4c63f1befa1063184c5982b11b6a2cc17d4e5815.zip"],
)

http_archive(
    name = "pybind11_abseil",
    sha256 = "0223b647b8cc817336a51e787980ebc299c8d5e64c069829bf34b69d72337449",
    strip_prefix = "pybind11_abseil-2c4932ed6f6204f1656e245838f4f5eae69d2e29",
    urls = ["https://github.com/pybind/pybind11_abseil/archive/2c4932ed6f6204f1656e245838f4f5eae69d2e29.tar.gz"],
)

http_archive(
    name = "com_github_nlohmann_json",
    build_file = "@//third_party:nlohmann.BUILD",
    sha256 = "6bea5877b1541d353bd77bdfbdb2696333ae5ed8f9e8cc22df657192218cad91",
    urls = ["https://github.com/nlohmann/json/releases/download/v3.9.1/include.zip"],
)

http_archive(
    name = "io_abseil_py",
    sha256 = "0fb3a4916a157eb48124ef309231cecdfdd96ff54adf1660b39c0d4a9790a2c0",
    strip_prefix = "abseil-py-1.4.0",
    urls = ["https://github.com/abseil/abseil-py/archive/refs/tags/v1.4.0.tar.gz"],
)

http_archive(
    name = "skia",
    sha256 = "2fe28173428f8eebf2aa8a665bad32136086cc065f50c7154678a96250d1cde1",
    strip_prefix = "skia-226ae9d866748a2e68b6dbf114b37129c380a298",
    urls = ["https://github.com/google/skia/archive/226ae9d866748a2e68b6dbf114b37129c380a298.zip"],
)

http_archive(
    name = "skia_user_config",
    sha256 = "2fe28173428f8eebf2aa8a665bad32136086cc065f50c7154678a96250d1cde1",
    strip_prefix = "skia-226ae9d866748a2e68b6dbf114b37129c380a298/include/config",
    urls = ["https://github.com/google/skia/archive/226ae9d866748a2e68b6dbf114b37129c380a298.zip"],
)

http_archive(
    name = "gradle_distribution",
    build_file = "@//third_party:gradle_distribution.BUILD",
    sha256 = "3e1af3ae886920c3ac87f7a91f816c0c7c436f276a6eefdb3da152100fef72ae",
    strip_prefix = "gradle-8.4",
    urls = ["https://services.gradle.org/distributions/gradle-8.4-bin.zip"],
)

http_archive(
    name = "boringssl",
    sha256 = "52e2d96759d483e384e3964a2513781ea05cb6b2d677f1f8f5a4049aea30535d",
    strip_prefix = "boringssl-0.20260211.0",
    url = "https://github.com/google/boringssl/archive/refs/tags/0.20260211.0.tar.gz",
)

http_archive(
    name = "libcurl",
    build_file = "@//third_party:curl.BUILD",
    sha256 = "d15ebab765d793e2e96db090f0e172d127859d78ca6f6391d7eafecfd894bbc0",
    strip_prefix = "curl-8.10.1",
    url = "https://curl.haxx.se/download/curl-8.10.1.tar.gz",
)

# LiteRT v2.1.6
# Fetch just the source tree and let it use our already-defined workspace
# dependencies (@org_tensorflow, @xla, etc.) to avoid collisions.
#
# IMPORTANT: LiteRT and org_tensorflow's TFLite both use `namespace tflite`.
# Do not mix @litert and @org_tensorflow//tensorflow/lite/ targets in the
# same binary to prevent duplicate-symbol/ODR violations.
http_archive(
    name = "litert",
    patch_args = ["-p1"],
    # LiteRT's BUILD/bzl files load py_test/py_library/py_binary from
    # "@xla//third_party/rules_python/python:*.bzl", a path that doesn't
    # exist at mediapipe's pinned org_tensorflow/XLA commit (LiteRT expects a
    # newer XLA layout). XLA's wrapper also adds a strict_deps attribute
    # standard rules_python doesn't have. This patch adds an in-repo compat
    # shim (rules_python_compat.bzl) that drops strict_deps and delegates to
    # mediapipe's own working @rules_python, redirects all the broken loads
    # to it, and strips the now-inapplicable strict_deps call-site
    # arguments (a lint-only attribute org_tensorflow's own
    # py_test/py_library/py_binary macros don't accept either at mediapipe's
    # pinned version) - rather than trying to reconcile XLA versions.
    patches = [
        "@//third_party:litert_rules_python_and_strict_deps.diff",
        "@//third_party:litert_custom_ops.diff",
        "@//third_party:litert_internal_fbs_fix.diff",
    ],
    sha256 = "f95fa96332c56b7103db7a02ab4edab845949c196a986db55bddaa70539ee45b",
    strip_prefix = "LiteRT-2.1.6",
    urls = ["https://github.com/google-ai-edge/LiteRT/archive/refs/tags/v2.1.6.tar.gz"],
)
