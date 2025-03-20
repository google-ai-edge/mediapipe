workspace(name = "mediapipe")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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

# ABSL on 2023-10-18
http_archive(
    name = "com_google_absl",
    patch_args = [
        "-p1",
    ],
    patches = [
        "@//third_party:com_google_absl_windows_patch.diff",
    ],
    sha256 = "f841f78243f179326f2a80b719f2887c38fe226d288ecdc46e2aa091e6aa43bc",
    strip_prefix = "abseil-cpp-9687a8ea750bfcddf790372093245a1d041b21a3",
    urls = ["https://github.com/abseil/abseil-cpp/archive//9687a8ea750bfcddf790372093245a1d041b21a3.tar.gz"],
)

http_archive(
    name = "rules_java",
    sha256 = "c73336802d0b4882e40770666ad055212df4ea62cfa6edf9cb0f9d29828a0934",
    url = "https://github.com/bazelbuild/rules_java/releases/download/5.3.5/rules_java-5.3.5.tar.gz",
)

http_archive(
    name = "rules_android_ndk",
    sha256 = "d230a980e0d3a42b85d5fce2cb17ec3ac52b88d2cff5aaf86bae0f05b48adc55",
    strip_prefix = "rules_android_ndk-d5c9d46a471e8fcd80e7ec5521b78bb2df48f4e0",
    url = "https://github.com/bazelbuild/rules_android_ndk/archive/d5c9d46a471e8fcd80e7ec5521b78bb2df48f4e0.zip",
)

load("@rules_android_ndk//:rules.bzl", "android_ndk_repository")

http_archive(
    name = "build_bazel_rules_apple",
    patch_args = [
        "-p1",
    ],
    patches = [
        # Bypass checking ios unit test runner when building MP ios applications.
        "@//third_party:build_bazel_rules_apple_bypass_test_runner_check.diff",
        # https://github.com/bazelbuild/rules_apple/commit/95b1305255dc29874cacc3dc7fdc017f16d8dbe8
        "@//third_party:build_bazel_rules_apple_multi_arch_split_with_new_transition.diff",
    ],
    sha256 = "3e2c7ae0ddd181c4053b6491dad1d01ae29011bc322ca87eea45957c76d3a0c3",
    url = "https://github.com/bazelbuild/rules_apple/releases/download/2.1.0/rules_apple.2.1.0.tar.gz",
)

http_archive(
    name = "com_google_protobuf",
    patch_args = [
        "-p1",
    ],
    patches = [
        "@//third_party:com_google_protobuf_fixes.diff",
    ],
    sha256 = "87407cd28e7a9c95d9f61a098a53cf031109d451a7763e7dd1253abf8b4df422",
    strip_prefix = "protobuf-3.19.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.19.1.tar.gz"],
)

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
    patch_args = [
        "-p1",
    ],
    patches = [
        "@//third_party:zlib.diff",
    ],
    sha256 = "b3a24de97a8fdbc835b9833169501030b8977031bcb54b3b3ac13740f846ab30",
    strip_prefix = "zlib-1.2.13",
    url = "http://zlib.net/fossils/zlib-1.2.13.tar.gz",
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
RULES_JVM_EXTERNAL_TAG = "4.0"

RULES_JVM_EXTERNAL_SHA = "31701ad93dbfe544d597dbe62c9a1fdd76d81d8a9150c2bf1ecf928ecdf97169"

http_archive(
    name = "rules_jvm_external",
    sha256 = RULES_JVM_EXTERNAL_SHA,
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
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

# XNNPACK on 2024-11-18
http_archive(
    name = "XNNPACK",
    # `curl -L <url> | shasum -a 256`
    sha256 = "af30fe2b301330a7e19cd422acf22991de3c1f5d91dda58e9ee67544d608fa51",
    strip_prefix = "XNNPACK-dc1549a7141c7a9496ae160bb27b8700f0f6e1f1",
    url = "https://github.com/google/XNNPACK/archive/dc1549a7141c7a9496ae160bb27b8700f0f6e1f1.zip",
)

# 2020-07-09
http_archive(
    name = "pybind11_bazel",
    sha256 = "75922da3a1bdb417d820398eb03d4e9bd067c4905a4246d35a44c01d62154d91",
    strip_prefix = "pybind11_bazel-203508e14aab7309892a1c5f7dd05debda22d9a5",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/203508e14aab7309892a1c5f7dd05debda22d9a5.zip"],
)

# 2022-10-20
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    sha256 = "fcf94065efcfd0a7a828bacf118fa11c43f6390d0c805e3e6342ac119f2e9976",
    strip_prefix = "pybind11-2.10.1",
    urls = [
        "https://github.com/pybind/pybind11/archive/v2.10.1.zip",
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

# KleidiAI is needed to get the best possible performance out of XNNPack
http_archive(
    name = "KleidiAI",
    sha256 = "ad37707084a6d4ff41be10cbe8540c75bea057ba79d0de6c367c1bfac6ba0852",
    strip_prefix = "kleidiai-40a926833857fb64786e02f97703e42b1537cb57",
    urls = [
        "https://gitlab.arm.com/kleidi/kleidiai/-/archive/40a926833857fb64786e02f97703e42b1537cb57/kleidiai-40a926833857fb64786e02f97703e42b1537cb57.zip",
    ],
)

http_archive(
    name = "cpuinfo",
    sha256 = "e2bd8049d29dfbed675a0bc7c01947f8b8bd3f17f706b827d3f6c1e5c64dd8c3",
    strip_prefix = "cpuinfo-8df44962d437a0477f07ba6b8843d0b6a48646a4",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/8df44962d437a0477f07ba6b8843d0b6a48646a4.zip",
    ],
)

# TF on 2024-09-24
_TENSORFLOW_GIT_COMMIT = "5329ec8dd396487982ef3e743f98c0195af39a6b"

# curl -L https://github.com/tensorflow/tensorflow/archive/<COMMIT>.tar.gz | shasum -a 256
_TENSORFLOW_SHA256 = "eb1f8d740d59ea3dee91108ab1fc19d91c4e9ac2fd17d9ab86d865c3c43d81c9"

http_archive(
    name = "org_tensorflow",
    patch_args = [
        "-p1",
    ],
    patches = [
        "@//third_party:org_tensorflow_c_api_experimental.diff",
        # Diff is generated with a script, don't update it manually.
        "@//third_party:org_tensorflow_custom_ops.diff",
        # Works around Bazel issue with objc_library.
        # See https://github.com/bazelbuild/bazel/issues/19912
        "@//third_party:org_tensorflow_objc_build_fixes.diff",
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

# TODO: This is an are indirect dependency. We should factor it out.
http_archive(
    name = "pthreadpool",
    sha256 = "a4cf06de57bfdf8d7b537c61f1c3071bce74e57524fe053e0bbd2332feca7f95",
    strip_prefix = "pthreadpool-4fe0e1e183925bf8cfa6aae24237e724a96479b8",
    urls = ["https://github.com/Maratyszcza/pthreadpool/archive/4fe0e1e183925bf8cfa6aae24237e724a96479b8.zip"],
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
    urls = ["https://chromium.googlesource.com/libyuv/libyuv/+archive/2525698acba9bf9b701ba6b4d9584291a1f62257.tar.gz"],
)

# Note: protobuf-javalite is no longer released as a separate download, it's included in the main Java download.
# ...but the Java download is currently broken, so we use the "source" download.
http_archive(
    name = "com_google_protobuf_javalite",
    sha256 = "87407cd28e7a9c95d9f61a098a53cf031109d451a7763e7dd1253abf8b4df422",
    strip_prefix = "protobuf-3.19.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.19.1.tar.gz"],
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
    sha256 = "c97f55d05c98da6fcaf7f9ecc6a6dc6bc5b18b8564465f77abff8879d446491c",
    strip_prefix = "darts-clone-e40ce4627526985a7767444b6ed6893ab6ff8983",
    urls = [
        "https://github.com/s-yata/darts-clone/archive/e40ce4627526985a7767444b6ed6893ab6ff8983.zip",
    ],
)

http_archive(
    name = "org_tensorflow_text",
    patch_args = ["-p1"],
    patches = [
        "@//third_party:tensorflow_text_remove_tf_deps.diff",
        "@//third_party:tensorflow_text_a0f49e63.diff",
    ],
    repo_mapping = {"@com_google_re2": "@com_googlesource_code_re2"},
    sha256 = "f64647276f7288d1b1fe4c89581d51404d0ce4ae97f2bcc4c19bd667549adca8",
    strip_prefix = "text-2.2.0",
    urls = [
        "https://github.com/tensorflow/text/archive/v2.2.0.zip",
    ],
)

http_archive(
    name = "com_googlesource_code_re2",
    sha256 = "ef516fb84824a597c4d5d0d6d330daedb18363b5a99eda87d027e6bdd9cba299",
    strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
    urls = [
        "https://github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz",
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

http_archive(
    name = "android_opencv",
    build_file = "@//third_party:opencv_android.BUILD",
    strip_prefix = "OpenCV-android-sdk",
    type = "zip",
    url = "https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-android-sdk.zip",
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

# Hermetic CUDA
load(
    "@org_tensorflow//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@org_tensorflow//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
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
    "@org_tensorflow//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
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

load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_repos", "rules_proto_grpc_toolchains")

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
    sha256 = "038d4a21f9c72d71ab49e3a7d7677b39585329465d093a4260b6c73d2f3984d6",
    strip_prefix = "skia-ac75382cb971d2f5465b4608a74561ecb68599c5",
    urls = ["https://github.com/google/skia/archive/ac75382cb971d2f5465b4608a74561ecb68599c5.zip"],
)

http_archive(
    name = "skia_user_config",
    sha256 = "038d4a21f9c72d71ab49e3a7d7677b39585329465d093a4260b6c73d2f3984d6",
    strip_prefix = "skia-ac75382cb971d2f5465b4608a74561ecb68599c5/include/config",
    urls = ["https://github.com/google/skia/archive/ac75382cb971d2f5465b4608a74561ecb68599c5.zip"],
)
