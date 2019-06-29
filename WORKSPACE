workspace(name = "mediapipe")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "bbccf674aa441c266df9894182d80de104cabd19be98be002f6d478aaa31574d",
    strip_prefix = "bazel-skylib-2169ae1c374aab4a09aa90e65efe1a3aad4e279b",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/2169ae1c374aab4a09aa90e65efe1a3aad4e279b.tar.gz"],
)
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check(minimum_bazel_version = "0.23.0")

# ABSL cpp library.
http_archive(
    name = "com_google_absl",
    # Head commit on 2019-04-12.
    # TODO: Switch to the latest absl version when the problem gets
    # fixed.
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/a02f62f456f2c4a7ecf2be3104fe0c6e16fbad9a.tar.gz",
    ],
    sha256 = "d437920d1434c766d22e85773b899c77c672b8b4865d5dc2cd61a29fdff3cf03",
    strip_prefix = "abseil-cpp-a02f62f456f2c4a7ecf2be3104fe0c6e16fbad9a",
)

# GoogleTest/GoogleMock framework. Used by most unit-tests.
http_archive(
     name = "com_google_googletest",
     urls = ["https://github.com/google/googletest/archive/master.zip"],
     strip_prefix = "googletest-master",
)

# Google Benchmark library.
http_archive(
    name = "com_google_benchmark",
    urls = ["https://github.com/google/benchmark/archive/master.zip"],
    strip_prefix = "benchmark-master",
    build_file = "@//third_party:benchmark.BUILD",
)

# gflags needed by glog
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "6e16c8bc91b1310a44f3965e616383dbda48f83e8c1eaa2370a215057b00cabe",
    strip_prefix = "gflags-77592648e3f3be87d6c7123eb81cbad75f9aef5a",
    urls = [
        "https://mirror.bazel.build/github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
        "https://github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
    ],
)

# glog
http_archive(
    name = "com_google_glog",
    url = "https://github.com/google/glog/archive/v0.3.5.zip",
    sha256 = "267103f8a1e9578978aa1dc256001e6529ef593e5aea38193d31c2872ee025e8",
    strip_prefix = "glog-0.3.5",
    build_file = "@//third_party:glog.BUILD",
)

# libyuv
http_archive(
    name = "libyuv",
    urls = ["https://chromium.googlesource.com/libyuv/libyuv/+archive/refs/heads/master.tar.gz"],
    build_file = "@//third_party:libyuv.BUILD",
)

http_archive(
    name = "com_google_protobuf_javalite",
    sha256 = "79d102c61e2a479a0b7e5fc167bcfaa4832a0c6aad4a75fa7da0480564931bcc",
    strip_prefix = "protobuf-384989534b2246d413dbcd750744faab2607b516",
    urls = ["https://github.com/google/protobuf/archive/384989534b2246d413dbcd750744faab2607b516.zip"],
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

# TensorFlow r1.14-rc0
http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-1.14.0-rc0",
    sha256 = "76404a6157a45e8d7a07e4f5690275256260130145924c2a7c73f6eda2a3de10",
    urls = ["https://github.com/tensorflow/tensorflow/archive/v1.14.0-rc0.zip"],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(tf_repo_name = "org_tensorflow")

# Please run $ sudo apt-get install libopencv-dev
new_local_repository(
    name = "linux_opencv",
    build_file = "@//third_party:opencv_linux.BUILD",
    path = "/usr",
)

# Please run $ brew install opencv
new_local_repository(
    name = "macos_opencv",
    build_file = "@//third_party:opencv_macos.BUILD",
    path = "/usr",
)

http_archive(
    name = "android_opencv",
    sha256="056b849842e4fa8751d09edbb64530cfa7a63c84ccd232d0ace330e27ba55d0b",
    build_file = "@//third_party:opencv_android.BUILD",
    strip_prefix = "OpenCV-android-sdk",
    type = "zip",
    url = "https://github.com/opencv/opencv/releases/download/4.1.0/opencv-4.1.0-android-sdk.zip",
)

RULES_JVM_EXTERNAL_TAG = "2.2"
RULES_JVM_EXTERNAL_SHA = "f1203ce04e232ab6fdd81897cf0ff76f2c04c0741424d192f28e65ae752ce2d6"

http_archive(
    name = "rules_jvm_external",
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    sha256 = RULES_JVM_EXTERNAL_SHA,
    url = "https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG,
)

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    artifacts = [
        "com.android.support.constraint:constraint-layout:aar:1.0.2",
        "androidx.appcompat:appcompat:aar:1.0.2",
    ],
    repositories = [
        "https://dl.google.com/dl/android/maven2",
    ],
)

maven_server(
    name = "google_server",
    url = "https://dl.google.com/dl/android/maven2",
)

maven_jar(
    name = "androidx_lifecycle",
    artifact = "androidx.lifecycle:lifecycle-common:2.0.0",
    sha1 = "e070ffae07452331bc5684734fce6831d531785c",
    server = "google_server",
)

maven_jar(
     name = "androidx_concurrent_futures",
     artifact = "androidx.concurrent:concurrent-futures:1.0.0-alpha03",
     sha1 = "b528df95c7e2fefa2210c0c742bf3e491c1818ae",
     server = "google_server",
)

maven_jar(
    name = "com_google_guava_android",
    artifact = "com.google.guava:guava:27.0.1-android",
    sha1 = "b7e1c37f66ef193796ccd7ea6e80c2b05426182d",
)

maven_jar(
    name = "com_google_common_flogger",
    artifact = "com.google.flogger:flogger:0.3.1",
    sha1 = "585030fe1ec709760cbef997a459729fb965df0e",
)

maven_jar(
    name = "com_google_common_flogger_system_backend",
    artifact = "com.google.flogger:flogger-system-backend:0.3.1",
    sha1 = "287b569d76abcd82f9de87fe41829fbc7ebd8ac9",
)

maven_jar(
    name = "com_google_code_findbugs",
    artifact = "com.google.code.findbugs:jsr305:3.0.2",
    sha1 = "25ea2e8b0c338a877313bd4672d3fe056ea78f0d",
)

# You may run setup_android.sh to install Android SDK and NDK.
android_ndk_repository(
    name = "androidndk",
)

android_sdk_repository(
    name = "androidsdk",
)
