workspace(name = "mediapipe")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

skylib_version = "0.9.0"
http_archive(
    name = "bazel_skylib",
    type = "tar.gz",
    url = "https://github.com/bazelbuild/bazel-skylib/releases/download/{}/bazel_skylib-{}.tar.gz".format (skylib_version, skylib_version),
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
)
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check(minimum_bazel_version = "2.0.0")


# ABSL cpp library lts_2020_02_25
http_archive(
    name = "com_google_absl",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/20200225.tar.gz",
    ],
    # Remove after https://github.com/abseil/abseil-cpp/issues/326 is solved.
    patches = [
        "@//third_party:com_google_absl_f863b622fe13612433fdf43f76547d5edda0c93001.diff"
    ],
    patch_args = [
        "-p1",
    ],
    strip_prefix = "abseil-cpp-20200225",
    sha256 = "728a813291bdec2aa46eab8356ace9f75ac2ed9dfe2df5ab603c4e6c09f1c353"
)

http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-master",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/master.zip"],
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
    name = "com_github_glog_glog",
    url = "https://github.com/google/glog/archive/v0.3.5.zip",
    sha256 = "267103f8a1e9578978aa1dc256001e6529ef593e5aea38193d31c2872ee025e8",
    strip_prefix = "glog-0.3.5",
    build_file = "@//third_party:glog.BUILD",
    patches = [
        "@//third_party:com_github_glog_glog_9779e5ea6ef59562b030248947f787d1256132ae.diff"
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
    urls = ["https://chromium.googlesource.com/libyuv/libyuv/+archive/refs/heads/master.tar.gz"],
    build_file = "@//third_party:libyuv.BUILD",
)

# Note: protobuf-javalite is no longer released as a separate download, it's included in the main Java download.
# ...but the Java download is currently broken, so we use the "source" download.
http_archive(
    name = "com_google_protobuf_javalite",
    sha256 = "a79d19dcdf9139fa4b81206e318e33d245c4c9da1ffed21c87288ed4380426f9",
    strip_prefix = "protobuf-3.11.4",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.11.4.tar.gz"],
)

http_archive(
    name = "com_google_audio_tools",
    strip_prefix = "multichannel-audio-tools-master",
    urls = ["https://github.com/google/multichannel-audio-tools/archive/master.zip"],
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

# 2020-04-01
_TENSORFLOW_GIT_COMMIT = "805e47cea96c7e8c6fccf494d40a2392dc99fdd8"
_TENSORFLOW_SHA256= "9ee3ae604c2e1345ac60345becee6d659364721513f9cb8652eb2e7138320ca5"
http_archive(
    name = "org_tensorflow",
    urls = [
      "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
      "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
    patches = [
        "@//third_party:org_tensorflow_compatibility_fixes.diff",
        "@//third_party:org_tensorflow_protobuf_updates.diff",
    ],
    patch_args = [
        "-p1",
    ],
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
    sha256 = _TENSORFLOW_SHA256,
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(tf_repo_name = "org_tensorflow")

http_archive(
    name = "ceres_solver",
    url = "https://github.com/ceres-solver/ceres-solver/archive/1.14.0.zip",
    patches = [
        "@//third_party:ceres_solver_9bf9588988236279e1262f75d7f4d85711dfa172.diff"
    ],
    patch_args = [
        "-p1",
    ],
    strip_prefix = "ceres-solver-1.14.0",
    sha256 = "5ba6d0db4e784621fda44a50c58bb23b0892684692f0c623e2063f9c19f192f1"
)

new_local_repository(
    name = "linux_opencv",
    build_file = "@//third_party:opencv_linux.BUILD",
    path = "/usr",
)

new_local_repository(
    name = "linux_ffmpeg",
    build_file = "@//third_party:ffmpeg_linux.BUILD",
    path = "/usr"
)

new_local_repository(
    name = "macos_opencv",
    build_file = "@//third_party:opencv_macos.BUILD",
    path = "/usr",
)

new_local_repository(
    name = "macos_ffmpeg",
    build_file = "@//third_party:ffmpeg_macos.BUILD",
    path = "/usr",
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

# You may run setup_android.sh to install Android SDK and NDK.
android_ndk_repository(
    name = "androidndk",
)

android_sdk_repository(
    name = "androidsdk",
)

# iOS basic build deps.

http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "7a7afdd4869bb201c9352eed2daf37294d42b093579b70423490c1b4d4f6ce42",
    url = "https://github.com/bazelbuild/rules_apple/releases/download/0.19.0/rules_apple.0.19.0.tar.gz",
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
    "@build_bazel_apple_support//lib:repositories.bzl",
    "apple_support_dependencies",
)

apple_support_dependencies()

# More iOS deps.

http_archive(
    name = "google_toolbox_for_mac",
    url = "https://github.com/google/google-toolbox-for-mac/archive/v2.2.1.zip",
    sha256 = "e3ac053813c989a88703556df4dc4466e424e30d32108433ed6beaec76ba4fdc",
    strip_prefix = "google-toolbox-for-mac-2.2.1",
    build_file = "@//third_party:google_toolbox_for_mac.BUILD",
)

# Maven dependencies.

RULES_JVM_EXTERNAL_TAG = "3.2"
RULES_JVM_EXTERNAL_SHA = "82262ff4223c5fda6fb7ff8bd63db8131b51b413d26eb49e3131037e79e324af"

http_archive(
    name = "rules_jvm_external",
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    sha256 = RULES_JVM_EXTERNAL_SHA,
    url = "https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG,
)

load("@rules_jvm_external//:defs.bzl", "maven_install")

# Important: there can only be one maven_install rule. Add new maven deps here.
maven_install(
    name = "maven",
    artifacts = [
        "junit:junit:4.12",
        "androidx.test.espresso:espresso-core:3.1.1",
        "org.hamcrest:hamcrest-library:1.3",
        "androidx.concurrent:concurrent-futures:1.0.0-alpha03",
        "androidx.lifecycle:lifecycle-common:2.2.0",
        "androidx.annotation:annotation:aar:1.1.0",
        "androidx.appcompat:appcompat:aar:1.1.0-rc01",
        "androidx.camera:camera-core:aar:1.0.0-alpha06",
        "androidx.camera:camera-camera2:aar:1.0.0-alpha06",
        "androidx.constraintlayout:constraintlayout:aar:1.1.3",
        "androidx.core:core:aar:1.1.0-rc03",
        "androidx.legacy:legacy-support-v4:aar:1.0.0",
        "androidx.recyclerview:recyclerview:aar:1.1.0-beta02",
        "com.google.android.material:material:aar:1.0.0-rc01",
        "com.google.code.findbugs:jsr305:3.0.2",
        "com.google.flogger:flogger-system-backend:0.3.1",
        "com.google.flogger:flogger:0.3.1",
        "com.google.guava:guava:27.0.1-android",
    ],
    repositories = [
        "https://jcenter.bintray.com",
        "https://maven.google.com",
        "https://dl.google.com/dl/android/maven2",
        "https://repo1.maven.org/maven2",
    ],
    fetch_sources = True,
    version_conflict_policy = "pinned",
)
