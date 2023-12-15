"""MediaPipe's shared dependencies that can be used by dependent projects. Includes build patches."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# ABSL cpp library lts_2023_01_25.
def mediapipe_absl():
    http_archive(
        name = "com_google_absl",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20230125.0.tar.gz",
        ],
        patches = [
            "@//third_party:com_google_absl_windows_patch.diff",
        ],
        patch_args = [
            "-p1",
        ],
        strip_prefix = "abseil-cpp-20230125.0",
        sha256 = "3ea49a7d97421b88a8c48a0de16c16048e17725c7ec0f1d3ea2683a2a75adc21",
    )

def mediapipe_sentencepiece():
    http_archive(
        name = "com_google_sentencepiece",
        strip_prefix = "sentencepiece-0.1.96",
        sha256 = "8409b0126ebd62b256c685d5757150cf7fcb2b92a2f2b98efb3f38fc36719754",
        urls = [
            "https://github.com/google/sentencepiece/archive/refs/tags/v0.1.96.zip",
        ],
        build_file = "@//third_party:sentencepiece.BUILD",
        patches = ["@//third_party:com_google_sentencepiece.diff"],
        patch_args = ["-p1"],
    )
