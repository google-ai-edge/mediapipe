"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-23.1.21",
        sha256 = "d84cb25686514348e615163b458ae0767001b24b42325f426fd56406fd384238",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/flatbuffers/archive/v23.1.21.tar.gz",
            "https://github.com/google/flatbuffers/archive/v23.1.21.tar.gz",
        ],
        build_file = "//third_party/flatbuffers:BUILD.bazel",
        delete = ["build_defs.bzl", "BUILD.bazel"],
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )
