"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-2.0.6",
        sha256 = "e2dc24985a85b278dd06313481a9ca051d048f9474e0f199e372fea3ea4248c9",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/flatbuffers/archive/v2.0.6.tar.gz",
            "https://github.com/google/flatbuffers/archive/v2.0.6.tar.gz",
        ],
        build_file = "//third_party/flatbuffers:BUILD.bazel",
        delete = ["build_defs.bzl", "BUILD.bazel"],
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )
