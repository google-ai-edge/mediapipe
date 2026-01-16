"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-595bf0007ab1929570c7671f091313c8fc20644e",
        sha256 = "987300083ec1f1b095d5596ef8fb657ba46c45d786bc866a5e9029d7590a5e48",
        urls = [
            "https://github.com/google/flatbuffers/archive/595bf0007ab1929570c7671f091313c8fc20644e.tar.gz",
            "https://github.com/google/flatbuffers/archive/595bf0007ab1929570c7671f091313c8fc20644e.tar.gz",
        ],
        build_file = "//third_party/flatbuffers:BUILD.bazel",
        delete = ["build_defs.bzl", "BUILD.bazel"],
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )
