"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-6ff9e90e7e399f3977e99a315856b57c8afe5b4d",
        sha256 = "f4b3dfed9f8f4f0fd9f857fe96a46199cb5745ddb458cad20caf6837230ea188",
        urls = [
            "https://github.com/google/flatbuffers/archive/6ff9e90e7e399f3977e99a315856b57c8afe5b4d.tar.gz",
            "https://github.com/google/flatbuffers/archive/6ff9e90e7e399f3977e99a315856b57c8afe5b4d.tar.gz",
        ],
        build_file = "//third_party/flatbuffers:BUILD.bazel",
        delete = ["build_defs.bzl", "BUILD.bazel"],
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )
