load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["metadata_schema.fbs"])

# Generic schema for model metadata.
flatbuffer_cc_library(
    name = "metadata_schema_cc",
    srcs = ["metadata_schema.fbs"],
)
