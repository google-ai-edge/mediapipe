"""MediaPipe's shared dependencies that can be used by dependent projects. Includes build patches."""

load("@//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# TensorFlow repo should always go after the other external dependencies.
# TF on 2023-07-26.
_TENSORFLOW_GIT_COMMIT = "e92261fd4cec0b726692081c4d2966b75abf31dd"

# curl -L https://github.com/tensorflow/tensorflow/archive/<TENSORFLOW_GIT_COMMIT>.tar.gz | shasum -a 256
_TENSORFLOW_SHA256 = "478a229bd4ec70a5b568ac23b5ea013d9fca46a47d6c43e30365a0412b9febf4"

# ABSL cpp library lts_2023_01_25.
def mediapipe_absl():
    """Exports the ABSL depedency on TensorFlow."""
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
    """Exports the Semtencepiece depedency on TensorFlow."""
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

def mediapipe_flatbuffers():
    """Exports the FlatBuffers depedency on TensorFlow."""
    flatbuffers()

def mediapipe_tensorflow():
    """Exports the MediaPipe depedency on TensorFlow."""

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

    http_archive(
        name = "org_tensorflow",
        urls = ["https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT],
        patches = [
            "@//third_party:org_tensorflow_compatibility_fixes.diff",
            "@//third_party:org_tensorflow_system_python.diff",
            # Diff is generated with a script, don't update it manually.
            "@//third_party:org_tensorflow_custom_ops.diff",
            # Works around Bazel issue with objc_library.
            # See https://github.com/bazelbuild/bazel/issues/19912
            "@//third_party:org_tensorflow_objc_build_fixes.diff",
        ],
        patch_args = [
            "-p1",
        ],
        strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
        sha256 = _TENSORFLOW_SHA256,
    )
