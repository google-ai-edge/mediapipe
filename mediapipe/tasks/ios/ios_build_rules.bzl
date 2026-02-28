"""This module contains utility macros for MediaPipe Tasks iOS BUILD files."""

load("@build_bazel_rules_apple//apple:apple.bzl", "apple_static_xcframework")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//mediapipe/framework/tool:ios.bzl",
    "MPP_TASK_MINIMUM_OS_VERSION",
)

def _dummy_objc_library(name):
    """Creates a dummy objc_library with a single dummy source file.

    This macro generates a genrule that creates a dummy Objective-C source file
    and an objc_library that compiles it. This is useful for creating dummy
    libraries to satisfy dependencies in apple_static_xcframework rules.

    Args:
      name: The root name of the dummy library.
    """
    genrule_name = "_%s_dummy_src" % name
    native.genrule(
        name = genrule_name,
        outs = ["%s_dummy.m" % name],
        cmd = "echo 'void mediapipe_tasks_%s_dummy(){}' > $@" % name,
        visibility = ["//visibility:private"],
    )

    objc_library(
        name = "%s_dummy" % name,
        srcs = [":" + genrule_name],
        visibility = ["//visibility:private"],
    )

def mediapipe_static_xcframework(name, **kwargs):
    """An apple_static_xcframework with a dummy library to allow for empty frameworks.

    Args:
      name: The name of the apple_static_xcframework target.
      **kwargs: Arguments passed to apple_static_xcframework.
    """
    if "deps" not in kwargs:
        _dummy_objc_library(name = name)
        kwargs["deps"] = [":%s_dummy" % name]

    if "ios" not in kwargs:
        kwargs["ios"] = {
            "simulator": [
                "arm64",
                "x86_64",
            ],
            "device": ["arm64"],
        }
    if "minimum_os_versions" not in kwargs:
        kwargs["minimum_os_versions"] = {
            "ios": MPP_TASK_MINIMUM_OS_VERSION,
        }

    apple_static_xcframework(
        name = name,
        **kwargs
    )
