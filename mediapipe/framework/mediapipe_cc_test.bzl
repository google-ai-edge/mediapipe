"""Macro for multi-platform C++ tests."""

DEFAULT_ADDITIONAL_TEST_DEPS = []

def mediapipe_cc_test(
        name,
        srcs = [],
        data = [],
        deps = [],
        size = None,
        timeout = None,
        additional_deps = DEFAULT_ADDITIONAL_TEST_DEPS,
        **kwargs):
    # Note: additional_deps are MediaPipe-specific test support deps added by default.
    # They are provided as a default argument so they can be disabled if desired.
    native.cc_library(
        name = name + "_lib",
        testonly = 1,
        srcs = srcs,
        data = data,
        deps = deps + additional_deps,
        alwayslink = 1,
    )

    native.cc_test(
        name = name,
        size = size,
        timeout = timeout,
        deps = [":{}_lib".format(name)],
    )
