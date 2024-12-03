# TODO: Split build rules into several individual files to make them
# more manageable.

""".bzl file for mediapipe open source build configs."""

load("@npm//@bazel/typescript:index.bzl", "ts_project")
load(
    "//mediapipe/framework/tool:mediapipe_proto.bzl",
    _mediapipe_cc_proto_library = "mediapipe_cc_proto_library",
    _mediapipe_proto_library = "mediapipe_proto_library",
)

# TODO: enable def_rewrite once alias proto sources are generated.
def mediapipe_proto_library(def_rewrite = False, **kwargs):
    _mediapipe_proto_library(def_rewrite = def_rewrite, **kwargs)

mediapipe_cc_proto_library = _mediapipe_cc_proto_library

def provided_args(**kwargs):
    """Returns the keyword arguments omitting None arguments."""
    return {k: v for k, v in kwargs.items() if v != None}

def replace_suffix(string, old, new):
    """Returns a string with an old suffix replaced by a new suffix."""
    return string.endswith(old) and string[:-len(old)] + new or string

def mediapipe_ts_library(
        name,
        srcs = [],
        visibility = None,
        deps = [],
        testonly = 0,
        allow_unoptimized_namespaces = False):
    """Generate ts_project for MediaPipe open source version.

    Args:
      name: the name of the mediapipe_ts_library.
      srcs: the .ts files of the mediapipe_ts_library for Bazel use.
      visibility: visibility of this target.
      deps: a list of dependency labels for Bazel use.
      testonly: test only or not.
      allow_unoptimized_namespaces: ignored, used only internally
    """
    _ignore = [allow_unoptimized_namespaces]  # buildifier: disable=unused-variable

    ts_project(**provided_args(
        name = name,
        srcs = srcs,
        visibility = visibility,
        deps = deps + [
            "@npm//@types/jasmine",
            "@npm//@types/node",
            "@npm//@types/offscreencanvas",
            "@npm//@types/google-protobuf",
            "@npm//@webgpu/types",
        ],
        testonly = testonly,
        declaration = True,
        tsconfig = "//:tsconfig.json",
    ))

def mediapipe_ts_declaration(
        name,
        srcs,
        visibility = None,
        deps = []):
    """Generate ts_declaration for MediaPipe open source version.

    Args:
      name: the name of the mediapipe_ts_declaration.
      srcs: the .d.ts files of the mediapipe_ts_declaration for Bazel use.
      visibility: visibility of this target.
      deps: a list of dependency labels for Bazel use
    """

    # Bazel does not create JS files for .d.ts files, which leads to import
    # failures in our open source build. We simply re-name the .d.ts files
    # to .ts to work around this problem.
    for src in srcs:
        native.genrule(
            name = replace_suffix(src, ".d.ts", "_d_ts"),
            srcs = [src],
            outs = [replace_suffix(src, ".d.ts", ".ts")],
            visibility = visibility,
            cmd = "cp -n $< $@;",
        )

    mediapipe_ts_library(
        name = name,
        srcs = [replace_suffix(src, ".d.ts", "_d_ts") for src in srcs],
        visibility = visibility,
        deps = deps,
    )
