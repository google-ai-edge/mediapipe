# TODO: Split build rules into several individual files to make them
# more manageable.

""".bzl file for mediapipe open source build configs."""

load("@aspect_rules_ts//ts:defs.bzl", "ts_project")
load(
    "//mediapipe/framework/tool:mediapipe_proto.bzl",
    _mediapipe_cc_proto_library = "mediapipe_cc_proto_library",
    _mediapipe_proto_library = "mediapipe_proto_library",
)

mediapipe_cc_proto_library = _mediapipe_cc_proto_library
mediapipe_proto_library = _mediapipe_proto_library

def provided_args(**kwargs):
    """Returns the keyword arguments omitting None arguments."""
    return {k: v for k, v in kwargs.items() if v != None}

def replace_suffix(string, old, new):
    """Returns a string with an old suffix replaced by a new suffix."""
    return string.endswith(old) and string[:-len(old)] + new or string

def ts_library(
        name,
        srcs = [],
        visibility = None,
        deps = [],
        testonly = 0,
        allow_unoptimized_namespaces = False):
    """Generate ts_project for MediaPipe open source version.

    Args:
        name: The name of the target.
        srcs: The list of source files.
        visibility: The visibility of the target.
        deps: The list of dependencies.
        testonly: Whether the target is testonly.
        allow_unoptimized_namespaces: Whether to allow unoptimized namespaces.
    """
    _ignore = [allow_unoptimized_namespaces]  # buildifier: disable=unused-variable

    # aspect_rules_ts automatically generates a `{name}_types`
    # target when `declaration = True`. By suffixing the underlying ts_project
    # target name and exposing an alias, we dodge collisions with existing
    # explicit `_types` targets (like ts_declaration) defined in MediaPipe
    # BUILDs.
    internal_name = name + "_internal"

    ts_project(**provided_args(
        name = internal_name,
        srcs = srcs,
        visibility = visibility,
        deps = deps + [
            "//:node_modules/@types/jasmine",
            "//:node_modules/@types/node",
            "//:node_modules/@types/offscreencanvas",
            "//:node_modules/@types/google-protobuf",
            "//:node_modules/@webgpu/types",
        ],
        testonly = testonly,
        declaration = True,
        transpiler = "tsc",
        tsconfig = "//:tsconfig",
    ))

    # Proxy the internal target back to the requested name.
    native.alias(
        name = name,
        actual = internal_name,
        visibility = visibility,
    )

def ts_declaration(
        name,
        srcs,
        visibility = None,
        deps = []):
    """Generate ts_declaration for MediaPipe open source version.

    Args:
        name: The name of the target.
        srcs: The list of source files.
        visibility: The visibility of the target.
        deps: The list of dependencies.
    """
    for src in srcs:
        native.genrule(
            name = replace_suffix(src, ".d.ts", "_d_ts"),
            srcs = [src],
            outs = [replace_suffix(src, ".d.ts", ".ts")],
            visibility = visibility,
            cmd = "cp -n $< $@;",
        )

    ts_library(
        name = name,
        srcs = [replace_suffix(src, ".d.ts", "_d_ts") for src in srcs],
        visibility = visibility,
        deps = deps,
    )
