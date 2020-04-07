# TODO: Split build rules into several individual files to make them
# more manageable.

""".bzl file for mediapipe open source build configs."""

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")

def provided_args(**kwargs):
    """Returns the keyword arguments omitting None arguments."""
    return {k: v for k, v in kwargs.items() if v != None}

def allowed_args(**kwargs):
    """Returns the keyword arguments allowed for proto_library().

    Args:
      **kwargs: the specified keyword arguments.
    Returns:
      the allowed keyword arguments.
    """
    result = dict(kwargs)
    result.pop("cc_api_version", None)
    return result

# TODO: load this macro from a common helper file.
def mediapipe_proto_library(
        name,
        srcs,
        deps = [],
        visibility = None,
        testonly = 0,
        compatible_with = [],
        def_proto = True,
        def_cc_proto = True,
        def_py_proto = True,
        def_java_lite_proto = True,
        def_portable_proto = True,
        portable_deps = None):
    """Defines the proto_library targets needed for all mediapipe platforms.

    Args:
      name: the new proto_library target name.
      srcs: the ".proto" source files to compile.
      deps: the proto_library targets for all referenced protobufs.
      portable_deps: the portable_proto_library targets for all referenced protobufs.
      visibility: visibility of this target.
      testonly: true means the proto can be used for testing only.
      compatible_with: see go/target-constraints.
      def_proto: define the proto_library target
      def_cc_proto: define the cc_proto_library target
      def_py_proto: define the py_proto_library target
      def_java_lite_proto: define the java_lite_proto_library target
      def_portable_proto: define the portable_proto_library target
    """
    _ignore = [def_portable_proto, portable_deps]

    # The proto_library targets for the compiled ".proto" source files.
    proto_deps = [":" + name]

    if def_proto:
        native.proto_library(**allowed_args(**provided_args(
            name = name,
            srcs = srcs,
            deps = deps,
            visibility = visibility,
            testonly = testonly,
            cc_api_version = 2,
            compatible_with = compatible_with,
        )))

    if def_cc_proto:
        cc_deps = [dep.replace("_proto", "_cc_proto") for dep in deps]
        mediapipe_cc_proto_library(**provided_args(
            name = name.replace("_proto", "_cc_proto"),
            srcs = srcs,
            deps = proto_deps,
            cc_deps = cc_deps,
            visibility = visibility,
            testonly = testonly,
        ))

    if def_py_proto:
        py_deps = [dep.replace("_proto", "_py_pb2") for dep in deps]
        mediapipe_py_proto_library(**provided_args(
            name = name.replace("_proto", "_py_pb2"),
            srcs = srcs,
            proto_deps = proto_deps,
            py_proto_deps = py_deps,
            visibility = visibility,
            api_version = 2,
        ))

    if def_java_lite_proto:
        native.java_lite_proto_library(**provided_args(
            name = name.replace("_proto", "_java_proto_lite"),
            deps = proto_deps,
            strict_deps = 0,
            visibility = visibility,
        ))

def mediapipe_py_proto_library(
        name,
        srcs,
        visibility,
        py_proto_deps = [],
        proto_deps = None,
        api_version = None):
    """Generate py_proto_library for mediapipe open source version.

    Args:
      name: the name of the py_proto_library.
      srcs: the .proto files of the py_proto_library for Bazel use.
      visibility: visibility of this target.
      py_proto_deps: a list of dependency labels for Bazel use; must be py_proto_library.
    """
    _ignore = [api_version, proto_deps]
    py_proto_library(**provided_args(
        name = name,
        srcs = srcs,
        visibility = visibility,
        default_runtime = "@com_google_protobuf//:protobuf_python",
        protoc = "@com_google_protobuf//:protoc",
        deps = py_proto_deps + ["@com_google_protobuf//:protobuf_python"],
    ))

def mediapipe_cc_proto_library(name, srcs, visibility, deps = [], cc_deps = [], testonly = 0):
    """Generate cc_proto_library for mediapipe open source version.

      Args:
        name: the name of the cc_proto_library.
        srcs: the .proto files of the cc_proto_library for Bazel use.
        visibility: visibility of this target.
        deps: a list of dependency labels for Bazel use; must be cc_proto_library.
        testonly: test only proto or not.
    """
    _ignore = [deps]
    cc_proto_library(**provided_args(
        name = name,
        srcs = srcs,
        visibility = visibility,
        deps = cc_deps,
        testonly = testonly,
        cc_libs = ["@com_google_protobuf//:protobuf"],
        protoc = "@com_google_protobuf//:protoc",
        default_runtime = "@com_google_protobuf//:protobuf",
        alwayslink = 1,
    ))
