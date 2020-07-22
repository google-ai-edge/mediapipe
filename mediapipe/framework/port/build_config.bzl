# TODO: Split build rules into several individual files to make them
# more manageable.

""".bzl file for mediapipe open source build configs."""

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")

def provided_args(**kwargs):
    """Returns the keyword arguments omitting None arguments."""
    return {k: v for k, v in kwargs.items() if v != None}

def replace_suffix(string, old, new):
    """Returns a string with an old suffix replaced by a new suffix."""
    return string.endswith(old) and string[:-len(old)] + new or string

def replace_deps(deps, old, new, drop_google_protobuf = True):
    """Returns deps with an old suffix replaced by a new suffix.

    Args:
      deps: the specified dep targets.
      old: the suffix to remove.
      new: the suffix to insert.
      drop_google_protobuf: if true, omit google/protobuf deps.
    Returns:
      the modified dep targets.
    """
    if drop_google_protobuf:
        deps = [dep for dep in deps if not dep.startswith("@com_google_protobuf//")]
    deps = [replace_suffix(dep, "any_proto", "cc_wkt_protos") for dep in deps]
    deps = [replace_suffix(dep, old, new) for dep in deps]
    return deps

# TODO: load this macro from a common helper file.
def mediapipe_proto_library(
        name,
        srcs,
        deps = [],
        exports = None,
        visibility = None,
        testonly = None,
        compatible_with = None,
        def_proto = True,
        def_cc_proto = True,
        def_py_proto = True,
        def_java_lite_proto = True,
        def_portable_proto = True,
        def_objc_proto = True,
        def_java_proto = True,
        def_jspb_proto = True,
        portable_deps = None):
    """Defines the proto_library targets needed for all mediapipe platforms.

    Args:
      name: the new proto_library target name.
      srcs: the ".proto" source files to compile.
      deps: the proto_library targets for all referenced protobufs.
      exports: deps that are published with "import public".
      portable_deps: the portable_proto_library targets for all referenced protobufs.
      visibility: visibility of this target.
      testonly: true means the proto can be used for testing only.
      compatible_with: see go/target-constraints.
      def_proto: define the proto_library target
      def_cc_proto: define the cc_proto_library target
      def_py_proto: define the py_proto_library target
      def_java_lite_proto: define the java_lite_proto_library target
      def_portable_proto: define the portable_proto_library target
      def_objc_proto: define the objc_proto_library target
      def_java_proto: define the java_proto_library target
      def_jspb_proto: define the jspb_proto_library target
    """
    _ignore = [def_portable_proto, def_objc_proto, def_java_proto, def_jspb_proto, portable_deps]

    # The proto_library targets for the compiled ".proto" source files.
    proto_deps = [":" + name]

    if def_proto:
        native.proto_library(**provided_args(
            name = name,
            srcs = srcs,
            deps = deps,
            exports = exports,
            visibility = visibility,
            testonly = testonly,
            compatible_with = compatible_with,
        ))

    if def_cc_proto:
        cc_deps = replace_deps(deps, "_proto", "_cc_proto", False)
        mediapipe_cc_proto_library(**provided_args(
            name = replace_suffix(name, "_proto", "_cc_proto"),
            srcs = srcs,
            deps = proto_deps,
            cc_deps = cc_deps,
            visibility = visibility,
            testonly = testonly,
        ))

    if def_py_proto:
        py_deps = replace_deps(deps, "_proto", "_py_pb2")
        mediapipe_py_proto_library(**provided_args(
            name = replace_suffix(name, "_proto", "_py_pb2"),
            srcs = srcs,
            proto_deps = proto_deps,
            py_proto_deps = py_deps,
            api_version = 2,
            visibility = visibility,
            testonly = testonly,
        ))

    if def_java_lite_proto:
        native.java_lite_proto_library(**provided_args(
            name = replace_suffix(name, "_proto", "_java_proto_lite"),
            deps = proto_deps,
            strict_deps = 0,
            visibility = visibility,
            testonly = testonly,
            compatible_with = compatible_with,
        ))

def mediapipe_py_proto_library(
        name,
        srcs,
        visibility = None,
        py_proto_deps = [],
        proto_deps = None,
        api_version = None,
        testonly = 0):
    """Generate py_proto_library for mediapipe open source version.

    Args:
      name: the name of the py_proto_library.
      api_version: api version for bazel use only.
      srcs: the .proto files of the py_proto_library for Bazel use.
      visibility: visibility of this target.
      py_proto_deps: a list of dependency labels for Bazel use; must be py_proto_library.
      proto_deps: a list of dependency labels for bazel use.
      testonly: test only proto or not.
    """
    _ignore = [api_version, proto_deps]
    py_proto_library(**provided_args(
        name = name,
        srcs = srcs,
        visibility = visibility,
        default_runtime = "@com_google_protobuf//:protobuf_python",
        protoc = "@com_google_protobuf//:protoc",
        deps = py_proto_deps + ["@com_google_protobuf//:protobuf_python"],
        testonly = testonly,
    ))

def mediapipe_cc_proto_library(name, srcs, visibility = None, deps = [], cc_deps = [], testonly = 0):
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
