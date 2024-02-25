"""Provides BUILD macros for MediaPipe proto-buffers.
"""

# buildifier: disable=out-of-order-load
load("//mediapipe/framework/tool:mediapipe_graph.bzl", "mediapipe_options_library")
load("//mediapipe/framework/tool:mediapipe_proto_allowlist.bzl", "rewrite_target_list")
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc//js:defs.bzl", "js_proto_library")

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
    if deps == None:
        return deps

    if drop_google_protobuf:
        deps = [dep for dep in deps if not dep.startswith("@com_google_protobuf//")]
    deps = [replace_suffix(dep, "any_proto", "cc_wkt_protos") for dep in deps]

    deps = [dep for dep in deps if not dep.endswith("_annotations")]
    deps = [replace_suffix(dep, old, new) for dep in deps]
    return deps

def mediapipe_proto_library_impl(
        name,
        srcs,
        deps = [],
        exports = None,
        visibility = None,
        testonly = None,
        compatible_with = None,
        alwayslink = None,
        def_proto = True,
        def_cc_proto = True,
        def_py_proto = True,
        def_java_lite_proto = True,
        def_kt_lite_proto = True,
        def_objc_proto = True,
        def_java_proto = True,
        def_jspb_proto = True,
        def_go_proto = True,
        def_dart_proto = True,
        def_options_lib = True):
    """Defines the proto_library targets needed for all mediapipe platforms.

    Args:
      name: the new proto_library target name.
      srcs: the ".proto" source files to compile.
      deps: the proto_library targets for all referenced protobufs.
      exports: deps that are published with "import public".
      visibility: visibility of this target.
      testonly: true means the proto can be used for testing only.
      compatible_with: a list of environments the rule is compatible with.
      alwayslink: any binary depending on the generated C++ library will link all object files,
          useful if the protocol buffer is looked up dynamically.
      def_proto: define the proto_library target
      def_cc_proto: define the cc_proto_library target
      def_py_proto: define the py_proto_library target
      def_java_lite_proto: define the java_lite_proto_library target
      def_kt_lite_proto: define the kt_lite_proto_library target
      def_objc_proto: define the objc_proto_library target
      def_java_proto: define the java_proto_library target
      def_jspb_proto: define the jspb_proto_library target
      def_go_proto: define the go_proto_library target
      def_dart_proto: define the dart_proto_library target
      def_options_lib: define the mediapipe_options_library target
    """

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
            compatible_with = compatible_with,
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
            compatible_with = compatible_with,
        ))

    if def_java_lite_proto:
        native.java_lite_proto_library(**provided_args(
            name = replace_suffix(name, "_proto", "_java_proto_lite"),
            deps = proto_deps,
            visibility = visibility,
            testonly = testonly,
            compatible_with = compatible_with,
        ))

    if def_java_proto:
        native.java_proto_library(**provided_args(
            name = replace_suffix(name, "_proto", "_java_proto"),
            deps = proto_deps,
            visibility = visibility,
            testonly = testonly,
            compatible_with = compatible_with,
        ))

    if def_jspb_proto:
        mediapipe_js_proto_library(**provided_args(
            name = replace_suffix(name, "_proto", "_jspb_proto"),
            srcs = srcs,
            deps = proto_deps,
            lib_proto_deps = deps,
            visibility = visibility,
            testonly = testonly,
            compatible_with = compatible_with,
        ))

    if def_options_lib:
        cc_deps = replace_deps(deps, "_proto", "_cc_proto")
        mediapipe_options_library(**provided_args(
            name = replace_suffix(name, "_proto", "_options_lib"),
            proto_lib = name,
            deps = cc_deps,
            visibility = visibility,
            testonly = testonly,
            compatible_with = compatible_with,
        ))

def SourceName(proto_target):
    "Returns the proto source file name for a target without extension."
    words = proto_target.split("_")[:-1]
    return "_".join(words)

def MessageName(proto_target):
    "Returns the proto message name for a target."
    words = proto_target.split("_")[:-1]
    words_upper = [w.capitalize() for w in words]
    return "".join(words_upper)

def SubsituteCommand(old, new):
    "Returns the shell command to replace a regex."
    old = old.replace("/", "[/]")
    new = new.replace('"', '\\"')
    return "awk '{print gensub(/" + old + '/, "' + new + '", "g");}' + "'"

rewrite_source_regex = "|".join([SourceName(t) for t in rewrite_target_list])
rewrite_message_regex = "|".join([MessageName(t) for t in rewrite_target_list])

# After rewrite, all intra-package references refer to "mediapipe",
# and all imports refer to rewritten proto source files.
def rewrite_mediapipe_proto(name, rewrite_proto, source_proto, **kwargs):
    """Generates a proto source file with package changed to mediapipe.

    Args:
      name: the new target name.
      rewrite_proto: the new ".proto" source file name.
      source_proto: the old ".proto" source file name.
      **kwargs: the remaining arguments.
    """

    split_path = r"(.*)/(" + rewrite_source_regex + ").proto"
    join_path = r"\\1/protobuf/\\2.proto"
    rewrite_package = SubsituteCommand(
        "package mediapipe;",
        "package mediapipe;",
    )
    rewrite_import = SubsituteCommand(
        'import "' + split_path + '";',
        'import "' + join_path + '";',
    )
    rewrite_import_public = SubsituteCommand(
        'import public "' + split_path + '";',
        'import public "' + join_path + '";',
    )
    rewrite_ref = SubsituteCommand(
        r"mediapipe\.(" + rewrite_message_regex + ")",
        r"mediapipe.\\1",
    )
    rewrite_objc = SubsituteCommand(
        r'objc_class_prefix = "MediaPipe"',
        r'objc_class_prefix = "MPP"',
    )
    native.genrule(
        name = name,
        outs = [rewrite_proto],
        srcs = [source_proto],
        cmd = "for OUT in $(OUTS); do \n" +
              "  cat $(location " + source_proto + ") " +
              "  | " + rewrite_package +
              "  | " + rewrite_import +
              "  | " + rewrite_import_public +
              "  | " + rewrite_ref +
              "  | " + rewrite_objc +
              "  > $$OUT \n" +
              "done",
        **kwargs
    )

# Returns the path to a rewritten protobuf file or label.
def mediapipe_proto_path(path):
    if path.startswith("@com_google_protobuf//"):
        return path
    i = max(path.rfind(":"), path.rfind("/")) + 1
    return path[0:i] + "protobuf/" + path[i:]

# Returns a list of paths to rewritten protobuf files or labels.
def mediapipe_proto_paths(paths):
    if paths == None:
        return paths
    return [mediapipe_proto_path(path) for path in paths]

def mediapipe_proto_library(
        name,
        srcs,
        deps = [],
        exports = None,
        visibility = None,
        testonly = None,
        compatible_with = None,
        alwayslink = None,
        def_proto = True,
        def_cc_proto = True,
        def_py_proto = True,
        def_java_lite_proto = True,
        def_kt_lite_proto = True,
        def_portable_proto = True,  # @unused
        def_objc_proto = True,
        def_java_proto = True,
        def_jspb_proto = True,
        def_go_proto = True,
        def_dart_proto = True,
        def_options_lib = True,
        def_rewrite = True,
        portable_deps = None):  # @unused
    """Defines the proto_library targets needed for all mediapipe platforms.

    Args:
      name: the new proto_library target name.
      srcs: the ".proto" source files to compile.
      deps: the proto_library targets for all referenced protobufs.
      exports: deps that are published with "import public".
      portable_deps: ignored since portable protos are gone.
      visibility: visibility of this target.
      testonly: true means the proto can be used for testing only.
      compatible_with: a list of environments the rule is compatible with.
      alwayslink: any binary depending on the generated C++ library will link all object files,
          useful if the protocol buffer is looked up dynamically.
      def_proto: define the proto_library target
      def_cc_proto: define the cc_proto_library target
      def_py_proto: define the py_proto_library target
      def_java_lite_proto: define the java_lite_proto_library target
      def_kt_lite_proto: define the kt_lite_proto_library target
      def_portable_proto: ignored since portable protos are gone
      def_objc_proto: define the objc_proto_library target
      def_java_proto: define the java_proto_library target
      def_jspb_proto: define the jspb_proto_library target
      def_go_proto: define the go_proto_library target
      def_dart_proto: define the dart_proto_library target
      def_options_lib: define the mediapipe_options_library target
      def_rewrite: define a sibling mediapipe_proto_library with package "mediapipe"
    """

    mediapipe_proto_library_impl(
        name = name,
        srcs = srcs,
        deps = deps,
        exports = exports,
        visibility = visibility,
        testonly = testonly,
        compatible_with = compatible_with,
        alwayslink = alwayslink,
        def_proto = def_proto,
        def_cc_proto = def_cc_proto,
        def_py_proto = def_py_proto,
        def_java_lite_proto = def_java_lite_proto,
        def_kt_lite_proto = def_kt_lite_proto,
        def_objc_proto = def_objc_proto,
        def_java_proto = def_java_proto,
        def_jspb_proto = def_jspb_proto,
        def_go_proto = def_go_proto,
        def_dart_proto = def_dart_proto,
        def_options_lib = def_options_lib,
    )

    if def_rewrite:
        source_proto = replace_suffix(name, "_proto", ".proto")
        rewrite_proto = mediapipe_proto_path(source_proto)
        rewrite_mediapipe_proto(
            name = replace_suffix(name, "_proto", "_rewrite"),
            rewrite_proto = rewrite_proto,
            source_proto = source_proto,
            compatible_with = compatible_with,
        )
        mediapipe_proto_library_impl(
            name = mediapipe_proto_path(name),
            srcs = [rewrite_proto],
            deps = mediapipe_proto_paths(deps),
            exports = mediapipe_proto_paths(exports),
            visibility = visibility,
            testonly = testonly,
            compatible_with = compatible_with,
            alwayslink = alwayslink,
            def_proto = def_proto,
            def_cc_proto = def_cc_proto,
            def_py_proto = def_py_proto,
            def_java_lite_proto = def_java_lite_proto,
            def_kt_lite_proto = def_kt_lite_proto,
            def_objc_proto = def_objc_proto,
            def_java_proto = def_java_proto,
            def_jspb_proto = def_jspb_proto,
            def_go_proto = def_go_proto,
            def_dart_proto = def_dart_proto,
            # A clone of mediapipe_options_library() will redefine some classes.
            def_options_lib = False,
        )

def mediapipe_py_proto_library_oss(
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

def mediapipe_cc_proto_library_oss(
        name,
        srcs,
        visibility = None,
        deps = [],
        cc_deps = [],
        testonly = 0):
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

def mediapipe_js_proto_library_oss(
        name,
        srcs,
        deps,
        lib_proto_deps,
        visibility = None,
        testonly = 0,
        compatible_with = None):
    """Generate js_proto_library for mediapipe open source version.

    Args:
      name: the name of the js_proto_library.
      srcs: the .proto files of the js_proto_library for Bazel use.
      deps: a list of dependency labels for bazel use ; must be proto_library.
      lib_proto_deps: a list of "_proto" dependency labels.
      visibility: Visibility of this target.
      testonly: test only proto or not.
      compatible_with: a list of environments the rule is compatible with.
    """
    _ignore = [deps, testonly, compatible_with]

    js_deps = replace_deps(lib_proto_deps, "_proto", "_jspb_proto", False)
    proto_library(
        name = replace_suffix(name, "_jspb_proto", "_lib_proto"),
        srcs = srcs,
        deps = lib_proto_deps,
        visibility = visibility,
    )
    js_proto_library(
        name = name,
        protos = [replace_suffix(name, "_jspb_proto", "_lib_proto")],
        output_mode = "NO_PREFIX_FLAT",
        # Need to specify this to work around bug in js_proto_library()
        # https://github.com/bazelbuild/rules_nodejs/issues/3503
        legacy_path = "unused",
        deps = js_deps,
        visibility = visibility,
    )

def mediapipe_py_proto_library(**kwargs):
    mediapipe_py_proto_library_oss(**kwargs)

def mediapipe_cc_proto_library(**kwargs):
    mediapipe_cc_proto_library_oss(**kwargs)

def mediapipe_js_proto_library(**kwargs):
    mediapipe_js_proto_library_oss(**kwargs)
