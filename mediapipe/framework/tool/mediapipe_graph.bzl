"""Provides BUILD macros for MediaPipe graphs.

mediapipe_binary_graph() converts a graph from text format to serialized binary
format.

Example:
  mediapipe_binary_graph(
    name = "make_graph_binarypb",
    graph = "//mediapipe/framework/tool/testdata:test_graph",
    output_name = "test.binarypb",
    deps = [
        "//video/annotation:graph_calculators_lib",
    ]
  )

"""

load("//mediapipe/framework:encode_binary_proto.bzl", "encode_binary_proto", "generate_proto_descriptor_set")
load("//mediapipe/framework:transitive_protos.bzl", "transitive_protos")
load("//mediapipe/framework/deps:expand_template.bzl", "expand_template")
load("//mediapipe/framework/tool:build_defs.bzl", "clean_dep")
load("//mediapipe/framework/deps:descriptor_set.bzl", "direct_descriptor_set", "transitive_descriptor_set")
load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

def mediapipe_binary_graph(name, graph = None, output_name = None, deps = [], testonly = False, **kwargs):
    """Converts a graph from text format to binary format."""

    if not graph:
        fail("No input graph file specified.")

    if not output_name:
        fail("Must specify the output_name.")

    transitive_protos(
        name = name + "_gather_cc_protos",
        deps = deps,
        testonly = testonly,
    )

    # Compile a simple proto parser binary using the deps.
    native.cc_binary(
        name = name + "_text_to_binary_graph",
        visibility = ["//visibility:private"],
        deps = [
            clean_dep("//mediapipe/framework/tool:text_to_binary_graph"),
            name + "_gather_cc_protos",
        ],
        tags = ["manual"],
        testonly = testonly,
    )

    # Invoke the proto parser binary.
    native.genrule(
        name = name,
        srcs = [graph],
        outs = [output_name],
        cmd = (
            "$(location " + name + "_text_to_binary_graph" + ") " +
            ("--proto_source=$(location %s) " % graph) +
            ("--proto_output=\"$@\" ")
        ),
        tools = [name + "_text_to_binary_graph"],
        testonly = testonly,
    )

def data_as_c_string(
        name,
        srcs,
        outs = None,
        testonly = None):
    """Encodes the data from a file as a C string literal.

    This produces a text file containing the quoted C string literal. It can be
    included directly in a C++ source file.

    Args:
      name: The name of the rule.
      srcs: A list containing a single item, the file to encode.
      outs: A list containing a single item, the name of the output text file.
            Defaults to the rule name.
      testonly: pass 1 if the graph is to be used only for tests.
    """
    if len(srcs) != 1:
        fail("srcs must be a single-element list")
    if outs == None:
        outs = [name]
    encode_as_c_string = clean_dep("//mediapipe/framework/tool:encode_as_c_string")
    native.genrule(
        name = name,
        srcs = srcs,
        outs = outs,
        cmd = "$(location %s) \"$<\" > \"$@\"" % encode_as_c_string,
        tools = [encode_as_c_string],
        testonly = testonly,
    )

def mediapipe_simple_subgraph(
        name,
        register_as,
        graph,
        deps = [],
        tflite_deps = None,
        visibility = None,
        testonly = None,
        **kwargs):
    """Defines a registered subgraph for inclusion in other graphs.

    Args:
      name: name of the subgraph target to define.
      register_as: name used to invoke this graph in supergraphs. Should be in
          CamelCase.
      graph: the BUILD label of a text-format MediaPipe graph.
      deps: any calculators or subgraphs used by this graph.
      tflite_deps: any calculators or subgraphs used by this graph that may use different TFLite implementation.
      visibility: The list of packages the subgraph should be visible to.
      testonly: pass 1 if the graph is to be used only for tests.
      **kwargs: Remaining keyword args, forwarded to cc_library.
    """
    graph_base_name = name
    mediapipe_binary_graph(
        name = name + "_graph",
        graph = graph,
        output_name = graph_base_name + ".binarypb",
        deps = deps,
        testonly = testonly,
    )
    data_as_c_string(
        name = name + "_inc",
        srcs = [graph_base_name + ".binarypb"],
        outs = [graph_base_name + ".inc"],
    )

    # cc_library for a linked mediapipe graph.
    expand_template(
        name = name + "_linked_cc",
        template = clean_dep("//mediapipe/framework/tool:simple_subgraph_template.cc"),
        out = name + "_linked.cc",
        substitutions = {
            "{{SUBGRAPH_CLASS_NAME}}": register_as,
            "{{SUBGRAPH_INC_FILE_PATH}}": native.package_name() + "/" + graph_base_name + ".inc",
        },
        testonly = testonly,
    )
    if not tflite_deps:
        native.cc_library(
            name = name,
            srcs = [
                name + "_linked.cc",
                graph_base_name + ".inc",
            ],
            deps = [
                clean_dep("//mediapipe/framework:calculator_framework"),
                clean_dep("//mediapipe/framework:subgraph"),
            ] + deps,
            alwayslink = 1,
            visibility = visibility,
            testonly = testonly,
            **kwargs
        )
    else:
        cc_library_with_tflite(
            name = name,
            srcs = [
                name + "_linked.cc",
                graph_base_name + ".inc",
            ],
            tflite_deps = tflite_deps,
            deps = [
                clean_dep("//mediapipe/framework:calculator_framework"),
                clean_dep("//mediapipe/framework:subgraph"),
            ] + deps,
            alwayslink = 1,
            visibility = visibility,
            testonly = testonly,
            **kwargs
        )

def mediapipe_reexport_library(
        name,
        actual,
        **kwargs):
    """Defines a cc_library that exports the headers of other libraries.

    Normally cc_library does not export the headers of its dependencies,
    and the clang "layering_check" requires clients to depend on them
    directly.  Header files can be exported by listing them in either
    cc_library's "hdrs" or "textual_hdrs" argument.  The "textual_hdrs"
    argument can also accept library targets and has the effect of
    exporting their header files and permitting client references to them.
    The result is a new library target that combines and exports the public
    interfaces of several existing library targets.

    Args:
      name: the name for the combined target.
      actual: the targets to combine and export together.
      **kwargs: Remaining keyword args, forwarded to cc_library.
    """
    native.cc_library(
        name = name,
        textual_hdrs = actual,
        deps = actual,
        **kwargs
    )

def mediapipe_options_library(
        name,
        proto_lib,
        deps = [],
        visibility = None,
        testonly = None,
        **kwargs):
    """Registers options protobuf metadata for defining options packets.

    Args:
      name: name of the options_lib target to define.
      proto_lib: the proto_library target to register.
      deps: any additional protobuf dependencies.
      visibility: The list of packages the subgraph should be visible to.
      testonly: pass 1 if the graph is to be used only for tests.
      **kwargs: Remaining keyword args, forwarded to cc_library.
    """

    transitive_descriptor_set(
        name = proto_lib + "_transitive",
        deps = [proto_lib],
        testonly = testonly,
    )
    direct_descriptor_set(
        name = proto_lib + "_direct",
        deps = [proto_lib],
        testonly = testonly,
    )
    data_as_c_string(
        name = name + "_inc",
        srcs = [proto_lib + "_transitive-transitive-descriptor-set.proto.bin"],
        outs = [proto_lib + "_descriptors.inc"],
    )
    native.genrule(
        name = name + "_type_name",
        srcs = [proto_lib + "_direct-direct-descriptor-set.proto.bin"],
        outs = [name + "_type_name.h"],
        cmd = ("$(location " + "//mediapipe/framework/tool:message_type_util" + ") " +
               ("--input_path=$(location %s) " % (proto_lib + "_direct-direct-descriptor-set.proto.bin")) +
               ("--root_type_macro_output_path=$(location %s) " % (name + "_type_name.h"))),
        tools = ["//mediapipe/framework/tool:message_type_util"],
        visibility = visibility,
        testonly = testonly,
    )
    expand_template(
        name = name + "_cc",
        template = clean_dep("//mediapipe/framework/tool:options_lib_template.cc"),
        out = name + ".cc",
        substitutions = {
            "{{MESSAGE_NAME_HEADER}}": native.package_name() + "/" + name + "_type_name.h",
            "{{MESSAGE_PROTO_HEADER}}": native.package_name() + "/" + proto_lib.replace("_proto", ".pb.h"),
            "{{DESCRIPTOR_INC_FILE_PATH}}": native.package_name() + "/" + proto_lib + "_descriptors.inc",
        },
        testonly = testonly,
    )
    native.cc_library(
        name = proto_lib.replace("_proto", "_options_registry"),
        srcs = [
            name + ".cc",
            proto_lib + "_descriptors.inc",
            name + "_type_name.h",
        ],
        deps = [
            clean_dep("//mediapipe/framework:calculator_framework"),
            clean_dep("//mediapipe/framework/port:advanced_proto"),
            clean_dep("//mediapipe/framework/tool:options_registry"),
            proto_lib.replace("_proto", "_cc_proto"),
        ] + deps,
        alwayslink = 1,
        visibility = visibility,
        testonly = testonly,
        features = ["-no_undefined"],
        **kwargs
    )
    mediapipe_reexport_library(
        name = name,
        actual = [
            proto_lib.replace("_proto", "_cc_proto"),
            proto_lib.replace("_proto", "_options_registry"),
        ],
        visibility = visibility,
        testonly = testonly,
        **kwargs
    )
