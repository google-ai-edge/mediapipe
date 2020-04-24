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
