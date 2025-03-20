# Copyright 2019-2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A rule for encoding a text format protocol buffer into binary.

Example usage:

    proto_library(
        name = "calculator_proto",
        srcs = ["calculator.proto"],
    )

    encode_binary_proto(
        name = "foo_binary",
        deps = [":calculator_proto"],
        message_type = "mediapipe.CalculatorGraphConfig",
        input = "foo.pbtxt",
    )

Args:
  name: The name of this target.
  deps: A list of proto_library targets that define messages that may be
        used in the input file.
  input: The text format protocol buffer.
  message_type: The root message of the buffer.
  output: The desired name of the output file. Optional.
"""

# buildifier: disable=out-of-order-load

load("@bazel_skylib//lib:paths.bzl", "paths")

PROTOC = "@com_google_protobuf//:protoc"

def _canonicalize_proto_path_oss(f):
    if not f.root.path:
        return struct(
            proto_path = ".",
            file_name = f.short_path,
        )

    # `f.path` looks like "<genfiles>/external/<repo>/(_virtual_imports/<library>/)?<file_name>"
    repo_name, _, file_name = f.path[len(paths.join(f.root.path, "external") + "/"):].partition("/")
    if file_name.startswith("_virtual_imports/"):
        # This is a virtual import; move "_virtual_imports/<library>" from `repo_name` to `file_name`.
        repo_name = paths.join(repo_name, *file_name.split("/", 2)[:2])
        file_name = file_name.split("/", 2)[-1]
    return struct(
        proto_path = paths.join(f.root.path, "external", repo_name),
        file_name = file_name,
    )

def _map_root_path(f):
    return _canonicalize_proto_path_oss(f).proto_path

def _map_short_path(f):
    return _canonicalize_proto_path_oss(f).file_name

def _get_proto_provider(dep):
    """Get the provider for protocol buffers from a dependnecy.

    Necessary because Bazel does not provide the .proto. provider but ProtoInfo
    cannot be created from Starlark at the moment.

    Returns:
      The provider containing information about protocol buffers.
    """
    if ProtoInfo in dep:
        return dep[ProtoInfo]

    elif hasattr(dep, "proto"):
        return dep.proto
    else:
        fail("cannot happen, rule definition requires .proto" +
             " or ProtoInfo")

def _encode_binary_proto_impl(ctx):
    """Implementation of the encode_binary_proto rule."""
    all_protos = depset(
        direct = [],
        transitive = [_get_proto_provider(dep).transitive_sources for dep in ctx.attr.deps],
    )

    textpb = ctx.file.input
    binarypb = ctx.outputs.output or ctx.actions.declare_file(
        textpb.basename.rsplit(".", 1)[0] + ".binarypb",
        sibling = textpb,
    )

    args = ctx.actions.args()
    args.add(textpb)
    args.add(binarypb)
    args.add(ctx.executable._proto_compiler)
    args.add(ctx.attr.message_type, format = "--encode=%s")
    args.add("--proto_path=.")
    args.add_all(
        all_protos,
        map_each = _map_root_path,
        format_each = "--proto_path=%s",
        uniquify = True,
    )
    args.add_all(
        all_protos,
        map_each = _map_short_path,
        uniquify = True,
    )

    # Note: the combination of absolute_paths and proto_path, as well as the exact
    # order of gendir before ., is needed for the proto compiler to resolve
    # import statements that reference proto files produced by a genrule.
    ctx.actions.run_shell(
        tools = depset(
            direct = [textpb, ctx.executable._proto_compiler],
            transitive = [all_protos],
        ),
        outputs = [binarypb],
        command = "${@:3} < $1 > $2",
        arguments = [args],
        mnemonic = "EncodeProto",
        toolchain = None,
    )

    output_depset = depset([binarypb])
    return [DefaultInfo(
        files = output_depset,
        data_runfiles = ctx.runfiles(transitive_files = output_depset),
    )]

_encode_binary_proto = rule(
    implementation = _encode_binary_proto_impl,
    attrs = {
        "_proto_compiler": attr.label(
            executable = True,
            default = Label(PROTOC),
            cfg = "exec",
        ),
        "deps": attr.label_list(
            providers = [
                [ProtoInfo],
                ["proto"],
            ],
        ),
        "input": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "message_type": attr.string(
            mandatory = True,
        ),
        "output": attr.output(),
    },
)

def encode_binary_proto(name, input, message_type, deps, **kwargs):
    if type(input) == type("string"):
        input_label = input
        srcs = [input]
    elif type(input) == type(dict()):
        # We cannot accept a select, as macros are unable to manipulate selects.
        input_label = select(input)
        srcs_dict = dict()
        for k, v in input.items():
            srcs_dict[k] = [v]
        srcs = select(srcs_dict)
    else:
        fail("input should be a string or a dict, got %s" % input)

    _encode_binary_proto(
        name = name,
        input = input_label,
        message_type = message_type,
        deps = deps,
        **kwargs
    )

def _generate_proto_descriptor_set_impl(ctx):
    """Implementation of the generate_proto_descriptor_set rule."""
    all_protos = depset(transitive = [
        _get_proto_provider(dep).transitive_sources
        for dep in ctx.attr.deps
        if (
            ProtoInfo in dep or
            hasattr(dep, "proto")
        )
    ])
    descriptor = ctx.outputs.output

    # Note: the combination of absolute_paths and proto_path, as well as the exact
    # order of gendir before ., is needed for the proto compiler to resolve
    # import statements that reference proto files produced by a genrule.
    ctx.actions.run(
        inputs = all_protos,
        tools = [ctx.executable._proto_compiler],
        outputs = [descriptor],
        executable = ctx.executable._proto_compiler,
        arguments = [
                        "--descriptor_set_out=%s" % descriptor.path,
                        "--proto_path=" + ctx.genfiles_dir.path,
                        "--proto_path=" + ctx.bin_dir.path,
                        "--proto_path=.",
                    ] +
                    [s.path for s in all_protos.to_list()],
        mnemonic = "GenerateProtoDescriptor",
    )

generate_proto_descriptor_set = rule(
    implementation = _generate_proto_descriptor_set_impl,
    attrs = {
        "_proto_compiler": attr.label(
            executable = True,
            default = Label(PROTOC),
            cfg = "exec",
        ),
        "deps": attr.label_list(
            providers = [
                [ProtoInfo],
                ["proto"],
            ],
        ),
    },
    outputs = {"output": "%{name}.proto.bin"},
)
