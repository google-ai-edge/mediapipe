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

PROTOC = "@com_google_protobuf//:protoc"

def _canonicalize_proto_path_oss(all_protos, genfile_path):
    """For the protos from external repository, canonicalize the proto path and the file name.

    Returns:
       Proto path list and proto source file list.
    """
    proto_paths = []
    proto_file_names = []
    for s in all_protos.to_list():
        if s.path.startswith(genfile_path):
            repo_name, _, file_name = s.path[len(genfile_path + "/external/"):].partition("/")

            # handle virtual imports
            if file_name.startswith("_virtual_imports"):
                repo_name = repo_name + "/" + "/".join(file_name.split("/", 2)[:2])
                file_name = file_name.split("/", 2)[-1]
            proto_paths.append(genfile_path + "/external/" + repo_name)
            proto_file_names.append(file_name)
        else:
            proto_file_names.append(s.path)
    return ([" --proto_path=" + path for path in proto_paths], proto_file_names)

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
        fail("cannot happen, rule definition requires .proto or ProtoInfo")

def _encode_binary_proto_impl(ctx):
    """Implementation of the encode_binary_proto rule."""
    all_protos = depset()
    for dep in ctx.attr.deps:
        provider = _get_proto_provider(dep)
        all_protos = depset(
            direct = [],
            transitive = [all_protos, provider.transitive_sources],
        )

    textpb = ctx.file.input
    binarypb = ctx.outputs.output or ctx.actions.declare_file(
        textpb.basename.rsplit(".", 1)[0] + ".binarypb",
        sibling = textpb,
    )

    path_list, file_list = _canonicalize_proto_path_oss(all_protos, ctx.genfiles_dir.path)

    # Note: the combination of absolute_paths and proto_path, as well as the exact
    # order of gendir before ., is needed for the proto compiler to resolve
    # import statements that reference proto files produced by a genrule.
    ctx.actions.run_shell(
        tools = all_protos.to_list() + [textpb, ctx.executable._proto_compiler],
        outputs = [binarypb],
        command = " ".join(
            [
                ctx.executable._proto_compiler.path,
                "--encode=" + ctx.attr.message_type,
                "--proto_path=" + ctx.genfiles_dir.path,
                "--proto_path=" + ctx.bin_dir.path,
                "--proto_path=.",
            ] + path_list + file_list +
            ["<", textpb.path, ">", binarypb.path],
        ),
        mnemonic = "EncodeProto",
    )

    output_depset = depset([binarypb])
    return [DefaultInfo(
        files = output_depset,
        data_runfiles = ctx.runfiles(transitive_files = output_depset),
    )]

encode_binary_proto = rule(
    implementation = _encode_binary_proto_impl,
    attrs = {
        "_proto_compiler": attr.label(
            executable = True,
            default = Label(PROTOC),
            cfg = "host",
        ),
        "deps": attr.label_list(
            providers = [[ProtoInfo], ["proto"]],
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

def _generate_proto_descriptor_set_impl(ctx):
    """Implementation of the generate_proto_descriptor_set rule."""
    all_protos = depset(transitive = [
        _get_proto_provider(dep).transitive_sources
        for dep in ctx.attr.deps
        if ProtoInfo in dep or hasattr(dep, "proto")
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
            cfg = "host",
        ),
        "deps": attr.label_list(
            providers = [[ProtoInfo], ["proto"]],
        ),
    },
    outputs = {"output": "%{name}.proto.bin"},
)
