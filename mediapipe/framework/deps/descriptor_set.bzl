"""Outputs a FileDescriptorSet with all transitive dependencies.
   Copied from tools/build_defs/proto/descriptor_set.bzl.
"""

TransitiveDescriptorInfo = provider(
    "The transitive descriptors from a set of protos.",
    fields = ["descriptors"],
)

DirectDescriptorInfo = provider(
    "The direct descriptors from a set of protos.",
    fields = ["descriptors"],
)

def calculate_transitive_descriptor_set(actions, deps, output):
    """Calculates the transitive dependencies of the deps.

    Args:
      actions: the actions (typically ctx.actions) used to run commands
      deps: the deps to get the transitive dependencies of
      output: the output file the data will be written to

    Returns:
      The same output file passed as the input arg, for convenience.
    """

    # Join all proto descriptors in a single file.
    transitive_descriptor_sets = depset(transitive = [
        dep[ProtoInfo].transitive_descriptor_sets if ProtoInfo in dep else dep[TransitiveDescriptorInfo].descriptors
        for dep in deps
    ])
    args = actions.args()
    args.use_param_file(param_file_arg = "--arg-file=%s")
    args.add_all(transitive_descriptor_sets)

    # Because `xargs` must take its arguments before the command to execute,
    # we cannot simply put a reference to the argument list at the end, as in the
    # case of param file spooling, since the entire argument list will get
    # replaced by "--arg-file=bazel-out/..." which needs to be an `xargs`
    # argument rather than a `cat` argument.
    #
    # We look to see if the first argument begins with a '--arg-file=' and
    # selectively choose xargs vs. just supplying the arguments to `cat`.
    actions.run_shell(
        outputs = [output],
        inputs = transitive_descriptor_sets,
        progress_message = "Joining descriptors.",
        command = ("if [[ \"$1\" =~ ^--arg-file=.* ]]; then xargs \"$1\" cat; " +
                   "else cat \"$@\"; fi >{output}".format(output = output.path)),
        arguments = [args],
    )
    return output

def _transitive_descriptor_set_impl(ctx):
    """Combine descriptors for all transitive proto dependencies into one file.

    Warning: Concatenating all of the descriptor files with a single `cat` command
    could exceed system limits (1MB+). For example, a dependency on gwslog.proto
    will trigger this edge case.

    When writing new code, prefer to accept a list of descriptor files instead of
    just one so that this limitation won't impact you.
    """
    output = ctx.actions.declare_file(ctx.attr.name + "-transitive-descriptor-set.proto.bin")
    calculate_transitive_descriptor_set(ctx.actions, ctx.attr.deps, output)
    return DefaultInfo(
        files = depset([output]),
        runfiles = ctx.runfiles(files = [output]),
    )

# transitive_descriptor_set outputs a single file containing a binary
# FileDescriptorSet with all transitive dependencies of the given proto
# dependencies.
#
# Example usage:
#
#     transitive_descriptor_set(
#         name = "my_descriptors",
#         deps = [":my_proto"],
#     )
transitive_descriptor_set = rule(
    attrs = {
        "deps": attr.label_list(providers = [[ProtoInfo], [TransitiveDescriptorInfo]]),
    },
    outputs = {
        "out": "%{name}-transitive-descriptor-set.proto.bin",
    },
    implementation = _transitive_descriptor_set_impl,
)

def calculate_direct_descriptor_set(actions, deps, output):
    """Calculates the direct dependencies of the deps.

    Args:
      actions: the actions (typically ctx.actions) used to run commands
      deps: the deps to get the direct dependencies of
      output: the output file the data will be written to

    Returns:
      The same output file passed as the input arg, for convenience.
    """
    descriptor_set = depset(
        [dep[ProtoInfo].direct_descriptor_set for dep in deps if ProtoInfo in dep],
        transitive = [dep[DirectDescriptorInfo].descriptors for dep in deps if ProtoInfo not in dep],
    )

    actions.run_shell(
        outputs = [output],
        inputs = descriptor_set,
        progress_message = "Joining direct descriptors.",
        command = ("cat %s > %s") % (
            " ".join([d.path for d in descriptor_set.to_list()]),
            output.path,
        ),
    )
    return output

def _direct_descriptor_set_impl(ctx):
    calculate_direct_descriptor_set(ctx.actions, ctx.attr.deps, ctx.outputs.out)

# direct_descriptor_set outputs a single file containing a binary
# FileDescriptorSet with all direct, non transitive dependencies of
# the given proto dependencies.
#
# Example usage:
#
#     direct_descriptor_set(
#         name = "my_direct_descriptors",
#         deps = [":my_proto"],
#     )
direct_descriptor_set = rule(
    attrs = {
        "deps": attr.label_list(providers = [[ProtoInfo], [DirectDescriptorInfo]]),
    },
    outputs = {
        "out": "%{name}-direct-descriptor-set.proto.bin",
    },
    implementation = _direct_descriptor_set_impl,
)
