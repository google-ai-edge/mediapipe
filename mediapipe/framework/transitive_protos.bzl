"""This rule gathers all .proto files used by all of its dependencies.

The entire dependency tree is searched. The search crosses through cc_library
rules and portable_proto_library rules to collect the transitive set of all
.proto dependencies. This is provided to other rules in the form of a "proto"
provider, using the transitive_sources field.

The basic rule is transitive_protos. Example:

proto_library(
    name = "a_proto_library",
    srcs = ["a.proto],
)

proto_library(
    name = "b_proto_library",
    srcs = ["b.proto],
)

cc_library(
    name = "a_cc_library",
    deps = ["b_proto_library],
)

transitive_protos(
    name = "all_my_protos",
    deps = [
        "a_proto_library",
        "a_cc_library",
    ],
)

all_my_protos will gather all proto files used in its dependency tree; in this
case, ["a.proto", "b.proto"]. These are provided as the default outputs of this
rule, so you can place the rule in any context that requires a list of files,
and also as a "proto" provider, for use by any rules that would normally depend
on proto_library.

The dependency tree is explored using an aspect, transitive_protos_aspect. This
aspect propagates across two attributes, "deps" and "hdrs". The latter is used
for compatibility with portable_proto_library; see comments below and in that
file for more details.

At each visited node in the tree, the aspect collects protos:
- direct_sources from the proto provider in the current node. This is filled in
  by proto_library nodes, and also by piggyback_header nodes (see
  portable_proto_build_defs.bzl).
- protos from the transitive_protos provider in dependency nodes, found from
  both the "deps" and the "hdrs" aspect.
Then it puts all the protos in the protos field of the transitive_protos
provider which it generates. This is how each node sends its gathered protos up
the tree.
"""

load("@com_google_protobuf//bazel/common:proto_info.bzl", "ProtoInfo")

_TransitiveProtoAspectInfo = provider(
    "TransitiveProtoAspectInfo",
    fields = [
        "protos",
        "descriptors",
        "proto_libs",
    ],
)

TransitiveProtoInfo = provider(
    "TransitiveProtoInfo",
    fields = [
        "transitive_descriptor_sets",
        "transitive_sources",
    ],
)

def _gather_transitive_protos_deps(deps, my_protos = [], my_descriptors = [], my_proto_libs = []):
    useful_deps = [dep for dep in deps if _TransitiveProtoAspectInfo in dep]
    protos = depset(
        my_protos,
        transitive = [dep[_TransitiveProtoAspectInfo].protos for dep in useful_deps],
    )
    proto_libs = depset(
        my_proto_libs,
        transitive = [dep[_TransitiveProtoAspectInfo].proto_libs for dep in useful_deps],
    )
    descriptors = depset(
        my_descriptors,
        transitive = [dep[_TransitiveProtoAspectInfo].descriptors for dep in useful_deps],
    )

    return _TransitiveProtoAspectInfo(
        protos = protos,
        descriptors = descriptors,
        proto_libs = proto_libs,
    )

def _transitive_protos_aspect_impl(target, ctx):
    """Implementation of the transitive_protos_aspect aspect.

    Args:
      target: The current target.
      ctx: The current rule context.
    Returns:
      A transitive_protos provider.
    """
    proto_provider = None
    if ProtoInfo in target:
        proto_provider = target[ProtoInfo]
    elif hasattr(target, "proto"):
        proto_provider = target.proto

    if proto_provider != None:
        protos = proto_provider.direct_sources
        if hasattr(proto_provider, "direct_descriptor_set"):
            descriptors = [proto_provider.direct_descriptor_set]
        else:
            descriptors = []
    else:
        protos = []
        descriptors = []

    deps = ctx.rule.attr.deps[:] if hasattr(ctx.rule.attr, "deps") else []

    proto_libs = []
    if ctx.rule.kind == "proto_library":
        proto_libs = [f for f in target.files.to_list() if f.extension == "a"]

    # Searching through the hdrs attribute is necessary because of
    # portable_proto_library. In portable mode, that macro
    # generates a cc_library that does not depend on any proto_libraries, so
    # the .proto files do not appear in its dependency tree.
    # portable_proto_library cannot add arbitrary providers or attributes to
    # a cc_library rule, so instead it piggybacks the provider on a rule that
    # generates a header, which occurs in the hdrs attribute of the cc_library.
    if hasattr(ctx.rule.attr, "hdrs"):
        deps += ctx.rule.attr.hdrs
    result = _gather_transitive_protos_deps(deps, protos, descriptors, proto_libs)
    return result

transitive_protos_aspect = aspect(
    implementation = _transitive_protos_aspect_impl,
    attr_aspects = ["deps", "hdrs"],
    attrs = {},
)

def _transitive_protos_impl(ctx):
    """Implementation of transitive_protos rule.

    Args:
      ctx: The rule context.
    Returns:
      A proto provider (with transitive_sources and transitive_descriptor_sets filled in),
      and marks all transitive sources as default output.
    """
    gathered = _gather_transitive_protos_deps(ctx.attr.deps)
    protos = gathered.protos
    descriptors = gathered.descriptors

    return [
        DefaultInfo(files = protos),
        TransitiveProtoInfo(
            transitive_descriptor_sets = descriptors,
            transitive_sources = protos,
        ),
    ]

transitive_protos = rule(
    implementation = _transitive_protos_impl,
    attrs = {
        "deps": attr.label_list(
            aspects = [transitive_protos_aspect],
        ),
    },
)

def _transitive_proto_cc_libs_impl(ctx):
    """Implementation of transitive_proto_cc_libs rule.

    NOTE: this only works on Bazel, not exobazel.

    Args:
      ctx: The rule context.

    Returns:
      All transitive proto C++ .a files as default output.
    """
    gathered = _gather_transitive_protos_deps(ctx.attr.deps)
    proto_libs = gathered.proto_libs
    return [DefaultInfo(files = proto_libs)]

transitive_proto_cc_libs = rule(
    implementation = _transitive_proto_cc_libs_impl,
    attrs = {
        "deps": attr.label_list(
            aspects = [transitive_protos_aspect],
        ),
    },
)

def _transitive_proto_descriptor_sets_impl(ctx):
    """Implementation of transitive_proto_descriptor_sets rule.

    Args:
      ctx: The rule context.

    Returns:
      All transitive proto descriptor files as default output.
    """
    gathered = _gather_transitive_protos_deps(ctx.attr.deps)
    descriptors = gathered.descriptors
    return [DefaultInfo(files = descriptors)]

transitive_proto_descriptor_sets = rule(
    implementation = _transitive_proto_descriptor_sets_impl,
    attrs = {
        "deps": attr.label_list(
            aspects = [transitive_protos_aspect],
        ),
    },
)
