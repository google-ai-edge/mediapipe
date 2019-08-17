"""Extract a cc_library compatible dependency with only the top level proto rules."""

ProtoLibsInfo = provider(fields = ["targets", "out"])

def _get_proto_rules(deps, proto_rules = None):
    useful_deps = [dep for dep in deps if hasattr(dep, "proto_rules")]
    if proto_rules == None:
        proto_rules = []
    for dep in useful_deps:
        proto_rules = proto_rules + dep.proto_rules
    return proto_rules

def _proto_rules_aspect_impl(target, ctx):
    # Make sure the rule has a srcs attribute.
    proto_rules = []
    found_cc_proto = False
    if hasattr(ctx.rule.attr, "srcs") and len(ctx.rule.attr.srcs) == 1:
        for f in ctx.rule.attr.srcs[0].files.to_list():
            if f.basename.endswith(".pb.cc"):
                proto_rules = [target[CcInfo]]
                found = True
                break

    if not found_cc_proto:
        deps = ctx.rule.attr.deps[:] if hasattr(ctx.rule.attr, "deps") else []
        proto_rules = _get_proto_rules(deps, proto_rules)

    return struct(
        proto_rules = proto_rules,
    )

proto_rules_aspect = aspect(
    implementation = _proto_rules_aspect_impl,
    attr_aspects = ["deps"],
)

def _transitive_protos_impl(ctx):
    """Implementation of transitive_protos rule.

    Args:
      ctx: The rule context.
    Returns:
      A proto provider (with transitive_sources and transitive_descriptor_sets filled in),
      and marks all transitive sources as default output.
    """
    cc_infos = []
    for dep in ctx.attr.deps:
        for dep_proto_rule in dep.proto_rules:
            cc_infos.append(dep_proto_rule)
    return [cc_common.merge_cc_infos(cc_infos = cc_infos)]

transitive_protos = rule(
    implementation = _transitive_protos_impl,
    attrs =
        {
            "deps": attr.label_list(
                aspects = [proto_rules_aspect],
            ),
        },
    provides = [CcInfo],
)
