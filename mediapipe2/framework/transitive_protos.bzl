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

"""Extract a cc_library compatible dependency with only the top level proto rules."""

ProtoLibsInfo = provider(fields = ["targets", "out"])

def _get_proto_rules(deps, proto_rules = None):
    useful_deps = [dep for dep in deps if hasattr(dep, "proto_rules")]
    if proto_rules == None:
        proto_rules = depset()
    proto_rules = depset(
        transitive = [proto_rules] + [dep.proto_rules for dep in useful_deps],
    )
    return proto_rules

def _proto_rules_aspect_impl(target, ctx):
    # Make sure the rule has a srcs attribute.
    proto_rules = depset()
    found_cc_proto = False
    if hasattr(ctx.rule.attr, "srcs") and len(ctx.rule.attr.srcs) == 1:
        for f in ctx.rule.attr.srcs[0].files.to_list():
            if f.basename.endswith(".pb.cc"):
                proto_rules = depset([target[CcInfo]])
                found_cc_proto = True
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
    cc_info_sets = []
    for dep in ctx.attr.deps:
        cc_info_sets.append(dep.proto_rules)
    cc_infos = depset(transitive = cc_info_sets).to_list()
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
