# Copyright 2026 The MediaPipe Authors.
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

"""Build definitions for MediaPipe Tasks Web."""

load("@aspect_rules_jasmine//jasmine:defs.bzl", "jasmine_test")
load("@aspect_rules_js//npm:defs.bzl", "npm_package")
load("@aspect_rules_rollup//rollup:defs.bzl", "rollup")

def _substitute_impl(ctx):
    """Implementation of substitution rule."""
    out = ctx.actions.declare_file(ctx.attr.out_name)
    ctx.actions.expand_template(
        template = ctx.file.src,
        output = out,
        substitutions = ctx.attr.substitutions,
    )
    return [DefaultInfo(files = depset([out]))]

_substitute = rule(
    implementation = _substitute_impl,
    attrs = {
        "src": attr.label(allow_single_file = True, mandatory = True),
        "out_name": attr.string(mandatory = True),
        "substitutions": attr.string_dict(mandatory = True),
    },
)

def jasmine_node_test(name, srcs, **kwargs):
    """Rule for generating a jasmine node test.

    Args:
        name: The name of the rule.
        srcs: The source files of the test.
        **kwargs: Additional arguments to pass to the jasmine_test rule.
    """
    jasmine_test(
        name = name,
        data = srcs,
        args = ["$(rootpaths %s)" % target for target in srcs],
        node_modules = "//:node_modules",
        **kwargs
    )

def mediapipe_oss_js(name, deps, js_flags = [], visibility = []):
    """Rule for generating JS files that can be used in the OSS pipeline.

    Args:
        name: The name of the rule.
        deps: The dependencies of the rule.
        js_flags: The JS flags to use.
        visibility: The visibility of the rule.
    """
    unused = [js_flags]  # buildifier: disable=unused-variable

    native.genrule(
        name = name + "_config_copy",
        srcs = ["//mediapipe/tasks/web:rollup.config.mjs"],
        outs = [name + "_rollup.config.mjs"],
        cmd = "cp $< $@",
    )

    rollup(
        name,
        "//:node_modules",
        config_file = name + "_config_copy",
        entry_point = "index.js",
        deps = deps,
        visibility = visibility,
    )

def rollup_bundle(name, entry_point, config_file, format, deps = [], **kwargs):
    """Rule for generating a rollup bundle.

    Args:
        name: The name of the rule.
        entry_point: The entry point of the rollup bundle.
        config_file: The config file of the rollup bundle.
        format: The format of the rollup bundle.
        deps: The dependencies of the rollup bundle.
        **kwargs: Additional arguments to pass to the rollup rule.
    """
    kwargs.pop("output_dir", None)

    native.genrule(
        name = name + "_config_copy",
        srcs = [config_file],
        outs = [name + "_rollup.config.mjs"],
        cmd = "cp $< $@",
    )

    rollup(
        name,
        "//:node_modules",
        config_file = name + "_config_copy",
        entry_point = entry_point.replace(".ts", ".js"),
        format = format,
        deps = deps,
        **kwargs
    )

def pkg_npm(name, package_json = None, srcs = [], substitutions = {}, deps = [], **kwargs):
    """Rule for generating an npm package.

    Args:
        name: The name of the rule.
        package_json: The package.json file of the npm package.
        srcs: The source files of the npm package.
        substitutions: The substitutions to make in the package.json file.
        deps: The dependencies of the npm package.
        **kwargs: Additional arguments to pass to the npm_package rule.
    """

    # Handle package.json substitution
    if package_json:
        if substitutions:
            _substitute(
                name = name + "_package_json",
                src = package_json,
                out_name = "package.json",
                substitutions = substitutions,
            )
            srcs.append(":" + name + "_package_json")
        else:
            srcs.append(package_json)

    npm_package(
        name = name,
        srcs = srcs + deps,
        **kwargs
    )
