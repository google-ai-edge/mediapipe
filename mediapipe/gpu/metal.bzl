"""Experimental Skylark rules for Apple's Metal.

This creates a .metallib file containing compiled Metal shaders.
Note that the default behavior in Xcode is to put all metal shaders into a
single "default.metallib", which can be loaded using the method
newDefaultLibrary in MTLDevice. Meanwhile, the metal_library rule creates a
named .metallib, which can be loaded using newLibraryWithFile:error:.

Example:

  metal_library(
      name = "my_shaders",
      srcs = ["my_shaders.metal"],
      hdrs = ["my_shaders.h"],
  )

This produces a "my_shaders.metallib".

The metal_library target can be added to the deps attribute of an objc_library.
The dependent objc_library can then access the headers declared by the
metal_library, if any.

The metal_library target can also be added to the resources attribute as a
simple data file, but in that case any declared headers are not visible to
dependent objc_library rules.
"""

load("@bazel_skylib//lib:dicts.bzl", "dicts")
load("@build_bazel_apple_support//lib:apple_support.bzl", "apple_support")

# This load statement is overriding the visibility of the internal implementation of rules_apple.
# This rule will be migrated to rules_apple in the future, hence the override. Please do not use
# this import anywhere else.
load(
    "@build_bazel_rules_apple//apple/internal:resources.bzl",
    "resources",
)

def _metal_compiler_args(ctx, src, obj, minimum_os_version, copts, diagnostics, deps_dump):
    """Returns arguments for metal compiler."""
    apple_fragment = ctx.fragments.apple

    platform = apple_fragment.single_arch_platform

    if not minimum_os_version:
        minimum_os_version = ctx.attr._xcode_config[apple_common.XcodeVersionConfig].minimum_os_for_platform_type(
            platform.platform_type,
        )

    args = copts + [
        "-arch",
        "air64",  # TODO: choose based on target device/cpu/platform?
        "-emit-llvm",
        "-c",
        "-gline-tables-only",
        "-isysroot",
        apple_support.path_placeholders.sdkroot(),
        "-ffast-math",
        "-serialize-diagnostics",
        diagnostics.path,
        "-o",
        obj.path,
        "-mios-version-min=%s" % minimum_os_version,
        "",
        src.path,
        "-MMD",
        "-MT",
        "dependencies",
        "-MF",
        deps_dump.path,
    ]
    return args

def _metal_compiler_inputs(srcs, hdrs, deps = []):
    """Determines the list of inputs required for a compile action."""

    cc_infos = [dep[CcInfo] for dep in deps if CcInfo in dep]

    dep_headers = depset(transitive = [
        cc_info.compilation_context.headers
        for cc_info in cc_infos
    ])

    return depset(srcs + hdrs, transitive = [dep_headers])

def _metal_library_impl(ctx):
    """Implementation for metal_library Skylark rule."""

    # A unique path for rule's outputs.
    objs_outputs_path = "{}.objs/".format(ctx.label.name)

    output_objs = []
    for src in ctx.files.srcs:
        basename = src.basename
        obj = ctx.actions.declare_file(objs_outputs_path + basename + ".air")
        output_objs.append(obj)
        diagnostics = ctx.actions.declare_file(objs_outputs_path + basename + ".dia")
        deps_dump = ctx.actions.declare_file(objs_outputs_path + basename + ".dat")

        args = (["metal"] +
                _metal_compiler_args(ctx, src, obj, ctx.attr.minimum_os_version, ctx.attr.copts, diagnostics, deps_dump))

        apple_support.run(
            actions = ctx.actions,
            xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
            apple_fragment = ctx.fragments.apple,
            xcode_path_resolve_level = apple_support.xcode_path_resolve_level.args,
            inputs = _metal_compiler_inputs(ctx.files.srcs, ctx.files.hdrs, ctx.attr.deps),
            outputs = [obj, diagnostics, deps_dump],
            mnemonic = "MetalCompile",
            executable = "/usr/bin/xcrun",
            toolchain = None,
            arguments = args,
            use_default_shell_env = False,
            progress_message = ("Compiling Metal shader %s" %
                                (basename)),
        )

    output_lib = ctx.actions.declare_file(ctx.label.name + ".metallib")
    args = [
        "metallib",
        "-split-module",
        "-o",
        output_lib.path,
    ] + [x.path for x in output_objs]

    apple_support.run(
        actions = ctx.actions,
        xcode_config = ctx.attr._xcode_config[apple_common.XcodeVersionConfig],
        apple_fragment = ctx.fragments.apple,
        xcode_path_resolve_level = apple_support.xcode_path_resolve_level.args,
        inputs = output_objs,
        outputs = (output_lib,),
        mnemonic = "MetalLink",
        executable = "/usr/bin/xcrun",
        toolchain = None,
        arguments = args,
        progress_message = (
            "Linking Metal library %s" % ctx.label.name
        ),
    )

    objc_provider = apple_common.new_objc_provider(
        providers = [x[apple_common.Objc] for x in ctx.attr.deps if apple_common.Objc in x],
    )

    cc_infos = [dep[CcInfo] for dep in ctx.attr.deps if CcInfo in dep]
    if ctx.files.hdrs:
        cc_infos.append(
            CcInfo(
                compilation_context = cc_common.create_compilation_context(
                    headers = depset([f for f in ctx.files.hdrs]),
                ),
            ),
        )

    return [
        DefaultInfo(
            files = depset([output_lib]),
        ),
        objc_provider,
        cc_common.merge_cc_infos(cc_infos = cc_infos),
        # Return the provider for the new bundling logic of rules_apple.
        resources.bucketize_typed(
            bucket_type = "unprocessed",
            expect_files = True,
            resources = [output_lib],
        ),
    ]

METAL_LIBRARY_ATTRS = dicts.add(apple_support.action_required_attrs(), {
    "srcs": attr.label_list(allow_files = [".metal"], allow_empty = False),
    "hdrs": attr.label_list(allow_files = [".h"]),
    "deps": attr.label_list(providers = [["objc", CcInfo], [apple_common.Objc, CcInfo]]),
    "copts": attr.string_list(),
    "minimum_os_version": attr.string(),
})

metal_library = rule(
    implementation = _metal_library_impl,
    attrs = METAL_LIBRARY_ATTRS,
    fragments = ["apple", "objc"],
)
"""
Builds a Metal library.

Args:
  srcs: Metal shader sources.
  hdrs: Header files used by the shader sources.
  deps: objc_library targets whose headers should be visible to the shaders.

The header files declared in this rule are also visible to any objc_library
rules that have it as a dependency, so that constants and typedefs can be
shared between Metal and Objective-C code.
"""
