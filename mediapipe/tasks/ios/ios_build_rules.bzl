"""This module contains utility macros for MediaPipe Tasks iOS BUILD files."""

load("@build_bazel_rules_apple//apple:apple.bzl", "apple_static_xcframework")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//mediapipe/framework/tool:ios.bzl",
    "MPP_TASK_MINIMUM_OS_VERSION",
)

def _dummy_objc_library(name):
    """Creates a dummy objc_library with a single dummy source file.

    This macro generates a genrule that creates a dummy Objective-C source file
    and an objc_library that compiles it. This is useful for creating dummy
    libraries to satisfy dependencies in apple_static_xcframework rules.

    Args:
      name: The root name of the dummy library.
    """
    genrule_name = "_%s_dummy_src" % name
    native.genrule(
        name = genrule_name,
        outs = ["%s_dummy.m" % name],
        cmd = "echo 'void mediapipe_tasks_%s_dummy(){}' > $@" % name,
        visibility = ["//visibility:private"],
    )

    objc_library(
        name = "%s_dummy" % name,
        srcs = [":" + genrule_name],
        visibility = ["//visibility:private"],
    )

def mediapipe_static_xcframework(name, **kwargs):
    """An apple_static_xcframework with a dummy library to allow for empty frameworks.

    Args:
      name: The name of the apple_static_xcframework target.
      **kwargs: Arguments passed to apple_static_xcframework.
    """

    kwargs.pop("bundle_format", None)
    kwargs.pop("bundle_id", None)
    kwargs.pop("infoplists", None)

    if "deps" not in kwargs or kwargs.get("bundle_format") == "framework":
        _dummy_objc_library(name = name)
        if "deps" in kwargs:
            kwargs["deps"] = kwargs["deps"] + [":%s_dummy" % name]
        else:
            kwargs["deps"] = [":%s_dummy" % name]

    if "ios" not in kwargs:
        kwargs["ios"] = {
            "simulator": [
                "arm64",
                "x86_64",
            ],
            "device": ["arm64"],
        }
    if "minimum_os_versions" not in kwargs:
        kwargs["minimum_os_versions"] = {
            "ios": MPP_TASK_MINIMUM_OS_VERSION,
        }

    public_hdrs = kwargs.get("public_hdrs", [])

    if not public_hdrs:
        # No issues, just build the xcframework as normal.
        apple_static_xcframework(
            name = name,
            **kwargs
        )
        return

    # WORKAROUND: b/504553290
    # We are unable to build an xcframework with multiple architectures and headers, due to
    # file name conflicts during the build process.
    # So we build a single architecture xcframework with headers and a multi-architecture xcframework
    # without headers and then stitch them together.

    # 1. Build for a single architecture with headers
    kwargs_single = dict(kwargs)
    kwargs_single["ios"] = {"simulator": ["x86_64"]}
    kwargs_single["bundle_name"] = kwargs.get("bundle_name", name) + "_single"
    single_arch_name = name + "_single_arch_with_hdrs"

    apple_static_xcframework(
        name = single_arch_name,
        **kwargs_single
    )

    # 2. Build for multi-architecture without headers
    kwargs_multi = dict(kwargs)
    kwargs_multi.pop("public_hdrs")
    multi_arch_name = name + "_multi_arch_headerless"

    apple_static_xcframework(
        name = multi_arch_name,
        **kwargs_multi
    )

    original_bundle_name = kwargs.get("bundle_name", name)
    single_bundle_name = kwargs_single["bundle_name"]

    stitch_cmd = """
    WORK_DIR=$$(mktemp -d)
    SINGLE_DIR=$$WORK_DIR/single
    MULTI_DIR=$$WORK_DIR/multi

    mkdir -p $$SINGLE_DIR $$MULTI_DIR
    unzip -q $(location :{single_arch_name}) -d $$SINGLE_DIR
    unzip -q $(location :{multi_arch_name}) -d $$MULTI_DIR

    SINGLE_FRAMEWORK=$$(find $$SINGLE_DIR -type d -name "*.framework" | head -n 1)
    MULTI_FRAMEWORKS=$$(find $$MULTI_DIR -type d -name "*.framework")

    for fw in $$MULTI_FRAMEWORKS; do
        if [ -d "$$SINGLE_FRAMEWORK/Headers" ]; then
            mkdir -p "$$fw/Headers"
            cp -R "$$SINGLE_FRAMEWORK/Headers/"* "$$fw/Headers/"
            if [ -f "$$fw/Headers/{single_bundle_name}.h" ]; then
                mv "$$fw/Headers/{single_bundle_name}.h" "$$fw/Headers/{original_bundle_name}.h"
            fi
            for hdr in "$$fw/Headers/"*.h; do
                if [ -f "$$hdr" ]; then
                    sed 's/{single_bundle_name}/{original_bundle_name}/g' "$$hdr" > "$$hdr.tmp"
                    mv "$$hdr.tmp" "$$hdr"
                fi
            done
        fi
        if [ -d "$$SINGLE_FRAMEWORK/Modules" ]; then
            mkdir -p "$$fw/Modules"
            cp -R "$$SINGLE_FRAMEWORK/Modules/"* "$$fw/Modules/"
            if [ -f "$$fw/Modules/module.modulemap" ]; then
                sed 's/{single_bundle_name}/{original_bundle_name}/g' "$$fw/Modules/module.modulemap" > "$$fw/Modules/module.modulemap.tmp"
                mv "$$fw/Modules/module.modulemap.tmp" "$$fw/Modules/module.modulemap"
            fi
        fi
    done

    # Zip exactly from the multi directory so the .xcframework is at the root
    pushd $$MULTI_DIR > /dev/null
    zip -qr output.zip *
    popd > /dev/null
    mv $$MULTI_DIR/output.zip $@
    rm -rf $$WORK_DIR
    """.format(
        single_arch_name = single_arch_name,
        multi_arch_name = multi_arch_name,
        single_bundle_name = single_bundle_name,
        original_bundle_name = original_bundle_name,
    )

    native.genrule(
        name = name,
        srcs = [":" + single_arch_name, ":" + multi_arch_name],
        outs = [name + ".xcframework.zip"],
        cmd = stitch_cmd,
        visibility = kwargs.get("visibility"),
    )
