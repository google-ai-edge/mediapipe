"""This module contains utility macros for MediaPipe Tasks iOS BUILD files."""

load(
    "@build_bazel_rules_apple//apple:apple.bzl",
    "apple_static_xcframework",
    "apple_xcframework",
)
load("@rules_cc//cc:find_cc_toolchain.bzl", "CC_TOOLCHAIN_ATTRS", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(
    "//mediapipe:version.bzl",
    "MEDIAPIPE_FULL_VERSION",
)
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

def _framework_infoplist(name):
    """Creates a genrule that generates a valid Info.plist for a framework."""
    plist_target = "_%s_infoplist" % name
    plist_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\\n' +
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" ' +
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\\n' +
        '<plist version="1.0">\\n' +
        "<dict>\\n" +
        "  <key>CFBundleInfoDictionaryVersion</key>\\n" +
        "  <string>6.0</string>\\n" +
        "  <key>CFBundlePackageType</key>\\n" +
        "  <string>FMWK</string>\\n" +
        "  <key>CFBundleShortVersionString</key>\\n" +
        "  <string>%s</string>\\n" % MEDIAPIPE_FULL_VERSION +
        "  <key>CFBundleVersion</key>\\n" +
        "  <string>%s</string>\\n" % MEDIAPIPE_FULL_VERSION +
        "</dict>\\n" +
        "</plist>"
    )
    native.genrule(
        name = plist_target,
        outs = ["%s_Info.plist" % name],
        cmd = "printf '%s\\n' > $@" % plist_content,
        visibility = ["//visibility:private"],
    )
    return ":" + plist_target

def _strip_rust_metadata(name, src, visibility):
    fail("strip_rust_metadata is not supported in the OSS build.")

_LLVM_TOOLS = []
_LLVM_TOOLS_SETUP_SH = 'NM_BIN="$$(xcrun -f llvm-nm 2>/dev/null || which llvm-nm)"; OBJCOPY_BIN="$$(xcrun -f llvm-objcopy 2>/dev/null || which llvm-objcopy)"'

def mediapipe_symbol_rename_map(
        name,
        srcs,
        visibility = ["//visibility:private"]):
    """Generates an llvm-objcopy --redefine-syms map prefixing non-MPP symbols.

    Args:
      name: The name of the genrule target.
      srcs: List of raw .xcframework.zip targets to extract global symbols from.
      visibility: Target visibility.
    """
    srcs_locations = " ".join(["$(execpath %s)" % s for s in srcs])
    cmd = """
    WORK_DIR=$$(mktemp -d)
    """ + _LLVM_TOOLS_SETUP_SH + """
    idx=0
    for z in {srcs_locations}; do
      idx=$$((idx + 1))
      mkdir -p "$$WORK_DIR/in_$$idx"
      unzip -q "$$z" -d "$$WORK_DIR/in_$$idx"
    done
    : > "$$WORK_DIR/raw_syms.txt"
    for archive in $$(find "$$WORK_DIR" -path "$$WORK_DIR/in_*" -type f \\
        \\( -name "*.a" -o -path "*.framework/*" \\) \\
        ! -path "*.swiftmodule/*" ! -name "*.swift*" ! -name "*.abi.json" \\
        ! -name "*.plist" ! -name "*.h" ! -name "*.modulemap" ! -name "*.xcprivacy"); do
      "$$NM_BIN" --arch=arm64 --defined-only -g "$$archive" \\
        2>/dev/null >> "$$WORK_DIR/raw_syms.txt" || true
    done
    awk 'NF==3 {{print $$3}}' "$$WORK_DIR/raw_syms.txt" \\
      | LC_ALL=C sort -u \\
      | awk '
        $$0 ~ /^_\\$$[sS]/ {{ next }}
        $$0 ~ /^___?swift/ {{ next }}
        $$0 ~ /^_?MPP/ {{ next }}
        $$0 ~ /^_Mp[A-Z]/ {{ next }}
        $$0 ~ /^_OBJC_(CLASS|METACLASS|IVAR)_\\$$_MPP/ {{ next }}
        $$0 ~ /^__?OBJC_(PROTOCOL|LABEL_PROTOCOL)_\\$$_/ {{ next }}
        $$0 ~ /^__Z(N|NK|NO|NKR|TV|TI|TS|TT|Thn|Tv|GV|Z)?N?St/ {{ next }}
        $$0 ~ /^___(cxa|gxx)_/ {{ next }}
        $$0 ~ /^_OBJC_(CLASS|METACLASS|IVAR)_\\$$_/ {{
          r = $$0
          sub(/^_OBJC_(CLASS|METACLASS|IVAR)_\\$$_/, "&MPP_", r)
          print $$0 " " r
          next
        }}
        $$0 ~ /^_/ {{ print $$0 " _MPP" $$0; next }}
        {{ print $$0 " MPP_" $$0 }}
      ' > $@
    rm -rf "$$WORK_DIR"
    """.format(srcs_locations = srcs_locations)

    native.genrule(
        name = name,
        srcs = srcs,
        outs = [name + ".txt"],
        cmd = cmd,
        tools = _LLVM_TOOLS,
        visibility = visibility,
    )

def mediapipe_static_xcframework(
        name,
        strip_rust_metadata = False,
        symbol_rename_map = None,
        **kwargs):
    """An apple_static_xcframework with a dummy library for empty frameworks.

    Args:
      name: The name of the apple_static_xcframework target.
      strip_rust_metadata: Whether to run `llvm-strip -S` on the binaries in the
        generated xcframework to remove Rust crate metadata. rules_rust does not strip
        it for Apple targets, and Apple's linker ignores it.
      symbol_rename_map: Optional label of a mediapipe_symbol_rename_map target
        used to prefix internal global symbols with MPP_ via llvm-objcopy.
      **kwargs: Arguments passed to apple_static_xcframework.
    """

    strip_rust_metadata = False

    # When stripping, the xcframework is built under an internal name and the final
    # target (with the original name and output file name) is the stripped copy.
    target_name = name + "_with_rust_metadata" if strip_rust_metadata else name
    final_visibility = kwargs.get("visibility")
    if strip_rust_metadata or symbol_rename_map:
        kwargs.setdefault("bundle_name", name)
        kwargs["visibility"] = ["//visibility:private"]

    if kwargs.get("bundle_format") == "framework":
        kwargs["infoplists"] = [_framework_infoplist(name)]

    kwargs.pop("bundle_format", None)
    kwargs.pop("bundle_id", None)
    kwargs.pop("infoplists", None)
    symbol_rename_map = None

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
            ],
            "device": ["arm64"],
        }
    if "minimum_os_versions" not in kwargs:
        kwargs["minimum_os_versions"] = {
            "ios": MPP_TASK_MINIMUM_OS_VERSION,
        }

    if not symbol_rename_map:
        apple_static_xcframework(
            name = target_name,
            **kwargs
        )
        if strip_rust_metadata:
            _strip_rust_metadata(name, ":" + target_name, final_visibility)
        return

    if kwargs.get("public_hdrs"):
        fail("symbol_rename_map is not supported together with public_hdrs")

    raw_name = name + "_raw"
    apple_static_xcframework(
        name = raw_name,
        **kwargs
    )
    rename_cmd = """
    WORK_DIR=$$(mktemp -d)
    OUT_DIR=$$WORK_DIR/out
    RENAME_MAP="$$PWD/$(execpath {symbol_rename_map})"
    """.format(symbol_rename_map = symbol_rename_map) + _LLVM_TOOLS_SETUP_SH + """
    mkdir -p "$$OUT_DIR"
    unzip -q $(execpath :{raw_name}) -d "$$OUT_DIR"
    for archive in $$(find "$$OUT_DIR" -type f \\
        \\( -name "*.a" -o -path "*.framework/*" \\) \\
        ! -path "*.swiftmodule/*" ! -name "*.swift*" ! -name "*.abi.json" \\
        ! -name "*.plist" ! -name "*.h" ! -name "*.modulemap" ! -name "*.xcprivacy"); do
      "$$OBJCOPY_BIN" --redefine-syms="$$RENAME_MAP" \\
        "$$archive" "$$archive.renamed"
      mv "$$archive.renamed" "$$archive"
    done
    pushd "$$OUT_DIR" > /dev/null
    zip -qr output.zip *
    popd > /dev/null
    mv "$$OUT_DIR/output.zip" $@
    rm -rf "$$WORK_DIR"
    """.format(raw_name = raw_name)

    native.genrule(
        name = target_name,
        srcs = [":" + raw_name, symbol_rename_map],
        outs = [target_name + ".xcframework.zip"],
        cmd = rename_cmd,
        tools = _LLVM_TOOLS,
        visibility = ["//visibility:private"] if strip_rust_metadata else final_visibility,
    )
    if strip_rust_metadata:
        _strip_rust_metadata(name, ":" + target_name, final_visibility)

def _mediapipe_alwayslink_libs_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    direct_owners = {dep.label: True for dep in ctx.attr.deps}
    new_linker_inputs = []
    seen = {}
    comp_contexts = []
    for dep in ctx.attr.deps:
        if CcInfo not in dep:
            continue
        cc_info = dep[CcInfo]
        comp_contexts.append(cc_info.compilation_context)
        if not cc_info.linking_context:
            continue
        for linker_input in cc_info.linking_context.linker_inputs.to_list():
            owner_key = linker_input.owner or linker_input
            if owner_key in seen:
                continue
            seen[owner_key] = True
            if linker_input.owner not in direct_owners:
                new_linker_inputs.append(linker_input)
                continue
            needs_wrap = any([
                not lib.alwayslink and (lib.static_library or lib.pic_static_library)
                for lib in linker_input.libraries
            ])
            if not needs_wrap:
                new_linker_inputs.append(linker_input)
                continue
            new_libraries = []
            for lib in linker_input.libraries:
                if lib.alwayslink or not (lib.static_library or lib.pic_static_library):
                    new_libraries.append(lib)
                    continue
                new_libraries.append(cc_common.create_library_to_link(
                    actions = ctx.actions,
                    feature_configuration = feature_configuration,
                    cc_toolchain = cc_toolchain,
                    static_library = lib.static_library,
                    pic_static_library = lib.pic_static_library,
                    alwayslink = True,
                ))
            add_inputs = linker_input.additional_inputs
            if type(add_inputs) != "depset":
                add_inputs = depset(add_inputs or [])
            new_linker_inputs.append(cc_common.create_linker_input(
                owner = linker_input.owner,
                libraries = depset(new_libraries, order = "topological"),
                user_link_flags = linker_input.user_link_flags,
                additional_inputs = add_inputs,
            ))

    comp_ctx = cc_common.merge_compilation_contexts(
        compilation_contexts = comp_contexts,
    ) if comp_contexts else cc_common.create_compilation_context()
    return [
        CcInfo(
            compilation_context = comp_ctx,
            linking_context = cc_common.create_linking_context(
                linker_inputs = depset(new_linker_inputs, order = "topological"),
            ),
        ),
    ]

mediapipe_alwayslink_libs = rule(
    implementation = _mediapipe_alwayslink_libs_impl,
    attrs = dict(
        CC_TOOLCHAIN_ATTRS,
        deps = attr.label_list(),
    ),
    doc = "Marks direct C/C++ library deps alwayslink=True without force-loading transitive third-party archives.",
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
)

def mediapipe_dynamic_xcframework(name, **kwargs):
    """An apple_xcframework configured with MediaPipe iOS defaults.

    Args:
      name: The name of the apple_xcframework target.
      **kwargs: Arguments passed to apple_xcframework.
    """
    kwargs.setdefault("bundle_name", name)
    kwargs["infoplists"] = [_framework_infoplist(name)]

    if "ios" not in kwargs:
        kwargs["ios"] = {
            "simulator": [
                "arm64",
            ],
            "device": ["arm64"],
        }
    if "minimum_os_versions" not in kwargs:
        kwargs["minimum_os_versions"] = {
            "ios": MPP_TASK_MINIMUM_OS_VERSION,
        }

    apple_xcframework(
        name = name,
        **kwargs
    )
