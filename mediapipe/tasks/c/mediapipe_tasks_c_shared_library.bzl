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
"""Macro for building a self-contained MediaPipe Tasks C API shared library.

Each MediaPipe Tasks domain (vision, text, audio, decision, retrieval) is built
as its own shared library that statically links everything it needs (MediaPipe
framework, TFLite, ...). Only the `Mp*` C symbols listed in the domain's version
script are exported; all other symbols are kept local so that multiple domain
libraries can be loaded into the same process without clashing.

The version script (GNU ld format) is the single source of truth for the
exported symbol set. On Apple platforms it is converted into an
`-exported_symbols_list` file at build time.

For every domain the macro produces:

*   `:<name>` - the raw `cc_binary(linkshared = True)` for the target platform.
*   `:lib<name>.so` / `:lib<name>.dylib` / `:<name>.dll` - genrules that copy
    the binary to its conventional file name. Prefer these over `:<name>` when
    building from the command line: they only depend on the library file itself
    and not on the (potentially unbuildable) runfiles of the `cc_binary`.
*   `:<name>_lib` - an alias that selects the right genrule for the target OS.
*   `:<framework_name>` - optionally, a dynamic `apple_xcframework` for iOS
    (device + simulator) that wraps the same code for use from Xcode, Swift
    Package Manager, CocoaPods or Flutter.
"""

load("@build_bazel_rules_apple//apple:apple.bzl", "apple_xcframework")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("//mediapipe:version.bzl", "MEDIAPIPE_FULL_VERSION")
load("//mediapipe/framework/tool:ios.bzl", "MPP_TASK_MINIMUM_OS_VERSION")

# Shell snippet that converts a GNU ld version script into an Apple
# `-exported_symbols_list` file: every `MpFoo*;` / `MpBar;` entry in the
# `global:` section becomes `_MpFoo*` / `_MpBar` (Mach-O symbols carry a
# leading underscore). Comments and the `local: *;` catch-all are dropped.
_LDS_TO_EXPORTED_SYMBOLS_CMD = (
    "sed -n 's/^[[:space:]]*\\(Mp[A-Za-z0-9_]*\\**\\);.*$$/_\\1/p' $< > $@"
)

_SHARED_LIBRARY_TAGS = [
    "manual",
    "nobuilder",
    "notap",
]

# Size-related linker flags. The libraries only ever expose the `Mp*` C API,
# so everything that is not reachable from it is dead weight for consumers:
#
# *   `--gc-sections` / `-dead_strip` drop unreferenced sections (crosstools
#     compile with `-ffunction-sections -fdata-sections`).
# *   `--icf=all` folds identical functions (ELF only; lld-link does this
#     by default together with `/OPT:REF`).
# *   `--strip-all` / `-x` drop the symbol table and debug info, which can be
#     a third of an unstripped library. The exported `Mp*` symbols are kept.
#
# Together with `-legacy_whole_archive` (see `cc_binary` below) this is what
# the MediaPipe Android AAR (`mediapipe_tasks_aar.bzl`) does.
_ELF_SIZE_LINKOPTS = [
    "-Wl,--gc-sections",
    "-Wl,--icf=all",
    "-Wl,--strip-all",
]

# Raw `ld64` options: `apple_xcframework` hands `linkopts` to the linker
# directly, `cc_binary` goes through the clang driver and needs `-Wl,`.
_MACHO_SIZE_LDFLAGS = [
    "-dead_strip",
    "-x",
]

# Consumers rewrite the install name and rpaths of the library when they
# bundle it (Dart/Flutter run `install_name_tool -id` with the absolute path
# of the copy). lld pads the Mach-O header by only 0x20 bytes, which left the
# macOS and iOS simulator libraries with 40-88 bytes of room and made that
# step fail with "larger updated load commands do not fit". Reserve room for
# maximum-length install names, like Xcode does for its own links.
_MACHO_HEADERPAD_LDFLAGS = [
    "-headerpad_max_install_names",
]

_MACHO_LDFLAGS = _MACHO_SIZE_LDFLAGS + _MACHO_HEADERPAD_LDFLAGS

_MACHO_LINKOPTS = ["-Wl," + flag for flag in _MACHO_LDFLAGS]

# Already the default of the lexan toolchain in opt mode (`/opt:ref` implies
# ICF in lld-link); listed explicitly so that the release does not depend on
# toolchain defaults.
_PE_SIZE_LINKOPTS = [
    "/OPT:REF",
    "/OPT:ICF",
]

def _framework_infoplist(name):
    """Generates a minimal Info.plist for a dynamic framework bundle."""
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
        cmd = "printf '%s' > $@" % plist_content,
        visibility = ["//visibility:private"],
    )
    return ":" + plist_target

def _copy_output_genrule(name, src, out, visibility):
    """Copies the shared library produced by `src` to `out`.

    `cc_binary` targets drag their runfiles (data deps of the whole transitive
    closure) along when built as a top-level target. Going through a genrule
    only depends on the library file itself, mirroring `:libmediapipe.so`.
    """
    native.genrule(
        name = name,
        srcs = [src],
        outs = [out],
        cmd = "cp $< $@",
        tags = _SHARED_LIBRARY_TAGS,
        visibility = visibility,
    )

def mediapipe_tasks_c_shared_library(
        name,
        version_script,
        deps,
        framework_name = None,
        framework_bundle_id = None,
        ios_minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
        visibility = None,
        **kwargs):
    """Builds a MediaPipe Tasks C API shared library for the current platform.

    Args:
      name: Name of the library, e.g. "mediapipe_tasks_vision".
      version_script: Label of the GNU ld version script that lists the
        exported `Mp*` symbols for this domain.
      deps: C API `cc_library` targets to link into the shared library.
      framework_name: If set, additionally builds a dynamic `apple_xcframework`
        with this bundle name (e.g. "MediaPipeTasksVisionC") containing iOS
        device (arm64) and simulator (arm64, x86_64) slices. Build it with
        `--config=ios`.
      framework_bundle_id: Bundle identifier of the framework. Required when
        `framework_name` is set.
      ios_minimum_os_version: Minimum iOS version of the framework.
      visibility: Visibility of the generated targets.
      **kwargs: Additional arguments forwarded to `cc_binary`.
    """
    exported_symbols = name + "_exported_symbols.exp"
    native.genrule(
        name = name + "_exported_symbols",
        srcs = [version_script],
        outs = [exported_symbols],
        cmd = _LDS_TO_EXPORTED_SYMBOLS_CMD,
        visibility = ["//visibility:private"],
    )

    cc_binary(
        name = name,
        additional_linker_inputs = [
            version_script,
            ":" + exported_symbols,
        ],
        linkopts = select({
            "@platforms//os:android": [
                "-Wl,-soname=lib%s.so" % name,
                "-Wl,--version-script=$(location %s)" % version_script,
                "-Wl,-Bsymbolic",
            ] + _ELF_SIZE_LINKOPTS,
            "@platforms//os:linux": [
                "-Wl,-soname=lib%s.so" % name,
                "-Wl,--version-script=$(location %s)" % version_script,
                "-Wl,-Bsymbolic",
            ] + _ELF_SIZE_LINKOPTS,
            "@platforms//os:osx": [
                "-Wl,-install_name,@rpath/lib%s.dylib" % name,
                "-Wl,-exported_symbols_list,$(location :%s)" % exported_symbols,
            ] + _MACHO_LINKOPTS,
            # Windows relies on `MP_EXPORT` (`__declspec(dllexport)`) to
            # restrict the exported symbol set.
            "@platforms//os:windows": _PE_SIZE_LINKOPTS,
            "//conditions:default": [],
        }),
        linkshared = True,
        linkstatic = True,
        # Do not embed build information (user, workspace, timestamp).
        stamp = 0,
        tags = _SHARED_LIBRARY_TAGS,
        visibility = visibility,
        deps = deps,
        # Only link what is referenced from the C API plus the `alwayslink`
        # libraries (the C API libraries themselves, calculators, task graphs
        # and other static registrations) instead of every object of every
        # dependency. Without this, the linker has nothing to garbage-collect
        # and the libraries carry the whole transitive closure.
        features = ["-legacy_whole_archive"],
        **kwargs
    )

    # Conventionally named copies of the library (see module docstring).
    _copy_output_genrule(
        name = name + "_linux",
        src = ":" + name,
        out = "lib%s.so" % name,
        visibility = visibility,
    )
    _copy_output_genrule(
        name = name + "_macos",
        src = ":" + name,
        out = "lib%s.dylib" % name,
        visibility = visibility,
    )
    _copy_output_genrule(
        name = name + "_windows",
        src = ":" + name,
        out = "%s.dll" % name,
        visibility = visibility,
    )
    native.alias(
        name = name + "_lib",
        actual = select({
            "@platforms//os:windows": ":%s.dll" % name,
            "@platforms//os:osx": ":lib%s.dylib" % name,
            "//conditions:default": ":lib%s.so" % name,
        }),
        tags = _SHARED_LIBRARY_TAGS,
        visibility = visibility,
    )

    if not framework_name:
        return
    if not framework_bundle_id:
        fail("framework_bundle_id is required when framework_name is set")

    # Dynamic framework for iOS. Static xcframeworks are not an option for
    # consumers that load the library at runtime (e.g. dart:ffi), and a dynamic
    # framework also lets us restrict the exported symbols to the C API so that
    # the domain frameworks can coexist in one app.
    apple_xcframework(
        name = framework_name,
        bundle_id = framework_bundle_id,
        bundle_name = framework_name,
        exported_symbols_lists = [":" + exported_symbols],
        features = ["exported_symbols"],
        infoplists = [_framework_infoplist(framework_name)],
        ios = {
            "device": ["arm64"],
            "simulator": [
                "arm64",
                "x86_64",
            ],
        },
        # rules_apple does not strip dynamic frameworks itself; without this
        # every slice carries ~150k local symbols.
        linkopts = _MACHO_LDFLAGS,
        minimum_os_versions = {
            "ios": ios_minimum_os_version,
        },
        tags = _SHARED_LIBRARY_TAGS,
        visibility = visibility,
        deps = deps,
    )
