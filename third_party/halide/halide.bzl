# Copyright 2023 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bazel build rules for Halide."""

load("@bazel_skylib//lib:collections.bzl", "collections")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "use_cpp_toolchain")

def halide_language_copts():
    _common_opts = [
        "-fPIC",
        "-frtti",
        "-Wno-conversion",
        "-Wno-sign-compare",
    ]
    _posix_opts = [
        "$(STACK_FRAME_UNLIMITED)",
        "-fno-exceptions",
        "-funwind-tables",
        "-fvisibility-inlines-hidden",
    ]
    _msvc_opts = [
        "-D_CRT_SECURE_NO_WARNINGS",
        "/MD",
    ]
    return _common_opts + select({
        "//conditions:default": _posix_opts,
        "@mediapipe//mediapipe:windows": _msvc_opts,
    })

def halide_language_linkopts():
    _linux_opts = [
        "-ldl",
        "-lpthread",
        "-lz",
        "-rdynamic",
    ]
    _osx_opts = [
        "-lz",
        "-Wl,-stack_size",
        "-Wl,1000000",
    ]
    _msvc_opts = []
    return select({
        "//conditions:default": _linux_opts,
        "@mediapipe//mediapipe:macos": _osx_opts,
        "@mediapipe//mediapipe:windows": _msvc_opts,
    })

def halide_runtime_linkopts():
    """ Return the linkopts needed when linking against halide_library_runtime.

    Returns:
        List to be used for linkopts.
    """
    _posix_opts = [
        "-ldl",
        "-lpthread",
    ]
    _android_opts = [
        "-llog",
    ]
    _msvc_opts = []

    return select({
        "//conditions:default": _posix_opts,
        "@mediapipe//mediapipe:android": _android_opts,
        "@mediapipe//mediapipe:windows": _msvc_opts,
    })

# Map of halide-target-base -> config_settings
_HALIDE_TARGET_CONFIG_SETTINGS_MAP = {
    # Android
    "arm-32-android": ["@mediapipe//mediapipe:android_arm"],
    "arm-64-android": ["@mediapipe//mediapipe:android_arm64"],
    "x86-32-android": ["@mediapipe//mediapipe:android_x86"],
    "x86-64-android": ["@mediapipe//mediapipe:android_x86_64"],
    # iOS
    "arm-32-ios": ["@mediapipe//mediapipe:ios_armv7"],
    "arm-64-ios": ["@mediapipe//mediapipe:ios_arm64", "@mediapipe//mediapipe:ios_arm64e"],
    # OSX (or iOS simulator)
    "x86-32-osx": ["@mediapipe//mediapipe:ios_i386"],
    "x86-64-osx": ["@mediapipe//mediapipe:macos_x86_64", "@mediapipe//mediapipe:ios_x86_64"],
    "arm-64-osx": ["@mediapipe//mediapipe:macos_arm64"],
    # Windows
    "x86-64-windows": ["@mediapipe//mediapipe:windows"],
    # Linux
    "x86-64-linux": ["@mediapipe//mediapipe:linux"],
    # Deliberately no //condition:default clause here.
}

_HALIDE_TARGET_MAP_DEFAULT = {
    "x86-64-linux": [
        "x86-64-linux-sse41-avx-avx2-fma",
        "x86-64-linux-sse41",
        "x86-64-linux",
    ],
    "x86-64-osx": [
        "x86-64-osx-sse41-avx-avx2-fma",
        "x86-64-osx-sse41",
        "x86-64-osx",
    ],
    "x86-64-windows": [
        "x86-64-windows-sse41-avx-avx2-fma",
        "x86-64-windows-sse41",
        "x86-64-windows",
    ],
}

def halide_library_default_target_map():
    return _HALIDE_TARGET_MAP_DEFAULT

# Alphabetizes the features part of the target to make sure they always match no
# matter the concatenation order of the target string pieces.
def _canonicalize_target(halide_target):
    if halide_target == "host":
        return halide_target
    if "," in halide_target:
        fail("Multitarget may not be specified here")
    tokens = halide_target.split("-")
    if len(tokens) < 3:
        fail("Illegal target: %s" % halide_target)

    # rejoin the tokens with the features sorted
    return "-".join(tokens[0:3] + sorted(tokens[3:]))

# Converts comma and dash separators to underscore and alphabetizes
# the features part of the target to make sure they always match no
# matter the concatenation order of the target string pieces.
def _halide_target_to_bazel_rule_name(multitarget):
    subtargets = multitarget.split(",")
    subtargets = [_canonicalize_target(st).replace("-", "_") for st in subtargets]
    return "_".join(subtargets)

# The second argument is True if there is a separate file generated
# for each subtarget of a multitarget output, False if not. The third
# argument is True if the output is a directory (vs. a single file).
# The fourth argument is a list of output group(s) that the files should
# be added to.

_is_multi = True
_is_single = False
_is_file = False

_output_extensions = {
    "assembly": ("s", _is_multi, _is_file, []),
    "bitcode": ("bc", _is_multi, _is_file, ["generated_bitcode"]),
    "c_header": ("h", _is_single, _is_file, ["generated_headers"]),
    "c_source": ("halide_generated.cpp", _is_multi, _is_file, []),
    "compiler_log": ("halide_compiler_log", _is_single, _is_file, ["generated_object", "generated_compiler_log"]),
    "cpp_stub": ("stub.h", _is_single, _is_file, []),
    "featurization": ("featurization", _is_multi, _is_file, []),
    "llvm_assembly": ("ll", _is_multi, _is_file, []),
    "object": ("o", _is_single, _is_file, ["generated_object"]),
    "python_extension": ("py.cpp", _is_single, _is_file, []),
    "registration": ("registration.cpp", _is_single, _is_file, ["generated_registration"]),
    "schedule": ("schedule.h", _is_single, _is_file, []),
    "static_library": ("a", _is_single, _is_file, ["generated_object"]),
    "stmt": ("stmt", _is_multi, _is_file, []),
    "stmt_html": ("stmt.html", _is_multi, _is_file, []),
}

def _add_output_file(f, fmt, output_files, output_dict, verbose_extra_outputs, verbose_output_paths):
    if fmt in verbose_extra_outputs:
        verbose_output_paths.append(f.path)
    output_files.append(f)
    if fmt in _output_extensions:
        for group in _output_extensions[fmt][3]:
            output_dict.setdefault(group, []).append(f)

HalideFunctionNameInfo = provider(fields = ["function_name"])
HalideGeneratorBinaryInfo = provider(fields = ["generator_binary"])
HalideGeneratorNameInfo = provider(fields = ["generator_name_"])
HalideGeneratorParamsInfo = provider(fields = ["generator_params"])
HalideLibraryNameInfo = provider(fields = ["library_name"])
HalideTargetFeaturesInfo = provider(fields = ["target_features"])

def _gengen_closure_impl(ctx):
    return [
        HalideGeneratorBinaryInfo(generator_binary = ctx.attr.generator_binary),
        HalideGeneratorNameInfo(generator_name_ = ctx.attr.generator_name_),
    ]

_gengen_closure = rule(
    implementation = _gengen_closure_impl,
    attrs = {
        "generator_binary": attr.label(
            executable = True,
            allow_files = True,
            mandatory = True,
            cfg = "exec",
        ),
        # "generator_name" is apparently reserved by Bazel for attrs in rules
        "generator_name_": attr.string(mandatory = True),
    },
    provides = [HalideGeneratorBinaryInfo, HalideGeneratorNameInfo],
)

def _halide_library_instance_impl(ctx):
    generator_binary = ctx.attr.generator_closure[HalideGeneratorBinaryInfo].generator_binary if ctx.attr.generator_closure else ""
    generator_name = ctx.attr.generator_closure[HalideGeneratorNameInfo].generator_name_ if ctx.attr.generator_closure else ""
    return [
        HalideFunctionNameInfo(function_name = ctx.attr.function_name),
        HalideGeneratorBinaryInfo(generator_binary = generator_binary),
        HalideGeneratorNameInfo(generator_name_ = generator_name),
        HalideGeneratorParamsInfo(generator_params = ctx.attr.generator_params),
        HalideLibraryNameInfo(library_name = ctx.attr.library_name),
        HalideTargetFeaturesInfo(target_features = ctx.attr.target_features),
    ]

_halide_library_instance = rule(
    implementation = _halide_library_instance_impl,
    attrs = {
        "function_name": attr.string(),
        "generator_closure": attr.label(
            cfg = "exec",
            providers = [HalideGeneratorBinaryInfo, HalideGeneratorNameInfo],
        ),
        "generator_params": attr.string_list(),
        "library_name": attr.string(),
        "target_features": attr.string_list(),
    },
    provides = [
        HalideFunctionNameInfo,
        HalideGeneratorBinaryInfo,
        HalideGeneratorNameInfo,
        HalideGeneratorParamsInfo,
        HalideLibraryNameInfo,
        HalideTargetFeaturesInfo,
    ],
)

def _gengen_impl(ctx):
    if _has_dupes(ctx.attr.requested_outputs):
        fail("Duplicate values in outputs: " + str(ctx.attr.requested_outputs))

    function_name = ctx.attr.function_name[HalideFunctionNameInfo].function_name if ctx.attr.function_name else ""
    generator_binary = ctx.attr.generator_binary[HalideGeneratorBinaryInfo].generator_binary if ctx.attr.generator_binary else ""
    generator_name_ = ctx.attr.generator_name_[HalideGeneratorNameInfo].generator_name_ if ctx.attr.generator_name_ else ""
    generator_params = ctx.attr.generator_params[HalideGeneratorParamsInfo].generator_params if ctx.attr.generator_params else []
    library_name = ctx.attr.library_name[HalideLibraryNameInfo].library_name if ctx.attr.library_name else ""
    target_features = ctx.attr.target_features[HalideTargetFeaturesInfo].target_features if ctx.attr.target_features else []

    for gp in generator_params:
        if " " in gp:
            fail("%s: Entries in generator_params must not contain spaces." % library_name)

    # Escape backslashes and double quotes.
    generator_params = [gp.replace("\\", '\\\\"').replace('"', '\\"') for gp in generator_params]

    execution_requirements = {}

    # --- Calculate the output type(s) we're going to produce (and which ones should be verbose)
    quiet_extra_outputs = []
    verbose_extra_outputs = []
    if ctx.attr.consider_halide_extra_outputs:
        if "halide_extra_outputs" in ctx.var:
            verbose_extra_outputs = ctx.var.get("halide_extra_outputs", "").split(",")
        if "halide_extra_outputs_quiet" in ctx.var:
            quiet_extra_outputs = ctx.var.get("halide_extra_outputs_quiet", "").split(",")
    requested_outputs = sorted(collections.uniq(ctx.attr.requested_outputs +
                                                verbose_extra_outputs +
                                                quiet_extra_outputs))

    # --- Assemble halide_target, adding extra features if necessary
    base_target = ctx.attr.halide_base_target
    if "," in base_target:
        fail("halide_base_target should never be a multitarget")
    if len(base_target.split("-")) != 3:
        fail("halide_base_target should have exactly 3 components")

    target_features = target_features + ctx.var.get("halide_target_features", "").split(",")

    if "no_runtime" in target_features:
        fail("Specifying 'no_runtime' in halide_target_features is not supported; " +
             "please add 'add_halide_runtime_deps = False' to the halide_library() rule instead.")

    for san in ["asan", "msan", "tsan"]:
        if san in target_features:
            fail("halide_library doesn't support '%s' in halide_target_features; please build with --config=%s instead." % (san, san))

    # Append the features common to everything.
    target_features.append("c_plus_plus_name_mangling")
    target_features.append("no_runtime")

    # Make it all neat and tidy.
    target_features = sorted(collections.uniq(target_features))

    # Get the multitarget list (if any) from halide_target_map
    halide_targets = ctx.attr.halide_target_map.get(base_target, [base_target])

    # Add the extra features to all of them
    halide_targets = _add_features_to_all(halide_targets, target_features)

    leaf_name = ctx.attr.filename.split("/")[-1]

    output_files = []
    output_dict = {}
    verbose_output_paths = []
    inputs = []

    env = {
        "HL_DEBUG_CODEGEN": str(ctx.var.get("halide_debug_codegen", 0)),
        # --define halide_llvm_args=-time-passes is a typical usage
        "HL_LLVM_ARGS": str(ctx.var.get("halide_llvm_args", "")),
    }

    be_very_quiet = ctx.var.get("halide_experimental_quiet", False)  # I'm hunting wabbit...

    # --- Calculate the final set of output files
    for fmt in requested_outputs:
        if fmt not in _output_extensions:
            fail("Unknown Halide output '%s'; known outputs are %s" %
                 (fmt, sorted(_output_extensions.keys())))
        ext, is_multiple, is_dir, _ = _output_extensions[fmt]

        # Special-case Windows file extensions
        if "windows" in halide_targets[-1]:
            if ext == "o":
                ext = "obj"
            if ext == "a":
                ext = "lib"
        if is_multiple and len(halide_targets) > 1:
            for h in halide_targets:
                suffix = _canonicalize_target(h)
                name = "%s-%s.%s" % (ctx.attr.filename, suffix, ext)
                f = ctx.actions.declare_directory(name) if is_dir else ctx.actions.declare_file(name)
                _add_output_file(f, fmt, output_files, output_dict, verbose_extra_outputs, verbose_output_paths)
        else:
            name = "%s.%s" % (ctx.attr.filename, ext)
            f = ctx.actions.declare_directory(name) if is_dir else ctx.actions.declare_file(name)
            _add_output_file(f, fmt, output_files, output_dict, verbose_extra_outputs, verbose_output_paths)

    # --- Progress message(s), including log info about any 'extra' files being output due to --define halide_extra_output
    progress_message = "Executing generator %s with target (%s) args (%s)." % (
        generator_name_,
        ",".join(halide_targets),
        " ".join(generator_params),
    )

    for f in output_files:
        if any([f.path.endswith(suf) for suf in [".h", ".a", ".o", ".lib", ".registration.cpp", ".bc", ".halide_compiler_log"]]):
            continue

        # If an extra output was specified via --define halide_extra_outputs=foo on the command line,
        # add to the progress message (so that it is ephemeral and doesn't clog stdout).
        #
        # (Trailing space is intentional since Starlark will append a period to the end,
        # making copy-n-paste harder than it might otherwise be...)
        if not be_very_quiet:
            extra_msg = "Emitting extra Halide output: %s " % f.path
            progress_message += "\n" + extra_msg
            if f.path in verbose_output_paths:
                # buildifier: disable=print
                print(extra_msg)

    # --- Construct the arguments list for the Generator
    arguments = ctx.actions.args()
    arguments.add("-o", output_files[0].dirname)
    if ctx.attr.generate_runtime:
        arguments.add("-r", leaf_name)
        if len(halide_targets) > 1:
            fail("Only one halide_target allowed when using generate_runtime")
        if function_name:
            fail("halide_function_name not allowed when using generate_runtime")
    else:
        arguments.add("-g", generator_name_)
        arguments.add("-n", leaf_name)
        if function_name:
            arguments.add("-f", function_name)

    if requested_outputs:
        arguments.add_joined("-e", requested_outputs, join_with = ",")

    # Can't use add_joined(), as it will insert a space after target=
    arguments.add("target=%s" % (",".join(halide_targets)))
    if generator_params:
        for p in generator_params:
            for s in ["target"]:
                if p.startswith("%s=" % s):
                    fail("You cannot specify %s in the generator_params parameter in bazel." % s)
        arguments.add_all(generator_params)

    show_gen_arg = ctx.var.get("halide_show_generator_command", "")

    # If it's an exact match of a fully qualified path, show just that one.
    # If it's * or "all", match everything.
    if library_name and show_gen_arg in [library_name, "all", "*"] and not ctx.attr.generate_runtime:
        # The 'Args' object can be printed, but can't be usefully converted to a string, or iterated,
        # so we'll reproduce the logic here. We'll also take the opportunity to add or augment
        # some args to be more useful to whoever runs it (eg, add `-v=1`, add some output files).
        sg_args = ["-v", "1"]
        sg_args += ["-o", "/tmp"]
        sg_args += ["-g", generator_name_]
        sg_args += ["-n", leaf_name]
        if function_name:
            sg_args += ["-f", function_name]
        if requested_outputs:
            # Ensure that several commonly-useful output are added
            ro = sorted(collections.uniq(requested_outputs + ["stmt", "assembly", "llvm_assembly"]))
            sg_args += ["-e", ",".join(ro)]
        sg_args.append("target=%s" % (",".join(halide_targets)))

        if generator_params:
            sg_args += generator_params

        # buildifier: disable=print
        print(
            "\n\nTo locally run the Generator for",
            library_name,
            "use the command:\n\n",
            "bazel run -c opt",
            generator_binary.label,
            "--",
            " ".join(sg_args),
            "\n\n",
        )

    # Finally... run the Generator.
    ctx.actions.run(
        execution_requirements = execution_requirements,
        arguments = [arguments],
        env = env,
        executable = generator_binary.files_to_run.executable,
        mnemonic = "ExecuteHalideGenerator",
        inputs = depset(direct = inputs),
        outputs = output_files,
        progress_message = progress_message,
        exec_group = "generator",
    )

    return [
        DefaultInfo(files = depset(direct = output_files)),
        OutputGroupInfo(**output_dict),
    ]

_gengen = rule(
    implementation = _gengen_impl,
    attrs = {
        "consider_halide_extra_outputs": attr.bool(),
        "filename": attr.string(),
        "generate_runtime": attr.bool(default = False),
        "generator_binary": attr.label(
            cfg = "exec",
            providers = [HalideGeneratorBinaryInfo],
        ),
        # "generator_name" is apparently reserved by Bazel for attrs in rules
        "generator_name_": attr.label(
            cfg = "exec",
            providers = [HalideGeneratorNameInfo],
        ),
        "halide_base_target": attr.string(),
        "function_name": attr.label(
            cfg = "target",
            providers = [
                HalideFunctionNameInfo,
            ],
        ),
        "generator_params": attr.label(
            cfg = "target",
            providers = [
                HalideGeneratorParamsInfo,
            ],
        ),
        "library_name": attr.label(
            cfg = "target",
            providers = [
                HalideLibraryNameInfo,
            ],
        ),
        "target_features": attr.label(
            cfg = "target",
            providers = [
                HalideTargetFeaturesInfo,
            ],
        ),
        "halide_target_map": attr.string_list_dict(),
        "requested_outputs": attr.string_list(),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
    exec_groups = {
        "generator": exec_group(),
    },
)

def _add_target_features(target, features):
    if "," in target:
        fail("Cannot use multitarget here")
    new_target = target.split("-")
    for f in features:
        if f and f not in new_target:
            new_target.append(f)
    return "-".join(new_target)

def _add_features_to_all(halide_targets, features):
    return [_canonicalize_target(_add_target_features(t, features)) for t in halide_targets]

def _has_dupes(some_list):
    clean = collections.uniq(some_list)
    return sorted(some_list) != sorted(clean)

# Target features which do not affect runtime compatibility.
_IRRELEVANT_FEATURES = collections.uniq([
    "arm_dot_prod",
    "arm_fp16",
    "c_plus_plus_name_mangling",
    "check_unsafe_promises",
    "embed_bitcode",
    "enable_llvm_loop_opt",
    "large_buffers",
    "no_asserts",
    "no_bounds_query",
    "profile",
    "strict_float",
    "sve",
    "sve2",
    "trace_loads",
    "trace_pipeline",
    "trace_realizations",
    "trace_stores",
    "user_context",
    "wasm_sat_float_to_int",
    "wasm_signext",
    "wasm_simd128",
])

def _discard_irrelevant_features(halide_target_features = []):
    return sorted(collections.uniq([f for f in halide_target_features if f not in _IRRELEVANT_FEATURES]))

def _halide_library_runtime_target_name(halide_target_features = []):
    return "_".join(["halide_library_runtime"] + _discard_irrelevant_features(halide_target_features))

def _define_halide_library_runtime(
        halide_target_features = [],
        compatible_with = []):
    target_name = _halide_library_runtime_target_name(halide_target_features)

    if not native.existing_rule("halide_library_runtime.generator"):
        halide_generator(
            name = "halide_library_runtime.generator",
            srcs = [],
            deps = [],
            visibility = ["//visibility:private"],
        )
    condition_deps = {}
    for base_target, cfgs in _HALIDE_TARGET_CONFIG_SETTINGS_MAP.items():
        target_features = _discard_irrelevant_features(halide_target_features)
        halide_target_name = _halide_target_to_bazel_rule_name(base_target)
        gengen_name = "%s_%s" % (halide_target_name, target_name)

        _halide_library_instance(
            name = "%s.library_instance" % gengen_name,
            compatible_with = compatible_with,
            function_name = "",
            generator_closure = ":halide_library_runtime.generator_closure",
            generator_params = [],
            library_name = "",
            target_features = target_features,
            visibility = ["//visibility:private"],
        )
        hl_instance = ":%s.library_instance" % gengen_name

        _gengen(
            name = gengen_name,
            compatible_with = compatible_with,
            filename = "%s/%s" % (halide_target_name, target_name),
            generate_runtime = True,
            generator_binary = hl_instance,
            generator_name_ = hl_instance,
            halide_base_target = base_target,
            requested_outputs = ["object"],
            tags = ["manual"],
            target_features = hl_instance,
            visibility = ["@halide//:__subpackages__"],
        )
        for cfg in cfgs:
            condition_deps[cfg] = [":%s" % gengen_name]

    deps = []
    native.cc_library(
        name = target_name,
        compatible_with = compatible_with,
        srcs = select(condition_deps),
        linkopts = halide_runtime_linkopts(),
        tags = ["manual"],
        deps = deps,
        visibility = ["//visibility:public"],
    )

    return target_name

def _standard_library_runtime_features():
    _standard_features = [
        [],
        ["cuda"],
        ["metal"],
        ["opencl"],
        ["openglcompute"],
        ["openglcompute", "egl"],
    ]
    return [f for f in _standard_features] + [f + ["debug"] for f in _standard_features]

def _standard_library_runtime_names():
    return collections.uniq([_halide_library_runtime_target_name(f) for f in _standard_library_runtime_features()])

def halide_library_runtimes(compatible_with = []):
    unused = [
        _define_halide_library_runtime(f, compatible_with = compatible_with)
        for f in _standard_library_runtime_features()
    ]
    unused = unused  # unused variable

def halide_generator(
        name,
        srcs,
        compatible_with = [],
        copts = [],
        deps = [],
        generator_name = "",
        includes = [],
        tags = [],
        testonly = False,
        visibility = None):
    if not name.endswith(".generator"):
        fail("halide_generator rules must end in .generator")

    basename = name[:-10]  # strip ".generator" suffix
    if not generator_name:
        generator_name = basename

    # Note: This target is public, but should not be needed by the vast
    # majority of users. Unless you are writing a custom Bazel rule that
    # involves Halide generation, you most probably won't need to depend on
    # this rule.
    native.cc_binary(
        name = name,
        copts = copts + halide_language_copts(),
        linkopts = halide_language_linkopts(),
        compatible_with = compatible_with,
        srcs = srcs,
        deps = [
            "@halide//:gengen",
            "@halide//:language",
        ] + deps,
        tags = ["manual"] + tags,
        testonly = testonly,
        visibility = ["//visibility:public"],
    )

    _gengen_closure(
        name = "%s_closure" % name,
        generator_binary = name,
        generator_name_ = generator_name,
        compatible_with = compatible_with,
        testonly = testonly,
        visibility = ["//visibility:private"],
    )

# This rule exists to allow us to select() on halide_target_features.
def _select_halide_library_runtime_impl(ctx):
    f = ctx.attr.halide_target_features

    standard_runtimes = {t.label.name: t for t in ctx.attr._standard_runtimes}

    f = sorted(_discard_irrelevant_features(collections.uniq(f)))
    runtime_name = _halide_library_runtime_target_name(f)
    if runtime_name not in standard_runtimes:
        fail(("There is no Halide runtime available for the feature set combination %s. " +
              "Please use contact information from halide-lang.org to contact the Halide " +
              "team to add the right combination.") % str(f))

    return standard_runtimes[runtime_name][CcInfo]

_select_halide_library_runtime = rule(
    implementation = _select_halide_library_runtime_impl,
    attrs = {
        "halide_target_features": attr.string_list(),
        "_standard_runtimes": attr.label_list(
            default = ["@halide//:%s" % n for n in _standard_library_runtime_names()],
            providers = [CcInfo],
        ),
    },
    provides = [CcInfo],
)

def halide_library_from_generator(
        name,
        generator,
        add_halide_runtime_deps = True,
        compatible_with = [],
        deps = [],
        function_name = None,
        generator_params = [],
        halide_target_features = [],
        halide_target_map = halide_library_default_target_map(),
        includes = [],
        namespace = None,
        tags = [],
        testonly = False,
        visibility = None):
    if not function_name:
        function_name = name

    if namespace:
        function_name = "%s::%s" % (namespace, function_name)

    generator_closure = "%s_closure" % generator

    _halide_library_instance(
        name = "%s.library_instance" % name,
        compatible_with = compatible_with,
        function_name = function_name,
        generator_closure = generator_closure,
        generator_params = generator_params,
        library_name = "//%s:%s" % (native.package_name(), name),
        target_features = halide_target_features,
        testonly = testonly,
        visibility = ["//visibility:private"],
    )
    hl_instance = ":%s.library_instance" % name

    condition_deps = {}
    for base_target, cfgs in _HALIDE_TARGET_CONFIG_SETTINGS_MAP.items():
        base_target_name = _halide_target_to_bazel_rule_name(base_target)
        gengen_name = "%s_%s" % (base_target_name, name)
        _gengen(
            name = gengen_name,
            compatible_with = compatible_with,
            consider_halide_extra_outputs = True,
            filename = "%s/%s" % (base_target_name, name),
            function_name = hl_instance,
            generator_binary = generator_closure,
            generator_name_ = generator_closure,
            generator_params = hl_instance,
            halide_base_target = base_target,
            halide_target_map = halide_target_map,
            library_name = hl_instance,
            requested_outputs = ["static_library"],
            tags = ["manual"] + tags,
            target_features = hl_instance,
            testonly = testonly,
        )
        for cfg in cfgs:
            condition_deps[cfg] = [":%s" % gengen_name]

    # Use a canonical target to build CC, regardless of config detected
    cc_base_target = "x86-64-linux"

    for output, target_name in [
        ("c_header", "%s_h" % name),
        ("c_source", "%s_cc" % name),
    ]:
        _gengen(
            name = target_name,
            compatible_with = compatible_with,
            filename = name,
            function_name = hl_instance,
            generator_binary = generator_closure,
            generator_name_ = generator_closure,
            generator_params = hl_instance,
            halide_base_target = cc_base_target,
            library_name = hl_instance,
            requested_outputs = [output],
            tags = ["manual"] + tags,
            target_features = hl_instance,
            testonly = testonly,
        )

    _select_halide_library_runtime(
        name = "%s.halide_library_runtime_deps" % name,
        halide_target_features = halide_target_features,
        compatible_with = compatible_with,
        tags = tags,
        visibility = ["//visibility:private"],
    )

    native.filegroup(
        name = "%s_object" % name,
        srcs = select(condition_deps),
        output_group = "generated_object",
        visibility = ["//visibility:private"],
        compatible_with = compatible_with,
        tags = tags,
        testonly = testonly,
    )

    native.cc_library(
        name = name,
        srcs = ["%s_object" % name],
        hdrs = [
            ":%s_h" % name,
        ],
        deps = deps +
               ["@halide//:runtime"] +  # for HalideRuntime.h, etc
               ([":%s.halide_library_runtime_deps" % name] if add_halide_runtime_deps else []),  # for the runtime implementation
        defines = ["HALIDE_FUNCTION_ATTRS=HALIDE_MUST_USE_RESULT"],
        compatible_with = compatible_with,
        includes = includes,
        tags = tags,
        testonly = testonly,
        visibility = visibility,
        linkstatic = 1,
    )

    # Return the fully-qualified built target name.
    return "//%s:%s" % (native.package_name(), name)

def halide_library(
        name,
        srcs = [],
        add_halide_runtime_deps = True,
        copts = [],
        compatible_with = [],
        filter_deps = [],
        function_name = None,
        generator_params = [],
        generator_deps = [],
        generator_name = None,
        halide_target_features = [],
        halide_target_map = halide_library_default_target_map(),
        includes = [],
        namespace = None,
        tags = [],
        testonly = False,
        visibility = None):
    if not srcs and not generator_deps:
        fail("halide_library needs at least one of srcs or generator_deps to provide a generator")

    halide_generator(
        name = "%s.generator" % name,
        srcs = srcs,
        compatible_with = compatible_with,
        generator_name = generator_name,
        deps = generator_deps,
        includes = includes,
        copts = copts,
        tags = tags,
        testonly = testonly,
        visibility = visibility,
    )

    return halide_library_from_generator(
        name = name,
        generator = ":%s.generator" % name,
        add_halide_runtime_deps = add_halide_runtime_deps,
        compatible_with = compatible_with,
        deps = filter_deps,
        function_name = function_name,
        generator_params = generator_params,
        halide_target_features = halide_target_features,
        halide_target_map = halide_target_map,
        includes = includes,
        namespace = namespace,
        tags = tags,
        testonly = testonly,
        visibility = visibility,
    )
