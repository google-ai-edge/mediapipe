# Copyright 2020 The MediaPipe Authors.
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

"""A rule for registering types with mediapipe.

Example usage:
    mediapipe_proto_library(
        name = "foo_proto",
        srcs = ["foo.proto"],
    )

    load("//mediapipe/framework:mediapipe_register_type.bzl",
         "mediapipe_register_type")

    # Creates rules "foo_registration"
    mediapipe_register_type(
        base_name = "foo"
        include_headers = ["mediapipe/framework/formats/foo.proto.h""],
        types = [
            "::mediapipe::Foo",
            "::mediapipe::FooList",
            "::std::vector<::mediapipe::Foo>",
        ],
        deps = [":foo_cc_proto"],
    )

Args
  base_name: The base name of the target
             (name + "_registration" will be created).
  types: A list of C++ classes to register using MEDIAPIPE_REGISTER_TYPE.
  deps: A list of cc deps.
  include_headers: A list of header files that must be included.
"""

load("//mediapipe/framework/tool:build_defs.bzl", "clean_dep")

def _mediapipe_register_type_generate_cc_impl(ctx):
    """Generate a cc file that registers types with mediapipe."""
    file_data_template = '''
{include_headers}
#include "mediapipe/framework/type_map.h"

{registration_commands}
'''
    header_lines = []
    for header in ctx.attr.include_headers:
        header_lines.append('#include "{}"'.format(header))
    registration_lines = []
    for registration_type in ctx.attr.types:
        if " " in registration_type:
            fail(('registration type "{}" should be fully qualified ' +
                  "and must not include spaces").format(registration_type))
        registration_lines.append(
            "#define TEMP_MP_TYPE {}".format(registration_type),
        )
        registration_lines.append(
            ("MEDIAPIPE_REGISTER_TYPE(\n" +
             "    TEMP_MP_TYPE,\n" +
             '    "{}",\n'.format(registration_type) +
             "    nullptr, nullptr);\n"),
        )
        registration_lines.append("#undef TEMP_MP_TYPE")

    file_data = file_data_template.format(
        include_headers = "\n".join(header_lines),
        registration_commands = "\n".join(registration_lines),
    )

    ctx.actions.write(ctx.outputs.output, file_data)

mediapipe_register_type_generate_cc = rule(
    implementation = _mediapipe_register_type_generate_cc_impl,
    attrs = {
        "deps": attr.label_list(),
        "types": attr.string_list(
            mandatory = True,
        ),
        "include_headers": attr.string_list(
            mandatory = True,
        ),
        "output": attr.output(),
    },
)

def mediapipe_register_type(
        base_name,
        types,
        deps = [],
        include_headers = [],
        visibility = ["//visibility:public"]):
    mediapipe_register_type_generate_cc(
        name = base_name + "_registration_cc",
        types = types,
        include_headers = include_headers,
        output = base_name + "_registration.cc",
    )

    native.cc_library(
        name = base_name + "_registration",
        srcs = [base_name + "_registration.cc"],
        deps = depset(deps + [
            clean_dep("//mediapipe/framework:type_map"),
        ]),
        visibility = visibility,
        alwayslink = 1,
    )
