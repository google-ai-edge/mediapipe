#
# Copyright (c) 2023 Intel Corporation
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
#

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
def _is_windows(ctx):
    return ctx.os.name.lower().find("windows") != -1

def _get_windows_build_file():
    build_file_content = """
load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")

visibility = ["//visibility:public"]

filegroup(
    name = "all_srcs",
    srcs = glob(["model_api/cpp/**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "model_api_cmake",
    build_args = [
        "--verbose",
        "-j 24",
    ],
    cache_entries = {{
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "OpenVINO_DIR": "{openvino_dir}",
        "OpenCV_DIR": "{opencv_dir}",
    }},
    env = {{
        "HTTP_PROXY": "{http_proxy}",
        "HTTPS_PROXY": "{https_proxy}",
    }},
    lib_source = ":all_srcs",
    out_static_libs = ["model_api.lib"],
    tags = ["requires-network"],
    deps = [
        "@mediapipe//mediapipe/framework/port:opencv_core",
        "@mediapipe//third_party:openvino",
    ],
)

cc_library(
    name = "model_api",
    deps = [
        "@mediapipe//mediapipe/framework/port:opencv_core",
        "@mediapipe//third_party:openvino",
        ":model_api_cmake",
    ],
    visibility = ["//visibility:public"],
)
"""
    
    return build_file_content

def _get_linux_build_file():
    build_file_content = """
load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")

visibility = ["//visibility:public"]

filegroup(
    name = "all_srcs",
    srcs = glob(["model_api/cpp/**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "model_api_cmake",
    build_args = [
        "--verbose",
        "--",  # <- Pass remaining options to the native tool.
        # https://github.com/bazelbuild/rules_foreign_cc/issues/329
        # there is no elegant paralell compilation support
        "VERBOSE=1",
        "-j 24",
    ],
    cache_entries = {{
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "OpenVINO_DIR": "/opt/intel/openvino/runtime/cmake",
    }},
    env = {{
        "HTTP_PROXY": "{http_proxy}",
        "HTTPS_PROXY": "{https_proxy}",
        "http_proxy": "{http_proxy}",
        "https_proxy": "{https_proxy}",
    }},
    lib_source = ":all_srcs",
    out_static_libs = ["libmodel_api.a"],
    tags = ["requires-network"],
    deps = [
        "@mediapipe//mediapipe/framework/port:opencv_core",
        "@mediapipe//third_party:openvino",
    ],
)

cc_library(
    name = "model_api",
    deps = [
        "@mediapipe//mediapipe/framework/port:opencv_core",
        "@mediapipe//third_party:openvino",
        ":model_api_cmake",
    ],
    visibility = ["//visibility:public"],
)"""
    return build_file_content

def _impl(repository_ctx):
    http_proxy = repository_ctx.os.environ.get("HTTP_PROXY", "")
    https_proxy = repository_ctx.os.environ.get("HTTPS_PROXY", "")
    openvino_dir = repository_ctx.os.environ.get("OpenVINO_DIR", "")
    opencv_dir = repository_ctx.os.environ.get("OpenCV_DIR", "")
    if not http_proxy:
        http_proxy = repository_ctx.os.environ.get("http_proxy", "")
    if not https_proxy:
        https_proxy = repository_ctx.os.environ.get("https_proxy", "")
        
    # Note we need to escape '{/}' by doubling them due to call to format
    if _is_windows(repository_ctx):
        openvino_dir = openvino_dir.replace("\\", "\\\\").replace("/", "\\\\")
        build_file_content = _get_windows_build_file()
    else:
        build_file_content = _get_linux_build_file()

    repository_ctx.file("BUILD", build_file_content.format(http_proxy=http_proxy, https_proxy=https_proxy, openvino_dir=openvino_dir, opencv_dir=opencv_dir))

model_api_repository = repository_rule(
    implementation = _impl,
    local=False,
)

def workspace_model_api():
    model_api_repository(name="_model-api")
    new_git_repository(
        name = "model_api",
        remote = "https:///github.com/openvinotoolkit/model_api/",
        build_file = "@_model-api//:BUILD",
        commit = "9b5d37c22d97603de2e7ece07bea2e24d5a199d8",
    )

