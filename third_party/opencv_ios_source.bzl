# Copyright 2023 The MediaPipe Authors. All rights reserved.
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

"""Custom rules for building iOS OpenCV xcframework from sources."""

load(
    "@//third_party:opencv_ios_xcframework_files.bzl",
    "OPENCV_XCFRAMEWORK_INFO_PLIST_PATH",
    "OPENCV_XCFRAMEWORK_IOS_DEVICE_FILE_PATHS",
    "OPENCV_XCFRAMEWORK_IOS_SIMULATOR_FILE_PATHS",
)

_OPENCV_XCFRAMEWORK_DIR_NAME = "opencv2.xcframework"
_OPENCV_FRAMEWORK_DIR_NAME = "opencv2.framework"
_OPENCV_SIMULATOR_PLATFORM_DIR_NAME = "ios-arm64_x86_64-simulator"
_OPENCV_DEVICE_PLATFORM_DIR_NAME = "ios-arm64"

def _select_headers_impl(ctx):
    # Should match with `/`. Othewise `ios-arm64` matches with `ios-arm64_x86-64`
    _files = [
        f
        for f in ctx.files.srcs
        if (f.basename.endswith(".h") or f.basename.endswith(".hpp")) and
           f.dirname.find(ctx.attr.platform + "/") != -1
    ]
    return [DefaultInfo(files = depset(_files))]

# This rule selects only the headers from an apple static xcframework filtered by
# an input platform string.
select_headers = rule(
    implementation = _select_headers_impl,
    attrs = {
        "srcs": attr.label_list(mandatory = True, allow_files = True),
        "platform": attr.string(mandatory = True),
    },
)

# This function declares and returns symlinks to the directories within each platform
# in `opencv2.xcframework` expected to be present.
# The symlinks are created according to the structure stipulated by apple xcframeworks
# do that they can be correctly consumed by `apple_static_xcframework_import` rule.
def _opencv2_directory_symlinks(ctx, platforms):
    basenames = ["Resources", "Headers", "Modules", "Versions/Current"]
    symlinks = []

    for platform in platforms:
        symlinks = symlinks + [
            ctx.actions.declare_symlink(
                _OPENCV_XCFRAMEWORK_DIR_NAME + "/{}/{}/{}".format(platform, _OPENCV_FRAMEWORK_DIR_NAME, name),
            )
            for name in basenames
        ]

    return symlinks

# This function declares and returns all the files for each platform expected
# to be present in  `opencv2.xcframework` after the unzipping action is run.
def _opencv2_file_list(ctx, platform_filepath_lists):
    binary_name = "opencv2"
    output_files = []
    binaries_to_symlink = []

    for (platform, filepaths) in platform_filepath_lists:
        for path in filepaths:
            file = ctx.actions.declare_file(path)
            output_files.append(file)
            if path.endswith(binary_name):
                symlink_output = ctx.actions.declare_file(
                    _OPENCV_XCFRAMEWORK_DIR_NAME + "/{}/{}/{}".format(
                        platform,
                        _OPENCV_FRAMEWORK_DIR_NAME,
                        binary_name,
                    ),
                )
                binaries_to_symlink.append((symlink_output, file))

    return output_files, binaries_to_symlink

def _unzip_opencv_xcframework_impl(ctx):
    # Array to iterate over the various platforms to declare output files and
    # symlinks.
    platform_filepath_lists = [
        (_OPENCV_SIMULATOR_PLATFORM_DIR_NAME, OPENCV_XCFRAMEWORK_IOS_SIMULATOR_FILE_PATHS),
        (_OPENCV_DEVICE_PLATFORM_DIR_NAME, OPENCV_XCFRAMEWORK_IOS_DEVICE_FILE_PATHS),
    ]

    # Gets an exhaustive list of output files which are  present in the xcframework.
    # Also gets array of `(binary simlink, binary)` pairs which are to be symlinked
    # using `ctx.actions.symlink()`.
    output_files, binaries_to_symlink = _opencv2_file_list(ctx, platform_filepath_lists)
    output_files.append(ctx.actions.declare_file(OPENCV_XCFRAMEWORK_INFO_PLIST_PATH))

    # xcframeworks have a directory structure in which the `opencv2.framework` folders for each
    # platform contain directories which are symlinked to the respective folders of the version
    # in use. Simply unzipping the zip of the framework will not make Bazel treat these
    # as symlinks. They have to be explicity declared as symlinks using `ctx.actions.declare_symlink()`.
    directory_symlinks = _opencv2_directory_symlinks(
        ctx,
        [_OPENCV_SIMULATOR_PLATFORM_DIR_NAME, _OPENCV_DEVICE_PLATFORM_DIR_NAME],
    )

    output_files = output_files + directory_symlinks

    args = ctx.actions.args()

    # Add the path of the zip file to be unzipped as an argument to be passed to
    # `run_shell` action.
    args.add(ctx.file.zip_file.path)

    # Add the path to the directory in which the framework is to be unzipped to.
    args.add(ctx.file.zip_file.dirname)

    ctx.actions.run_shell(
        inputs = [ctx.file.zip_file],
        outputs = output_files,
        arguments = [args],
        progress_message = "Unzipping %s" % ctx.file.zip_file.short_path,
        command = "unzip -qq $1 -d $2",
    )

    # The symlinks of the opencv2 binaries for each platform in the xcframework
    # have to be symlinked using the `ctx.actions.symlink` unlike the directory
    # symlinks which can be expected to be valid when unzipping is completed.
    # Otherwise, when tests are run, the linker complaints that the binary is
    # not found.
    binary_symlink_files = []
    for (symlink_output, binary_file) in binaries_to_symlink:
        ctx.actions.symlink(output = symlink_output, target_file = binary_file)
        binary_symlink_files.append(symlink_output)

    # Return all the declared output files and symlinks as the output of this
    # rule.
    return [DefaultInfo(files = depset(output_files + binary_symlink_files))]

# This rule unzips an `opencv2.xcframework.zip` created by a genrule that
# invokes a python script in the opencv 4.5.1 github archive.
# It returns all the contents of  opencv2.xcframework as a list of files in the
# output. This rule works by explicitly declaring files at hardcoded
# paths in the opencv2 xcframework bundle which are expected to be present when
# the zip file is unzipped. This is a prerequisite since the outputs of this rule
# will be consumed by apple_static_xcframework_import which can only take a list
# of files as inputs.
unzip_opencv_xcframework = rule(
    implementation = _unzip_opencv_xcframework_impl,
    attrs = {
        "zip_file": attr.label(mandatory = True, allow_single_file = True),
    },
)
