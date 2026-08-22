#!/bin/bash
set -eu
set -o pipefail

# Adds modulemap & header files to an iOS Framework
# generated with bazel and encapsulating Mediapipe.
#
# This makes it so that the patched .framework can be imported into Xcode.
# For a long term solution track the following issue:
#   https://github.com/bazelbuild/rules_apple/issues/355

[[ $# -lt 2 ]] && echo "Usage: $0  <path/to/zipped .framework> <hdrs>..." && exit 1
zipped=$(python -c "import os; print(os.path.realpath('$1'))"); shift
name=$(basename "$zipped" .zip)
parent=$(dirname "$zipped")
named="$parent"/"$name".framework

unzip "$zipped" -d "$parent"

mkdir "$named"/Modules
cat << EOF >"$named"/Modules/module.modulemap
framework module $name {
  umbrella header "$name.h"

  export *
  module * { export * }

  link framework "AVFoundation"
  link framework "Accelerate"
  link framework "AssetsLibrary"
  link framework "CoreFoundation"
  link framework "CoreGraphics"
  link framework "CoreImage"
  link framework "CoreMedia"
  link framework "CoreVideo"
  link framework "GLKit"
  link framework "Metal"
  link framework "MetalKit"
  link framework "OpenGLES"
  link framework "QuartzCore"
  link framework "UIKit"
}
EOF
# NOTE: All these linked frameworks are required by mediapipe/objc.

cat << EOF >"$named"/Headers/$name.h
//
//  $name.h
//  $name
//

#import <Foundation/Foundation.h>

//! Project version number for $name.
FOUNDATION_EXPORT double ${name}VersionNumber;

//! Project version string for $name.
FOUNDATION_EXPORT const unsigned char ${name}VersionString[];

// In this header, you should import all the public headers of your framework using statements like #import <$name/PublicHeader.h>

EOF
until [[ $# -eq 0 ]]; do
  printf '#import "'"$1"'"\n' "$1" >>"$named"/Headers/$name.h
  shift
done
