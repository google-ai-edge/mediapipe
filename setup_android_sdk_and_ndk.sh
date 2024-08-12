#!/bin/bash
# Copyright 2019-2020 The MediaPipe Authors.
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
# =========================================================================
#
# Script to setup Android SDK and NDK.
# usage:
# $ cd <mediapipe root dir>
# $ bash ./setup_android_sdk_and_ndk.sh ~/Android/Sdk ~/Android/Ndk r26d [--accept-licenses]

set -e

if [ "$(uname)" == "Darwin" ]; then
  platform="darwin"
  platform_android_sdk="mac"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  platform="linux"
  platform_android_sdk="linux"
fi

if [[ $ANDROID_HOME ]] && [[ $ANDROID_NDK_HOME ]]
then
  echo "Found existing \$ANDROID_HOME="$ANDROID_HOME" and \$ANDROID_NDK_HOME="$ANDROID_NDK_HOME
  echo "Bazel will locate Android SDK and NDK automatically."
  exit 0
fi

android_sdk_path=$1
android_ndk_path=$2
ndk_version=$3
licenses=$4

if [ -z $1 ]
then
  echo "Warning: android_sdk_path (argument 1) is not specified. Fallback to ~/Android/Sdk/"
  android_sdk_path=$HOME"/Android/Sdk"
fi

if [ -z $2 ]
then
  echo "Warning: android_ndk_path (argument 2) is not specified. Fallback to ~/Android/Sdk/ndk-bundle/android-ndk-<NDK_VERSION>/"
  android_ndk_path=$HOME"/Android/Sdk/ndk-bundle"
fi

if [ -z $3 ]
then
  echo "Warning: ndk_version (argument 3) is not specified. Fallback to r26d."
  ndk_version="r26d"
fi

if [ -d "$android_sdk_path" ]
then
  echo "Warning: android_sdk_path is non empty. Installation of the Android SDK will be skipped."
else
  rm -rf /tmp/android_sdk/
  mkdir  /tmp/android_sdk/
  curl https://dl.google.com/android/repository/commandlinetools-${platform_android_sdk}-7583922_latest.zip -o /tmp/android_sdk/commandline_tools.zip
  unzip /tmp/android_sdk/commandline_tools.zip -d /tmp/android_sdk/
  mkdir -p $android_sdk_path
  /tmp/android_sdk/cmdline-tools/bin/sdkmanager --update --sdk_root=${android_sdk_path}
  if [ "$licenses" == "--accept-licenses" ]
  then
    yes | /tmp/android_sdk/cmdline-tools/bin/sdkmanager --licenses --sdk_root=${android_sdk_path}
  fi
  /tmp/android_sdk/cmdline-tools/bin/sdkmanager "build-tools;30.0.3" "platform-tools" "platforms;android-30" "extras;android;m2repository" --sdk_root=${android_sdk_path}
  rm -rf /tmp/android_sdk/
  echo "Android SDK is now installed. Consider setting \$ANDROID_HOME environment variable to be ${android_sdk_path}"
fi

if [ -d "${android_ndk_path}/android-ndk-${ndk_version}" ]
then
  echo "Warning: android_ndk_path is non empty. Android NDK Installation will be ignored."
else
  rm -rf /tmp/android_ndk/
  mkdir /tmp/android_ndk/
  curl https://dl.google.com/android/repository/android-ndk-${ndk_version}-${platform}.zip -o /tmp/android_ndk/android_ndk.zip
  mkdir -p ${android_ndk_path}/android-ndk-${ndk_version}
  unzip /tmp/android_ndk/android_ndk.zip -d ${android_ndk_path}
  rm -rf /tmp/android_ndk/
  echo "Android NDK is now installed. Consider setting \$ANDROID_NDK_HOME environment variable to be ${android_ndk_path}/android-ndk-${ndk_version}"
fi

echo "Set android_ndk_repository and android_sdk_repository in WORKSPACE"
workspace_file="$( cd "$(dirname "$0")" ; pwd -P )"/WORKSPACE
echo "android_sdk_repository(name = \"androidsdk\", path = \"${android_sdk_path}\")" >> $workspace_file
echo "android_ndk_repository(name = \"androidndk\", api_level=21, path = \"${android_ndk_path}/android-ndk-${ndk_version}\")" >> $workspace_file
# See https://github.com/bazelbuild/rules_android_ndk/issues/31#issuecomment-1396182185
echo "bind(name = \"android/crosstool\", actual = \"@androidndk//:toolchain\")" >> $workspace_file
echo "Done"
