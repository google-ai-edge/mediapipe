#!/bin/bash
# Copyright 2019 The MediaPipe Authors.
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
# $ chmod +x ./setup_android_sdk_and_ndk.sh
# $ ./setup_android_sdk_and_ndk.sh ~/Android/Sdk ~/Android/Ndk r18b

set -e

if [ "$(uname)" == "Darwin" ]; then
  platform="darwin"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  platform="linux"
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
  echo "Warning: ndk_version (argument 3) is not specified. Fallback to r18b."
  ndk_version="r18b"
fi

if [ -d "$android_sdk_path" ]
then
  echo "Warning: android_sdk_path is non empty. Android SDK Installation will be ignored."
else
  rm -rf /tmp/android_sdk/
  mkdir  /tmp/android_sdk/
  curl https://dl.google.com/android/repository/sdk-tools-${platform}-4333796.zip -o /tmp/android_sdk/android_sdk.zip
  unzip /tmp/android_sdk/android_sdk.zip -d /tmp/android_sdk/
  mkdir -p $android_sdk_path
  /tmp/android_sdk/tools/bin/sdkmanager --update
  /tmp/android_sdk/tools/bin/sdkmanager "build-tools;29.0.1" "platform-tools" "platforms;android-29" --sdk_root=${android_sdk_path}
  rm -rf /tmp/android_sdk/
  echo "Android SDK is now installed. Consider setting \$ANDROID_HOME environment variable to be ${android_sdk_path}"
fi

if [ -d "${android_ndk_path}/android-ndk-${ndk_version}" ]
then
  echo "Warning: android_ndk_path is non empty. Android NDK Installation will be ignored."
else
  rm -rf /tmp/android_ndk/
  mkdir /tmp/android_ndk/
  curl https://dl.google.com/android/repository/android-ndk-${ndk_version}-${platform}-x86_64.zip -o /tmp/android_ndk/android_ndk.zip
  mkdir -p ${android_ndk_path}/android-ndk-${ndk_version}
  unzip /tmp/android_ndk/android_ndk.zip -d ${android_ndk_path}
  rm -rf /tmp/android_ndk/
  echo "Android NDK is now installed. Consider setting \$ANDROID_NDK_HOME environment variable to be ${android_ndk_path}/android-ndk-${ndk_version}"
fi

echo "Set android_ndk_repository and android_sdk_repository in WORKSPACE"
workspace_file="$( cd "$(dirname "$0")" ; pwd -P )"/WORKSPACE

ndk_block=$(grep -n 'android_ndk_repository(' $workspace_file | awk -F  ":" '{print $1}')
ndk_path_line=$((ndk_block+2))'i'
sdk_block=$(grep -n 'android_sdk_repository(' $workspace_file | awk -F  ":" '{print $1}')
sdk_path_line=$((sdk_block+3))'i'

if [ $platform == "darwin" ]; then
  sed -i -e "$ndk_path_line\\
  \ \ \ \ path = \"${android_ndk_path}/android-ndk-${ndk_version}\",
  " $workspace_file
  sed -i -e "$sdk_path_line\\
  \ \ \ \ path = \"${android_sdk_path}\",
  " $workspace_file
elif [ $platform == "linux" ]; then
  sed -i "$ndk_path_line \    path = \"${android_ndk_path}/android-ndk-${ndk_version}\"," $workspace_file
  sed -i "$sdk_path_line \    path = \"${android_sdk_path}\"," $workspace_file
fi

echo "Done"
