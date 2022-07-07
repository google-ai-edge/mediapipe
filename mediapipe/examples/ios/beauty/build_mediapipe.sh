#!/bin/sh

mkdir -p ./frameworkbuild/MediaPipeFramework/arm64
mkdir -p ./frameworkbuild/MediaPipeFramework/x86_64
mkdir -p ./frameworkbuild/MediaPipeFramework/xcframework

bazel build --config=ios_arm64 mediapipe/examples/ios/facemeshgpu:MediaPipeFramework
./mediapipe/examples/ios/facemeshgpu/patch_ios_framework.sh ./bazel-out/applebin_ios-ios_arm64-fastbuild-ST-2967bd56a867/bin/mediapipe/examples/ios/facemeshgpu/MediaPipeFramework.zip MediaPipeController.h
cp -a ./bazel-out/applebin_ios-ios_arm64-fastbuild-ST-2967bd56a867/bin/mediapipe/examples/ios/facemeshgpu/MediaPipeFramework.framework ./frameworkbuild/MediaPipeFramework/arm64

bazel build --config=ios_x86_64 mediapipe/examples/ios/facemeshgpu:MediaPipeFramework
./mediapipe/examples/ios/facemeshgpu/patch_ios_framework.sh ./bazel-out/applebin_ios-ios_x86_64-fastbuild-ST-2967bd56a867/bin/mediapipe/examples/ios/facemeshgpu/MediaPipeFramework.zip MediaPipeController.h
cp -a ./bazel-out/applebin_ios-ios_x86_64-fastbuild-ST-2967bd56a867/bin/mediapipe/examples/ios/facemeshgpu/MediaPipeFramework.framework ./frameworkbuild/MediaPipeFramework/x86_64

xcodebuild -create-xcframework \
  -framework ./frameworkbuild/MediaPipeFramework/x86_64/MediaPipeFramework.framework \
  -framework ./frameworkbuild/MediaPipeFramework/arm64/MediaPipeFramework.framework \
  -output ./frameworkbuild/MediaPipeFramework/xcframework/MediaPipeFramework.xcframework

