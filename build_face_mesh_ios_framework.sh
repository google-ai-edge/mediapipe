#!/bin/sh

sudo rm -rf frameworkbuild

# Create output directories~
mkdir -p ./frameworkbuild/FaceMeshSDK/arm64
# XCFramework is how we're going to use it.
mkdir -p ./frameworkbuild/FaceMeshSDK/xcframework

# Interesting fact. Bazel `build` command stores cached files in `/private/var/tmp/...` folders
# and when you run build, if it finds cached files, it kind of symlinks the files/folders
# into the `bazel-bin` folder found in the project root. So don't be afraid of re-running builds
# because the files are cached.

# build the arm64 binary framework
# bazel build --copt=-fembed-bitcode --apple_bitcode=embedded --config=ios_arm64 mediapipe/examples/ios/facemeshioslib:FaceMeshIOSLibFramework
bazel build -c opt --config=ios_arm64 mediapipe/examples/ios/facemeshioslib:FaceMeshSDK
# use --cxxopt=-O3 to reduce framework size
# bazel build --copt=-O3 --cxxopt=-O3 --config=ios_arm64 mediapipe/examples/ios/facemeshioslib:FaceMeshIOSLibFramework

# The arm64 framework zip will be located at //bazel-bin/mediapipe/examples/ios/facemeshioslib/FaceMeshIOSLibFramework.zip

# Call the framework patcher (First argument = compressed framework.zip, Second argument = header file's name(in this case FaceMeshIOSLib.h))
sudo bash ./mediapipe/examples/ios/facemeshioslib/patch_ios_framework.sh ./bazel-bin/mediapipe/examples/ios/facemeshioslib/FaceMeshSDK.zip FaceMesh.h

# There will be a resulting patched .framework folder at the same directory, this is our arm64 one, we copy it to our arm64 folder
sudo cp -a ./bazel-bin/mediapipe/examples/ios/facemeshioslib/FaceMeshSDK.framework ./frameworkbuild/FaceMeshSDK/arm64

# Create xcframework (because the classic lipo method with normal .framework no longer works (shows Building for iOS Simulator, but the linked and embedded framework was built for iOS + iOS Simulator))

sudo xcodebuild -create-xcframework \
    -framework ./frameworkbuild/FaceMeshSDK/arm64/FaceMeshSDK.framework \
    -output ./frameworkbuild/FaceMeshSDK/xcframework/FaceMeshSDK.xcframework
