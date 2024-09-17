#!/bin/bash
#
set -e

export QNN_SDK_ROOT=third_party/qairt/

# Build test and copy it to the device
bazel build --android_ndk_min_sdk_version=30 //mediapipe/calculators/tensor:inference_calculator_qnn_test --config=android_arm64;
adb push bazel-bin/mediapipe/calculators/tensor/inference_calculator_qnn_test /data/local/tmp/qnn_delegate/

# Push QNN libraries
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnTFLiteDelegate.so /data/local/tmp/qnn_delegate/
# Push QNN delegate
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so /data/local/tmp/qnn_delegate/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so /data/local/tmp/qnn_delegate/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so /data/local/tmp/qnn_delegate/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV68Stub.so /data/local/tmp/qnn_delegate/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV69Stub.so /data/local/tmp/qnn_delegate/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV73Stub.so /data/local/tmp/qnn_delegate/
adb push $QNN_SDK_ROOT/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so /data/local/tmp/qnn_delegate/
adb push $QNN_SDK_ROOT/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so /data/local/tmp/qnn_delegate/
adb push $QNN_SDK_ROOT/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so /data/local/tmp/qnn_delegate/
adb push  $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV75Stub.so /data/local/tmp/qnn_delegate
adb push  $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so /data/local/tmp/qnn_delegate/
adb push  $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75.so /data/local/tmp/qnn_delegate/

# Run test
adb shell "cd /data/local/tmp/qnn_delegate; LD_LIBRARY_PATH="/data/local/tmp/qnn_delegate/" ./inference_calculator_qnn_test" --alsologtostderr
