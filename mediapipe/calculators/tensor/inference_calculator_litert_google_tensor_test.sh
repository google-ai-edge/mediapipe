#!/bin/bash
#
set -e

ADB=($(which adb))

# Override the `adb` command
adb() {
  "${ADB[@]}" "$@"
}

# Build Google Tensor dispatch library
bazel build //third_party/odml/litert/litert/vendors/google_tensor/dispatch:dispatch_api_so --config=android_arm64 --//third_party/android/ndk:min_sdk_version=26
adb push bazel-bin/third_party/odml/litert/litert/vendors/google_tensor/dispatch/libLiteRtDispatch_GoogleTensor.so /data/local/tmp/
bazel build //third_party/odml/litert/litert/c:litert_runtime_c_api_so --config=android_arm64 --//third_party/android/ndk:min_sdk_version=26
adb push bazel-bin/third_party/odml/litert/litert/c/libLiteRt.so /data/local/tmp/

# Build test and copy it to the device
bazel build //mediapipe/calculators/tensor:inference_calculator_litert_test --copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1 --config=android_arm64 --//third_party/android/ndk:min_sdk_version=26
adb push bazel-bin/mediapipe/calculators/tensor/inference_calculator_litert_test /data/local/tmp;

# adb push tflite model to the device
adb push mediapipe/app/aimatter/segmentation/test_data/simple_model_npu_google_tensor_precompiled.tflite /data/local/tmp/

# Run test
adb shell "cd /data/local/tmp; LD_LIBRARY_PATH=. ./inference_calculator_litert_test --model_filename=simple_model_npu_google_tensor_precompiled.tflite"
