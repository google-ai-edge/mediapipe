#!/bin/bash
#
set -e

ADB=($(which adb))

EXTRA_COPTS=""

HWASAN_ENABLED=false


# Override the `adb` command
adb() {
  "${ADB[@]}" "$@"
}

# Build test and copy it to the device
if $HWASAN_ENABLED; then
bazel build //mediapipe/calculators/tensor:inference_calculator_litert_test $EXTRA_COPTS --config=android_arm64 --features=hwasan -c dbg --dynamic_mode=off --//third_party/android/ndk:min_sdk_version=29
else
bazel build //mediapipe/calculators/tensor:inference_calculator_litert_test $EXTRA_COPTS --config=android_arm64 -c dbg --dynamic_mode=off --//third_party/android/ndk:min_sdk_version=29
fi
adb push bazel-bin/mediapipe/calculators/tensor/inference_calculator_litert_test /data/local/tmp;

# adb push tflite model to the device
bazel build //third_party/odml/litert/litert/test:testdata/simple_model.tflite
adb push bazel-genfiles/third_party/odml/litert/litert/test/testdata/simple_model.tflite /data/local/tmp/

# Run test
if $HWASAN_ENABLED; then
adb shell "cd /data/local/tmp; LD_HWASAN=1 LD_LIBRARY_PATH=. ./inference_calculator_litert_test --model_filename=simple_model.tflite" \
 --gunit_filter=InferenceCalculatorLiteRtTest.LiteRtInferenceWithGpu --alsologtostderr -v=1
else
adb shell "cd /data/local/tmp; LD_LIBRARY_PATH=. ./inference_calculator_litert_test --model_filename=simple_model.tflite" \
 --gunit_filter=InferenceCalculatorLiteRtTest.LiteRtInferenceWithGpu --alsologtostderr -v=1
fi
