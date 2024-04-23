#ifndef MEDIAPIPE_UTIL_TFLITE_TFLITE_SIGNATURE_READER_H_
#define MEDIAPIPE_UTIL_TFLITE_TFLITE_SIGNATURE_READER_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {

using TensorNames = std::vector<std::string>;

// Returns names of input and output tensors from TfLite signatures in the order
// the TfLite model / InferenceCalculators expects them. The interpreter
// argument must be initialized with a TfLite model. If the optional signature
// key is provided, the model matching the signature will be queried. Returns an
// error if the signature is not found. If signature_key is not provided, a
// single TfLite signature is expected. Returns pair of input and output tensor
// names.
absl::StatusOr<std::pair<TensorNames, TensorNames>>
GetInputOutputTensorNamesFromTfliteSignature(
    const tflite::Interpreter& interpreter,
    const std::string* signature_key = nullptr);

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_TFLITE_TFLITE_SIGNATURE_READER_H_
