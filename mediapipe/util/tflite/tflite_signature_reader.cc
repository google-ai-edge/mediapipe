#include "mediapipe/util/tflite/tflite_signature_reader.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {

namespace {

// Flips the key-value pairs in a map.
absl::flat_hash_map<uint32_t, std::string> FlipKVInMap(
    const std::map<std::string, uint32_t>& map) {
  absl::flat_hash_map<uint32_t, std::string> flipped;
  for (const auto& kv : map) {
    flipped[kv.second] = kv.first;
  }
  return flipped;
}

}  // namespace

absl::StatusOr<std::pair<TensorNames, TensorNames>>
GetInputOutputTensorNamesFromTfliteSignature(
    const tflite::Interpreter& interpreter,
    std::optional<std::string> signature_key) {
  std::vector<const std::string*> model_signature_keys =
      interpreter.signature_keys();
  if (model_signature_keys.empty()) {
    return absl::InvalidArgumentError("No signatures found.");
  }
  if (!signature_key.has_value() && model_signature_keys.size() > 1) {
    return absl::InvalidArgumentError(
        "Model contains multiple signatures but no signature key specified.");
  }
  const std::string signature_key_str =
      signature_key.has_value() ? *signature_key : *model_signature_keys[0];

  const absl::flat_hash_map<uint32_t, std::string>
      model_input_tensor_id_to_name_map =
          FlipKVInMap(interpreter.signature_inputs(signature_key_str.c_str()));
  const absl::flat_hash_map<uint32_t, std::string>
      model_output_tensor_id_to_name_map =
          FlipKVInMap(interpreter.signature_outputs(signature_key_str.c_str()));

  // Maps the model input and outputs to internal model tensor ids.
  const std::vector<int>& model_input_tensor_ids = interpreter.inputs();
  const std::vector<int>& model_output_tensor_ids = interpreter.outputs();

  std::vector<std::string> input_names;
  input_names.reserve(model_input_tensor_ids.size());
  for (int i = 0; i < model_input_tensor_ids.size(); ++i) {
    const auto it =
        model_input_tensor_id_to_name_map.find(model_input_tensor_ids[i]);
    if (it == model_input_tensor_id_to_name_map.end()) {
      return absl::InternalError(absl::StrCat("Input tensor id ",
                                              model_input_tensor_ids[i],
                                              " not found in signature."));
    }
    input_names.push_back(it->second);
  }

  std::vector<std::string> output_names;
  output_names.reserve(model_output_tensor_ids.size());
  for (int i = 0; i < model_output_tensor_ids.size(); ++i) {
    const auto it =
        model_output_tensor_id_to_name_map.find(model_output_tensor_ids[i]);
    if (it == model_output_tensor_id_to_name_map.end()) {
      return absl::InternalError(absl::StrCat("Output tensor id ",
                                              model_output_tensor_ids[i],
                                              " not found in signature."));
    }
    output_names.push_back(it->second);
  }
  return std::make_pair(input_names, output_names);
}

}  // namespace mediapipe
