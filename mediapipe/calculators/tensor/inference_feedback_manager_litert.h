#ifndef MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_FEEDBACK_MANAGER_LITERT_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_FEEDBACK_MANAGER_LITERT_H_

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "litert/cc/internal/litert_extended_model.h"  // from @litert
#include "litert/cc/litert_common.h"                   // from @litert
#include "litert/cc/litert_compiled_model.h"           // from @litert
#include "litert/cc/litert_expected.h"                 // from @litert
#include "litert/cc/litert_model.h"                    // from @litert
#include "litert/cc/litert_tensor_buffer.h"            // from @litert
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"

namespace mediapipe {

// Feedback tensors are pairs of model output / input tensors where the
// model output is used as model input in the next model invocation. This allows
// to manage a notion of temporal state by continuously feeding from the model's
// output to the model's input during each inference step. The
// InferenceFeedbackManagerLiteRt initializes the feedback input tensors with
// zeros and efficiently swaps them from output to input with zero copies.
class InferenceFeedbackManagerLiteRt {
 public:
  // Initializes the feedback tensors with zeros and generates
  // feedback_tensor_indices_links_. The provided interpreter must outlive the
  // InferenceFeedbackManagerLiteRt instance.
  absl::Status Init(
      const mediapipe::InferenceCalculatorOptions::InputOutputConfig& io_config,
      const InputOutputTensorNames& input_output_tensor_names_map,
      const litert::Subgraph* subgraph, const litert::Model* model,
      const litert::CompiledModel* compiled_model, int signature_index);

  // Returns true if a feedback tensor for the given tensor name is present.
  bool Contains(absl::string_view tensor_name) const {
    return feedback_tensor_buffers_.contains(tensor_name);
  }

  // Returns the feedback tensor buffer for the given tensor name. The tensor
  // buffer is owned by the InferenceFeedbackManagerLiteRt instance. Will return
  // an error if the tensor name is not found.
  absl::StatusOr<const litert::TensorBuffer&> GetFeedbackTensorBuffer(
      absl::string_view tensor_name) {
    if (!Contains(tensor_name)) {
      return absl::InvalidArgumentError("Feedback tensor not found.");
    }
    return (feedback_tensor_buffers_[tensor_name]);
  }

  // Returns the number of expected non-feedback tensors. This can be used to
  // confirm the number of input tensors to the InferenceRunner implementation.
  int GetNumberOfNonFeedbackInputTensors() const;

  //  Returns the number of feedback tensor pairs.
  int GetNumberOfFeedbackTensors() const;

  // Returns true if the tensor at the given index is a feedback input tensor.
  bool IsFeedbackInputTensorAtIndex(int idx) const;

  // Returns true if the tensor at the given index is a feedback output tensor.
  bool IsFeedbackOutputTensorAtIndex(int idx) const;

 private:
  // Creates feedback input tensor buffers.
  absl::Status CreateFeedbackTensorBuffers(
      const litert::Subgraph* subgraph, const litert::Model* model,
      const litert::CompiledModel* compiled_model, int signature_index);

  // Links between feedback tensors defined by model tensor indices.
  struct TensorFeedbackIndicesLink {
    int from_idx;
    int to_idx;
  };

  // Translates the tensor names from the input/output config into the
  // corresponding TfLite tensor indices.
  static absl::StatusOr<std::vector<TensorFeedbackIndicesLink>>
  ConvertSignatureTensorNamesToModelIndices(
      const mediapipe::InferenceCalculatorOptions::InputOutputConfig& io_config,
      const InputOutputTensorNames& input_output_tensor_names_map);

  void StoreFeedbackTensorNames(
      const mediapipe::InferenceCalculatorOptions::InputOutputConfig&
          io_config);

  // List of tensor feedback pairs defined by model tensor indices.
  std::vector<TensorFeedbackIndicesLink> feedback_tensor_indices_links_;

  // Feedback tensors come in pairs. There is a "from" tensor and a "to" tensor,
  // this map stores the reverse mapping from (the "to" tensor to the "from"
  // tensor) to allow us to easily get the output tensor name from the input
  // tensor name when we process the model inputs.
  absl::flat_hash_map<std::string, std::string> feedback_tensor_map_;

  // Maps InferenceRunner input indices to TfLiteModel input indices.
  std::vector<int> input_tensor_to_model_indices_;

  // Set of feedback input model tensor indices.
  absl::flat_hash_set<int> feedback_input_indices_;

  // Set of feedback output model tensor indices.
  absl::flat_hash_set<int> feedback_output_indices_;

  // Map of feedback input tensor signature names to tensor buffer. The
  // inference runner owns the tensor buffer.
  absl::flat_hash_map<std::string, litert::TensorBuffer>
      feedback_tensor_buffers_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_FEEDBACK_MANAGER_LITERT_H_
