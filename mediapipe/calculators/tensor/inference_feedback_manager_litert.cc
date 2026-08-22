#include "mediapipe/calculators/tensor/inference_feedback_manager_litert.h"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/c/litert_model_types.h"               // from @litert
#include "litert/cc/internal/litert_extended_model.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"           // from @litert
#include "litert/cc/litert_macros.h"                   // from @litert
#include "litert/cc/litert_model.h"                    // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"       // from @litert
#include "litert/cc/litert_tensor_buffer.h"            // from @litert
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/tflite_signature_reader.h"

namespace mediapipe {

namespace {

// TODO: Move to a litert utility file.
bool LiteRtTensorSpecEqual(const litert::Tensor& a, const litert::Tensor& b) {
  if (!(a.ElementType() == b.ElementType() && a.TypeId() == b.TypeId() &&
        a.HasQuantization() == b.HasQuantization())) {
    return false;
  }

  if (!a.HasQuantization()) {
    return true;
  }

  if (a.QTypeId() == kLiteRtQuantizationPerTensor) {
    return a.PerTensorQuantization().scale == b.PerTensorQuantization().scale &&
           a.PerTensorQuantization().zero_point ==
               b.PerTensorQuantization().zero_point;
  } else if (a.QTypeId() == kLiteRtQuantizationPerChannel) {
    // Check that the quantization dimension and number of channels are the
    // same.
    if (!(a.PerChannelQuantization().quantized_dimension ==
              b.PerChannelQuantization().quantized_dimension &&
          a.PerChannelQuantization().num_channels ==
              b.PerChannelQuantization().num_channels)) {
      return false;
    }
    for (int i = 0; i < a.PerChannelQuantization().num_channels; ++i) {
      if (a.PerChannelQuantization().scales[i] !=
              b.PerChannelQuantization().scales[i] ||
          a.PerChannelQuantization().zero_points[i] !=
              b.PerChannelQuantization().zero_points[i]) {
        return false;
      }
    }
  }
  return true;
}

absl::flat_hash_map<std::string, int> CreateNameToIndexMap(
    const std::vector<std::string>& names) {
  absl::flat_hash_map<std::string, int> name_to_index_map;
  for (int i = 0; i < names.size(); ++i) {
    name_to_index_map[names[i]] = i;
  }
  return name_to_index_map;
}

// TODO: Move to a litert utility file.
bool IsDynamic(const litert::Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto ranked_tensor_type, tensor.RankedTensorType());
  absl::Span<const int> shape = ranked_tensor_type.Layout().Dimensions();
  return absl::c_linear_search(shape, -1);
}

}  // namespace

// Initializes the feedback manager: Performs basic sanity checks on feedback
// tensor configuration, and initializes feedback tensor buffers.
// The passed-in pointers (subgraph, model, compiled_model) are only used to
// create the feedback tensor buffers, and are not stored after Init.
absl::Status InferenceFeedbackManagerLiteRt::Init(
    const InferenceCalculatorOptions::InputOutputConfig& io_config,
    const InputOutputTensorNames& input_output_tensor_names,
    const litert::Subgraph* subgraph, const litert::Model* model,
    const litert::CompiledModel* compiled_model, int signature_index) {
  StoreFeedbackTensorNames(io_config);

  ABSL_ASSIGN_OR_RETURN(feedback_tensor_indices_links_,
                        ConvertSignatureTensorNamesToModelIndices(
                            io_config, input_output_tensor_names));

  for (const auto& link : feedback_tensor_indices_links_) {
    const auto [output_unused_iter, output_was_inserted] =
        feedback_output_indices_.insert(link.from_idx);
    RET_CHECK(output_was_inserted) << "Feedback output tensors must be unique.";
    litert::SubgraphOutputs subgraph_outputs = subgraph->Outputs();
    const litert::Tensor& from_tensor = subgraph_outputs[link.from_idx];

    RET_CHECK(!IsDynamic(from_tensor))
        << "Feedback output tensors must not be dynamic.";
    const auto [input_unused_iter, input_was_inserted] =
        feedback_input_indices_.insert(link.to_idx);
    RET_CHECK(input_was_inserted) << "Feedback input tensors must be unique.";

    litert::SubgraphInputs subgraph_inputs = subgraph->Inputs();
    const litert::Tensor& to_tensor = subgraph_inputs[link.to_idx];

    RET_CHECK(!IsDynamic(to_tensor))
        << "Feedback input tensors must not be dynamic.";

    RET_CHECK(LiteRtTensorSpecEqual(from_tensor, to_tensor))
        << "Feedback tensors must have the same spec.";
  }

  // Populate input_tensor_to_model_indices_ which maps InferenceRunner input
  // tensors indices to the model input indices.
  input_tensor_to_model_indices_.reserve(subgraph->Inputs().size());
  for (int i = 0; i < subgraph->Inputs().size(); ++i) {
    if (!feedback_input_indices_.contains(i)) {
      input_tensor_to_model_indices_.push_back(i);
    }
  }
  return CreateFeedbackTensorBuffers(subgraph, model, compiled_model,
                                     signature_index);
}

// static
absl::StatusOr<
    std::vector<InferenceFeedbackManagerLiteRt::TensorFeedbackIndicesLink>>
InferenceFeedbackManagerLiteRt::ConvertSignatureTensorNamesToModelIndices(
    const InferenceCalculatorOptions::InputOutputConfig& io_config,
    const InputOutputTensorNames& input_output_tensor_names_map) {
  std::vector<TensorFeedbackIndicesLink> indices_links;
  if (input_output_tensor_names_map.empty() ||
      input_output_tensor_names_map.size() > 1) {
    // Fail gracefully by returning an empty TensorFeedbackIndicesLink list if
    // SignatureDef is not available or not supported.
    ABSL_LOG(WARNING)
        << "Feedback manager requires a model with a single signature "
           "inference. Disabling support for feedback tensors.";
    return indices_links;
  }
  // Obtain reference to single-signature in input_output_tensor_names_map.
  const auto& input_output_tensor_names =
      input_output_tensor_names_map.begin()->second;

  const auto input_name_to_index_map =
      CreateNameToIndexMap(input_output_tensor_names.input_tensor_names);
  const auto output_name_to_index_map =
      CreateNameToIndexMap(input_output_tensor_names.output_tensor_names);

  // Create a set of all input/output tensor names used for InferenceCalculator
  // I/O mapping.
  absl::flat_hash_set<std::string> input_output_mapping_tensor_names;
  for (const auto& name : io_config.input_tensor_names_map().tensor_names()) {
    input_output_mapping_tensor_names.insert(name);
  }
  for (const auto& name : io_config.output_tensor_names_map().tensor_names()) {
    input_output_mapping_tensor_names.insert(name);
  }

  for (const auto& link : io_config.feedback_tensor_links()) {
    RET_CHECK(!input_output_mapping_tensor_names.contains(
        link.from_output_tensor_name()))
        << absl::StrFormat(
               "Feedback output tensor [%s] cannot be used for input/output "
               "mapping. Input/output mapping tensor names: [%s]",
               link.from_output_tensor_name(),
               absl::StrJoin(input_output_mapping_tensor_names, ", "));
    RET_CHECK(!input_output_mapping_tensor_names.contains(
        link.to_input_tensor_name()))
        << absl::StrFormat(
               "Feedback input tensor [%s] cannot be used for input/output "
               "mapping. Input/output mapping tensor names: [%s]",
               link.to_input_tensor_name(),
               absl::StrJoin(input_output_mapping_tensor_names, ", "));
    TensorFeedbackIndicesLink indices_link;
    auto from_it =
        output_name_to_index_map.find(link.from_output_tensor_name());
    RET_CHECK(from_it != output_name_to_index_map.end())
        << "Output tensor name not found: " << link.from_output_tensor_name();
    auto to_it = input_name_to_index_map.find(link.to_input_tensor_name());
    RET_CHECK(to_it != input_name_to_index_map.end())
        << "Input tensor name not found: " << link.to_input_tensor_name();
    indices_link.from_idx = from_it->second;
    indices_link.to_idx = to_it->second;
    indices_links.push_back(indices_link);
  }
  return indices_links;
}

void InferenceFeedbackManagerLiteRt::StoreFeedbackTensorNames(
    const mediapipe::InferenceCalculatorOptions::InputOutputConfig& io_config) {
  for (const auto& link : io_config.feedback_tensor_links()) {
    feedback_tensor_map_[link.to_input_tensor_name()] =
        link.from_output_tensor_name();
  }
}

bool InferenceFeedbackManagerLiteRt::IsFeedbackInputTensorAtIndex(
    int idx) const {
  return feedback_input_indices_.contains(idx);
}

bool InferenceFeedbackManagerLiteRt::IsFeedbackOutputTensorAtIndex(
    int idx) const {
  return feedback_output_indices_.contains(idx);
}

int InferenceFeedbackManagerLiteRt::GetNumberOfNonFeedbackInputTensors() const {
  return input_tensor_to_model_indices_.size();
}

int InferenceFeedbackManagerLiteRt::GetNumberOfFeedbackTensors() const {
  return feedback_tensor_indices_links_.size();
}

absl::Status InferenceFeedbackManagerLiteRt::CreateFeedbackTensorBuffers(
    const litert::Subgraph* subgraph, const litert::Model* model,
    const litert::CompiledModel* compiled_model, int signature_index) {
  // TODO: Create dedicated input and output tensor buffers
  // for feedback tensors and perform an actual swap, instead of 'Duplicating'
  // the input buffer.  It's hard to verify that the 'Duplicated' approach will
  // actually work for all OPs, use cases and accelerators.
  const litert::SubgraphInputs& model_input_tensors = subgraph->Inputs();
  LITERT_ASSIGN_OR_RETURN(
      std::vector<absl::string_view> model_input_tensor_names,
      model->GetSignatureInputNames(signature_index));

  LITERT_ASSIGN_OR_RETURN(auto signature, model->GetSignature(signature_index));

  for (int i = 0; i < model_input_tensors.size(); ++i) {
    if (!IsFeedbackInputTensorAtIndex(i)) {
      continue;
    }
    const std::string& model_input_tensor_name =
        std::string(model_input_tensor_names[i]);
    // Feedback tensors are stripped from the InferenceRunner input.
    // We need to create them if they are not already created.
    if (feedback_tensor_buffers_.contains(model_input_tensor_name)) {
      return absl::InternalError(
          "Feedback tensor buffer already exists for tensor: " +
          model_input_tensor_name);
    }
    // We use the compiled model to create the tensor buffer because it will use
    // the correct buffer type based on the compiled model options.
    LITERT_ASSIGN_OR_RETURN(litert::TensorBuffer feedback_input_tensor_buffer,
                            compiled_model->CreateInputBuffer(
                                signature.Key(), model_input_tensor_names[i]));

    // Zero out the feedback input tensor buffer.
    LITERT_ASSIGN_OR_RETURN(auto feedback_input_tensor_buffer_size,
                            feedback_input_tensor_buffer.Size());
    LITERT_ASSIGN_OR_RETURN(auto feedback_input_tensor_buffer_lock_and_addr,
                            ::litert::TensorBufferScopedLock::Create(
                                feedback_input_tensor_buffer,
                                ::litert::TensorBuffer::LockMode::kWrite));
    auto* feedback_input_tensor_buffer_ptr =
        static_cast<char*>(feedback_input_tensor_buffer_lock_and_addr.second);
    memset(feedback_input_tensor_buffer_ptr, 0,
           feedback_input_tensor_buffer_size);

    LITERT_ASSIGN_OR_RETURN(litert::TensorBuffer feedback_output_tensor_buffer,
                            feedback_input_tensor_buffer.Duplicate());

    feedback_tensor_buffers_[model_input_tensor_name] =
        std::move(feedback_input_tensor_buffer);
    feedback_tensor_buffers_
        [feedback_tensor_map_[model_input_tensor_names[i]]] =
            std::move(feedback_output_tensor_buffer);
  }

  return absl::OkStatus();
}

}  // namespace mediapipe
