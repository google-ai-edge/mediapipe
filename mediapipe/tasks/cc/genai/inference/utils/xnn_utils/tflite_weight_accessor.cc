// Copyright 2024 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/tflite_weight_accessor.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/vector.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"
#include "xnnpack.h"  // from @XNNPACK
// clang-format off
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/memory_mapped_file.h"
// clang-format on
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe::tasks::genai::xnn_utils {

using ::mediapipe::tasks::genai::llm_utils::MemoryMappedFile;

TfLiteWeightAccessor::TfLiteWeightAccessor(
    std::shared_ptr<const tflite::Model> tflite_model, char* data)
    : tflite_model_(tflite_model) {
  BuildWeightsMapFromTfliteModel(data);
}

TfLiteWeightAccessor::TfLiteWeightAccessor(absl::string_view filename) {
  std::shared_ptr<MemoryMappedFile> mmap_file =
      MemoryMappedFile::Create(filename).value_or(nullptr);
  if (mmap_file) {
    tflite_model_ = std::shared_ptr<const ::tflite::Model>(
        mmap_file, ::tflite::GetModel(mmap_file->data()));
    BuildWeightsMapFromTfliteModel(static_cast<char*>(mmap_file->data()));
  }
}

void TfLiteWeightAccessor::BuildWeightsMapFromTfliteModel(char* data) {
  if (!tflite_model_) return;
  const flatbuffers::Vector<flatbuffers::Offset<::tflite::Buffer>>& buffers =
      *tflite_model_->buffers();
  for (const tflite::SubGraph* subgraph : *tflite_model_->subgraphs()) {
    for (const tflite::Tensor* tfl_tensor : *subgraph->tensors()) {
      absl::string_view tensor_name = absl::string_view(
          tfl_tensor->name()->data(), tfl_tensor->name()->size());
      Tensor::DimsType dims(tfl_tensor->shape()->begin(),
                            tfl_tensor->shape()->end());
      std::shared_ptr<Tensor> tensor;
      ABSL_DCHECK_LT(tfl_tensor->buffer(), buffers.size());
      const tflite::Buffer& tfl_buffer = *buffers.Get(tfl_tensor->buffer());
      switch (tfl_tensor->type()) {
        case tflite::TensorType::TensorType_FLOAT32:
          tensor = std::make_shared<Tensor>(std::move(dims), xnn_datatype_fp32);
          ABSL_DCHECK_EQ(tfl_buffer.size(),
                         tensor->num_elements * sizeof(float));
          tensor->flat_data =
              std::shared_ptr<char>(tflite_model_, data + tfl_buffer.offset());
          break;
        case tflite::TensorType::TensorType_INT8:
          tensor = std::make_shared<QCTensor>(std::move(dims), /*dim_scale=*/0,
                                              xnn_datatype_qcint8);
          ABSL_DCHECK_EQ(tfl_buffer.size(), tensor->num_elements);
          tensor->flat_data =
              std::shared_ptr<char>(tflite_model_, data + tfl_buffer.offset());
          break;
        case tflite::TensorType::TensorType_INT4:
          tensor = std::make_shared<QCTensor>(std::move(dims), /*dim_scale=*/0,
                                              xnn_datatype_qcint4);
          ABSL_DCHECK_EQ(tfl_buffer.size(), tensor->num_elements / 2);
          tensor->flat_data =
              std::shared_ptr<char>(tflite_model_, data + tfl_buffer.offset());
          break;
        default:
          ABSL_LOG(DFATAL) << "Unsupported tensor type: " << tfl_tensor->type();
          break;
      }
      weights_[tensor_name] = tensor;
    }
  }
}

absl::StatusOr<std::shared_ptr<Tensor>> TfLiteWeightAccessor::LoadWeight(
    absl::string_view tensor_name, Tensor::DimsType expected_dims,
    size_t dim_scale_if_any) const {
  RET_CHECK(tflite_model_);
  if (!weights_.contains(tensor_name)) {
    ABSL_DLOG(WARNING) << "Tensor not found: " << tensor_name;
    return nullptr;
  }
  // qtensor means potentially quantized tensor.
  std::shared_ptr<Tensor> qtensor = weights_.at(tensor_name);
  // Check dimensions.
  {
    bool correct_dimension = true;
    correct_dimension &= (qtensor->dims.size() == expected_dims.size());
    for (int i = 0; i < qtensor->dims.size(); ++i) {
      correct_dimension &= (qtensor->dims[i] == expected_dims[i]);
    }
    if (!correct_dimension) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Dimension mismatch at ", tensor_name, " with expected dimensions [",
          absl::StrJoin(expected_dims, ", "), "] actual dimensions [",
          absl::StrJoin(qtensor->dims, ", "), "]"));
    }
  }
  if (qtensor->datatype == xnn_datatype_fp32) {
    return qtensor;
  }

  // Following are logic for quantized weights.
  std::shared_ptr<QCTensor> result;
  std::string scale_tensor_name =
      absl::StrCat(tensor_name, kQuantizedScaleSuffix);
  if (!weights_.contains(scale_tensor_name)) {
    return absl::NotFoundError(
        absl::StrCat("Scale tensor not found: ", scale_tensor_name));
  }
  std::shared_ptr<Tensor> scale_tensor = weights_.at(scale_tensor_name);
  RET_CHECK_EQ(expected_dims[dim_scale_if_any], scale_tensor->num_elements);
  switch (qtensor->datatype) {
    case xnn_datatype_qcint8:
      result = std::make_shared<QCTensor>(
          std::move(expected_dims), dim_scale_if_any, xnn_datatype_qcint8);
      result->flat_data = qtensor->flat_data;
      result->scale_data = std::shared_ptr<float>(
          scale_tensor->flat_data,
          reinterpret_cast<float*>(scale_tensor->flat_data.get()));
      break;
    case xnn_datatype_qcint4:
      result = std::make_shared<QCTensor>(
          std::move(expected_dims), dim_scale_if_any, xnn_datatype_qcint4);
      result->flat_data = qtensor->flat_data;
      result->scale_data = std::shared_ptr<float>(
          scale_tensor->flat_data,
          reinterpret_cast<float*>(scale_tensor->flat_data.get()));
      break;
    default:
      return absl::InvalidArgumentError("Unsupported tensor type");
      break;
  }
  return result;
}

absl::StatusOr<std::shared_ptr<Tensor>>
TfLiteWeightAccessor::LoadTransposedWeight(absl::string_view tensor_name,
                                           Tensor::DimsType expected_dims,
                                           size_t dim_scale_if_any) const {
  RET_CHECK(tflite_model_);
  RET_CHECK_EQ(expected_dims.size(), 2);
  return LoadWeight(
      tensor_name,
      Tensor::DimsType(expected_dims.rbegin(), expected_dims.rend()),
      1 - dim_scale_if_any);
}

}  // namespace mediapipe::tasks::genai::xnn_utils
