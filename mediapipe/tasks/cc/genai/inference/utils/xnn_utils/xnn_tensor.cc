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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"

#include <fcntl.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/utils.h"
#include "xnnpack.h"  // from @XNNPACK

namespace mediapipe::tasks::genai {
namespace xnn_utils {
namespace {

// Same as numpy isclose()
bool IsClose(float actual, float expected, float atol, float rtol) {
  float tolerance = std::abs(expected * rtol) + std::abs(atol);
  float diff = std::abs(actual - expected);
  return diff <= tolerance;
}

}  // namespace

std::ostream& operator<<(std::ostream& os,
                         const absl::flat_hash_map<std::string, int> map) {
  os << "{";
  int cnt = 0;
  for (const auto& [key, value] : map) {
    os << key << ":" << value;
    if (cnt++ == 0 && map.size() != 1) {
      os << ", ";
    }
  }
  os << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  os << "Tensor{dims=[" << tensor.dims << "], datatype=" << tensor.datatype
     << ", num_elements=" << tensor.num_elements
     << ", metadata=" << tensor.metadata << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const QCTensor& tensor) {
  os << "QCTensor{dims=[" << tensor.dims << "], dim_scale=" << tensor.dim_scale
     << " datatype=" << tensor.datatype
     << ", num_elements=" << tensor.num_elements
     << ", metadata=" << tensor.metadata << "}";
  return os;
}

bool Tensor::operator==(const Tensor& other) const {
  if (dims.size() != other.dims.size()) {
    return false;
  } else if (datatype != other.datatype) {
    return false;
  } else {
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] != other.dims[i]) {
        return false;
      }
    }
  }
  return 0 == memcmp(Data(), other.Data(), ElementSize(num_elements));
}

std::optional<int> Tensor::GetMetadata(absl::string_view key) const {
  if (metadata.contains(key)) {
    return metadata.at(key);
  }
  return std::nullopt;
}

int Tensor::GetMetadata(absl::string_view key, int default_value) const {
  if (metadata.contains(key)) {
    return metadata.at(key);
  }
  return default_value;
}

void Tensor::AllocateBufferIfNeeded() {
  if (!flat_data) {
    auto real_buffer = std::make_shared<std::vector<char>>(
        ElementSize(num_elements) + XNN_EXTRA_BYTES, 0x00);
    elements_capacity = num_elements;
    flat_data = std::shared_ptr<char>(real_buffer, real_buffer->data());
  }
}

void* Tensor::Data() {
  ABSL_DCHECK(flat_data)
      << "If this is weight, you may need to call one of the LoadFrom*()";
  return flat_data.get();
}

std::shared_ptr<Tensor> Tensor::Slice(DimsType offset) {
  ABSL_DCHECK(flat_data);
  ABSL_CHECK_EQ(offset.size(), dims.size()) << offset << " vs. " << dims;
  // offset: [0, k, 0, 0], dims: [1, K, _, _]. dims before k must be 1.
  bool found_non_zero_offset = false;
  int index_k = -1;
  for (int i = 0; i < dims.size(); ++i) {
    if (found_non_zero_offset) {
      ABSL_DCHECK_EQ(offset[i], 0);
    } else if (offset[i] != 0) {
      found_non_zero_offset = true;
      index_k = i;
    }
  }
  ABSL_DCHECK(found_non_zero_offset) << offset;

  return Slice(index_k, offset[index_k]);
}

std::shared_ptr<Tensor> Tensor::Slice(size_t index, size_t offset) {
  size_t num_elements_offset = 1;
  DimsType new_dim = dims;
  for (int i = 0; i < dims.size(); ++i) {
    if (i < index) {
      ABSL_DCHECK_EQ(dims[i], 1);
    } else if (i == index) {
      ABSL_DCHECK_LT(offset, dims[i]);
      num_elements_offset *= offset;
      new_dim[i] = 1;
    } else {
      num_elements_offset *= dims[i];
    }
  }

  auto result =
      std::make_shared<Tensor>(std::move(new_dim), datatype, is_sparse());
  result->flat_data = std::shared_ptr<char>(
      flat_data, flat_data.get() + ElementSize(num_elements_offset));
  result->elements_capacity = result->num_elements;
  return result;
}

Tensor& Tensor::Borrow(std::shared_ptr<Tensor> other, size_t element_offset) {
  ABSL_DCHECK_EQ(datatype, other->datatype);
  ABSL_DCHECK_EQ(dims.size(), other->dims.size());
  flat_data = std::shared_ptr<char>(
      other->flat_data, other->flat_data.get() + ElementSize(element_offset));
  elements_capacity = other->elements_capacity - element_offset;
  return *this;
}

Tensor& Tensor::Resize(DimsType new_dims) {
  ABSL_DCHECK(!new_dims.empty());
  const size_t old_num_elements = num_elements;
  internal_dims = std::move(new_dims);
  internal_num_elements = std::accumulate(dims.begin(), dims.end(), size_t(1),
                                          std::multiplies<size_t>());
  ABSL_DCHECK_NE(internal_num_elements, 0);
  if (num_elements > elements_capacity) {
    auto old_flat_data = std::move(flat_data);
    AllocateBufferIfNeeded();
    memcpy(Data(), old_flat_data.get(), ElementSize(old_num_elements));
  }
  return *this;
}

const void* Tensor::Data() const { return const_cast<Tensor*>(this)->Data(); }

absl::Status Tensor::DefineInSubgraph(xnn_subgraph& subgraph, uint32_t flags) {
  uint32_t id;
  switch (datatype) {
    case xnn_datatype_fp32: {
      RET_CHECK_EQ(xnn_status_success,
                   xnn_define_tensor_value(
                       &subgraph, datatype, dims.size(), dims.data(),
                       /*data=*/nullptr,
                       /*external_id=*/tensor_id(&subgraph), flags, &id));
      break;
    }
    case xnn_datatype_qdint8: {
      // Set num_non_batch_dims=1, the last dim is # of channels, the other dims
      // are flattened and treated as batch size.
      RET_CHECK_EQ(xnn_status_success,
                   xnn_define_dynamically_quantized_tensor_value(
                       &subgraph, datatype, dims.size(),
                       /*num_non_batch_dims=*/1, dims.data(),
                       /*external_id=*/tensor_id(&subgraph), flags, &id))
          << dims;
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported datatype: ", datatype));
  }
  if (tensor_id(&subgraph) == XNN_INVALID_VALUE_ID) {
    RET_CHECK_NE(id, XNN_INVALID_VALUE_ID);
    map_subgraph_to_tensor_id[&subgraph] = id;
  } else {
    RET_CHECK_EQ(id, tensor_id(&subgraph));
  }
  return absl::OkStatus();
}

absl::Status Tensor::DefineAsInput(xnn_subgraph& subgraph) {
  return DefineInSubgraph(subgraph, XNN_VALUE_FLAG_EXTERNAL_INPUT);
}

absl::Status Tensor::DefineAsOutput(xnn_subgraph& subgraph) {
  return DefineInSubgraph(subgraph, XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
}

absl::Status Tensor::DefineAsIntermediateTensor(xnn_subgraph& subgraph) {
  RET_CHECK_EQ(tensor_id(&subgraph), XNN_INVALID_VALUE_ID);
  return DefineInSubgraph(subgraph, 0);
}

absl::Status Tensor::DefineWeight(xnn_subgraph& subgraph, uint32_t flags) {
  uint32_t assigned_tensor_id;
  RET_CHECK_EQ(xnn_status_success,
               xnn_define_tensor_value(
                   &subgraph, datatype, dims.size(), dims.data(), Data(),
                   tensor_id(&subgraph), flags, &assigned_tensor_id));
  RET_CHECK_NE(assigned_tensor_id, XNN_INVALID_VALUE_ID);
  map_subgraph_to_tensor_id[&subgraph] = assigned_tensor_id;
  return absl::OkStatus();
}

absl::Status Tensor::DefineWeight(xnn_subgraph& subgraph) {
  RET_CHECK_EQ(tensor_id(&subgraph), XNN_INVALID_VALUE_ID);
  return DefineWeight(subgraph, 0);
}

uint32_t Tensor::tensor_id(xnn_subgraph_t subgraph) {
  if (map_subgraph_to_tensor_id.contains(subgraph)) {
    return map_subgraph_to_tensor_id.at(subgraph);
  }
  return XNN_INVALID_VALUE_ID;
}

void Tensor::set_tensor_id(xnn_subgraph_t subgraph, uint32_t id) {
  map_subgraph_to_tensor_id[subgraph] = id;
}

absl::Status Tensor::LoadFromBuffer(const void* buffer) {
  AllocateBufferIfNeeded();
  memcpy(Data(), buffer, ElementSize(num_elements));
  return absl::OkStatus();
}

absl::Status Tensor::LoadFromVec(const std::vector<float>& data,
                                 bool exact_match) {
  AllocateBufferIfNeeded();
  if (exact_match) {
    RET_CHECK_EQ(ElementSize(num_elements), data.size() * sizeof(float));
  }

  memcpy(Data(), data.data(), data.size() * sizeof(float));

  return absl::OkStatus();
}

absl::Status Tensor::DumpToBuffer(void* buffer) {
  memcpy(buffer, Data(), ElementSize(num_elements));
  return absl::OkStatus();
}

absl::Status Tensor::DumpToVec(std::vector<float>& out_data, bool exact_match) {
  if (exact_match) {
    RET_CHECK_EQ(ElementSize(num_elements), out_data.size() * sizeof(float));
  } else {
    out_data.resize(num_elements);
  }
  memcpy(out_data.data(), Data(), ElementSize(num_elements));
  return absl::OkStatus();
}

absl::Status Tensor::DumpToFile(absl::string_view file_path) {
  return mediapipe::file::SetContents(
      file_path, absl::string_view(flat_data.get(), ElementSize(num_elements)));
}

absl::Status Tensor::LoadFromFile(absl::string_view file_path, bool use_mmap,
                                  bool exact_match) {
  const size_t expected_size_in_bytes =
      exact_match ? ElementSize(num_elements) : 0;

  size_t buffer_size;
  MP_ASSIGN_OR_RETURN(auto tmp_flat_data,
                      LoadBufferFromFile(file_path, &buffer_size, use_mmap,
                                         expected_size_in_bytes));
  if (!flat_data) {
    flat_data = tmp_flat_data;
    elements_capacity = num_elements;
  } else {
    memcpy(flat_data.get(), tmp_flat_data.get(), buffer_size);
  }
  source = mediapipe::file::Basename(file_path);

  return absl::OkStatus();
}

std::shared_ptr<Tensor> Tensor::Transpose() {
  ABSL_DCHECK_EQ(dims.size(), 2);
  DimsType out_dims{dims.rbegin(), dims.rend()};
  auto result =
      std::make_shared<Tensor>(std::move(out_dims), datatype, is_sparse());
  result->AllocateBufferIfNeeded();
  xnn_status s;
  const DimsType perm{1, 0};
  if (datatype == xnn_datatype_fp32) {
    s = xnn_run_transpose_nd_x32(Data(), result->Data(), dims.size(),
                                 dims.data(), perm.data(),
                                 /*flags=*/0, /*threadpool=*/nullptr);
  } else {
    ABSL_LOG(FATAL) << "Need update to support new type";
  }
  ABSL_DCHECK_EQ(s, xnn_status_success);
  return (s == xnn_status_success) ? result : nullptr;
}

absl::StatusOr<std::shared_ptr<Tensor>> Tensor::ConvertToF32() {
  auto result = std::make_shared<Tensor>(dims, xnn_datatype_fp32, is_sparse());
  MP_RETURN_IF_ERROR(result->LoadFromBuffer(Data()));
  return result;
}

absl::StatusOr<::mediapipe::Tensor> Tensor::ConvertToMediapipeTensor() {
  RET_CHECK_EQ(datatype, xnn_datatype_fp32) << "Try ConvertToF32 then convert";
  ::mediapipe::Tensor mp_tensor(
      ::mediapipe::Tensor::ElementType::kFloat32,
      ::mediapipe::Tensor::Shape(std::vector<int>(dims.begin(), dims.end())));
  void* mp_tensor_buffer = mp_tensor.GetCpuWriteView().buffer<float>();
  std::memcpy(mp_tensor_buffer, Data(), ElementSize(num_elements));
  return mp_tensor;
}

absl::Status Tensor::IsCloseTo(const Tensor& expected_tensor, float atol,
                               float rtol) {
  RET_CHECK_EQ(datatype, xnn_datatype_fp32) << "Try ConvertToF32";
  RET_CHECK_EQ(dims.size(), expected_tensor.dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    RET_CHECK_EQ(dims[i], expected_tensor.dims[i])
        << dims << " v.s. " << expected_tensor.dims;
  }

  const auto* actual = static_cast<const float*>(Data());
  const auto* expected = static_cast<const float*>(expected_tensor.Data());
  std::stringstream ss;

  size_t total_print = 0;
#define LOG_AND_COUNT() \
  ++total_print;        \
  ss << "\n" << i << ", expect: " << expected[i] << ", actual: " << actual[i];

  for (size_t i = 0; i < expected_tensor.num_elements; ++i) {
    if (std::isnan(actual[i]) || std::isnan(expected[i])) {
      LOG_AND_COUNT()
    } else if (!IsClose(actual[i], expected[i], atol, rtol)) {
      LOG_AND_COUNT()
    }
    if (total_print > 100) {
      ss << "\nand more...";
      return absl::InternalError(ss.str());
    }
  }
#undef LOG_AND_COUNT

  return absl::OkStatus();
}

absl::Status QCTensor::LoadFromFile(absl::string_view quantized_weight_filename,
                                    absl::string_view scale_filename,
                                    bool use_mmap, bool exact_match) {
  size_t scale_element_size = dims[dim_scale];

  size_t buffer_size, scale_buffer_size;
  MP_ASSIGN_OR_RETURN(
      auto tmp_flat_data,
      LoadBufferFromFile(quantized_weight_filename, &buffer_size, use_mmap,
                         exact_match ? ElementSize(num_elements) : 0));
  MP_ASSIGN_OR_RETURN(
      auto tmp_scale_data,
      LoadBufferFromFile<float>(
          scale_filename, &scale_buffer_size, use_mmap,
          exact_match ? scale_element_size * sizeof(float) : 0));
  if (!flat_data) {
    flat_data = tmp_flat_data;
    scale_data = tmp_scale_data;
    elements_capacity = num_elements;
  } else {
    memcpy(flat_data.get(), tmp_flat_data.get(), buffer_size);
    memcpy(scale_data.get(), tmp_scale_data.get(), scale_buffer_size);
  }
  source = mediapipe::file::Basename(quantized_weight_filename);

  return absl::OkStatus();
}

absl::Status QCTensor::DumpToFile(absl::string_view file_path) {
  MP_RETURN_IF_ERROR(mediapipe::file::SetContents(
      file_path,
      absl::string_view(flat_data.get(), ElementSize(num_elements))));
  return mediapipe::file::SetContents(
      absl::StrCat(file_path, kQuantizedScaleSuffix),
      absl::string_view(reinterpret_cast<char*>(scale_data.get()),
                        dims[dim_scale] * sizeof(float)));
}

absl::Status QCTensor::DefineWeight(xnn_subgraph& subgraph, uint32_t flags) {
  uint32_t assigned_tensor_id;
  RET_CHECK_EQ(xnn_status_success,
               xnn_define_channelwise_quantized_tensor_value_v2(
                   &subgraph, datatype, zero_point, scale_data.get(),
                   dims.size(), dim_scale, dims.data(), Data(),
                   XNN_INVALID_VALUE_ID, flags, &assigned_tensor_id))
      << *this;
  RET_CHECK_NE(assigned_tensor_id, XNN_INVALID_VALUE_ID);
  map_subgraph_to_tensor_id[&subgraph] = assigned_tensor_id;
  return absl::OkStatus();
}

void QCTensor::AllocateBufferIfNeeded() {
  Tensor::AllocateBufferIfNeeded();
  if (!scale_data) {
    auto real_buffer = std::make_shared<std::vector<float>>();
    real_buffer->resize(dims[dim_scale]);
    scale_data = std::shared_ptr<float>(real_buffer, real_buffer->data());
  }
}

std::shared_ptr<Tensor> QCTensor::Transpose() {
  ABSL_DCHECK_EQ(dims.size(), 2);
  size_t channel_size = dims[dim_scale];
  DimsType out_dims{dims.rbegin(), dims.rend()};
  auto result = std::make_shared<QCTensor>(std::move(out_dims), 1 - dim_scale,
                                           datatype, is_sparse());
  result->zero_point = zero_point;
  result->AllocateBufferIfNeeded();
  memcpy(result->scale_data.get(), scale_data.get(),
         channel_size * sizeof(float));
  xnn_status s;
  const DimsType perm{1, 0};
  switch (datatype) {
    case xnn_datatype_qcint8:
      s = xnn_run_transpose_nd_x8(Data(), result->Data(), dims.size(),
                                  dims.data(), perm.data(),
                                  /*flags=*/0, /*threadpool=*/nullptr);
      break;
    case xnn_datatype_qcint4: {
      std::vector<uint8_t> unpacked =
          xnn_utils::UnpackInt8ToInt4(
              absl::Span<uint8_t>(static_cast<uint8_t*>(Data()),
                                  ElementSize(num_elements)))
              .value();
      std::vector<uint8_t> transposed_unpacked(unpacked.size());
      s = xnn_run_transpose_nd_x8(unpacked.data(), transposed_unpacked.data(),
                                  dims.size(), dims.data(), perm.data(),
                                  /*flags=*/0, /*threadpool=*/nullptr);
      std::vector<uint8_t> packed =
          xnn_utils::PackInt4ToInt8(
              absl::Span<uint8_t>(transposed_unpacked.data(),
                                  transposed_unpacked.size()))
              .value();
      ABSL_CHECK_OK(result->LoadFromBuffer(packed.data()));
      break;
    }
    default:
      ABSL_LOG(FATAL) << "Need update to support new type";
  }
  ABSL_DCHECK_EQ(s, xnn_status_success);
  return (s == xnn_status_success) ? result : nullptr;
}

absl::StatusOr<std::shared_ptr<Tensor>> QCTensor::ConvertToF32() {
  RET_CHECK_EQ(dims.size(), 2)
      << "QCTensor is usually weight for FullConn" << dims;
  auto result = std::make_shared<Tensor>(dims, xnn_datatype_fp32, is_sparse());
  MP_RETURN_IF_ERROR(result->LoadFromVec({}, /*exact_match=*/false));
  float* scaled_data = result->DataAs<float>();

  switch (datatype) {
    case xnn_datatype_qcint8: {
      auto* quantized_data = DataAs<int8_t>();
      for (size_t i = 0; i < dims[0]; ++i) {
        for (size_t j = 0; j < dims[1]; ++j) {
          float scale = dim_scale ? scale_data.get()[j] : scale_data.get()[i];
          *scaled_data = *quantized_data * scale;
          ++scaled_data;
          ++quantized_data;
        }
      }
      break;
    }
    case xnn_datatype_qcint4: {
      uint8_t* quantized_data = static_cast<uint8_t*>(Data());
      RET_CHECK_EQ(dims[1] % 2, 0);
      for (size_t i = 0; i < dims[0]; ++i) {
        for (size_t j = 0; j < dims[1] / 2; ++j) {
          {
            // first element
            float scale =
                dim_scale ? scale_data.get()[j * 2] : scale_data.get()[i];
            *scaled_data =
                (static_cast<int32_t>(*quantized_data & 0x0f) - zero_point) *
                scale;
            ++scaled_data;
          }
          {
            // second element
            float scale =
                dim_scale ? scale_data.get()[j * 2 + 1] : scale_data.get()[i];
            *scaled_data =
                (static_cast<int32_t>(*quantized_data >> 4) - zero_point) *
                scale;
            ++scaled_data;
          }
          ++quantized_data;
        }
      }
      break;
    }
    default: {
      return absl::InvalidArgumentError("Need update to support new type");
    }
  }

  return result;
}

std::shared_ptr<Tensor> QCTensor::Slice(size_t index, size_t offset) {
  ABSL_CHECK_LE(index, 1);
  ABSL_CHECK_EQ(index, dim_scale);

  std::shared_ptr<QCTensor> result;
  if (index == 0) {
    result = std::make_shared<QCTensor>(DimsType{1, dims[1]}, dim_scale,
                                        datatype, is_sparse());
    result->flat_data = std::shared_ptr<char>(
        flat_data, flat_data.get() + ElementSize(dims[1] * offset));
    result->scale_data = std::make_shared<float>(*(scale_data.get() + offset));
  } else {
    result = std::make_shared<QCTensor>(DimsType{dims[0], 1}, dim_scale,
                                        datatype, is_sparse());
    result->flat_data = std::shared_ptr<char>(
        flat_data, flat_data.get() + ElementSize(dims[1] * offset));
    result->scale_data = std::make_shared<float>(*(scale_data.get() + offset));
  }
  result->elements_capacity = result->num_elements;
  return result;
}

}  // namespace xnn_utils
}  // namespace mediapipe::tasks::genai
