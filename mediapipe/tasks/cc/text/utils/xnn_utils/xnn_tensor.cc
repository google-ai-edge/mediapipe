#include "mediapipe/tasks/cc/text/utils/xnn_utils/xnn_tensor.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "file/base/helpers.h"
#include "file/base/options.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/utils.h"
#include "third_party/XNNPACK/include/xnnpack.h"

namespace mediapipe {
namespace xnn_utils {

absl::Status FillXnnRoPEWeights(Tensor& out_seg_pos) {
  RET_CHECK_EQ(out_seg_pos.dims.size(), 2);
  const size_t max_seq_len = out_seg_pos.dims[0];
  const size_t num_channels = out_seg_pos.dims[1];
  return out_seg_pos.LoadFromVec(FillXnnRoPEWeights(max_seq_len, num_channels));
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  os << "Tensor{dims=[" << tensor.dims << "], datatype=" << tensor.datatype
     << ", num_elements=" << tensor.num_elements << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const QCTensor& tensor) {
  os << "QCTensor{dims=[" << tensor.dims << "], dim_scale=" << tensor.dim_scale
     << " datatype=" << tensor.datatype
     << ", num_elements=" << tensor.num_elements << "}";
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
  return 0 == memcmp(Data(), other.Data(), num_elements * ElementSize());
}

void Tensor::AllocateBufferIfNeeded() {
  if (!flat_data) {
    auto real_buffer = std::make_shared<std::string>();
    real_buffer->reserve(num_elements * ElementSize() + XNN_EXTRA_BYTES);
    flat_data = std::shared_ptr<char>(real_buffer, real_buffer->data());
  }
}

void* Tensor::Data() {
  DCHECK(flat_data)
      << "If this is weight, you may need to call one of the LoadFrom*()";
  return flat_data.get();
}

std::shared_ptr<Tensor> Tensor::Slice(DimsType offset) {
  DCHECK(flat_data);
  CHECK_EQ(offset.size(), dims.size()) << offset << " vs. " << dims;
  // offset: [0, k, 0, 0], dims: [1, K, _, _]. dims before k must be 1.
  bool found_non_zero_offset = false;
  int index_k = -1;
  for (int i = 0; i < dims.size(); ++i) {
    if (found_non_zero_offset) {
      DCHECK_EQ(offset[i], 0);
    } else if (offset[i] != 0) {
      found_non_zero_offset = true;
      index_k = i;
    }
  }
  DCHECK(found_non_zero_offset) << offset;

  return Slice(index_k, offset[index_k]);
}

std::shared_ptr<Tensor> Tensor::Slice(size_t index, size_t offset) {
  size_t num_elements_offset = 1;
  DimsType new_dim = dims;
  for (int i = 0; i < dims.size(); ++i) {
    if (i < index) {
      DCHECK_EQ(dims[i], 1);
    } else if (i == index) {
      num_elements_offset *= offset;
      new_dim[i] = 1;
    } else {
      num_elements_offset *= dims[i];
    }
  }

  auto result = std::make_shared<Tensor>(std::move(new_dim), datatype);
  result->flat_data = std::shared_ptr<char>(
      flat_data, flat_data.get() + num_elements_offset * ElementSize());
  return result;
}

Tensor& Tensor::Borrow(std::shared_ptr<Tensor> other, size_t element_offset) {
  DCHECK_EQ(datatype, other->datatype);
  DCHECK_EQ(dims.size(), other->dims.size());
  flat_data = std::shared_ptr<char>(
      other->flat_data,
      other->flat_data.get() + element_offset * ElementSize());
  return *this;
}

std::shared_ptr<Tensor> Tensor::View() { return View(dims); }

std::shared_ptr<Tensor> Tensor::View(DimsType as_dims, size_t) {
  auto result = std::make_shared<Tensor>(as_dims, datatype);
  DCHECK_LE(result->num_elements, num_elements);
  result->flat_data = flat_data;
  return result;
}

const void* Tensor::Data() const { return const_cast<Tensor*>(this)->Data(); }

absl::Status Tensor::DefineAsExternal(xnn_subgraph& subgraph, uint32_t flags) {
  uint32_t id;
  RET_CHECK_EQ(xnn_status_success,
               xnn_define_tensor_value(&subgraph, datatype, dims.size(),
                                       dims.data(), /*data=*/nullptr,
                                       /*external_id=*/tensor_id, flags, &id));
  if (tensor_id == XNN_INVALID_VALUE_ID) {
    RET_CHECK_NE(id, XNN_INVALID_VALUE_ID);
    tensor_id = id;
  } else {
    RET_CHECK_EQ(id, tensor_id);
  }
  return absl::OkStatus();
}

absl::Status Tensor::DefineAsInput(xnn_subgraph& subgraph) {
  return DefineAsExternal(subgraph, XNN_VALUE_FLAG_EXTERNAL_INPUT);
}

absl::Status Tensor::DefineAsOutput(xnn_subgraph& subgraph) {
  return DefineAsExternal(subgraph, XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
}

absl::Status Tensor::DefineAsIntermediateTensor(xnn_subgraph& subgraph) {
  RET_CHECK_EQ(tensor_id, XNN_INVALID_VALUE_ID);
  return DefineAsExternal(subgraph, 0);
}

absl::Status Tensor::DefineWeight(xnn_subgraph& subgraph, uint32_t flags) {
  RET_CHECK_EQ(
      xnn_status_success,
      xnn_define_tensor_value(&subgraph, datatype, dims.size(), dims.data(),
                              Data(), tensor_id, flags, &tensor_id));
  RET_CHECK_NE(tensor_id, XNN_INVALID_VALUE_ID);
  return absl::OkStatus();
}

absl::Status Tensor::DefineWeight(xnn_subgraph& subgraph) {
  RET_CHECK_EQ(tensor_id, XNN_INVALID_VALUE_ID);
  return DefineWeight(subgraph, 0);
}

absl::Status Tensor::DefineRope(xnn_subgraph& subgraph) {
  RET_CHECK_NE(tensor_id, XNN_INVALID_VALUE_ID);
  return DefineWeight(subgraph, XNN_VALUE_FLAG_EXTERNAL_INPUT);
}

absl::Status Tensor::LoadFromBuffer(const void* buffer) {
  AllocateBufferIfNeeded();
  memcpy(Data(), buffer, num_elements * ElementSize());
  return absl::OkStatus();
}

absl::Status Tensor::LoadFromVec(const std::vector<float>& data,
                                 bool exact_match) {
  AllocateBufferIfNeeded();
  if (exact_match) {
    RET_CHECK_EQ(num_elements * ElementSize(), data.size() * sizeof(float));
  }

  memcpy(Data(), data.data(), data.size() * sizeof(float));

  return absl::OkStatus();
}

absl::Status Tensor::LoadFromVec(std::vector<float>&& data, bool exact_match) {
  if (exact_match) {
    RET_CHECK_EQ(num_elements * ElementSize(), data.size() * sizeof(float));
  }

  auto real_buffer = std::make_shared<std::vector<float>>(std::move(data));
  if (real_buffer->size() < num_elements) {
    real_buffer->resize(num_elements);
  }
  flat_data = std::shared_ptr<char>(
      real_buffer, reinterpret_cast<char*>(real_buffer->data()));

  return absl::OkStatus();
}

absl::Status Tensor::DumpToBuffer(void* buffer) {
  memcpy(buffer, Data(), num_elements * ElementSize());
  return absl::OkStatus();
}

absl::Status Tensor::DumpToVec(std::vector<float>& out_data, bool exact_match) {
  if (exact_match) {
    RET_CHECK_EQ(num_elements * ElementSize(), out_data.size() * sizeof(float));
  } else {
    out_data.resize(num_elements);
  }
  memcpy(out_data.data(), Data(), num_elements * ElementSize());
  return absl::OkStatus();
}

absl::Status Tensor::DumpToFile(absl::string_view file_path) {
  return file::SetContents(
      file_path,
      absl::string_view(flat_data.get(), num_elements * ElementSize()),
      file::Defaults());
}

absl::Status Tensor::LoadFromFile(absl::string_view file_path, bool use_mmap,
                                  bool exact_match) {
  const size_t expected_size_in_bytes =
      exact_match ? num_elements * ElementSize() : 0;

  ASSIGN_OR_RETURN(flat_data, LoadBufferFromFile(file_path, use_mmap,
                                                 expected_size_in_bytes));
  return absl::OkStatus();
}

std::shared_ptr<Tensor> Tensor::Transpose() {
  DCHECK_EQ(dims.size(), 2);
  DimsType out_dims{dims.rbegin(), dims.rend()};
  auto result = std::make_shared<Tensor>(std::move(out_dims), datatype);
  result->AllocateBufferIfNeeded();
  xnn_status s;
  const DimsType perm{1, 0};
  if (datatype == xnn_datatype_fp32) {
    s = xnn_run_transpose_nd_x32(Data(), result->Data(), dims.size(),
                                 dims.data(), perm.data(),
                                 /*flags=*/0, /*threadpool=*/nullptr);
  } else {
    LOG(FATAL) << "Need update to support new type";
  }
  DCHECK_EQ(s, xnn_status_success);
  return (s == xnn_status_success) ? result : nullptr;
}

absl::StatusOr<std::shared_ptr<Tensor>> Tensor::ConvertToF32() {
  auto result = std::make_shared<Tensor>(dims, xnn_datatype_fp32);
  MP_RETURN_IF_ERROR(result->LoadFromBuffer(Data()));
  return result;
}

absl::Status QCTensor::LoadFromFile(absl::string_view quantized_weight_filename,
                                    absl::string_view scale_filename,
                                    bool use_mmap, bool exact_match) {
  size_t scale_element_size = dims[dim_scale];

  ASSIGN_OR_RETURN(flat_data,
                   LoadBufferFromFile(quantized_weight_filename, use_mmap,
                                      exact_match ? num_elements : 0));
  ASSIGN_OR_RETURN(scale_data,
                   LoadBufferFromFile<float>(
                       scale_filename, use_mmap,
                       exact_match ? scale_element_size * sizeof(float) : 0));
  return absl::OkStatus();
}

absl::Status QCTensor::DumpToFile(absl::string_view file_path) {
  MP_RETURN_IF_ERROR(file::SetContents(
      file_path,
      absl::string_view(flat_data.get(), num_elements * ElementSize()),
      file::Defaults()));
  return file::SetContents(
      absl::StrCat(file_path, kQuantizedScaleSuffix),
      absl::string_view(reinterpret_cast<char*>(scale_data.get()),
                        dims[dim_scale] * sizeof(float)),
      file::Defaults());
}

absl::Status QCTensor::DefineWeight(xnn_subgraph& subgraph, uint32_t flags) {
  RET_CHECK_EQ(
      xnn_status_success,
      xnn_define_channelwise_quantized_tensor_value(
          &subgraph, datatype, scale_data.get(), dims.size(), dim_scale,
          dims.data(), Data(), XNN_INVALID_VALUE_ID, flags, &tensor_id))
      << *this;
  RET_CHECK_NE(tensor_id, XNN_INVALID_VALUE_ID);
  return absl::OkStatus();
}

void QCTensor::AllocateBufferIfNeeded() {
  Tensor::AllocateBufferIfNeeded();
  if (!scale_data) {
    auto real_buffer = std::make_shared<std::vector<float>>();
    real_buffer->reserve(dims[dim_scale]);
    scale_data = std::shared_ptr<float>(real_buffer, real_buffer->data());
  }
}

std::shared_ptr<Tensor> QCTensor::Transpose() {
  DCHECK_EQ(dims.size(), 2);
  size_t channel_size = dims[dim_scale];
  DimsType out_dims{dims.rbegin(), dims.rend()};
  auto result = std::make_shared<QCTensor>(std::move(out_dims), 1 - dim_scale);
  result->AllocateBufferIfNeeded();
  memcpy(result->scale_data.get(), scale_data.get(),
         channel_size * sizeof(float));
  xnn_status s;
  const DimsType perm{1, 0};
  if (datatype == xnn_datatype_qcint8) {
    s = xnn_run_transpose_nd_x8(Data(), result->Data(), dims.size(),
                                dims.data(), perm.data(),
                                /*flags=*/0, /*threadpool=*/nullptr);
  } else {
    LOG(FATAL) << "Need update to support new type";
  }
  DCHECK_EQ(s, xnn_status_success);
  return (s == xnn_status_success) ? result : nullptr;
}

absl::StatusOr<std::shared_ptr<Tensor>> QCTensor::ConvertToF32() {
  auto result = std::make_shared<Tensor>(dims, xnn_datatype_fp32);
  // TODO: proper implement.
  LOG(WARNING) << "This is fake impl";
  MP_RETURN_IF_ERROR(result->LoadFromVec({}, /*exact_match=*/false));
  return result;
}

std::shared_ptr<Tensor> QCTensor::View(DimsType as_dims,
                                       size_t dim_scale_if_any) {
  auto result = std::make_shared<QCTensor>(as_dims, dim_scale_if_any);
  DCHECK_LE(result->num_elements, num_elements);
  result->flat_data = flat_data;
  result->scale_data = scale_data;
  return result;
}

}  // namespace xnn_utils
}  // namespace mediapipe
