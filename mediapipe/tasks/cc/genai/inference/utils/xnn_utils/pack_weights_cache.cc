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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/pack_weights_cache.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/memory_mapped_file.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/named_buffer_generated.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"
#include "xnnpack.h"  // from @XNNPACK

namespace mediapipe::tasks::genai::xnn_utils {

namespace {

bool operator==(const xnn_weights_cache_look_up_key& lhs,
                const xnn_weights_cache_look_up_key& rhs) {
  return lhs.kernel == rhs.kernel && lhs.bias == rhs.bias &&
         lhs.seed == rhs.seed;
}

}  // namespace

PackWeightsCache::PackWeightsCache(absl::string_view cache_path)
    : cache_path_(cache_path) {
  xnn_weights_cache = &cache_provider_;
}

PackWeightsCache::~PackWeightsCache() { xnn_weights_cache = nullptr; }

absl::Status PackWeightsCache::Initialize() {
  mmap_file_ = GetMmapFile(cache_path_);
  if (mmap_file_) {
    MP_RETURN_IF_ERROR(InitializeFromCache(mmap_file_));
  } else {
    builder_ = std::make_unique<flatbuffers::FlatBufferBuilder>();
  }

  cache_provider_.context = this;
  cache_provider_.look_up = (size_t (*)(
      void*, const xnn_weights_cache_look_up_key*))PackWeightsCache::look_up;
  cache_provider_.reserve_space =
      (void* (*)(void*, size_t))PackWeightsCache::reserve_space;
  cache_provider_.look_up_or_insert =
      (size_t (*)(void*, const xnn_weights_cache_look_up_key*, void*,
                  size_t))PackWeightsCache::look_up_or_insert;
  cache_provider_.is_finalized =
      (bool (*)(void*))PackWeightsCache::is_finalized;
  cache_provider_.offset_to_addr =
      (void* (*)(void*, size_t))PackWeightsCache::offset_to_addr;
  cache_provider_.delete_cache =
      (enum xnn_status(*)(void*))PackWeightsCache::delete_cache;

  return absl::OkStatus();
}

absl::Status PackWeightsCache::AddUnpackedWeight(
    absl::string_view name, std::shared_ptr<Tensor> weight) {
  RET_CHECK(!name.empty());
  RET_CHECK(weight->Data());
  RET_CHECK(!kernel_to_name_.contains(weight->Data()));

  kernel_to_name_[weight->Data()] = name;
  return absl::OkStatus();
}

absl::Status PackWeightsCache::Finalize() {
  MP_RETURN_IF_ERROR(error_status_);

  is_finalized_ = true;

  if (!builder_) {
    return absl::OkStatus();
  }

  std::vector<flatbuffers::Offset<Buffer>> buffers;
  for (const auto& [name, offset_size] : name_to_offset_size_) {
    auto new_buffer =
        CreateBuffer(*builder_, builder_->CreateString(std::string(name)),
                     offset_size.first, offset_size.second);
    buffers.push_back(new_buffer);
  }

  auto named_buffers = CreateNamedBuffers(
      *builder_, builder_->CreateVector(buffers), /*flatbuffer_size=*/1);
  FinishNamedBuffersBuffer(*builder_, named_buffers);

  auto serialized =
      std::string(reinterpret_cast<const char*>(builder_->GetBufferPointer()),
                  builder_->GetSize());

  // Ensure 64 aligned.
  while (serialized.size() % 64 != 0) serialized += '\0';
  const size_t flatbuffer_size = serialized.size();

  {
    NamedBuffers* mutable_named_buffer =
        GetMutableNamedBuffers(serialized.data());
    RET_CHECK(mutable_named_buffer->mutate_flatbuffer_size(flatbuffer_size));
  }

  MP_RETURN_IF_ERROR(Prepend(serialized));
  builder_.reset();

  mmap_file_ = GetMmapFile(cache_path_);
  RET_CHECK(mmap_file_);

  return InitializeFromCache(mmap_file_);
}

bool PackWeightsCache::ShouldDoubleCheckCompatibility(
    const xnn_weights_cache_look_up_key* cache_key) {
  if (builder_) return false;
  if (key_sent_for_double_check_.has_value()) return false;

  if (auto entry = kernel_to_name_.find(cache_key->kernel);
      entry != kernel_to_name_.end()) {
    absl::string_view name = entry->second;
    // Usually only the fully_connect op in LLM needs packing, so here we just
    // double check the first FullConn kernel, and assume others are good.
    if (absl::StrContains(name, ".w")) {
      key_sent_for_double_check_ = *cache_key;
      return true;
    }
  }

  return false;
}

std::shared_ptr<llm_utils::MemoryMappedFile> PackWeightsCache::GetMmapFile(
    absl::string_view filename) {
  return mediapipe::file::Exists(filename).ok()
             ? llm_utils::MemoryMappedFile::CreateMutable(filename).value_or(
                   nullptr)
             : nullptr;
}

absl::Status PackWeightsCache::InitializeFromCache(
    std::shared_ptr<llm_utils::MemoryMappedFile> mmap_cache) {
  name_to_offset_size_.clear();
  named_buffers_ = std::shared_ptr<const NamedBuffers>(
      mmap_cache, GetNamedBuffers(mmap_cache->data()));
  for (const Buffer* buffer : *named_buffers_->buffers()) {
    absl::string_view name =
        absl::string_view(buffer->name()->c_str(), buffer->name()->size());
    name_to_offset_size_[name] =
        std::make_pair(buffer->offset(), buffer->size());
  }
  is_finalized_ = true;
  return absl::OkStatus();
}

absl::Status PackWeightsCache::Append(absl::string_view filename,
                                      absl::string_view data) {
  return mediapipe::file::AppendStringToFile(filename, data);
}

absl::Status PackWeightsCache::Prepend(absl::string_view filename,
                                       absl::string_view data) {
  // Append `data` to the end of the file to ensure the file is large enough.
  // Then move chunk_size of bytes towards the end of the file each time.
  // Finally copy `data` to position 0 of the file.
  MP_RETURN_IF_ERROR(Append(filename, data));
  auto mmap_file = GetMmapFile(filename);
  RET_CHECK(mmap_file);
  size_t src_offset = mmap_file->length() - data.size();
  do {
    size_t chunk_size = std::min(src_offset, data.size());
    src_offset -= chunk_size;
    memcpy(static_cast<char*>(mmap_file->data()) + src_offset + data.size(),
           static_cast<char*>(mmap_file->data()) + src_offset, chunk_size);
  } while (src_offset > 0);
  memcpy(mmap_file->data(), data.data(), data.size());
  return absl::OkStatus();
}

absl::Status PackWeightsCache::Append(absl::string_view data) {
  return Append(cache_path_, data);
}

absl::Status PackWeightsCache::Prepend(absl::string_view data) {
  return Prepend(cache_path_, data);
}

size_t PackWeightsCache::look_up(
    PackWeightsCache* context, const xnn_weights_cache_look_up_key* cache_key) {
  ABSL_CHECK(cache_key);

  // TODO: b/319561597 - take seed and bias into consideration.
  if (auto entry = context->kernel_to_name_.find(cache_key->kernel);
      entry != context->kernel_to_name_.end()) {
    absl::string_view name = entry->second;
    if (auto entry = context->name_to_offset_size_.find(name);
        entry != context->name_to_offset_size_.end() &&
        !context->ShouldDoubleCheckCompatibility(cache_key)) {
      return entry->second.first;
    }
  }

  return SIZE_MAX;
}

void* PackWeightsCache::reserve_space(PackWeightsCache* context, size_t n) {
  context->tmp_buffer_to_pack_weight_.resize(n);
  return context->tmp_buffer_to_pack_weight_.data();
}

size_t PackWeightsCache::look_up_or_insert(
    PackWeightsCache* context, const xnn_weights_cache_look_up_key* cache_key,
    void* ptr, size_t size) {
  ABSL_CHECK(cache_key);

  if (context->key_sent_for_double_check_.has_value() &&
      *cache_key == *context->key_sent_for_double_check_) {
    size_t ref_offset = look_up(context, cache_key);
    void* ref_ptr = offset_to_addr(context, ref_offset);
    if (0 == memcmp(ptr, ref_ptr, size)) {
      return ref_offset;
    }
    const absl::string_view error_message =
        "Packed weights is different from cache, it's likely the cache is out "
        "dated.";
    ABSL_LOG(DFATAL) << error_message;
    context->error_status_ = absl::FailedPreconditionError(error_message);
    return SIZE_MAX;
  }

  size_t offset = look_up(context, cache_key);
  if (offset != SIZE_MAX) {
    return offset;
  }

  if (!context->builder_) {
    const absl::string_view error_message =
        "insersion is not supported for an existing cache, consider clear and "
        "rebuild the cache.";
    ABSL_LOG(DFATAL) << error_message;
    context->error_status_ = absl::FailedPreconditionError(error_message);
    return SIZE_MAX;
  }

  if (auto entry = context->kernel_to_name_.find(cache_key->kernel);
      entry != context->kernel_to_name_.end()) {
    absl::string_view name = entry->second;

    size_t offset = context->blob_size_;
    context->name_to_offset_size_[name] = std::make_pair(offset, size);
    if (auto s =
            context->Append(absl::string_view(static_cast<char*>(ptr), size));
        !s.ok()) {
      return SIZE_MAX;
    }
    context->blob_size_ += size;
    return offset;
  }

  return SIZE_MAX;
}

bool PackWeightsCache::is_finalized(PackWeightsCache* context) {
  return context->is_finalized_;
}

void* PackWeightsCache::offset_to_addr(PackWeightsCache* context,
                                       size_t offset) {
  ABSL_DCHECK(is_finalized(context));
  ABSL_DCHECK(!context->builder_);

  uint32_t fb_size = context->named_buffers_->flatbuffer_size();
  void* r = static_cast<char*>(context->mmap_file_->data()) + fb_size + offset;
  return r;
}

absl::StatusOr<std::shared_ptr<Tensor>>
WeightAccessorCompositeWithCache::LoadWeight(absl::string_view tensor_name,
                                             Tensor::DimsType expected_dims,
                                             size_t dim_scale_if_any) const {
  MP_ASSIGN_OR_RETURN(
      auto r, accessor_->LoadWeight(tensor_name, std::move(expected_dims),
                                    dim_scale_if_any));
  // Some weights are not defined in some models and should be left empty.
  if (r == nullptr) {
    return r;
  }
  MP_RETURN_IF_ERROR(weights_cache_->AddUnpackedWeight(tensor_name, r));
  return r;
}

absl::StatusOr<std::shared_ptr<Tensor>>
WeightAccessorCompositeWithCache::LoadTransposedWeight(
    absl::string_view tensor_name, Tensor::DimsType expected_dims,
    size_t dim_scale_if_any) const {
  MP_ASSIGN_OR_RETURN(
      auto r, accessor_->LoadTransposedWeight(
                  tensor_name, std::move(expected_dims), dim_scale_if_any));
  // Some weights are not defined in some models and should be left empty.
  if (r == nullptr) {
    return r;
  }
  MP_RETURN_IF_ERROR(weights_cache_->AddUnpackedWeight(tensor_name, r));
  return r;
}

}  // namespace mediapipe::tasks::genai::xnn_utils
