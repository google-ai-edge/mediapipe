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

#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/model_data.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/vector.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/proto/llm_file_metadata.pb.h"
#include "mediapipe/tasks/cc/genai/inference/proto/llm_params.pb.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/memory_mapped_file.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/metadata_utils.h"
#include "mediapipe/tasks/cc/genai/inference/utils/llm_utils/scoped_file.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe::tasks::genai::llm_utils {
namespace {

// The maximum size of the tflite::Model (excluding buffers).
constexpr uint64_t kTfliteBaseSize = 1024 * 1024;

class SpanHolder : public DataHolder<uint8_t> {
 public:
  explicit SpanHolder(absl::Span<uint8_t> data) : data_(data) {}

  absl::Span<uint8_t> GetData() const override { return data_; }

 private:
  absl::Span<uint8_t> data_;
};

class FreeingSpanHolder : public DataHolder<uint8_t> {
 public:
  explicit FreeingSpanHolder(absl::Span<uint8_t> data) : data_(data) {}
  ~FreeingSpanHolder() override { free(data_.data()); }

  absl::Span<uint8_t> GetData() const override { return data_; }

 private:
  absl::Span<uint8_t> data_;
};

// Base class for loading models from a tflite file.
class TfliteModelData : public ModelData {
 public:
  explicit TfliteModelData(std::shared_ptr<tflite::FlatBufferModel> model)
      : model_(std::move(model)) {}
  ~TfliteModelData() override = default;

  std::optional<odml::infra::proto::LlmModelType> GetModelType() override {
    const tflite::Metadata* metadata = GetMetadata(kLlmModelTypeName);
    if (metadata == nullptr) {
      return std::nullopt;
    }
    return static_cast<odml::infra::proto::LlmModelType>(metadata->buffer());
  }

  std::optional<int> LoRARank() override {
    const tflite::Metadata* metadata = GetMetadata(kLoRARank);
    if (metadata == nullptr) return std::nullopt;
    return static_cast<int>(metadata->buffer());
  }

  const odml::infra::proto::LlmParameters& GetLlmParameters() override {
    return llm_parameters_;
  }

  absl::StatusOr<std::string> ReadMetadata(absl::string_view name) override {
    const tflite::Metadata* metadata = GetMetadata(name);
    if (metadata == nullptr) {
      return absl::NotFoundError(
          absl::StrCat("Failed to get metadata: ", name));
    }
    const tflite::Buffer* backend_buffer =
        model_->GetModel()->buffers()->Get(metadata->buffer());
    MP_ASSIGN_OR_RETURN(
        auto data, ReadData(backend_buffer->offset(), backend_buffer->size()));
    return std::string(reinterpret_cast<const char*>(data->GetData().data()),
                       data->GetData().size());
  }

  uint64_t GetMaxTensorSize() const override {
    uint64_t max_size = 0;
    const tflite::Model* tflite_model = model_->GetModel();
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>& buffers =
        *tflite_model->buffers();
    for (const tflite::SubGraph* subgraph : *tflite_model->subgraphs()) {
      for (const tflite::Tensor* tfl_tensor : *subgraph->tensors()) {
        if (tfl_tensor->buffer() >= buffers.size()) {
          continue;
        }
        max_size =
            std::max(max_size, buffers.Get(tfl_tensor->buffer())->size());
      }
    }
    return max_size;
  }

  uint64_t GetTensorSize(absl::string_view name) const override {
    const tflite::Buffer* buffer = GetBuffer(name);
    if (buffer) {
      return buffer->size();
    }
    return 0;
  }

  absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadTensor(
      absl::string_view name) override {
    const tflite::Buffer* buffer = GetBuffer(name);
    if (buffer) {
      return ReadData(buffer->offset(), buffer->size());
    }
    return nullptr;
  }

  absl::Status InitLlmParameters() {
    MP_ASSIGN_OR_RETURN(std::string proto_str,
                        ReadMetadata(llm_parameters_.GetTypeName()));
    RET_CHECK(llm_parameters_.ParseFromString(proto_str));
    return absl::OkStatus();
  }

 protected:
  virtual absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadData(
      uint64_t offset, uint64_t size) = 0;

  std::shared_ptr<tflite::FlatBufferModel> model_;
  odml::infra::proto::LlmParameters llm_parameters_;

 private:
  const tflite::Buffer* GetBuffer(absl::string_view name) const {
    const tflite::Model* tflite_model = model_->GetModel();
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>& buffers =
        *tflite_model->buffers();
    for (const tflite::SubGraph* subgraph : *tflite_model->subgraphs()) {
      for (const tflite::Tensor* tfl_tensor : *subgraph->tensors()) {
        if (name != tfl_tensor->name()->c_str()) {
          continue;
        }
        if (tfl_tensor->buffer() >= buffers.size()) {
          continue;
        }
        return buffers.Get(tfl_tensor->buffer());
      }
    }
    return nullptr;
  }

  const tflite::Metadata* GetMetadata(absl::string_view name) {
    const tflite::Model* tflite_model = model_->GetModel();
    if (tflite_model->metadata() == nullptr) {
      return nullptr;
    }

    for (const tflite::Metadata* metadata : *tflite_model->metadata()) {
      if (name == metadata->name()->c_str()) {
        return metadata;
      }
    }
    return nullptr;
  }
};

// Loads from a tflite model which includes all buffers in the allocation.
class InMemoryTfliteModelData : public TfliteModelData {
 public:
  explicit InMemoryTfliteModelData(
      std::shared_ptr<tflite::FlatBufferModel> model)
      : TfliteModelData(std::move(model)) {}
  ~InMemoryTfliteModelData() override = default;

  void Clear() override {}

 protected:
  absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadData(
      uint64_t offset, uint64_t size) override {
    return std::make_unique<SpanHolder>(absl::MakeSpan(
        const_cast<uint8_t*>(
            static_cast<const uint8_t*>(model_->allocation()->base())) +
            offset,
        size));
  }
};

// Loads tflite data from a file as needed.
class FileTfliteModelData : public TfliteModelData {
 public:
  FileTfliteModelData(std::shared_ptr<tflite::FlatBufferModel> model,
                      std::unique_ptr<DataHolder<const uint8_t>> model_data,
                      ScopedFile file)
      : TfliteModelData(std::move(model)),
        file_(std::move(file)),
        model_data_(std::move(model_data)) {}
  ~FileTfliteModelData() override = default;

  void Clear() override {
    file_ = ScopedFile();
    model_data_.reset();
  }

 protected:
  absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadData(
      uint64_t offset, uint64_t size) override {
    return CreateMemoryMappedDataHolder<uint8_t>(file_.file(), offset, size,
                                                 key_);
  }

 private:
  static uint32_t next_key_;
  std::string key_{absl::StrCat("FileTfliteModelData_", next_key_++)};
  ScopedFile file_;
  std::unique_ptr<DataHolder<const uint8_t>> model_data_;
};

uint32_t FileTfliteModelData::next_key_ = 0;

// Loads tflite data from the provided function. This owns any data returned
// from the read data function.
class FunctionTfliteModelData : public TfliteModelData {
 public:
  FunctionTfliteModelData(std::shared_ptr<tflite::FlatBufferModel> model,
                          ModelData::ReadDataFn fn)
      : TfliteModelData(std::move(model)), fn_(std::move(fn)) {}
  ~FunctionTfliteModelData() override { Clear(); }

  void Clear() override {
    free(const_cast<void*>(model_->allocation()->base()));
    fn_(0, 0, ReadMode::DISCARD_ALL);
  }

 protected:
  absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadData(
      uint64_t offset, uint64_t size) override {
    void* data = fn_(offset, size, ReadMode::DISCARD);
    RET_CHECK(data) << "Error fetching data.";
    return std::make_unique<FreeingSpanHolder>(
        absl::MakeSpan(static_cast<uint8_t*>(data), size));
  }

 private:
  ModelData::ReadDataFn fn_;
};

// Loads the model using the passed metadata to point to offsets in the file.
class CustomModelData : public ModelData {
 public:
  CustomModelData(const odml::infra::proto::LlmFileMetadata& metadata,
                  ScopedFile file,
                  std::unique_ptr<DataHolder<const uint8_t>> spm_data)
      : metadata_(metadata),
        file_(std::move(file)),
        spm_data_(std::move(spm_data)) {
    for (const auto& tensor : metadata_.tensors()) {
      tensors_[tensor.name()] = tensor;
    }
  }
  ~CustomModelData() override = default;

  std::optional<odml::infra::proto::LlmModelType> GetModelType() override {
    return std::nullopt;
  }

  std::optional<int> LoRARank() override {
    if (metadata_.lora_rank() > 0) {
      return metadata_.lora_rank();
    }
    return std::nullopt;
  }

  const odml::infra::proto::LlmParameters& GetLlmParameters() override {
    return metadata_.model_params();
  }

  absl::StatusOr<std::string> ReadMetadata(absl::string_view name) override {
    if (name == kSpmVocabName && spm_data_) {
      auto spm_data = std::move(spm_data_);
      return std::string(
          reinterpret_cast<const char*>(spm_data->GetData().data()),
          spm_data->GetData().size());
    } else if (name == kLlmBackendName) {
      return "gpu";
    }
    return absl::NotFoundError(absl::StrCat("Failed to get metadata: ", name));
  }

  uint64_t GetMaxTensorSize() const override {
    uint64_t max_size = 0;
    for (const auto& kv : tensors_) {
      max_size = std::max(
          max_size,
          GetAlignedOffsetAndSize(kv.second.offset(), kv.second.size()).size);
    }
    return max_size;
  }

  uint64_t GetTensorSize(absl::string_view name) const override {
    if (auto it = tensors_.find(name); it != tensors_.end()) {
      return it->second.size();
    }
    return 0;
  }

  absl::StatusOr<std::unique_ptr<DataHolder<uint8_t>>> ReadTensor(
      absl::string_view name) override {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      return nullptr;
    }
    return CreateMemoryMappedDataHolder<uint8_t>(
        file_.file(), it->second.offset(), it->second.size(), key_);
  }

  void Clear() override {
    file_ = ScopedFile();
    spm_data_.reset();
  }

 private:
  odml::infra::proto::LlmFileMetadata metadata_;
  ScopedFile file_;
  std::unique_ptr<DataHolder<const uint8_t>> spm_data_;
  absl::flat_hash_map<std::string,
                      odml::infra::proto::LlmFileMetadata::TensorInfo>
      tensors_;

  static uint32_t next_key_;
  std::string key_{absl::StrCat("CustomModelData_", next_key_++)};
};
uint32_t CustomModelData::next_key_ = 0;

uint64_t AlignByN(uint64_t number, uint64_t n) {
  const uint64_t q = number / n;
  return (number % n == 0 ? q : q + 1) * n;
}

}  // namespace

OffsetAndSize GetAlignedOffsetAndSize(uint64_t base_offset,
                                      uint64_t base_size) {
  const size_t kAlignment = MemoryMappedFile::GetOffsetAlignment();
  uint64_t offset = (base_offset / kAlignment) * kAlignment;
  uint64_t size = AlignByN(base_offset - offset + base_size, kAlignment);
  return {.offset = offset, .size = size};
}

// static
absl::StatusOr<std::shared_ptr<ModelData>> ModelData::Create(
    std::unique_ptr<DataHolder<const uint8_t>> sp_model_proto,
    std::unique_ptr<DataHolder<const uint8_t>> llm_model_proto,
    ScopedFile file) {
  odml::infra::proto::LlmFileMetadata file_metadata;
  RET_CHECK(file_metadata.ParseFromArray(llm_model_proto->GetData().data(),
                                         llm_model_proto->GetData().size()));
  return std::make_shared<CustomModelData>(file_metadata, std::move(file),
                                           std::move(sp_model_proto));
}

// static
absl::StatusOr<std::shared_ptr<ModelData>> ModelData::Create(
    std::shared_ptr<tflite::FlatBufferModel> model) {
  auto model_data = std::make_shared<InMemoryTfliteModelData>(std::move(model));
  MP_RETURN_IF_ERROR(model_data->InitLlmParameters());
  return model_data;
}

// static
absl::StatusOr<std::shared_ptr<ModelData>> ModelData::Create(ScopedFile file) {
  // Load the first chunk of the file as a tflite model, and load the rest
  // on-demand when needed.
  MP_ASSIGN_OR_RETURN(auto data,
                      CreateMemoryMappedDataHolder<const uint8_t>(
                          file.file(), /*offset=*/0, /*size=*/kTfliteBaseSize));
  auto model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(data->GetData().data()),
      data->GetData().size());
  RET_CHECK(model) << "Error building tflite model.";
  auto model_data = std::make_shared<FileTfliteModelData>(
      std::move(model), std::move(data), std::move(file));
  MP_RETURN_IF_ERROR(model_data->InitLlmParameters());
  return model_data;
}

// static
absl::StatusOr<std::shared_ptr<ModelData>> ModelData::Create(ReadDataFn fn) {
  // Load the first chunk of the file as a tflite model, and load the rest
  // on-demand when needed.
  void* data = fn(0, kTfliteBaseSize, ReadMode::KEEP);
  RET_CHECK(data) << "Error fetching data.";
  auto model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(data), kTfliteBaseSize);
  RET_CHECK(model) << "Error building tflite model.";
  auto model_data = std::make_shared<FunctionTfliteModelData>(std::move(model),
                                                              std::move(fn));
  MP_RETURN_IF_ERROR(model_data->InitLlmParameters());
  return model_data;
}

// static
absl::StatusOr<std::shared_ptr<ModelData>> ModelData::Create(
    absl::string_view weight_path, absl::string_view spm_path) {
  // If the path is not a directory, it should be a tflite file.
  if (!mediapipe::file::IsDirectory(weight_path).ok()) {
    MP_ASSIGN_OR_RETURN(auto tflite_file, ScopedFile::Open(weight_path));
    return ModelData::Create(std::move(tflite_file));
  }

  // If model proto exists, it should be a gpu combined model format.
  auto model_proto_path =
      mediapipe::file::JoinPath(weight_path, kBasePbFileName);
  MP_RETURN_IF_ERROR(mediapipe::file::Exists(model_proto_path));

  MP_ASSIGN_OR_RETURN(auto model_proto_file,
                      ScopedFile::Open(model_proto_path));
  MP_ASSIGN_OR_RETURN(auto weights_file,
                      ScopedFile::Open(mediapipe::file::JoinPath(
                          weight_path, kBaseWeightsFileName)));
  MP_ASSIGN_OR_RETURN(
      auto model_proto_data,
      CreateMemoryMappedDataHolder<const uint8_t>(model_proto_file.file()));
  // If spm_path is empty, we don't need to load SPM data separately.
  if (spm_path.empty()) {
    return ModelData::Create(nullptr, std::move(model_proto_data),
                             std::move(weights_file));
  }
  MP_ASSIGN_OR_RETURN(auto spm_model_file, ScopedFile::Open(spm_path));
  MP_ASSIGN_OR_RETURN(
      auto spm_data,
      CreateMemoryMappedDataHolder<const uint8_t>(spm_model_file.file()));
  return ModelData::Create(std::move(spm_data), std::move(model_proto_data),
                           std::move(weights_file));
}

// static
absl::StatusOr<std::shared_ptr<ModelData>> ModelData::CreateLoRAFromPath(
    absl::string_view lora_path) {
  // If the path is not a directory, it should be a tflite file.
  if (!mediapipe::file::IsDirectory(lora_path).ok()) {
    MP_ASSIGN_OR_RETURN(auto tflite_file, ScopedFile::Open(lora_path));
    return ModelData::Create(std::move(tflite_file));
  }

  // Otherwise, we expect the combined GPU model format.
  MP_ASSIGN_OR_RETURN(
      auto model_proto_file,
      ScopedFile::Open(mediapipe::file::JoinPath(lora_path, kLoraPbFileName)));
  MP_ASSIGN_OR_RETURN(auto weights_file,
                      ScopedFile::Open(mediapipe::file::JoinPath(
                          lora_path, kLoraWeightsFileName)));
  MP_ASSIGN_OR_RETURN(
      auto model_proto_data,
      CreateMemoryMappedDataHolder<const uint8_t>(model_proto_file.file()));
  return ModelData::Create(nullptr, std::move(model_proto_data),
                           std::move(weights_file));
}

}  // namespace mediapipe::tasks::genai::llm_utils
