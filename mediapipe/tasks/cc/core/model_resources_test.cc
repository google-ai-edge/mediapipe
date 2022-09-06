/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/core/model_resources.h"

#include <fcntl.h>

#include <cstddef>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/shims/cc/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"
#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace custom {
TfLiteRegistration* Register_MY_CUSTOM_OP() {
  // Dummy implementation of custom OP
  static TfLiteRegistration r;
  return &r;
}
}  // namespace custom
}  // namespace ops
}  // namespace tflite

namespace mediapipe {
namespace tasks {
namespace core {

using ::mediapipe::tasks::metadata::ModelMetadataExtractor;

namespace {

constexpr char kTestModelResourcesTag[] = "test_model_resources";

constexpr char kTestModelPath[] =
    "mediapipe/tasks/testdata/core/"
    "test_model_without_custom_op.tflite";

constexpr char kTestModelWithCustomOpsPath[] =
    "mediapipe/tasks/testdata/core/"
    "test_model_with_custom_op.tflite";

constexpr char kTestModelWithMetadataPath[] =
    "mediapipe/tasks/testdata/core/"
    "mobilenet_v1_0.25_224_quant.tflite";

constexpr char kInvalidTestModelPath[] =
    "mediapipe/tasks/testdata/core/"
    "i_do_not_exist.tflite";

// This file is a corrupted version of the original file. Some bytes have been
// trimmed as follow:
//
// tail -c +3 mobilenet_v1_0.25_224_1_default_1.tflite \
// > corrupted_mobilenet_v1_0.25_224_1_default_1.tflite
constexpr char kCorruptedModelPath[] =
    "mediapipe/tasks/testdata/core/"
    "corrupted_mobilenet_v1_0.25_224_1_default_1.tflite";

std::string LoadBinaryContent(const char* filename) {
  std::ifstream input_file(filename, std::ios::binary | std::ios::ate);
  // Find buffer size from input file, and load the buffer.
  size_t buffer_size = input_file.tellg();
  std::string buffer(buffer_size, '\0');
  input_file.seekg(0, std::ios::beg);
  input_file.read(const_cast<char*>(buffer.c_str()), buffer_size);
  return buffer;
}

void AssertStatusHasMediaPipeTasksStatusCode(
    absl::Status status, MediaPipeTasksStatus mediapipe_tasks_code) {
  EXPECT_THAT(
      status.GetPayload(kMediaPipeTasksPayload),
      testing::Optional(absl::Cord(absl::StrCat(mediapipe_tasks_code))));
}

void CheckModelResourcesPackets(const ModelResources* model_resources) {
  Packet model_packet = model_resources->GetModelPacket();
  ASSERT_FALSE(model_packet.IsEmpty());
  MP_ASSERT_OK(model_packet.ValidateAsType<ModelResources::ModelPtr>());
  EXPECT_TRUE(model_packet.Get<ModelResources::ModelPtr>()->initialized());

  Packet op_resolver_packet = model_resources->GetOpResolverPacket();
  ASSERT_FALSE(op_resolver_packet.IsEmpty());
  MP_EXPECT_OK(op_resolver_packet.ValidateAsType<tflite::OpResolver>());

  EXPECT_TRUE(model_resources->GetMetadataExtractor());
  Packet metadata_extractor_packet =
      model_resources->GetMetadataExtractorPacket();
  ASSERT_FALSE(metadata_extractor_packet.IsEmpty());
  MP_EXPECT_OK(
      metadata_extractor_packet.ValidateAsType<ModelMetadataExtractor>());
}

}  // namespace

class ModelResourcesTest : public tflite_shims::testing::Test {};

TEST_F(ModelResourcesTest, CreateFromBinaryContent) {
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_content(LoadBinaryContent(kTestModelPath));
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      ModelResources::Create(kTestModelResourcesTag, std::move(model_file)));
  CheckModelResourcesPackets(model_resources.get());
}

TEST_F(ModelResourcesTest, CreateFromFile) {
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kTestModelPath);
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      ModelResources::Create(kTestModelResourcesTag, std::move(model_file)));
  CheckModelResourcesPackets(model_resources.get());
}

TEST_F(ModelResourcesTest, CreateFromFileDescriptor) {
  const int model_file_descriptor = open(kTestModelPath, O_RDONLY);
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->mutable_file_descriptor_meta()->set_fd(model_file_descriptor);
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      ModelResources::Create(kTestModelResourcesTag, std::move(model_file)));
  CheckModelResourcesPackets(model_resources.get());
}

TEST_F(ModelResourcesTest, CreateFromInvalidFile) {
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kInvalidTestModelPath);
  auto status_or_model_resources =
      ModelResources::Create(kTestModelResourcesTag, std::move(model_file));

  EXPECT_EQ(status_or_model_resources.status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_THAT(status_or_model_resources.status().message(),
              testing::HasSubstr("Unable to open file"));
  AssertStatusHasMediaPipeTasksStatusCode(
      status_or_model_resources.status(),
      MediaPipeTasksStatus::kFileNotFoundError);
}

TEST_F(ModelResourcesTest, CreateFromInvalidFileDescriptor) {
  const int model_file_descriptor = open(kInvalidTestModelPath, O_RDONLY);
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->mutable_file_descriptor_meta()->set_fd(model_file_descriptor);
  auto status_or_model_resources =
      ModelResources::Create(kTestModelResourcesTag, std::move(model_file));

  EXPECT_EQ(status_or_model_resources.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status_or_model_resources.status().message(),
      testing::HasSubstr("Provided file descriptor is invalid: -1 < 0"));
  AssertStatusHasMediaPipeTasksStatusCode(
      status_or_model_resources.status(),
      MediaPipeTasksStatus::kInvalidArgumentError);
}

TEST_F(ModelResourcesTest, CreateFailWithCorruptedFile) {
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kCorruptedModelPath);
  auto status_or_model_resources =
      ModelResources::Create(kTestModelResourcesTag, std::move(model_file));

  EXPECT_EQ(status_or_model_resources.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_model_resources.status().message(),
              testing::HasSubstr("The model is not a valid Flatbuffer"));
  AssertStatusHasMediaPipeTasksStatusCode(
      status_or_model_resources.status(),
      MediaPipeTasksStatus::kInvalidFlatBufferError);
}

// Load a model with a custom OP, create a MutableOpResolver to provide dummy
// implementation of the OP.
TEST_F(ModelResourcesTest, CreateSuccessWithCustomOpsFromFile) {
  static constexpr char kCustomOpName[] = "MY_CUSTOM_OP";
  tflite::MutableOpResolver resolver;
  resolver.AddBuiltin(::tflite::BuiltinOperator_ADD,
                      ::tflite_shims::ops::builtin::Register_ADD());
  resolver.AddCustom(kCustomOpName,
                     ::tflite::ops::custom::Register_MY_CUSTOM_OP());

  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kTestModelWithCustomOpsPath);
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      ModelResources::Create(
          kTestModelResourcesTag, std::move(model_file),
          absl::make_unique<tflite::MutableOpResolver>(resolver)));

  EXPECT_EQ(kTestModelResourcesTag, model_resources->GetTag());
  CheckModelResourcesPackets(model_resources.get());
  Packet op_resolver_packet = model_resources->GetOpResolverPacket();
  EXPECT_EQ(kCustomOpName, op_resolver_packet.Get<tflite::OpResolver>()
                               .FindOp(kCustomOpName, 1)
                               ->custom_name);
}

TEST_F(ModelResourcesTest, CreateSuccessFromFileWithMetadata) {
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kTestModelWithMetadataPath);
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      ModelResources::Create(kTestModelResourcesTag, std::move(model_file)));

  CheckModelResourcesPackets(model_resources.get());
  auto metadata_extractor = model_resources->GetMetadataExtractorPacket().Get();
  EXPECT_TRUE(metadata_extractor.GetModelMetadata()->subgraph_metadata());
}

TEST_F(ModelResourcesTest, CreateSuccessFromBufferWithMetadata) {
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_content(LoadBinaryContent(kTestModelWithMetadataPath));
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      ModelResources::Create(kTestModelResourcesTag, std::move(model_file)));

  CheckModelResourcesPackets(model_resources.get());
  auto metadata_extractor = model_resources->GetMetadataExtractorPacket().Get();
  EXPECT_TRUE(metadata_extractor.GetModelMetadata()->subgraph_metadata());
}

TEST_F(ModelResourcesTest, CreateWithEmptyOpResolverPacket) {
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kTestModelPath);
  api2::Packet<tflite::OpResolver> empty_packet;
  auto status_or_model_resources = ModelResources::Create(
      kTestModelResourcesTag, std::move(model_file), empty_packet);

  EXPECT_EQ(status_or_model_resources.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_model_resources.status().message(),
              testing::HasSubstr("op resolver packet must be non-empty"));
  AssertStatusHasMediaPipeTasksStatusCode(
      status_or_model_resources.status(),
      MediaPipeTasksStatus::kInvalidArgumentError);
}

TEST_F(ModelResourcesTest, CreateSuccessWithCustomOpsPacket) {
  static constexpr char kCustomOpName[] = "MY_CUSTOM_OP";
  tflite::MutableOpResolver resolver;
  resolver.AddBuiltin(::tflite::BuiltinOperator_ADD,
                      ::tflite_shims::ops::builtin::Register_ADD());
  resolver.AddCustom(kCustomOpName,
                     ::tflite::ops::custom::Register_MY_CUSTOM_OP());

  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kTestModelWithCustomOpsPath);
  auto external_op_resolver_packet = api2::PacketAdopting<tflite::OpResolver>(
      absl::make_unique<tflite::MutableOpResolver>(resolver));
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_resources,
      ModelResources::Create(kTestModelResourcesTag, std::move(model_file),
                             external_op_resolver_packet));
  EXPECT_EQ(kTestModelResourcesTag, model_resources->GetTag());
  CheckModelResourcesPackets(model_resources.get());
  Packet model_op_resolver_packet = model_resources->GetOpResolverPacket();
  EXPECT_EQ(kCustomOpName, model_op_resolver_packet.Get<tflite::OpResolver>()
                               .FindOp(kCustomOpName, 1)
                               ->custom_name);
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
