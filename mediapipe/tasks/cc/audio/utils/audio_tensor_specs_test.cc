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

#include "mediapipe/tasks/cc/audio/utils/audio_tensor_specs.h"

#include <stdint.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"

namespace mediapipe {
namespace tasks {
namespace audio {
namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::testing::Optional;
using ::tflite::EnumNameTensorType;

constexpr char kTestModelResourcesTag[] = "test_model_resources";
constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/audio/";
constexpr char kModelWithMetadata[] =
    "yamnet_audio_classifier_with_metadata.tflite";
constexpr char kModelWithoutMetadata[] = "model_without_metadata.tflite";

class AudioTensorSpecsTest : public tflite_shims::testing::Test {};

TEST_F(AudioTensorSpecsTest,
       BuildInputAudioTensorSpecsWithoutMetdataOptionsFails) {
  auto model_file = std::make_unique<core::proto::ExternalFile>();
  model_file->set_file_name(
      JoinPath("./", kTestDataDirectory, kModelWithoutMetadata));
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          core::ModelResources::Create(kTestModelResourcesTag,
                                                       std::move(model_file)));
  const tflite::Model& model = *model_resources->GetTfLiteModel();
  ASSERT_EQ(model.subgraphs()->size(), 1);
  const auto* primary_subgraph = (*model.subgraphs())[0];
  ASSERT_EQ(primary_subgraph->inputs()->size(), 1);
  auto* input_tensor =
      (*primary_subgraph->tensors())[(*primary_subgraph->inputs())[0]];
  const ModelMetadataExtractor& metadata_extractor =
      *model_resources->GetMetadataExtractor();
  MP_ASSERT_OK_AND_ASSIGN(auto* metadata,
                          GetAudioTensorMetadataIfAny(metadata_extractor, 0));
  absl::StatusOr<AudioTensorSpecs> input_specs_or =
      BuildInputAudioTensorSpecs(*input_tensor, metadata);
  EXPECT_THAT(input_specs_or, StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(input_specs_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(
                  absl::StrCat(MediaPipeTasksStatus::kMetadataNotFoundError))));
}

TEST_F(AudioTensorSpecsTest, BuildInputAudioTensorSpecsWorks) {
  auto model_file = std::make_unique<core::proto::ExternalFile>();
  model_file->set_file_name(
      JoinPath("./", kTestDataDirectory, kModelWithMetadata));
  MP_ASSERT_OK_AND_ASSIGN(auto model_resources,
                          core::ModelResources::Create(kTestModelResourcesTag,
                                                       std::move(model_file)));

  const tflite::Model& model = *model_resources->GetTfLiteModel();
  ASSERT_EQ(model.subgraphs()->size(), 1);
  const auto* primary_subgraph = (*model.subgraphs())[0];
  ASSERT_EQ(primary_subgraph->inputs()->size(), 1);
  auto* input_tensor =
      (*primary_subgraph->tensors())[(*primary_subgraph->inputs())[0]];
  const ModelMetadataExtractor& metadata_extractor =
      *model_resources->GetMetadataExtractor();
  MP_ASSERT_OK_AND_ASSIGN(auto* metadata,
                          GetAudioTensorMetadataIfAny(metadata_extractor, 0));
  absl::StatusOr<AudioTensorSpecs> input_specs_or =
      BuildInputAudioTensorSpecs(*input_tensor, metadata);
  MP_ASSERT_OK(input_specs_or);

  const AudioTensorSpecs& input_specs = input_specs_or.value();
  EXPECT_EQ(input_specs.num_channels, 1);
  EXPECT_EQ(input_specs.num_samples, 15600);
  EXPECT_EQ(input_specs.sample_rate, 16000);
  EXPECT_STREQ(EnumNameTensorType(input_specs.tensor_type),
               EnumNameTensorType(tflite::TensorType_FLOAT32));
  EXPECT_EQ(input_specs.num_overlapping_samples, 0);
}

}  // namespace
}  // namespace audio
}  // namespace tasks
}  // namespace mediapipe
