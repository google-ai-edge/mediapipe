/* Copyright 2022 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"

#include <fcntl.h>

#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/utils/zip_utils.h"

namespace mediapipe {
namespace tasks {
namespace core {
namespace {

constexpr char kTestModelResourcesTag[] = "test_model_asset_resources";

constexpr char kTestModelBundleResourcesTag[] =
    "test_model_asset_bundle_resources";

// Models files in dummy_gesture_recognizer.task:
//   gesture_recognizer.task
//     dummy_gesture_recognizer.tflite
//     dummy_hand_landmarker.task
//       dummy_hand_detector.tflite
//       dummy_hand_landmarker.tflite
constexpr char kTestModelBundlePath[] =
    "mediapipe/tasks/testdata/core/dummy_gesture_recognizer.task";

constexpr char kInvalidTestModelBundlePath[] =
    "mediapipe/tasks/testdata/core/i_do_not_exist.task";

}  // namespace

TEST(ModelAssetBundleResourcesTest, CreateFromBinaryContent) {
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_content(LoadBinaryContent(kTestModelBundlePath));
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_bundle_resources,
      ModelAssetBundleResources::Create(kTestModelBundleResourcesTag,
                                        std::move(model_file)));
  MP_EXPECT_OK(
      model_bundle_resources->GetFile("dummy_hand_landmarker.task").status());
  MP_EXPECT_OK(
      model_bundle_resources->GetFile("dummy_gesture_recognizer.tflite")
          .status());
}

TEST(ModelAssetBundleResourcesTest, CreateFromFile) {
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kTestModelBundlePath);
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_bundle_resources,
      ModelAssetBundleResources::Create(kTestModelBundleResourcesTag,
                                        std::move(model_file)));
  MP_EXPECT_OK(
      model_bundle_resources->GetFile("dummy_hand_landmarker.task").status());
  MP_EXPECT_OK(
      model_bundle_resources->GetFile("dummy_gesture_recognizer.tflite")
          .status());
}

#ifndef _WIN32
TEST(ModelAssetBundleResourcesTest, CreateFromFileDescriptor) {
  const int model_file_descriptor = open(kTestModelBundlePath, O_RDONLY);
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->mutable_file_descriptor_meta()->set_fd(model_file_descriptor);
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_bundle_resources,
      ModelAssetBundleResources::Create(kTestModelBundleResourcesTag,
                                        std::move(model_file)));
  MP_EXPECT_OK(
      model_bundle_resources->GetFile("dummy_hand_landmarker.task").status());
  MP_EXPECT_OK(
      model_bundle_resources->GetFile("dummy_gesture_recognizer.tflite")
          .status());
}
#endif  // _WIN32

TEST(ModelAssetBundleResourcesTest, CreateFromFilePointer) {
  auto file_content = LoadBinaryContent(kTestModelBundlePath);
  auto model_file = std::make_unique<proto::ExternalFile>();
  metadata::SetExternalFile(file_content, model_file.get());
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_bundle_resources,
      ModelAssetBundleResources::Create(kTestModelBundleResourcesTag,
                                        std::move(model_file)));
  MP_EXPECT_OK(
      model_bundle_resources->GetFile("dummy_hand_landmarker.task").status());
  MP_EXPECT_OK(
      model_bundle_resources->GetFile("dummy_gesture_recognizer.tflite")
          .status());
}

TEST(ModelAssetBundleResourcesTest, CreateFromInvalidFile) {
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kInvalidTestModelBundlePath);
  auto status_or_model_bundle_resources = ModelAssetBundleResources::Create(
      kTestModelBundleResourcesTag, std::move(model_file));

  EXPECT_EQ(status_or_model_bundle_resources.status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_THAT(status_or_model_bundle_resources.status().message(),
              testing::HasSubstr("Unable to open file"));
  EXPECT_THAT(status_or_model_bundle_resources.status().GetPayload(
                  kMediaPipeTasksPayload),
              testing::Optional(absl::Cord(
                  absl::StrCat(MediaPipeTasksStatus::kFileNotFoundError))));
}

TEST(ModelAssetBundleResourcesTest, ExtractValidModelBundleFile) {
  // Creates top-level model asset bundle resources.
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kTestModelBundlePath);
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_bundle_resources,
      ModelAssetBundleResources::Create(kTestModelBundleResourcesTag,
                                        std::move(model_file)));
  auto status_or_model_bundle_file =
      model_bundle_resources->GetFile("dummy_hand_landmarker.task");
  MP_EXPECT_OK(status_or_model_bundle_file.status());

  // Creates sub-task model asset bundle resources.
  auto hand_landmaker_model_file = std::make_unique<proto::ExternalFile>();
  metadata::SetExternalFile(status_or_model_bundle_file.value(),
                            hand_landmaker_model_file.get());
  MP_ASSERT_OK_AND_ASSIGN(
      auto hand_landmaker_model_bundle_resources,
      ModelAssetBundleResources::Create(kTestModelBundleResourcesTag,
                                        std::move(hand_landmaker_model_file)));
  MP_EXPECT_OK(hand_landmaker_model_bundle_resources
                   ->GetFile("dummy_hand_detector.tflite")
                   .status());
  MP_EXPECT_OK(hand_landmaker_model_bundle_resources
                   ->GetFile("dummy_hand_landmarker.tflite")
                   .status());
}

TEST(ModelAssetBundleResourcesTest, ExtractValidTFLiteModelFile) {
  // Creates top-level model asset bundle resources.
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kTestModelBundlePath);
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_bundle_resources,
      ModelAssetBundleResources::Create(kTestModelBundleResourcesTag,
                                        std::move(model_file)));
  auto status_or_model_bundle_file =
      model_bundle_resources->GetFile("dummy_gesture_recognizer.tflite");
  MP_EXPECT_OK(status_or_model_bundle_file.status());

  // Verify tflite model works.
  auto hand_detector_model_file = std::make_unique<proto::ExternalFile>();
  metadata::SetExternalFile(status_or_model_bundle_file.value(),
                            hand_detector_model_file.get());
  MP_ASSERT_OK_AND_ASSIGN(
      auto hand_detector_model_resources,
      ModelResources::Create(kTestModelResourcesTag,
                             std::move(hand_detector_model_file)));
  Packet model_packet = hand_detector_model_resources->GetModelPacket();
  ASSERT_FALSE(model_packet.IsEmpty());
  MP_ASSERT_OK(model_packet.ValidateAsType<ModelResources::ModelPtr>());
  EXPECT_TRUE(model_packet.Get<ModelResources::ModelPtr>()->initialized());
}

TEST(ModelAssetBundleResourcesTest, ExtractInvalidModelFile) {
  // Creates top-level model asset bundle resources.
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kTestModelBundlePath);
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_bundle_resources,
      ModelAssetBundleResources::Create(kTestModelBundleResourcesTag,
                                        std::move(model_file)));
  auto status = model_bundle_resources->GetFile("not_found.task").status();
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr("No file with name: not_found.task. All files in "
                         "the model asset bundle are: "));
  EXPECT_THAT(status.GetPayload(kMediaPipeTasksPayload),
              testing::Optional(absl::Cord(
                  absl::StrCat(MediaPipeTasksStatus::kFileNotFoundError))));
}

TEST(ModelAssetBundleResourcesTest, ListModelFiles) {
  // Creates top-level model asset bundle resources.
  auto model_file = std::make_unique<proto::ExternalFile>();
  model_file->set_file_name(kTestModelBundlePath);
  MP_ASSERT_OK_AND_ASSIGN(
      auto model_bundle_resources,
      ModelAssetBundleResources::Create(kTestModelBundleResourcesTag,
                                        std::move(model_file)));
  auto model_files = model_bundle_resources->ListFiles();
  std::vector<std::string> expected_model_files = {
      "dummy_gesture_recognizer.tflite", "dummy_hand_landmarker.task"};
  std::sort(model_files.begin(), model_files.end());
  EXPECT_THAT(expected_model_files, testing::ElementsAreArray(model_files));
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
