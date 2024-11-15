// Copyright 2019 The MediaPipe Authors.
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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/image/opencv_image_encoder_calculator.pb.h"
#include "mediapipe/calculators/tensorflow/pack_media_sequence_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/location_opencv.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/util/sequence/media_sequence.h"
#include "mediapipe/util/sequence/media_sequence_util.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {
namespace {

namespace tf = ::tensorflow;
namespace mpms = mediapipe::mediasequence;

constexpr char kBboxTag[] = "BBOX";
constexpr char kEncodedMediaStartTimestampTag[] =
    "ENCODED_MEDIA_START_TIMESTAMP";
constexpr char kEncodedMediaTag[] = "ENCODED_MEDIA";
constexpr char kClassSegmentationTag[] = "CLASS_SEGMENTATION";
constexpr char kKeypointsTestTag[] = "KEYPOINTS_TEST";
constexpr char kBboxPredictedTag[] = "BBOX_PREDICTED";
constexpr char kAudioOtherTag[] = "AUDIO_OTHER";
constexpr char kAudioTestTag[] = "AUDIO_TEST";
constexpr char kBytesFeatureOtherTag[] = "BYTES_FEATURE_OTHER";
constexpr char kBytesFeatureTestTag[] = "BYTES_FEATURE_TEST";
constexpr char kForwardFlowEncodedTag[] = "FORWARD_FLOW_ENCODED";
constexpr char kFloatContextFeatureOtherTag[] = "FLOAT_CONTEXT_FEATURE_OTHER";
constexpr char kFloatContextFeatureTestTag[] = "FLOAT_CONTEXT_FEATURE_TEST";
constexpr char kIntsContextFeatureTestTag[] = "INTS_CONTEXT_FEATURE_TEST";
constexpr char kIntsContextFeatureOtherTag[] = "INTS_CONTEXT_FEATURE_OTHER";
constexpr char kBytesContextFeatureTestTag[] = "BYTES_CONTEXT_FEATURE_TEST";
constexpr char kBytesContextFeatureOtherTag[] = "BYTES_CONTEXT_FEATURE_OTHER";
constexpr char kFloatFeatureOtherTag[] = "FLOAT_FEATURE_OTHER";
constexpr char kFloatFeatureTestTag[] = "FLOAT_FEATURE_TEST";
constexpr char kIntFeatureOtherTag[] = "INT_FEATURE_OTHER";
constexpr char kIntFeatureTestTag[] = "INT_FEATURE_TEST";
constexpr char kImageLabelTestTag[] = "IMAGE_LABEL_TEST";
constexpr char kImageLabelOtherTag[] = "IMAGE_LABEL_OTHER";
constexpr char kImagePrefixTag[] = "IMAGE_PREFIX";
constexpr char kSequenceExampleTag[] = "SEQUENCE_EXAMPLE";
constexpr char kImageTag[] = "IMAGE";
constexpr char kClipMediaIdTag[] = "CLIP_MEDIA_ID";
constexpr char kClipLabelTestTag[] = "CLIP_LABEL_TEST";
constexpr char kClipLabelOtherTag[] = "CLIP_LABEL_OTHER";
constexpr char kClipLabelAnotherTag[] = "CLIP_LABEL_ANOTHER";

class PackMediaSequenceCalculatorTest : public ::testing::Test {
 protected:
  void SetUpCalculator(const std::vector<std::string>& input_streams,
                       const tf::Features& features,
                       const bool output_only_if_all_present,
                       const bool replace_instead_of_append,
                       const bool output_as_zero_timestamp = false,
                       const bool add_empty_labels = false,
                       const std::vector<std::string>& input_side_packets = {
                           "SEQUENCE_EXAMPLE:input_sequence"}) {
    CalculatorGraphConfig::Node config;
    config.set_calculator("PackMediaSequenceCalculator");
    for (const std::string& side_packet : input_side_packets) {
      config.add_input_side_packet(side_packet);
    }
    config.add_output_stream("SEQUENCE_EXAMPLE:output_sequence");
    for (const std::string& stream : input_streams) {
      config.add_input_stream(stream);
    }
    auto options = config.mutable_options()->MutableExtension(
        PackMediaSequenceCalculatorOptions::ext);
    *options->mutable_context_feature_map() = features;
    options->set_output_only_if_all_present(output_only_if_all_present);
    options->set_replace_data_instead_of_append(replace_instead_of_append);
    options->set_output_as_zero_timestamp(output_as_zero_timestamp);
    options->set_add_empty_labels(add_empty_labels);
    runner_ = ::absl::make_unique<CalculatorRunner>(config);
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(PackMediaSequenceCalculatorTest, PacksTwoImages) {
  SetUpCalculator({"IMAGE:images"}, {}, false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());
  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(
      cv::imencode(".jpg", image, bytes, {cv::IMWRITE_HDR_COMPRESSION, 1}));
  OpenCvImageEncoderCalculatorResults encoded_image;
  encoded_image.set_encoded_image(bytes.data(), bytes.size());
  encoded_image.set_width(2);
  encoded_image.set_height(1);

  int num_images = 2;
  for (int i = 0; i < num_images; ++i) {
    auto image_ptr =
        ::absl::make_unique<OpenCvImageEncoderCalculatorResults>(encoded_image);
    runner_->MutableInputs()->Tag(kImageTag).packets.push_back(
        Adopt(image_ptr.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(test_video_id, mpms::GetClipMediaId(output_sequence));
  ASSERT_EQ(num_images, mpms::GetImageTimestampSize(output_sequence));
  ASSERT_EQ(num_images, mpms::GetImageEncodedSize(output_sequence));
  for (int i = 0; i < num_images; ++i) {
    ASSERT_EQ(i, mpms::GetImageTimestampAt(output_sequence, i));
    ASSERT_EQ(encoded_image.encoded_image(),
              mpms::GetImageEncodedAt(output_sequence, i));
  }
}

TEST_F(PackMediaSequenceCalculatorTest, PacksTwoPrefixedImages) {
  std::string prefix = "PREFIX";
  SetUpCalculator({"IMAGE_PREFIX:images"}, {}, false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());
  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(
      cv::imencode(".jpg", image, bytes, {cv::IMWRITE_HDR_COMPRESSION, 1}));
  OpenCvImageEncoderCalculatorResults encoded_image;
  encoded_image.set_encoded_image(bytes.data(), bytes.size());
  encoded_image.set_width(2);
  encoded_image.set_height(1);

  int num_images = 2;
  for (int i = 0; i < num_images; ++i) {
    auto image_ptr =
        ::absl::make_unique<OpenCvImageEncoderCalculatorResults>(encoded_image);
    runner_->MutableInputs()
        ->Tag(kImagePrefixTag)
        .packets.push_back(Adopt(image_ptr.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(test_video_id, mpms::GetClipMediaId(output_sequence));
  ASSERT_EQ(num_images, mpms::GetImageTimestampSize(prefix, output_sequence));
  ASSERT_EQ(num_images, mpms::GetImageEncodedSize(prefix, output_sequence));
  for (int i = 0; i < num_images; ++i) {
    ASSERT_EQ(i, mpms::GetImageTimestampAt(prefix, output_sequence, i));
    ASSERT_EQ(encoded_image.encoded_image(),
              mpms::GetImageEncodedAt(prefix, output_sequence, i));
  }
}

TEST_F(PackMediaSequenceCalculatorTest, PacksTwoFloatLists) {
  SetUpCalculator({"FLOAT_FEATURE_TEST:test", "FLOAT_FEATURE_OTHER:test2"}, {},
                  false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  int num_timesteps = 2;
  for (int i = 0; i < num_timesteps; ++i) {
    auto vf_ptr = ::absl::make_unique<std::vector<float>>(2, 2 << i);
    runner_->MutableInputs()
        ->Tag(kFloatFeatureTestTag)
        .packets.push_back(Adopt(vf_ptr.release()).At(Timestamp(i)));
    vf_ptr = ::absl::make_unique<std::vector<float>>(2, 2 << i);
    runner_->MutableInputs()
        ->Tag(kFloatFeatureOtherTag)
        .packets.push_back(Adopt(vf_ptr.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(num_timesteps,
            mpms::GetFeatureTimestampSize("TEST", output_sequence));
  ASSERT_EQ(num_timesteps, mpms::GetFeatureFloatsSize("TEST", output_sequence));
  ASSERT_EQ(num_timesteps,
            mpms::GetFeatureTimestampSize("OTHER", output_sequence));
  ASSERT_EQ(num_timesteps,
            mpms::GetFeatureFloatsSize("OTHER", output_sequence));
  for (int i = 0; i < num_timesteps; ++i) {
    ASSERT_EQ(i, mpms::GetFeatureTimestampAt("TEST", output_sequence, i));
    ASSERT_THAT(mpms::GetFeatureFloatsAt("TEST", output_sequence, i),
                ::testing::ElementsAreArray(std::vector<float>(2, 2 << i)));
    ASSERT_EQ(i, mpms::GetFeatureTimestampAt("OTHER", output_sequence, i));
    ASSERT_THAT(mpms::GetFeatureFloatsAt("OTHER", output_sequence, i),
                ::testing::ElementsAreArray(std::vector<float>(2, 2 << i)));
  }
}

TEST_F(PackMediaSequenceCalculatorTest, PacksTwoIntLists) {
  SetUpCalculator({"INT_FEATURE_TEST:test", "INT_FEATURE_OTHER:test2"}, {},
                  false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  int num_timesteps = 2;
  for (int i = 0; i < num_timesteps; ++i) {
    auto vi_ptr = ::absl::make_unique<std::vector<int64_t>>(2, 2 << i);
    runner_->MutableInputs()
        ->Tag(kIntFeatureTestTag)
        .packets.push_back(Adopt(vi_ptr.release()).At(Timestamp(i)));
    vi_ptr = ::absl::make_unique<std::vector<int64_t>>(2, 2 << i);
    runner_->MutableInputs()
        ->Tag(kIntFeatureOtherTag)
        .packets.push_back(Adopt(vi_ptr.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(num_timesteps,
            mpms::GetFeatureTimestampSize("TEST", output_sequence));
  ASSERT_EQ(num_timesteps, mpms::GetFeatureIntsSize("TEST", output_sequence));
  ASSERT_EQ(num_timesteps,
            mpms::GetFeatureTimestampSize("OTHER", output_sequence));
  ASSERT_EQ(num_timesteps, mpms::GetFeatureIntsSize("OTHER", output_sequence));
  for (int i = 0; i < num_timesteps; ++i) {
    ASSERT_EQ(i, mpms::GetFeatureTimestampAt("TEST", output_sequence, i));
    ASSERT_THAT(mpms::GetFeatureIntsAt("TEST", output_sequence, i),
                ::testing::ElementsAreArray(std::vector<int64_t>(2, 2 << i)));
    ASSERT_EQ(i, mpms::GetFeatureTimestampAt("OTHER", output_sequence, i));
    ASSERT_THAT(mpms::GetFeatureIntsAt("OTHER", output_sequence, i),
                ::testing::ElementsAreArray(std::vector<int64_t>(2, 2 << i)));
  }
}

TEST_F(PackMediaSequenceCalculatorTest, PacksTwoBytesLists) {
  SetUpCalculator({"BYTES_FEATURE_TEST:test", "BYTES_FEATURE_OTHER:test2"}, {},
                  false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  int num_timesteps = 2;
  for (int i = 0; i < num_timesteps; ++i) {
    auto vs_ptr = ::absl::make_unique<std::vector<std::string>>(
        2, absl::StrCat("foo", 2 << i));
    runner_->MutableInputs()
        ->Tag(kBytesFeatureTestTag)
        .packets.push_back(Adopt(vs_ptr.release()).At(Timestamp(i)));
    vs_ptr = ::absl::make_unique<std::vector<std::string>>(
        2, absl::StrCat("bar", 2 << i));
    runner_->MutableInputs()
        ->Tag(kBytesFeatureOtherTag)
        .packets.push_back(Adopt(vs_ptr.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(num_timesteps,
            mpms::GetFeatureTimestampSize("TEST", output_sequence));
  ASSERT_EQ(num_timesteps, mpms::GetFeatureBytesSize("TEST", output_sequence));
  ASSERT_EQ(num_timesteps,
            mpms::GetFeatureTimestampSize("OTHER", output_sequence));
  ASSERT_EQ(num_timesteps, mpms::GetFeatureBytesSize("OTHER", output_sequence));
  for (int i = 0; i < num_timesteps; ++i) {
    ASSERT_EQ(i, mpms::GetFeatureTimestampAt("TEST", output_sequence, i));
    ASSERT_THAT(mpms::GetFeatureBytesAt("TEST", output_sequence, i),
                ::testing::ElementsAreArray(
                    std::vector<std::string>(2, absl::StrCat("foo", 2 << i))));
    ASSERT_EQ(i, mpms::GetFeatureTimestampAt("OTHER", output_sequence, i));
    ASSERT_THAT(mpms::GetFeatureBytesAt("OTHER", output_sequence, i),
                ::testing::ElementsAreArray(
                    std::vector<std::string>(2, absl::StrCat("bar", 2 << i))));
  }
}

TEST_F(PackMediaSequenceCalculatorTest, PacksTwoImageLabels) {
  SetUpCalculator(
      {"IMAGE_LABEL_TEST:test_labels", "IMAGE_LABEL_OTHER:test_labels2"}, {},
      false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  int num_timesteps = 2;
  for (int i = 0; i < num_timesteps; ++i) {
    Detection detection1;
    detection1.add_label(absl::StrCat("foo", 2 << i));
    detection1.add_label_id(i);
    detection1.add_score(0.1 * i);
    detection1.add_label(absl::StrCat("foo", 2 << i));
    detection1.add_label_id(i);
    detection1.add_score(0.1 * i);
    auto label_ptr1 = ::absl::make_unique<Detection>(detection1);
    runner_->MutableInputs()
        ->Tag(kImageLabelTestTag)
        .packets.push_back(Adopt(label_ptr1.release()).At(Timestamp(i)));
    Detection detection2;
    detection2.add_label(absl::StrCat("bar", 2 << i));
    detection2.add_score(0.2 * i);
    detection2.add_label(absl::StrCat("bar", 2 << i));
    detection2.add_score(0.2 * i);
    auto label_ptr2 = ::absl::make_unique<Detection>(detection2);
    runner_->MutableInputs()
        ->Tag(kImageLabelOtherTag)
        .packets.push_back(Adopt(label_ptr2.release()).At(Timestamp(i)));
  }
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(num_timesteps,
            mpms::GetImageTimestampSize("TEST", output_sequence));
  ASSERT_EQ(num_timesteps,
            mpms::GetImageLabelStringSize("TEST", output_sequence));
  ASSERT_EQ(num_timesteps,
            mpms::GetImageLabelConfidenceSize("TEST", output_sequence));
  ASSERT_EQ(num_timesteps,
            mpms::GetImageTimestampSize("OTHER", output_sequence));
  ASSERT_EQ(num_timesteps,
            mpms::GetImageLabelStringSize("OTHER", output_sequence));
  ASSERT_EQ(num_timesteps,
            mpms::GetImageLabelConfidenceSize("OTHER", output_sequence));
  for (int i = 0; i < num_timesteps; ++i) {
    ASSERT_EQ(i, mpms::GetImageTimestampAt("TEST", output_sequence, i));
    ASSERT_THAT(mpms::GetImageLabelStringAt("TEST", output_sequence, i),
                ::testing::ElementsAreArray(
                    std::vector<std::string>(2, absl::StrCat("foo", 2 << i))));
    ASSERT_THAT(mpms::GetImageLabelIndexAt("TEST", output_sequence, i),
                ::testing::ElementsAreArray(std::vector<int32_t>(2, i)));
    ASSERT_THAT(mpms::GetImageLabelConfidenceAt("TEST", output_sequence, i),
                ::testing::ElementsAreArray(std::vector<float>(2, 0.1 * i)));
    ASSERT_EQ(i, mpms::GetImageTimestampAt("OTHER", output_sequence, i));
    ASSERT_THAT(mpms::GetImageLabelStringAt("OTHER", output_sequence, i),
                ::testing::ElementsAreArray(
                    std::vector<std::string>(2, absl::StrCat("bar", 2 << i))));
    ASSERT_THAT(mpms::GetImageLabelConfidenceAt("OTHER", output_sequence, i),
                ::testing::ElementsAreArray(std::vector<float>(2, 0.2 * i)));
  }
}

TEST_F(PackMediaSequenceCalculatorTest, OutputAsZeroTimestamp) {
  SetUpCalculator({"FLOAT_FEATURE_TEST:test"}, {}, false, true, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  int num_timesteps = 2;
  for (int i = 0; i < num_timesteps; ++i) {
    auto vf_ptr = ::absl::make_unique<std::vector<float>>(2, 2 << i);
    runner_->MutableInputs()
        ->Tag("FLOAT_FEATURE_TEST")
        .packets.push_back(Adopt(vf_ptr.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag("SEQUENCE_EXAMPLE") =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag("SEQUENCE_EXAMPLE").packets;
  ASSERT_EQ(1, output_packets.size());
  EXPECT_EQ(output_packets[0].Timestamp().Value(), 0ll);
}

TEST_F(PackMediaSequenceCalculatorTest, PacksTwoContextFloatLists) {
  SetUpCalculator(
      {"FLOAT_CONTEXT_FEATURE_TEST:test", "FLOAT_CONTEXT_FEATURE_OTHER:test2"},
      {}, false, true);
  auto input_sequence = absl::make_unique<tf::SequenceExample>();

  auto vf_ptr = absl::make_unique<std::vector<float>>(2, 3);
  runner_->MutableInputs()
      ->Tag(kFloatContextFeatureTestTag)
      .packets.push_back(Adopt(vf_ptr.release()).At(Timestamp::PostStream()));
  vf_ptr = absl::make_unique<std::vector<float>>(2, 4);
  runner_->MutableInputs()
      ->Tag(kFloatContextFeatureOtherTag)
      .packets.push_back(Adopt(vf_ptr.release()).At(Timestamp::PostStream()));

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_THAT(mpms::GetContextFeatureFloats("TEST", output_sequence),
              testing::ElementsAre(3, 3));
  ASSERT_THAT(mpms::GetContextFeatureFloats("OTHER", output_sequence),
              testing::ElementsAre(4, 4));
}

TEST_F(PackMediaSequenceCalculatorTest, ReplaceTwoContextFloatLists) {
  SetUpCalculator(
      /*input_streams=*/{"FLOAT_CONTEXT_FEATURE_TEST:test",
                         "FLOAT_CONTEXT_FEATURE_OTHER:test2"},
      /*features=*/{},
      /*output_only_if_all_present=*/false, /*replace_instead_of_append=*/true);
  auto input_sequence = std::make_unique<tf::SequenceExample>();
  mpms::SetContextFeatureFloats("TEST", {2, 3}, input_sequence.get());
  mpms::SetContextFeatureFloats("OTHER", {2, 4}, input_sequence.get());

  const std::vector<float> vf_1 = {5, 6};
  runner_->MutableInputs()
      ->Tag(kFloatContextFeatureTestTag)
      .packets.push_back(
          MakePacket<std::vector<float>>(vf_1).At(Timestamp::PostStream()));
  const std::vector<float> vf_2 = {7, 8};
  runner_->MutableInputs()
      ->Tag(kFloatContextFeatureOtherTag)
      .packets.push_back(
          MakePacket<std::vector<float>>(vf_2).At(Timestamp::PostStream()));

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_THAT(mpms::GetContextFeatureFloats("TEST", output_sequence),
              testing::ElementsAre(5, 6));
  ASSERT_THAT(mpms::GetContextFeatureFloats("OTHER", output_sequence),
              testing::ElementsAre(7, 8));
}

TEST_F(PackMediaSequenceCalculatorTest, AppendTwoContextFloatLists) {
  SetUpCalculator(
      /*input_streams=*/{"FLOAT_CONTEXT_FEATURE_TEST:test",
                         "FLOAT_CONTEXT_FEATURE_OTHER:test2"},
      /*features=*/{},
      /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/false);
  auto input_sequence = std::make_unique<tf::SequenceExample>();
  mpms::SetContextFeatureFloats("TEST", {2, 3}, input_sequence.get());
  mpms::SetContextFeatureFloats("OTHER", {2, 4}, input_sequence.get());

  const std::vector<float> vf_1 = {5, 6};
  runner_->MutableInputs()
      ->Tag(kFloatContextFeatureTestTag)
      .packets.push_back(
          MakePacket<std::vector<float>>(vf_1).At(Timestamp::PostStream()));
  const std::vector<float> vf_2 = {7, 8};
  runner_->MutableInputs()
      ->Tag(kFloatContextFeatureOtherTag)
      .packets.push_back(
          MakePacket<std::vector<float>>(vf_2).At(Timestamp::PostStream()));

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  EXPECT_THAT(mpms::GetContextFeatureFloats("TEST", output_sequence),
              testing::ElementsAre(2, 3, 5, 6));
  EXPECT_THAT(mpms::GetContextFeatureFloats("OTHER", output_sequence),
              testing::ElementsAre(2, 4, 7, 8));
}

TEST_F(PackMediaSequenceCalculatorTest, PackTwoContextIntLists) {
  SetUpCalculator(
      /*input_streams=*/{"INTS_CONTEXT_FEATURE_TEST:test",
                         "INTS_CONTEXT_FEATURE_OTHER:test2"},
      /*features=*/{},
      /*output_only_if_all_present=*/false, /*replace_instead_of_append=*/true);
  auto input_sequence = absl::make_unique<tf::SequenceExample>();

  const std::vector<int64_t> vi_1 = {2, 3};
  runner_->MutableInputs()
      ->Tag(kIntsContextFeatureTestTag)
      .packets.push_back(
          MakePacket<std::vector<int64_t>>(vi_1).At(Timestamp::PostStream()));
  const std::vector<int64_t> vi_2 = {2, 4};
  runner_->MutableInputs()
      ->Tag(kIntsContextFeatureOtherTag)
      .packets.push_back(
          MakePacket<std::vector<int64_t>>(vi_2).At(Timestamp::PostStream()));

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_THAT(mpms::GetContextFeatureInts("TEST", output_sequence),
              testing::ElementsAre(2, 3));
  ASSERT_THAT(mpms::GetContextFeatureInts("OTHER", output_sequence),
              testing::ElementsAre(2, 4));
}

TEST_F(PackMediaSequenceCalculatorTest, ReplaceTwoContextIntLists) {
  SetUpCalculator(
      /*input_streams=*/{"INTS_CONTEXT_FEATURE_TEST:test",
                         "INTS_CONTEXT_FEATURE_OTHER:test2"},
      /*features=*/{},
      /*output_only_if_all_present=*/false, /*replace_instead_of_append=*/true);
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  mpms::SetContextFeatureInts("TEST", {2, 3}, input_sequence.get());
  mpms::SetContextFeatureInts("OTHER", {2, 4}, input_sequence.get());

  const std::vector<int64_t> vi_1 = {5, 6};
  runner_->MutableInputs()
      ->Tag(kIntsContextFeatureTestTag)
      .packets.push_back(
          MakePacket<std::vector<int64_t>>(vi_1).At(Timestamp::PostStream()));
  const std::vector<int64_t> vi_2 = {7, 8};
  runner_->MutableInputs()
      ->Tag(kIntsContextFeatureOtherTag)
      .packets.push_back(
          MakePacket<std::vector<int64_t>>(vi_2).At(Timestamp::PostStream()));

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_THAT(mpms::GetContextFeatureInts("TEST", output_sequence),
              testing::ElementsAre(5, 6));
  ASSERT_THAT(mpms::GetContextFeatureInts("OTHER", output_sequence),
              testing::ElementsAre(7, 8));
}

TEST_F(PackMediaSequenceCalculatorTest, AppendTwoContextIntLists) {
  SetUpCalculator(
      /*input_streams=*/{"INTS_CONTEXT_FEATURE_TEST:test",
                         "INTS_CONTEXT_FEATURE_OTHER:test2"},
      /*features=*/{},
      /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/false);
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  mpms::SetContextFeatureInts("TEST", {2, 3}, input_sequence.get());
  mpms::SetContextFeatureInts("OTHER", {2, 4}, input_sequence.get());

  const std::vector<int64_t> vi_1 = {5, 6};
  runner_->MutableInputs()
      ->Tag(kIntsContextFeatureTestTag)
      .packets.push_back(
          MakePacket<std::vector<int64_t>>(vi_1).At(Timestamp::PostStream()));
  const std::vector<int64_t> vi_2 = {7, 8};
  runner_->MutableInputs()
      ->Tag(kIntsContextFeatureOtherTag)
      .packets.push_back(
          MakePacket<std::vector<int64_t>>(vi_2).At(Timestamp::PostStream()));

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_THAT(mpms::GetContextFeatureInts("TEST", output_sequence),
              testing::ElementsAre(2, 3, 5, 6));
  ASSERT_THAT(mpms::GetContextFeatureInts("OTHER", output_sequence),
              testing::ElementsAre(2, 4, 7, 8));
}

TEST_F(PackMediaSequenceCalculatorTest, PackTwoContextByteLists) {
  SetUpCalculator(
      /*input_streams=*/{"BYTES_CONTEXT_FEATURE_TEST:test",
                         "BYTES_CONTEXT_FEATURE_OTHER:test2"},
      /*features=*/{},
      /*output_only_if_all_present=*/false, /*replace_instead_of_append=*/true);
  auto input_sequence = absl::make_unique<tf::SequenceExample>();

  const std::vector<std::string> vb_1 = {"value_1", "value_2"};
  runner_->MutableInputs()
      ->Tag(kBytesContextFeatureTestTag)
      .packets.push_back(MakePacket<std::vector<std::string>>(vb_1).At(
          Timestamp::PostStream()));
  const std::vector<std::string> vb_2 = {"value_3", "value_4"};
  runner_->MutableInputs()
      ->Tag(kBytesContextFeatureOtherTag)
      .packets.push_back(MakePacket<std::vector<std::string>>(vb_2).At(
          Timestamp::PostStream()));

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_THAT(mpms::GetContextFeatureBytes("TEST", output_sequence),
              testing::ElementsAre("value_1", "value_2"));
  ASSERT_THAT(mpms::GetContextFeatureBytes("OTHER", output_sequence),
              testing::ElementsAre("value_3", "value_4"));
}

TEST_F(PackMediaSequenceCalculatorTest, ReplaceTwoContextByteLists) {
  SetUpCalculator(
      /*input_streams=*/{"BYTES_CONTEXT_FEATURE_TEST:test",
                         "BYTES_CONTEXT_FEATURE_OTHER:test2"},
      /*features=*/{},
      /*output_only_if_all_present=*/false, /*replace_instead_of_append=*/true);
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  mpms::SetContextFeatureBytes("TEST", {"existing_value_1", "existing_value_2"},
                               input_sequence.get());
  mpms::SetContextFeatureBytes(
      "OTHER", {"existing_value_3", "existing_value_4"}, input_sequence.get());

  const std::vector<std::string> vb_1 = {"value_1", "value_2"};
  runner_->MutableInputs()
      ->Tag(kBytesContextFeatureTestTag)
      .packets.push_back(MakePacket<std::vector<std::string>>(vb_1).At(
          Timestamp::PostStream()));
  const std::vector<std::string> vb_2 = {"value_3", "value_4"};
  runner_->MutableInputs()
      ->Tag(kBytesContextFeatureOtherTag)
      .packets.push_back(MakePacket<std::vector<std::string>>(vb_2).At(
          Timestamp::PostStream()));

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_THAT(mpms::GetContextFeatureBytes("TEST", output_sequence),
              testing::ElementsAre("value_1", "value_2"));
  ASSERT_THAT(mpms::GetContextFeatureBytes("OTHER", output_sequence),
              testing::ElementsAre("value_3", "value_4"));
}

TEST_F(PackMediaSequenceCalculatorTest, AppendTwoContextByteLists) {
  SetUpCalculator(
      /*input_streams=*/{"BYTES_CONTEXT_FEATURE_TEST:test",
                         "BYTES_CONTEXT_FEATURE_OTHER:test2"},
      /*features=*/{},
      /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/false);
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  mpms::SetContextFeatureBytes("TEST", {"existing_value_1", "existing_value_2"},
                               input_sequence.get());
  mpms::SetContextFeatureBytes(
      "OTHER", {"existing_value_3", "existing_value_4"}, input_sequence.get());

  const std::vector<std::string> vb_1 = {"value_1", "value_2"};
  runner_->MutableInputs()
      ->Tag(kBytesContextFeatureTestTag)
      .packets.push_back(MakePacket<std::vector<std::string>>(vb_1).At(
          Timestamp::PostStream()));
  const std::vector<std::string> vb_2 = {"value_3", "value_4"};
  runner_->MutableInputs()
      ->Tag(kBytesContextFeatureOtherTag)
      .packets.push_back(MakePacket<std::vector<std::string>>(vb_2).At(
          Timestamp::PostStream()));

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_THAT(mpms::GetContextFeatureBytes("TEST", output_sequence),
              testing::ElementsAre("existing_value_1", "existing_value_2",
                                   "value_1", "value_2"));
  ASSERT_THAT(mpms::GetContextFeatureBytes("OTHER", output_sequence),
              testing::ElementsAre("existing_value_3", "existing_value_4",
                                   "value_3", "value_4"));
}

TEST_F(PackMediaSequenceCalculatorTest, PacksAdditionalContext) {
  tf::Features context;
  (*context.mutable_feature())["TEST"].mutable_bytes_list()->add_value("YES");
  (*context.mutable_feature())["OTHER"].mutable_bytes_list()->add_value("NO");
  SetUpCalculator({"IMAGE:images"}, context, false, true);

  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());
  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(
      cv::imencode(".jpg", image, bytes, {cv::IMWRITE_HDR_COMPRESSION, 1}));
  OpenCvImageEncoderCalculatorResults encoded_image;
  encoded_image.set_encoded_image(bytes.data(), bytes.size());
  auto image_ptr =
      ::absl::make_unique<OpenCvImageEncoderCalculatorResults>(encoded_image);
  runner_->MutableInputs()->Tag(kImageTag).packets.push_back(
      Adopt(image_ptr.release()).At(Timestamp(0)));

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_TRUE(mpms::HasContext(output_sequence, "TEST"));
  ASSERT_TRUE(mpms::HasContext(output_sequence, "OTHER"));
  ASSERT_EQ(mpms::GetContext(output_sequence, "TEST").bytes_list().value(0),
            "YES");
  ASSERT_EQ(mpms::GetContext(output_sequence, "OTHER").bytes_list().value(0),
            "NO");
}

TEST_F(PackMediaSequenceCalculatorTest, PacksTwoForwardFlowEncodeds) {
  SetUpCalculator({"FORWARD_FLOW_ENCODED:flow"}, {}, false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());

  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(
      cv::imencode(".jpg", image, bytes, {cv::IMWRITE_HDR_COMPRESSION, 1}));
  std::string test_flow_string(bytes.begin(), bytes.end());
  OpenCvImageEncoderCalculatorResults encoded_flow;
  encoded_flow.set_encoded_image(test_flow_string);
  encoded_flow.set_width(2);
  encoded_flow.set_height(1);

  int num_flows = 2;
  for (int i = 0; i < num_flows; ++i) {
    auto flow_ptr =
        ::absl::make_unique<OpenCvImageEncoderCalculatorResults>(encoded_flow);
    runner_->MutableInputs()
        ->Tag(kForwardFlowEncodedTag)
        .packets.push_back(Adopt(flow_ptr.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(test_video_id, mpms::GetClipMediaId(output_sequence));
  ASSERT_EQ(num_flows, mpms::GetForwardFlowTimestampSize(output_sequence));
  ASSERT_EQ(num_flows, mpms::GetForwardFlowEncodedSize(output_sequence));
  for (int i = 0; i < num_flows; ++i) {
    ASSERT_EQ(i, mpms::GetForwardFlowTimestampAt(output_sequence, i));
    ASSERT_EQ(test_flow_string,
              mpms::GetForwardFlowEncodedAt(output_sequence, i));
  }
}

TEST_F(PackMediaSequenceCalculatorTest, PacksTwoBBoxDetections) {
  SetUpCalculator({"BBOX_PREDICTED:detections"}, {}, false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());
  int height = 480;
  int width = 640;
  mpms::SetImageHeight(height, input_sequence.get());
  mpms::SetImageWidth(width, input_sequence.get());

  int num_vectors = 2;
  for (int i = 0; i < num_vectors; ++i) {
    auto detections = ::absl::make_unique<::std::vector<Detection>>();
    Detection detection;
    detection.add_label("absolute bbox");
    detection.add_label_id(0);
    detection.add_score(0.5);
    Location::CreateBBoxLocation(0, height / 2, width / 2, height / 2)
        .ConvertToProto(detection.mutable_location_data());
    detections->push_back(detection);

    detection = Detection();
    detection.add_label("relative bbox");
    detection.add_label_id(1);
    detection.add_score(0.75);
    Location::CreateRelativeBBoxLocation(0, 0.5, 0.5, 0.5)
        .ConvertToProto(detection.mutable_location_data());
    detections->push_back(detection);

    // The mask detection should be ignored in the output.
    detection = Detection();
    detection.add_label("mask");
    detection.add_score(1.0);
    cv::Mat image(2, 3, CV_8UC1, cv::Scalar(0));
    mediapipe::CreateCvMaskLocation<uint8_t>(image).ConvertToProto(
        detection.mutable_location_data());
    detections->push_back(detection);

    runner_->MutableInputs()
        ->Tag(kBboxPredictedTag)
        .packets.push_back(Adopt(detections.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(test_video_id, mpms::GetClipMediaId(output_sequence));
  ASSERT_EQ(height, mpms::GetImageHeight(output_sequence));
  ASSERT_EQ(width, mpms::GetImageWidth(output_sequence));
  ASSERT_EQ(num_vectors, mpms::GetPredictedBBoxSize(output_sequence));
  ASSERT_EQ(num_vectors, mpms::GetPredictedBBoxTimestampSize(output_sequence));
  ASSERT_EQ(0, mpms::GetClassSegmentationEncodedSize(output_sequence));
  ASSERT_EQ(0, mpms::GetClassSegmentationTimestampSize(output_sequence));
  for (int i = 0; i < num_vectors; ++i) {
    ASSERT_EQ(i, mpms::GetPredictedBBoxTimestampAt(output_sequence, i));
    auto bboxes = mpms::GetPredictedBBoxAt(output_sequence, i);
    ASSERT_EQ(2, bboxes.size());
    for (int j = 0; j < bboxes.size(); ++j) {
      auto rect = bboxes[j].GetRelativeBBox();
      ASSERT_NEAR(0, rect.xmin(), 0.001);
      ASSERT_NEAR(0.5, rect.ymin(), 0.001);
      ASSERT_NEAR(0.5, rect.xmax(), 0.001);
      ASSERT_NEAR(1.0, rect.ymax(), 0.001);
    }
    auto class_strings =
        mpms::GetPredictedBBoxLabelStringAt(output_sequence, i);
    ASSERT_EQ("absolute bbox", class_strings[0]);
    ASSERT_EQ("relative bbox", class_strings[1]);
    auto class_indices = mpms::GetPredictedBBoxLabelIndexAt(output_sequence, i);
    ASSERT_EQ(0, class_indices[0]);
    ASSERT_EQ(1, class_indices[1]);
    auto class_scores =
        mpms::GetPredictedBBoxLabelConfidenceAt(output_sequence, i);
    ASSERT_FLOAT_EQ(0.5, class_scores[0]);
    ASSERT_FLOAT_EQ(0.75, class_scores[1]);
  }
}

TEST_F(PackMediaSequenceCalculatorTest, PacksBBoxWithoutImageDims) {
  SetUpCalculator({"BBOX_PREDICTED:detections"}, {}, false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());
  int height = 480;
  int width = 640;
  int num_vectors = 2;
  for (int i = 0; i < num_vectors; ++i) {
    auto detections = ::absl::make_unique<::std::vector<Detection>>();
    Detection detection;
    detection.add_label("absolute bbox");
    detection.add_label_id(0);
    detection.add_score(0.5);
    Location::CreateBBoxLocation(0, height / 2, width / 2, height / 2)
        .ConvertToProto(detection.mutable_location_data());
    detections->push_back(detection);

    detection = Detection();
    detection.add_label("relative bbox");
    detection.add_label_id(1);
    detection.add_score(0.75);
    Location::CreateRelativeBBoxLocation(0, 0.5, 0.5, 0.5)
        .ConvertToProto(detection.mutable_location_data());
    detections->push_back(detection);

    // The mask detection should be ignored in the output.
    detection = Detection();
    detection.add_label("mask");
    detection.add_score(1.0);
    cv::Mat image(2, 3, CV_8UC1, cv::Scalar(0));
    mediapipe::CreateCvMaskLocation<uint8_t>(image).ConvertToProto(
        detection.mutable_location_data());
    detections->push_back(detection);

    runner_->MutableInputs()
        ->Tag(kBboxPredictedTag)
        .packets.push_back(Adopt(detections.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  auto status = runner_->Run();
  EXPECT_EQ(absl::StatusCode::kInvalidArgument, status.code());
}

TEST_F(PackMediaSequenceCalculatorTest, PacksBBoxWithImages) {
  SetUpCalculator({"BBOX_PREDICTED:detections", "IMAGE:images"}, {}, false,
                  true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());
  int height = 480;
  int width = 640;
  int num_vectors = 2;
  for (int i = 0; i < num_vectors; ++i) {
    auto detections = ::absl::make_unique<::std::vector<Detection>>();
    Detection detection;
    detection.add_label("absolute bbox");
    detection.add_label_id(0);
    detection.add_score(0.5);
    Location::CreateBBoxLocation(0, height / 2, width / 2, height / 2)
        .ConvertToProto(detection.mutable_location_data());
    detections->push_back(detection);

    detection = Detection();
    detection.add_label("relative bbox");
    detection.add_label_id(1);
    detection.add_score(0.75);
    Location::CreateRelativeBBoxLocation(0, 0.5, 0.5, 0.5)
        .ConvertToProto(detection.mutable_location_data());
    detections->push_back(detection);

    // The mask detection should be ignored in the output.
    detection = Detection();
    detection.add_label("mask");
    detection.add_score(1.0);
    cv::Mat image(2, 3, CV_8UC1, cv::Scalar(0));
    mediapipe::CreateCvMaskLocation<uint8_t>(image).ConvertToProto(
        detection.mutable_location_data());
    detections->push_back(detection);

    runner_->MutableInputs()
        ->Tag(kBboxPredictedTag)
        .packets.push_back(Adopt(detections.release()).At(Timestamp(i)));
  }
  cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(
      cv::imencode(".jpg", image, bytes, {cv::IMWRITE_HDR_COMPRESSION, 1}));
  OpenCvImageEncoderCalculatorResults encoded_image;
  encoded_image.set_encoded_image(bytes.data(), bytes.size());
  encoded_image.set_width(width);
  encoded_image.set_height(height);

  int num_images = 2;
  for (int i = 0; i < num_images; ++i) {
    auto image_ptr =
        ::absl::make_unique<OpenCvImageEncoderCalculatorResults>(encoded_image);
    runner_->MutableInputs()->Tag(kImageTag).packets.push_back(
        Adopt(image_ptr.release()).At(Timestamp(i)));
  }
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(test_video_id, mpms::GetClipMediaId(output_sequence));
  ASSERT_EQ(height, mpms::GetImageHeight(output_sequence));
  ASSERT_EQ(width, mpms::GetImageWidth(output_sequence));
  ASSERT_EQ(num_vectors, mpms::GetPredictedBBoxSize(output_sequence));
  ASSERT_EQ(num_vectors, mpms::GetPredictedBBoxTimestampSize(output_sequence));
  ASSERT_EQ(0, mpms::GetClassSegmentationEncodedSize(output_sequence));
  ASSERT_EQ(0, mpms::GetClassSegmentationTimestampSize(output_sequence));
  for (int i = 0; i < num_vectors; ++i) {
    ASSERT_EQ(i, mpms::GetPredictedBBoxTimestampAt(output_sequence, i));
    auto bboxes = mpms::GetPredictedBBoxAt(output_sequence, i);
    ASSERT_EQ(2, bboxes.size());
    for (int j = 0; j < bboxes.size(); ++j) {
      auto rect = bboxes[j].GetRelativeBBox();
      ASSERT_NEAR(0, rect.xmin(), 0.001);
      ASSERT_NEAR(0.5, rect.ymin(), 0.001);
      ASSERT_NEAR(0.5, rect.xmax(), 0.001);
      ASSERT_NEAR(1.0, rect.ymax(), 0.001);
    }
    auto class_strings =
        mpms::GetPredictedBBoxLabelStringAt(output_sequence, i);
    ASSERT_EQ("absolute bbox", class_strings[0]);
    ASSERT_EQ("relative bbox", class_strings[1]);
    auto class_indices = mpms::GetPredictedBBoxLabelIndexAt(output_sequence, i);
    ASSERT_EQ(0, class_indices[0]);
    ASSERT_EQ(1, class_indices[1]);
    auto class_scores =
        mpms::GetPredictedBBoxLabelConfidenceAt(output_sequence, i);
    ASSERT_FLOAT_EQ(0.5, class_scores[0]);
    ASSERT_FLOAT_EQ(0.75, class_scores[1]);
  }
}

TEST_F(PackMediaSequenceCalculatorTest, PacksTwoKeypoints) {
  SetUpCalculator({"KEYPOINTS_TEST:keypoints"}, {}, false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());

  absl::flat_hash_map<std::string, std::vector<std::pair<float, float>>>
      points = {{"HEAD", {{0.1, 0.2}, {0.3, 0.4}}}, {"TAIL", {{0.5, 0.6}}}};
  runner_->MutableInputs()
      ->Tag(kKeypointsTestTag)
      .packets.push_back(PointToForeign(&points).At(Timestamp(0)));
  runner_->MutableInputs()
      ->Tag(kKeypointsTestTag)
      .packets.push_back(PointToForeign(&points).At(Timestamp(1)));
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(test_video_id, mpms::GetClipMediaId(output_sequence));
  ASSERT_EQ(2, mpms::GetBBoxPointSize("TEST/HEAD", output_sequence));
  ASSERT_EQ(2, mpms::GetBBoxPointSize("TEST/TAIL", output_sequence));
  ASSERT_NEAR(0.2,
              mpms::GetBBoxPointAt("TEST/HEAD", output_sequence, 0)[0].second,
              0.001);
  ASSERT_NEAR(0.5,
              mpms::GetBBoxPointAt("TEST/TAIL", output_sequence, 1)[0].first,
              0.001);
}

TEST_F(PackMediaSequenceCalculatorTest, PacksTwoMaskDetections) {
  SetUpCalculator({"CLASS_SEGMENTATION:detections"}, {}, false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());
  int height = 480;
  int width = 640;
  mpms::SetImageHeight(height, input_sequence.get());
  mpms::SetImageWidth(width, input_sequence.get());

  int num_vectors = 2;
  for (int i = 0; i < num_vectors; ++i) {
    auto detections = ::absl::make_unique<::std::vector<Detection>>();
    Detection detection;
    detection = Detection();
    detection.add_label("mask");
    detection.add_score(1.0);
    cv::Mat image(2, 3, CV_8UC1, cv::Scalar(0));
    mediapipe::CreateCvMaskLocation<uint8_t>(image).ConvertToProto(
        detection.mutable_location_data());

    detections->push_back(detection);

    runner_->MutableInputs()
        ->Tag(kClassSegmentationTag)
        .packets.push_back(Adopt(detections.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(test_video_id, mpms::GetClipMediaId(output_sequence));
  ASSERT_EQ(height, mpms::GetImageHeight(output_sequence));
  ASSERT_EQ(width, mpms::GetImageWidth(output_sequence));
  ASSERT_EQ(2, mpms::GetClassSegmentationEncodedSize(output_sequence));
  ASSERT_EQ(2, mpms::GetClassSegmentationTimestampSize(output_sequence));
  for (int i = 0; i < num_vectors; ++i) {
    ASSERT_EQ(i, mpms::GetClassSegmentationTimestampAt(output_sequence, i));
  }
  ASSERT_THAT(mpms::GetClassSegmentationClassLabelString(output_sequence),
              testing::ElementsAreArray(::std::vector<std::string>({"mask"})));
}

TEST_F(PackMediaSequenceCalculatorTest, PackThreeClipLabels) {
  SetUpCalculator(
      /*input_streams=*/{"CLIP_LABEL_TEST:test", "CLIP_LABEL_OTHER:test2",
                         "CLIP_LABEL_ANOTHER:test3"},
      /*features=*/{}, /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  Detection detection_1;
  detection_1.add_label("label_1");
  detection_1.add_label("label_2");
  detection_1.add_label_id(1);
  detection_1.add_label_id(2);
  detection_1.add_score(0.1);
  detection_1.add_score(0.2);
  runner_->MutableInputs()
      ->Tag(kClipLabelTestTag)
      .packets.push_back(MakePacket<Detection>(detection_1).At(Timestamp(1)));
  // No label ID for detection_2.
  Detection detection_2;
  detection_2.add_label("label_3");
  detection_2.add_label("label_4");
  detection_2.add_score(0.3);
  detection_2.add_score(0.4);
  runner_->MutableInputs()
      ->Tag(kClipLabelOtherTag)
      .packets.push_back(MakePacket<Detection>(detection_2).At(Timestamp(2)));
  // No label for detection_3.
  Detection detection_3;
  detection_3.add_label_id(3);
  detection_3.add_label_id(4);
  detection_3.add_score(0.3);
  detection_3.add_score(0.4);
  runner_->MutableInputs()
      ->Tag(kClipLabelAnotherTag)
      .packets.push_back(MakePacket<Detection>(detection_3).At(Timestamp(3)));

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_THAT(mpms::GetClipLabelString("TEST", output_sequence),
              testing::ElementsAre("label_1", "label_2"));
  ASSERT_THAT(mpms::GetClipLabelIndex("TEST", output_sequence),
              testing::ElementsAre(1, 2));
  ASSERT_THAT(mpms::GetClipLabelConfidence("TEST", output_sequence),
              testing::ElementsAre(0.1, 0.2));
  ASSERT_THAT(mpms::GetClipLabelString("OTHER", output_sequence),
              testing::ElementsAre("label_3", "label_4"));
  ASSERT_FALSE(mpms::HasClipLabelIndex("OTHER", output_sequence));
  ASSERT_THAT(mpms::GetClipLabelConfidence("OTHER", output_sequence),
              testing::ElementsAre(0.3, 0.4));
  ASSERT_FALSE(mpms::HasClipLabelString("ANOTHER", output_sequence));
  ASSERT_THAT(mpms::GetClipLabelIndex("ANOTHER", output_sequence),
              testing::ElementsAre(3, 4));
  ASSERT_THAT(mpms::GetClipLabelConfidence("ANOTHER", output_sequence),
              testing::ElementsAre(0.3, 0.4));
}

TEST_F(PackMediaSequenceCalculatorTest, PackTwoClipLabels_EmptyScore) {
  SetUpCalculator(
      /*input_streams=*/{"CLIP_LABEL_TEST:test", "CLIP_LABEL_OTHER:test2"},
      /*features=*/{}, /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  // No score in detection_1. detection_1 is ignored.
  Detection detection_1;
  detection_1.add_label("label_1");
  detection_1.add_label("label_2");
  runner_->MutableInputs()
      ->Tag(kClipLabelTestTag)
      .packets.push_back(MakePacket<Detection>(detection_1).At(Timestamp(1)));
  Detection detection_2;
  detection_2.add_label("label_3");
  detection_2.add_label("label_4");
  detection_2.add_score(0.3);
  detection_2.add_score(0.4);
  runner_->MutableInputs()
      ->Tag(kClipLabelOtherTag)
      .packets.push_back(MakePacket<Detection>(detection_2).At(Timestamp(2)));
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_FALSE(mpms::HasClipLabelString("TEST", output_sequence));
  ASSERT_FALSE(mpms::HasClipLabelIndex("TEST", output_sequence));
  ASSERT_FALSE(mpms::HasClipLabelConfidence("TEST", output_sequence));
  ASSERT_THAT(mpms::GetClipLabelString("OTHER", output_sequence),
              testing::ElementsAre("label_3", "label_4"));
  ASSERT_FALSE(mpms::HasClipLabelIndex("OTHER", output_sequence));
  ASSERT_THAT(mpms::GetClipLabelConfidence("OTHER", output_sequence),
              testing::ElementsAre(0.3, 0.4));
}

TEST_F(PackMediaSequenceCalculatorTest, PackTwoClipLabels_NoLabelOrLabelIndex) {
  SetUpCalculator(
      /*input_streams=*/{"CLIP_LABEL_TEST:test", "CLIP_LABEL_OTHER:test2"},
      /*features=*/{}, /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  // No label or label_index in detection_1.
  Detection detection_1;
  detection_1.add_score(0.1);
  runner_->MutableInputs()
      ->Tag(kClipLabelTestTag)
      .packets.push_back(MakePacket<Detection>(detection_1).At(Timestamp(1)));
  Detection detection_2;
  detection_2.add_label("label_3");
  detection_2.add_label("label_4");
  detection_2.add_score(0.3);
  detection_2.add_score(0.4);
  runner_->MutableInputs()
      ->Tag(kClipLabelOtherTag)
      .packets.push_back(MakePacket<Detection>(detection_2).At(Timestamp(2)));
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  ASSERT_THAT(
      runner_->Run(),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "detection.label and detection.label_id can't be both empty")));
}

TEST_F(PackMediaSequenceCalculatorTest, PackTwoClipLabels_AddEmptyLabels) {
  SetUpCalculator(
      /*input_streams=*/{"CLIP_LABEL_TEST:test"},
      /*features=*/{}, /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/true, /*output_as_zero_timestamp=*/false,
      /*add_empty_labels=*/true);
  auto input_sequence = std::make_unique<tf::SequenceExample>();

  // No label or label_index in detection_1.
  Detection detection;
  runner_->MutableInputs()
      ->Tag(kClipLabelTestTag)
      .packets.push_back(MakePacket<Detection>(detection).At(Timestamp(1)));
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_THAT(mpms::GetClipLabelString("TEST", output_sequence),
              testing::ElementsAre());
  ASSERT_THAT(mpms::GetClipLabelConfidence("TEST", output_sequence),
              testing::ElementsAre());
}

TEST_F(PackMediaSequenceCalculatorTest,
       PackTwoClipLabels_DifferentLabelScoreSize) {
  SetUpCalculator(
      /*input_streams=*/{"CLIP_LABEL_TEST:test", "CLIP_LABEL_OTHER:test2"},
      /*features=*/{}, /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  // 2 labels and 1 score in detection_1.
  Detection detection_1;
  detection_1.add_label("label_1");
  detection_1.add_label("label_2");
  detection_1.add_score(0.1);
  runner_->MutableInputs()
      ->Tag(kClipLabelTestTag)
      .packets.push_back(MakePacket<Detection>(detection_1).At(Timestamp(1)));
  Detection detection_2;
  detection_2.add_label("label_3");
  detection_2.add_label("label_4");
  detection_2.add_score(0.3);
  detection_2.add_score(0.4);
  runner_->MutableInputs()
      ->Tag(kClipLabelOtherTag)
      .packets.push_back(MakePacket<Detection>(detection_2).At(Timestamp(2)));
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  ASSERT_THAT(
      runner_->Run(),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Different size of detection.label and detection.score")));
}

TEST_F(PackMediaSequenceCalculatorTest,
       PackTwoClipLabels_DifferentLabelIdSize) {
  SetUpCalculator(
      /*input_streams=*/{"CLIP_LABEL_TEST:test", "CLIP_LABEL_OTHER:test2"},
      /*features=*/{}, /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  // 2 scores and 1 label_id in detection_1.
  Detection detection_1;
  detection_1.add_label("label_1");
  detection_1.add_label("label_2");
  detection_1.add_label_id(1);
  detection_1.add_score(0.1);
  detection_1.add_score(0.2);
  runner_->MutableInputs()
      ->Tag(kClipLabelTestTag)
      .packets.push_back(MakePacket<Detection>(detection_1).At(Timestamp(1)));
  Detection detection_2;
  detection_2.add_label("label_3");
  detection_2.add_label("label_4");
  detection_2.add_score(0.3);
  detection_2.add_score(0.4);
  runner_->MutableInputs()
      ->Tag(kClipLabelOtherTag)
      .packets.push_back(MakePacket<Detection>(detection_2).At(Timestamp(2)));
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  ASSERT_THAT(
      runner_->Run(),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Different size of detection.label_id and detection.score")));
}

TEST_F(PackMediaSequenceCalculatorTest, ReplaceTwoClipLabels) {
  // Replace existing clip/label/string and clip/label/confidence values for
  // the prefixes.
  SetUpCalculator(
      /*input_streams=*/{"CLIP_LABEL_TEST:test", "CLIP_LABEL_OTHER:test2"},
      /*features=*/{}, /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  mpms::SetClipLabelString("TEST", {"old_label_1", "old_label_2"},
                           input_sequence.get());
  mpms::SetClipLabelConfidence("TEST", {0.1, 0.2}, input_sequence.get());
  mpms::SetClipLabelString("OTHER", {"old_label_3", "old_label_4"},
                           input_sequence.get());
  mpms::SetClipLabelConfidence("OTHER", {0.3, 0.4}, input_sequence.get());

  Detection detection_1;
  detection_1.add_label("label_1");
  detection_1.add_label("label_2");
  detection_1.add_label_id(1);
  detection_1.add_label_id(2);
  detection_1.add_score(0.9);
  detection_1.add_score(0.8);
  runner_->MutableInputs()
      ->Tag(kClipLabelTestTag)
      .packets.push_back(MakePacket<Detection>(detection_1).At(Timestamp(1)));
  Detection detection_2;
  detection_2.add_label("label_3");
  detection_2.add_label("label_4");
  detection_2.add_label_id(3);
  detection_2.add_label_id(4);
  detection_2.add_score(0.7);
  detection_2.add_score(0.6);
  runner_->MutableInputs()
      ->Tag(kClipLabelOtherTag)
      .packets.push_back(MakePacket<Detection>(detection_2).At(Timestamp(2)));
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_THAT(mpms::GetClipLabelString("TEST", output_sequence),
              testing::ElementsAre("label_1", "label_2"));
  ASSERT_THAT(mpms::GetClipLabelIndex("TEST", output_sequence),
              testing::ElementsAre(1, 2));
  ASSERT_THAT(mpms::GetClipLabelConfidence("TEST", output_sequence),
              testing::ElementsAre(0.9, 0.8));
  ASSERT_THAT(mpms::GetClipLabelString("OTHER", output_sequence),
              testing::ElementsAre("label_3", "label_4"));
  ASSERT_THAT(mpms::GetClipLabelIndex("OTHER", output_sequence),
              testing::ElementsAre(3, 4));
  ASSERT_THAT(mpms::GetClipLabelConfidence("OTHER", output_sequence),
              testing::ElementsAre(0.7, 0.6));
}

TEST_F(PackMediaSequenceCalculatorTest, AppendTwoClipLabels) {
  // Append to the existing clip/label/string and clip/label/confidence values
  // for the prefixes.
  SetUpCalculator(
      /*input_streams=*/{"CLIP_LABEL_TEST:test", "CLIP_LABEL_OTHER:test2"},
      /*features=*/{}, /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/false);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  mpms::SetClipLabelString("TEST", {"old_label_1", "old_label_2"},
                           input_sequence.get());
  mpms::SetClipLabelIndex("TEST", {1, 2}, input_sequence.get());
  mpms::SetClipLabelConfidence("TEST", {0.1, 0.2}, input_sequence.get());
  mpms::SetClipLabelString("OTHER", {"old_label_3", "old_label_4"},
                           input_sequence.get());
  mpms::SetClipLabelIndex("OTHER", {3, 4}, input_sequence.get());
  mpms::SetClipLabelConfidence("OTHER", {0.3, 0.4}, input_sequence.get());

  Detection detection_1;
  detection_1.add_label("label_1");
  detection_1.add_label("label_2");
  detection_1.add_label_id(9);
  detection_1.add_label_id(8);
  detection_1.add_score(0.9);
  detection_1.add_score(0.8);
  runner_->MutableInputs()
      ->Tag(kClipLabelTestTag)
      .packets.push_back(MakePacket<Detection>(detection_1).At(Timestamp(1)));
  Detection detection_2;
  detection_2.add_label("label_3");
  detection_2.add_label("label_4");
  detection_2.add_label_id(7);
  detection_2.add_label_id(6);
  detection_2.add_score(0.7);
  detection_2.add_score(0.6);
  runner_->MutableInputs()
      ->Tag(kClipLabelOtherTag)
      .packets.push_back(MakePacket<Detection>(detection_2).At(Timestamp(2)));
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_THAT(
      mpms::GetClipLabelString("TEST", output_sequence),
      testing::ElementsAre("old_label_1", "old_label_2", "label_1", "label_2"));
  ASSERT_THAT(mpms::GetClipLabelIndex("TEST", output_sequence),
              testing::ElementsAre(1, 2, 9, 8));
  ASSERT_THAT(mpms::GetClipLabelConfidence("TEST", output_sequence),
              testing::ElementsAre(0.1, 0.2, 0.9, 0.8));
  ASSERT_THAT(
      mpms::GetClipLabelString("OTHER", output_sequence),
      testing::ElementsAre("old_label_3", "old_label_4", "label_3", "label_4"));
  ASSERT_THAT(mpms::GetClipLabelIndex("OTHER", output_sequence),
              testing::ElementsAre(3, 4, 7, 6));
  ASSERT_THAT(mpms::GetClipLabelConfidence("OTHER", output_sequence),
              testing::ElementsAre(0.3, 0.4, 0.7, 0.6));
}

TEST_F(PackMediaSequenceCalculatorTest,
       DifferentClipLabelScoreAndConfidenceSize) {
  SetUpCalculator(
      /*input_streams=*/{"CLIP_LABEL_TEST:test", "CLIP_LABEL_OTHER:test2"},
      /*features=*/{}, /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  Detection detection_1;
  // 2 labels and 1 score.
  detection_1.add_label("label_1");
  detection_1.add_label("label_2");
  detection_1.add_score(0.1);
  runner_->MutableInputs()
      ->Tag(kClipLabelTestTag)
      .packets.push_back(MakePacket<Detection>(detection_1).At(Timestamp(1)));
  Detection detection_2;
  detection_2.add_label("label_3");
  detection_2.add_label("label_4");
  detection_2.add_score(0.3);
  detection_2.add_score(0.4);
  runner_->MutableInputs()
      ->Tag(kClipLabelOtherTag)
      .packets.push_back(MakePacket<Detection>(detection_2).At(Timestamp(2)));
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  ASSERT_THAT(runner_->Run(),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(PackMediaSequenceCalculatorTest, AddClipMediaId) {
  SetUpCalculator(
      /*input_streams=*/{"FLOAT_FEATURE_TEST:test",
                         "FLOAT_FEATURE_OTHER:test2"},
      /*features=*/{},
      /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/true,
      /*output_as_zero_timestamp=*/false, /*add_empty_labels=*/false,
      /*input_side_packets=*/
      {"SEQUENCE_EXAMPLE:input_sequence", "CLIP_MEDIA_ID:video_id"});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  const std::string test_video_id = "test_video_id";

  int num_timesteps = 2;
  for (int i = 0; i < num_timesteps; ++i) {
    auto vf_ptr = ::absl::make_unique<std::vector<float>>(2, 2 << i);
    runner_->MutableInputs()
        ->Tag(kFloatFeatureTestTag)
        .packets.push_back(Adopt(vf_ptr.release()).At(Timestamp(i)));
    vf_ptr = ::absl::make_unique<std::vector<float>>(2, 2 << i);
    runner_->MutableInputs()
        ->Tag(kFloatFeatureOtherTag)
        .packets.push_back(Adopt(vf_ptr.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kClipMediaIdTag) =
      MakePacket<std::string>(test_video_id);
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(test_video_id, mpms::GetClipMediaId(output_sequence));
}

TEST_F(PackMediaSequenceCalculatorTest, ReplaceClipMediaId) {
  SetUpCalculator(
      /*input_streams=*/{"FLOAT_FEATURE_TEST:test",
                         "FLOAT_FEATURE_OTHER:test2"},
      /*features=*/{},
      /*output_only_if_all_present=*/false,
      /*replace_instead_of_append=*/true,
      /*output_as_zero_timestamp=*/false, /*add_empty_labels=*/false,
      /*input_side_packets=*/
      {"SEQUENCE_EXAMPLE:input_sequence", "CLIP_MEDIA_ID:video_id"});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  const std::string existing_video_id = "existing_video_id";
  mpms::SetClipMediaId(existing_video_id, input_sequence.get());
  const std::string test_video_id = "test_video_id";

  int num_timesteps = 2;
  for (int i = 0; i < num_timesteps; ++i) {
    auto vf_ptr = ::absl::make_unique<std::vector<float>>(2, 2 << i);
    runner_->MutableInputs()
        ->Tag(kFloatFeatureTestTag)
        .packets.push_back(Adopt(vf_ptr.release()).At(Timestamp(i)));
    vf_ptr = ::absl::make_unique<std::vector<float>>(2, 2 << i);
    runner_->MutableInputs()
        ->Tag(kFloatFeatureOtherTag)
        .packets.push_back(Adopt(vf_ptr.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kClipMediaIdTag) =
      MakePacket<std::string>(test_video_id).At(Timestamp(0));
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(test_video_id, mpms::GetClipMediaId(output_sequence));
}

TEST_F(PackMediaSequenceCalculatorTest, MissingStreamOK) {
  SetUpCalculator(
      {"FORWARD_FLOW_ENCODED:flow", "FLOAT_FEATURE_I3D_FLOW:feature"}, {},
      false, false);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());

  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(
      cv::imencode(".jpg", image, bytes, {cv::IMWRITE_HDR_COMPRESSION, 1}));
  std::string test_flow_string(bytes.begin(), bytes.end());
  OpenCvImageEncoderCalculatorResults encoded_flow;
  encoded_flow.set_encoded_image(test_flow_string);
  encoded_flow.set_width(2);
  encoded_flow.set_height(1);

  int num_flows = 2;
  for (int i = 0; i < num_flows; ++i) {
    auto flow_ptr =
        ::absl::make_unique<OpenCvImageEncoderCalculatorResults>(encoded_flow);
    runner_->MutableInputs()
        ->Tag(kForwardFlowEncodedTag)
        .packets.push_back(Adopt(flow_ptr.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(test_video_id, mpms::GetClipMediaId(output_sequence));
  ASSERT_EQ(num_flows, mpms::GetForwardFlowTimestampSize(output_sequence));
  ASSERT_EQ(num_flows, mpms::GetForwardFlowEncodedSize(output_sequence));
  for (int i = 0; i < num_flows; ++i) {
    ASSERT_EQ(i, mpms::GetForwardFlowTimestampAt(output_sequence, i));
    ASSERT_EQ(test_flow_string,
              mpms::GetForwardFlowEncodedAt(output_sequence, i));
  }
}

TEST_F(PackMediaSequenceCalculatorTest, MissingStreamNotOK) {
  SetUpCalculator(
      {"FORWARD_FLOW_ENCODED:flow", "FLOAT_FEATURE_I3D_FLOW:feature"}, {}, true,
      false);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());
  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(
      cv::imencode(".jpg", image, bytes, {cv::IMWRITE_HDR_COMPRESSION, 1}));
  std::string test_flow_string(bytes.begin(), bytes.end());
  OpenCvImageEncoderCalculatorResults encoded_flow;
  encoded_flow.set_encoded_image(test_flow_string);
  encoded_flow.set_width(2);
  encoded_flow.set_height(1);

  int num_flows = 2;
  for (int i = 0; i < num_flows; ++i) {
    auto flow_ptr =
        ::absl::make_unique<OpenCvImageEncoderCalculatorResults>(encoded_flow);
    runner_->MutableInputs()
        ->Tag(kForwardFlowEncodedTag)
        .packets.push_back(Adopt(flow_ptr.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  absl::Status status = runner_->Run();
  EXPECT_FALSE(status.ok());
}

TEST_F(PackMediaSequenceCalculatorTest, TestReplacingImages) {
  SetUpCalculator({"IMAGE:images"}, {}, false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());
  mpms::AddImageEncoded("one", input_sequence.get());
  mpms::AddImageEncoded("two", input_sequence.get());
  mpms::AddImageTimestamp(1, input_sequence.get());
  mpms::AddImageTimestamp(2, input_sequence.get());

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(test_video_id, mpms::GetClipMediaId(output_sequence));
  ASSERT_EQ(0, mpms::GetImageTimestampSize(output_sequence));
  ASSERT_EQ(0, mpms::GetImageEncodedSize(output_sequence));
}

TEST_F(PackMediaSequenceCalculatorTest, TestReplacingFlowImages) {
  SetUpCalculator({"FORWARD_FLOW_ENCODED:images"}, {}, false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());
  mpms::AddForwardFlowEncoded("one", input_sequence.get());
  mpms::AddForwardFlowEncoded("two", input_sequence.get());
  mpms::AddForwardFlowTimestamp(1, input_sequence.get());
  mpms::AddForwardFlowTimestamp(2, input_sequence.get());

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(test_video_id, mpms::GetClipMediaId(output_sequence));
  ASSERT_EQ(0, mpms::GetForwardFlowTimestampSize(output_sequence));
  ASSERT_EQ(0, mpms::GetForwardFlowEncodedSize(output_sequence));
}

TEST_F(PackMediaSequenceCalculatorTest, TestReplacingFloatVectors) {
  SetUpCalculator({"FLOAT_FEATURE_TEST:test", "FLOAT_FEATURE_OTHER:test2"}, {},
                  false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  int num_timesteps = 2;
  for (int i = 0; i < num_timesteps; ++i) {
    auto vf_ptr = ::absl::make_unique<std::vector<float>>(2, 2 << i);
    mpms::AddFeatureFloats("TEST", *vf_ptr, input_sequence.get());
    mpms::AddFeatureTimestamp("TEST", i, input_sequence.get());
    vf_ptr = ::absl::make_unique<std::vector<float>>(2, 2 << i);
    mpms::AddFeatureFloats("OTHER", *vf_ptr, input_sequence.get());
    mpms::AddFeatureTimestamp("OTHER", i, input_sequence.get());
  }
  ASSERT_EQ(num_timesteps,
            mpms::GetFeatureTimestampSize("TEST", *input_sequence));
  ASSERT_EQ(num_timesteps, mpms::GetFeatureFloatsSize("TEST", *input_sequence));
  ASSERT_EQ(num_timesteps,
            mpms::GetFeatureTimestampSize("OTHER", *input_sequence));
  ASSERT_EQ(num_timesteps,
            mpms::GetFeatureFloatsSize("OTHER", *input_sequence));
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(0, mpms::GetFeatureTimestampSize("TEST", output_sequence));
  ASSERT_EQ(0, mpms::GetFeatureFloatsSize("TEST", output_sequence));
  ASSERT_EQ(0, mpms::GetFeatureTimestampSize("OTHER", output_sequence));
  ASSERT_EQ(0, mpms::GetFeatureFloatsSize("OTHER", output_sequence));
}

TEST_F(PackMediaSequenceCalculatorTest, TestReplacingBytesVectors) {
  SetUpCalculator({"BYTES_FEATURE_TEST:test", "BYTES_FEATURE_OTHER:test2"}, {},
                  false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  int num_timesteps = 2;
  for (int i = 0; i < num_timesteps; ++i) {
    auto vs_ptr = ::absl::make_unique<std::vector<std::string>>(
        2, absl::StrCat("foo", 2 << i));
    mpms::AddFeatureBytes("TEST", *vs_ptr, input_sequence.get());
    mpms::AddFeatureTimestamp("TEST", i, input_sequence.get());
    vs_ptr = ::absl::make_unique<std::vector<std::string>>(
        2, absl::StrCat("bar", 2 << i));
    mpms::AddFeatureBytes("OTHER", *vs_ptr, input_sequence.get());
    mpms::AddFeatureTimestamp("OTHER", i, input_sequence.get());
  }
  ASSERT_EQ(num_timesteps,
            mpms::GetFeatureTimestampSize("TEST", *input_sequence));
  ASSERT_EQ(num_timesteps, mpms::GetFeatureBytesSize("TEST", *input_sequence));
  ASSERT_EQ(num_timesteps,
            mpms::GetFeatureTimestampSize("OTHER", *input_sequence));
  ASSERT_EQ(num_timesteps, mpms::GetFeatureBytesSize("OTHER", *input_sequence));
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(0, mpms::GetFeatureTimestampSize("TEST", output_sequence));
  ASSERT_EQ(0, mpms::GetFeatureFloatsSize("TEST", output_sequence));
  ASSERT_EQ(0, mpms::GetFeatureTimestampSize("OTHER", output_sequence));
  ASSERT_EQ(0, mpms::GetFeatureFloatsSize("OTHER", output_sequence));
}

TEST_F(PackMediaSequenceCalculatorTest, TestReconcilingAnnotations) {
  SetUpCalculator({"IMAGE:images"}, {}, false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(
      cv::imencode(".jpg", image, bytes, {cv::IMWRITE_HDR_COMPRESSION, 1}));
  OpenCvImageEncoderCalculatorResults encoded_image;
  encoded_image.set_encoded_image(bytes.data(), bytes.size());
  encoded_image.set_width(2);
  encoded_image.set_height(1);

  int num_images = 5;  // Timestamps: 10, 20, 30, 40, 50
  for (int i = 0; i < num_images; ++i) {
    auto image_ptr =
        ::absl::make_unique<OpenCvImageEncoderCalculatorResults>(encoded_image);
    runner_->MutableInputs()->Tag(kImageTag).packets.push_back(
        Adopt(image_ptr.release()).At(Timestamp((i + 1) * 10)));
  }

  mpms::AddBBoxTimestamp(9, input_sequence.get());
  mpms::AddBBoxTimestamp(21, input_sequence.get());
  mpms::AddBBoxTimestamp(22, input_sequence.get());

  mpms::AddBBoxTimestamp("PREFIX", 8, input_sequence.get());
  mpms::AddBBoxTimestamp("PREFIX", 9, input_sequence.get());
  mpms::AddBBoxTimestamp("PREFIX", 22, input_sequence.get());

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());
  MP_ASSERT_OK(runner_->Run());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kSequenceExampleTag).packets;
  ASSERT_EQ(1, output_packets.size());
  const tf::SequenceExample& output_sequence =
      output_packets[0].Get<tf::SequenceExample>();

  ASSERT_EQ(mpms::GetBBoxTimestampSize(output_sequence), 5);
  ASSERT_EQ(mpms::GetBBoxTimestampAt(output_sequence, 0), 10);
  ASSERT_EQ(mpms::GetBBoxTimestampAt(output_sequence, 1), 20);
  ASSERT_EQ(mpms::GetBBoxTimestampAt(output_sequence, 2), 30);
  ASSERT_EQ(mpms::GetBBoxTimestampAt(output_sequence, 3), 40);
  ASSERT_EQ(mpms::GetBBoxTimestampAt(output_sequence, 4), 50);

  ASSERT_EQ(mpms::GetBBoxTimestampSize("PREFIX", output_sequence), 5);
  ASSERT_EQ(mpms::GetBBoxTimestampAt("PREFIX", output_sequence, 0), 10);
  ASSERT_EQ(mpms::GetBBoxTimestampAt("PREFIX", output_sequence, 1), 20);
  ASSERT_EQ(mpms::GetBBoxTimestampAt("PREFIX", output_sequence, 2), 30);
  ASSERT_EQ(mpms::GetBBoxTimestampAt("PREFIX", output_sequence, 3), 40);
  ASSERT_EQ(mpms::GetBBoxTimestampAt("PREFIX", output_sequence, 4), 50);
}

TEST_F(PackMediaSequenceCalculatorTest, TestOverwritingAndReconciling) {
  SetUpCalculator({"IMAGE:images", "BBOX:bbox"}, {}, false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(
      cv::imencode(".jpg", image, bytes, {cv::IMWRITE_HDR_COMPRESSION, 1}));
  OpenCvImageEncoderCalculatorResults encoded_image;
  encoded_image.set_encoded_image(bytes.data(), bytes.size());
  int height = 2;
  int width = 2;
  encoded_image.set_width(width);
  encoded_image.set_height(height);

  int num_images = 5;  // Timestamps: 10, 20, 30, 40, 50
  for (int i = 0; i < num_images; ++i) {
    auto image_ptr =
        ::absl::make_unique<OpenCvImageEncoderCalculatorResults>(encoded_image);
    runner_->MutableInputs()->Tag(kImageTag).packets.push_back(
        Adopt(image_ptr.release()).At(Timestamp(i)));
  }

  for (int i = 0; i < num_images; ++i) {
    auto detections = ::absl::make_unique<::std::vector<Detection>>();
    Detection detection;
    detection = Detection();
    detection.add_label("relative bbox");
    detection.add_label_id(1);
    detection.add_score(0.75);
    Location::CreateRelativeBBoxLocation(0, 0.5, 0.5, 0.5)
        .ConvertToProto(detection.mutable_location_data());
    detections->push_back(detection);
    runner_->MutableInputs()->Tag(kBboxTag).packets.push_back(
        Adopt(detections.release()).At(Timestamp(i)));
  }

  for (int i = 0; i < 10; ++i) {
    mpms::AddBBoxTimestamp(-1, input_sequence.get());
    mpms::AddBBoxIsAnnotated(-1, input_sequence.get());
    mpms::AddBBoxNumRegions(-1, input_sequence.get());
    mpms::AddBBoxLabelString({"anything"}, input_sequence.get());
    mpms::AddBBoxLabelIndex({-1}, input_sequence.get());
    mpms::AddBBoxLabelConfidence({-1}, input_sequence.get());
    mpms::AddBBoxClassString({"anything"}, input_sequence.get());
    mpms::AddBBoxClassIndex({-1}, input_sequence.get());
    mpms::AddBBoxTrackString({"anything"}, input_sequence.get());
    mpms::AddBBoxTrackIndex({-1}, input_sequence.get());
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());
  // If the all the previous values aren't cleared, this assert will fail.
  MP_ASSERT_OK(runner_->Run());
}

TEST_F(PackMediaSequenceCalculatorTest, TestTooLargeInputFailsSoftly) {
  SetUpCalculator({"FLOAT_FEATURE_TEST:test"}, {}, false, true);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();

  // 1 billion floats should be > 1GB which can't be serialized. It should fail
  // gracefully with this input.
  int num_timesteps = 1000;
  for (int i = 0; i < num_timesteps; ++i) {
    auto vf_ptr = ::absl::make_unique<std::vector<float>>(1000000, i);
    runner_->MutableInputs()
        ->Tag(kFloatFeatureTestTag)
        .packets.push_back(Adopt(vf_ptr.release()).At(Timestamp(i)));
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());
  ASSERT_FALSE(runner_->Run().ok());
}

}  // namespace
}  // namespace mediapipe
