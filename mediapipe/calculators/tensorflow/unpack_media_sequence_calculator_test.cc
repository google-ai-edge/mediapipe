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

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "mediapipe/calculators/core/packet_resampler_calculator.pb.h"
#include "mediapipe/calculators/tensorflow/unpack_media_sequence_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/rectangle.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/audio_decoder.pb.h"
#include "mediapipe/util/sequence/media_sequence.h"
#include "tensorflow/core/example/example.pb.h"

namespace mediapipe {
namespace {

namespace tf = ::tensorflow;
namespace mpms = mediapipe::mediasequence;

constexpr char kImageFrameRateTag[] = "IMAGE_FRAME_RATE";
constexpr char kEncodedMediaStartTimestampTag[] =
    "ENCODED_MEDIA_START_TIMESTAMP";
constexpr char kEncodedMediaTag[] = "ENCODED_MEDIA";
constexpr char kResamplerOptionsTag[] = "RESAMPLER_OPTIONS";
constexpr char kSandboxedDecoderOptionsTag[] = "SANDBOXED_DECODER_OPTIONS";
constexpr char kDecoderOptionsTag[] = "DECODER_OPTIONS";
constexpr char kAudioDecoderOptionsTag[] = "AUDIO_DECODER_OPTIONS";
constexpr char kDataPathTag[] = "DATA_PATH";
constexpr char kDatasetRootTag[] = "DATASET_ROOT";
constexpr char kMediaIdTag[] = "MEDIA_ID";
constexpr char kFloatFeatureFdenseMaxTag[] = "FLOAT_FEATURE_FDENSE_MAX";
constexpr char kFloatFeatureFdenseAvgTag[] = "FLOAT_FEATURE_FDENSE_AVG";
constexpr char kAudioOtherTag[] = "AUDIO_OTHER";
constexpr char kAudioTestTag[] = "AUDIO_TEST";
constexpr char kFloatFeatureOtherTag[] = "FLOAT_FEATURE_OTHER";
constexpr char kFloatFeatureTestTag[] = "FLOAT_FEATURE_TEST";
constexpr char kBboxPrefixTag[] = "BBOX_PREFIX";
constexpr char kKeypointsTag[] = "KEYPOINTS";
constexpr char kBboxTag[] = "BBOX";
constexpr char kForwardFlowEncodedTag[] = "FORWARD_FLOW_ENCODED";
constexpr char kImagePrefixTag[] = "IMAGE_PREFIX";
constexpr char kImageTag[] = "IMAGE";
constexpr char kFloatContextFeatureOtherTag[] = "FLOAT_CONTEXT_FEATURE_OTHER";
constexpr char kFloatContextFeatureTestTag[] = "FLOAT_CONTEXT_FEATURE_TEST";
constexpr char kSequenceExampleTag[] = "SEQUENCE_EXAMPLE";

class UnpackMediaSequenceCalculatorTest : public ::testing::Test {
 protected:
  void SetUpCalculator(const std::vector<std::string>& output_streams,
                       const std::vector<std::string>& output_side_packets,
                       const std::vector<std::string>& input_side_packets = {},
                       const CalculatorOptions* options = nullptr) {
    CalculatorGraphConfig::Node config;
    config.set_calculator("UnpackMediaSequenceCalculator");
    config.add_input_side_packet("SEQUENCE_EXAMPLE:input_sequence");
    for (const std::string& stream : output_streams) {
      config.add_output_stream(stream);
    }
    for (const std::string& side_packet : output_side_packets) {
      config.add_output_side_packet(side_packet);
    }
    for (const std::string& side_packet : input_side_packets) {
      config.add_input_side_packet(side_packet);
    }
    if (options != nullptr) {
      *config.mutable_options() = *options;
    }
    LOG(INFO) << config.DebugString();
    runner_ = absl::make_unique<CalculatorRunner>(config);
  }

  void SetUp() override {
    sequence_ = absl::make_unique<tf::SequenceExample>();
    mpms::SetClipMediaId(video_id_, sequence_.get());
    mpms::SetClipDataPath(data_path_, sequence_.get());
    mpms::SetClipStartTimestamp(start_time_, sequence_.get());
    mpms::SetClipEndTimestamp(end_time_, sequence_.get());
    mpms::SetClipEncodedMediaBytes(encoded_video_data_, sequence_.get());
    mpms::SetClipEncodedMediaStartTimestamp(encoded_video_start_timestamp_,
                                            sequence_.get());
    mpms::SetImageFrameRate(image_frame_rate_, sequence_.get());
  }

  std::unique_ptr<tf::SequenceExample> sequence_;
  std::unique_ptr<CalculatorRunner> runner_;
  const std::string video_id_ = "test_video_id";
  const std::string data_path_ = "test_directory";
  const int64 start_time_ = 3000000;
  const int64 end_time_ = 5000000;
  const std::string encoded_video_data_ = "encoded_video_data";
  const int64 encoded_video_start_timestamp_ = 1000000;
  const double image_frame_rate_ = 1.0;
};

TEST_F(UnpackMediaSequenceCalculatorTest, UnpacksOneImage) {
  SetUpCalculator({"IMAGE:images"}, {});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());

  std::string test_image_string = "test_image_string";

  int num_images = 1;
  for (int i = 0; i < num_images; ++i) {
    mpms::AddImageTimestamp(i, input_sequence.get());
    mpms::AddImageEncoded(test_image_string, input_sequence.get());
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kImageTag).packets;
  ASSERT_EQ(num_images, output_packets.size());

  for (int i = 0; i < num_images; ++i) {
    const std::string& output_image = output_packets[i].Get<std::string>();
    ASSERT_EQ(output_image, test_image_string);
  }
}

TEST_F(UnpackMediaSequenceCalculatorTest, UnpacksTwoImages) {
  SetUpCalculator({"IMAGE:images"}, {});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());

  std::string test_image_string = "test_image_string";

  int num_images = 2;
  for (int i = 0; i < num_images; ++i) {
    mpms::AddImageTimestamp(i, input_sequence.get());
    mpms::AddImageEncoded(test_image_string, input_sequence.get());
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kImageTag).packets;
  ASSERT_EQ(num_images, output_packets.size());

  for (int i = 0; i < num_images; ++i) {
    const std::string& output_image = output_packets[i].Get<std::string>();
    ASSERT_EQ(output_image, test_image_string);
  }
}

TEST_F(UnpackMediaSequenceCalculatorTest, UnpacksTwoPrefixedImages) {
  std::string prefix = "PREFIX";
  SetUpCalculator({"IMAGE_PREFIX:images"}, {});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());

  std::string test_image_string = "test_image_string";

  int num_images = 2;
  for (int i = 0; i < num_images; ++i) {
    mpms::AddImageTimestamp(prefix, i, input_sequence.get());
    mpms::AddImageEncoded(prefix, test_image_string, input_sequence.get());
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kImagePrefixTag).packets;
  ASSERT_EQ(num_images, output_packets.size());

  for (int i = 0; i < num_images; ++i) {
    const std::string& output_image = output_packets[i].Get<std::string>();
    ASSERT_EQ(output_image, test_image_string);
  }
}

TEST_F(UnpackMediaSequenceCalculatorTest, UnpacksOneForwardFlowImage) {
  SetUpCalculator({"FORWARD_FLOW_ENCODED:flow_images"}, {});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  const std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());

  const std::string test_image_string = "test_image_string";
  const int num_forward_flow_images = 1;
  for (int i = 0; i < num_forward_flow_images; ++i) {
    mpms::AddForwardFlowTimestamp(i, input_sequence.get());
    mpms::AddForwardFlowEncoded(test_image_string, input_sequence.get());
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kForwardFlowEncodedTag).packets;
  ASSERT_EQ(num_forward_flow_images, output_packets.size());

  for (int i = 0; i < num_forward_flow_images; ++i) {
    const std::string& output_image = output_packets[i].Get<std::string>();
    ASSERT_EQ(output_image, test_image_string);
    ASSERT_EQ(output_packets[i].Timestamp().Value(), static_cast<int64>(i));
  }
}

TEST_F(UnpackMediaSequenceCalculatorTest, UnpacksTwoForwardFlowImages) {
  SetUpCalculator({"FORWARD_FLOW_ENCODED:flow_images"}, {});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  const std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());

  const std::string test_image_strings[2] = {"test_image_string0",
                                             "test_image_string1"};
  const int num_forward_flow_images = 2;
  for (int i = 0; i < num_forward_flow_images; ++i) {
    mpms::AddForwardFlowTimestamp(i, input_sequence.get());
    mpms::AddForwardFlowEncoded(test_image_strings[i], input_sequence.get());
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kForwardFlowEncodedTag).packets;
  ASSERT_EQ(num_forward_flow_images, output_packets.size());

  for (int i = 0; i < num_forward_flow_images; ++i) {
    const std::string& output_image = output_packets[i].Get<std::string>();
    ASSERT_EQ(output_image, test_image_strings[i]);
    ASSERT_EQ(output_packets[i].Timestamp().Value(), static_cast<int64>(i));
  }
}

TEST_F(UnpackMediaSequenceCalculatorTest, UnpacksBBoxes) {
  SetUpCalculator({"BBOX:test", "FLOAT_FEATURE_OTHER:other"}, {});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();

  std::vector<std::vector<Location>> bboxes = {
      {Location::CreateRelativeBBoxLocation(0.1, 0.2, 0.7, 0.7),
       Location::CreateRelativeBBoxLocation(0.3, 0.4, 0.2, 0.1)},
      {Location::CreateRelativeBBoxLocation(0.2, 0.3, 0.4, 0.5)}};

  for (int i = 0; i < bboxes.size(); ++i) {
    mpms::AddBBox(bboxes[i], input_sequence.get());
    mpms::AddBBoxTimestamp(i, input_sequence.get());
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kBboxTag).packets;
  ASSERT_EQ(bboxes.size(), output_packets.size());

  for (int i = 0; i < bboxes.size(); ++i) {
    const auto& output_vector =
        output_packets[i].Get<std::vector<::mediapipe::Location>>();
    for (int j = 0; j < bboxes[i].size(); ++j) {
      ASSERT_EQ(output_vector[j].GetRelativeBBox(),
                bboxes[i][j].GetRelativeBBox());
    }
  }
}

TEST_F(UnpackMediaSequenceCalculatorTest, UnpacksPrefixedBBoxes) {
  std::string prefix = "PREFIX";
  SetUpCalculator({"BBOX_PREFIX:test", "FLOAT_FEATURE_OTHER:other"}, {});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();

  std::vector<std::vector<Location>> bboxes = {
      {Location::CreateRelativeBBoxLocation(0.1, 0.2, 0.7, 0.7),
       Location::CreateRelativeBBoxLocation(0.3, 0.4, 0.2, 0.1)},
      {Location::CreateRelativeBBoxLocation(0.2, 0.3, 0.4, 0.5)}};

  for (int i = 0; i < bboxes.size(); ++i) {
    mpms::AddBBox(prefix, bboxes[i], input_sequence.get());
    mpms::AddBBoxTimestamp(prefix, i, input_sequence.get());
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kBboxPrefixTag).packets;
  ASSERT_EQ(bboxes.size(), output_packets.size());

  for (int i = 0; i < bboxes.size(); ++i) {
    const auto& output_vector =
        output_packets[i].Get<std::vector<::mediapipe::Location>>();
    for (int j = 0; j < bboxes[i].size(); ++j) {
      ASSERT_EQ(output_vector[j].GetRelativeBBox(),
                bboxes[i][j].GetRelativeBBox());
    }
  }
}

TEST_F(UnpackMediaSequenceCalculatorTest, UnpacksTwoFloatLists) {
  SetUpCalculator({"FLOAT_FEATURE_TEST:test", "FLOAT_FEATURE_OTHER:other"}, {});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();

  int num_float_lists = 2;
  for (int i = 0; i < num_float_lists; ++i) {
    std::vector<float> data(2, 2 << i);
    mpms::AddFeatureFloats("TEST", data, input_sequence.get());
    mpms::AddFeatureFloats("OTHER", data, input_sequence.get());
    mpms::AddFeatureTimestamp("TEST", i, input_sequence.get());
    mpms::AddFeatureTimestamp("OTHER", i, input_sequence.get());
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kFloatFeatureTestTag).packets;
  ASSERT_EQ(num_float_lists, output_packets.size());

  for (int i = 0; i < num_float_lists; ++i) {
    const auto& output_vector = output_packets[i].Get<std::vector<float>>();
    ASSERT_THAT(output_vector,
                ::testing::ElementsAreArray(std::vector<float>(2, 2 << i)));
  }

  const std::vector<Packet>& output_packets_other =
      runner_->Outputs().Tag(kFloatFeatureOtherTag).packets;
  ASSERT_EQ(num_float_lists, output_packets_other.size());

  for (int i = 0; i < num_float_lists; ++i) {
    const auto& output_vector =
        output_packets_other[i].Get<std::vector<float>>();
    ASSERT_THAT(output_vector,
                ::testing::ElementsAreArray(std::vector<float>(2, 2 << i)));
  }
}

TEST_F(UnpackMediaSequenceCalculatorTest, UnpacksNonOverlappingTimestamps) {
  SetUpCalculator({"IMAGE:images", "FLOAT_FEATURE_OTHER:other"}, {});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());

  std::string test_image_string = "test_image_string";
  int num_images = 2;
  for (int i = 0; i < num_images; ++i) {
    mpms::AddImageTimestamp(i, input_sequence.get());
    mpms::AddImageEncoded(test_image_string, input_sequence.get());
  }
  int num_float_lists = 2;
  for (int i = 0; i < num_float_lists; ++i) {
    std::vector<float> data(2, 2 << i);
    mpms::AddFeatureFloats("OTHER", data, input_sequence.get());
    mpms::AddFeatureTimestamp("OTHER", i + 5, input_sequence.get());
  }

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kImageTag).packets;
  ASSERT_EQ(num_images, output_packets.size());

  for (int i = 0; i < num_images; ++i) {
    const std::string& output_image = output_packets[i].Get<std::string>();
    ASSERT_EQ(output_image, test_image_string);
  }

  const std::vector<Packet>& output_packets_other =
      runner_->Outputs().Tag(kFloatFeatureOtherTag).packets;
  ASSERT_EQ(num_float_lists, output_packets_other.size());

  for (int i = 0; i < num_float_lists; ++i) {
    const auto& output_vector =
        output_packets_other[i].Get<std::vector<float>>();
    ASSERT_THAT(output_vector,
                ::testing::ElementsAreArray(std::vector<float>(2, 2 << i)));
  }
}

TEST_F(UnpackMediaSequenceCalculatorTest, UnpacksTwoPostStreamFloatLists) {
  SetUpCalculator(
      {"FLOAT_FEATURE_FDENSE_AVG:avg", "FLOAT_FEATURE_FDENSE_MAX:max"}, {});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  mpms::AddFeatureFloats("FDENSE_AVG", {1.0f, 2.0f}, input_sequence.get());
  mpms::AddFeatureTimestamp("FDENSE_AVG", Timestamp::PostStream().Value(),
                            input_sequence.get());

  mpms::AddFeatureFloats("FDENSE_MAX", {3.0f, 4.0f}, input_sequence.get());
  mpms::AddFeatureTimestamp("FDENSE_MAX", Timestamp::PostStream().Value(),
                            input_sequence.get());

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& fdense_avg_packets =
      runner_->Outputs().Tag(kFloatFeatureFdenseAvgTag).packets;
  ASSERT_EQ(fdense_avg_packets.size(), 1);
  const auto& fdense_avg_vector =
      fdense_avg_packets[0].Get<std::vector<float>>();
  ASSERT_THAT(fdense_avg_vector, ::testing::ElementsAreArray({1.0f, 2.0f}));
  ASSERT_THAT(fdense_avg_packets[0].Timestamp(),
              ::testing::Eq(Timestamp::PostStream()));

  const std::vector<Packet>& fdense_max_packets =
      runner_->Outputs().Tag(kFloatFeatureFdenseMaxTag).packets;
  ASSERT_EQ(fdense_max_packets.size(), 1);
  const auto& fdense_max_vector =
      fdense_max_packets[0].Get<std::vector<float>>();
  ASSERT_THAT(fdense_max_vector, ::testing::ElementsAreArray({3.0f, 4.0f}));
  ASSERT_THAT(fdense_max_packets[0].Timestamp(),
              ::testing::Eq(Timestamp::PostStream()));
}

TEST_F(UnpackMediaSequenceCalculatorTest, UnpacksImageWithPostStreamFloatList) {
  SetUpCalculator({"IMAGE:images"}, {});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());

  std::string test_image_string = "test_image_string";

  int num_images = 1;
  for (int i = 0; i < num_images; ++i) {
    mpms::AddImageTimestamp(i, input_sequence.get());
    mpms::AddImageEncoded(test_image_string, input_sequence.get());
  }

  mpms::AddFeatureFloats("FDENSE_MAX", {3.0f, 4.0f}, input_sequence.get());
  mpms::AddFeatureTimestamp("FDENSE_MAX", Timestamp::PostStream().Value(),
                            input_sequence.get());

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kImageTag).packets;
  ASSERT_EQ(num_images, output_packets.size());

  for (int i = 0; i < num_images; ++i) {
    const std::string& output_image = output_packets[i].Get<std::string>();
    ASSERT_EQ(output_image, test_image_string);
  }
}

TEST_F(UnpackMediaSequenceCalculatorTest, UnpacksPostStreamFloatListWithImage) {
  SetUpCalculator({"FLOAT_FEATURE_FDENSE_MAX:max"}, {});
  auto input_sequence = absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());

  std::string test_image_string = "test_image_string";

  int num_images = 1;
  for (int i = 0; i < num_images; ++i) {
    mpms::AddImageTimestamp(i, input_sequence.get());
    mpms::AddImageEncoded(test_image_string, input_sequence.get());
  }

  mpms::AddFeatureFloats("FDENSE_MAX", {3.0f, 4.0f}, input_sequence.get());
  mpms::AddFeatureTimestamp("FDENSE_MAX", Timestamp::PostStream().Value(),
                            input_sequence.get());

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(input_sequence.release());

  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& fdense_max_packets =
      runner_->Outputs().Tag(kFloatFeatureFdenseMaxTag).packets;
  ASSERT_EQ(fdense_max_packets.size(), 1);
  const auto& fdense_max_vector =
      fdense_max_packets[0].Get<std::vector<float>>();
  ASSERT_THAT(fdense_max_vector, ::testing::ElementsAreArray({3.0f, 4.0f}));
  ASSERT_THAT(fdense_max_packets[0].Timestamp(),
              ::testing::Eq(Timestamp::PostStream()));
}

TEST_F(UnpackMediaSequenceCalculatorTest, GetDatasetFromPacket) {
  SetUpCalculator({}, {"DATA_PATH:data_path"}, {"DATASET_ROOT:root"});

  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(sequence_.release());

  std::string root = "test_root";
  runner_->MutableSidePackets()->Tag(kDatasetRootTag) = PointToForeign(&root);
  MP_ASSERT_OK(runner_->Run());

  MP_ASSERT_OK(runner_->OutputSidePackets()
                   .Tag(kDataPathTag)
                   .ValidateAsType<std::string>());
  ASSERT_EQ(runner_->OutputSidePackets().Tag(kDataPathTag).Get<std::string>(),
            root + "/" + data_path_);
}

TEST_F(UnpackMediaSequenceCalculatorTest, GetDatasetFromOptions) {
  CalculatorOptions options;
  std::string root = "test_root";
  options.MutableExtension(UnpackMediaSequenceCalculatorOptions::ext)
      ->set_dataset_root_directory(root);
  SetUpCalculator({}, {"DATA_PATH:data_path"}, {}, &options);
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(sequence_.release());

  MP_ASSERT_OK(runner_->Run());

  MP_ASSERT_OK(runner_->OutputSidePackets()
                   .Tag(kDataPathTag)
                   .ValidateAsType<std::string>());
  ASSERT_EQ(runner_->OutputSidePackets().Tag(kDataPathTag).Get<std::string>(),
            root + "/" + data_path_);
}

TEST_F(UnpackMediaSequenceCalculatorTest, GetDatasetFromExample) {
  SetUpCalculator({}, {"DATA_PATH:data_path"});
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(sequence_.release());
  MP_ASSERT_OK(runner_->Run());

  MP_ASSERT_OK(runner_->OutputSidePackets()
                   .Tag(kDataPathTag)
                   .ValidateAsType<std::string>());
  ASSERT_EQ(runner_->OutputSidePackets().Tag(kDataPathTag).Get<std::string>(),
            data_path_);
}

TEST_F(UnpackMediaSequenceCalculatorTest, GetAudioDecoderOptions) {
  CalculatorOptions options;
  options.MutableExtension(UnpackMediaSequenceCalculatorOptions::ext)
      ->set_padding_before_label(1);
  options.MutableExtension(UnpackMediaSequenceCalculatorOptions::ext)
      ->set_padding_after_label(2);
  SetUpCalculator({}, {"AUDIO_DECODER_OPTIONS:audio_decoder_options"}, {},
                  &options);
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(sequence_.release());
  MP_ASSERT_OK(runner_->Run());

  MP_EXPECT_OK(runner_->OutputSidePackets()
                   .Tag(kAudioDecoderOptionsTag)
                   .ValidateAsType<AudioDecoderOptions>());
  EXPECT_NEAR(runner_->OutputSidePackets()
                  .Tag(kAudioDecoderOptionsTag)
                  .Get<AudioDecoderOptions>()
                  .start_time(),
              2.0, 1e-5);
  EXPECT_NEAR(runner_->OutputSidePackets()
                  .Tag(kAudioDecoderOptionsTag)
                  .Get<AudioDecoderOptions>()
                  .end_time(),
              7.0, 1e-5);
}

TEST_F(UnpackMediaSequenceCalculatorTest, GetAudioDecoderOptionsOverride) {
  CalculatorOptions options;
  options.MutableExtension(UnpackMediaSequenceCalculatorOptions::ext)
      ->set_padding_before_label(1);
  options.MutableExtension(UnpackMediaSequenceCalculatorOptions::ext)
      ->set_padding_after_label(2);
  options.MutableExtension(UnpackMediaSequenceCalculatorOptions::ext)
      ->set_force_decoding_from_start_of_media(true);
  SetUpCalculator({}, {"AUDIO_DECODER_OPTIONS:audio_decoder_options"}, {},
                  &options);
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(sequence_.release());
  MP_ASSERT_OK(runner_->Run());

  MP_EXPECT_OK(runner_->OutputSidePackets()
                   .Tag(kAudioDecoderOptionsTag)
                   .ValidateAsType<AudioDecoderOptions>());
  EXPECT_NEAR(runner_->OutputSidePackets()
                  .Tag(kAudioDecoderOptionsTag)
                  .Get<AudioDecoderOptions>()
                  .start_time(),
              0.0, 1e-5);
  EXPECT_NEAR(runner_->OutputSidePackets()
                  .Tag(kAudioDecoderOptionsTag)
                  .Get<AudioDecoderOptions>()
                  .end_time(),
              7.0, 1e-5);
}

TEST_F(UnpackMediaSequenceCalculatorTest, GetPacketResamplingOptions) {
  // TODO: Suport proto3 proto.Any in CalculatorOptions.
  // TODO: Avoid proto2 extensions in "RESAMPLER_OPTIONS".
  CalculatorOptions options;
  options.MutableExtension(UnpackMediaSequenceCalculatorOptions::ext)
      ->set_padding_before_label(1);
  options.MutableExtension(UnpackMediaSequenceCalculatorOptions::ext)
      ->set_padding_after_label(2);
  options.MutableExtension(UnpackMediaSequenceCalculatorOptions::ext)
      ->mutable_base_packet_resampler_options()
      ->set_frame_rate(1.0);
  SetUpCalculator({}, {"RESAMPLER_OPTIONS:resampler_options"}, {}, &options);
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(sequence_.release());
  MP_ASSERT_OK(runner_->Run());

  MP_EXPECT_OK(runner_->OutputSidePackets()
                   .Tag(kResamplerOptionsTag)
                   .ValidateAsType<CalculatorOptions>());
  EXPECT_NEAR(runner_->OutputSidePackets()
                  .Tag(kResamplerOptionsTag)
                  .Get<CalculatorOptions>()
                  .GetExtension(PacketResamplerCalculatorOptions::ext)
                  .start_time(),
              2000000, 1);
  EXPECT_NEAR(runner_->OutputSidePackets()
                  .Tag(kResamplerOptionsTag)
                  .Get<CalculatorOptions>()
                  .GetExtension(PacketResamplerCalculatorOptions::ext)
                  .end_time(),
              7000000, 1);
  EXPECT_NEAR(runner_->OutputSidePackets()
                  .Tag(kResamplerOptionsTag)
                  .Get<CalculatorOptions>()
                  .GetExtension(PacketResamplerCalculatorOptions::ext)
                  .frame_rate(),
              1.0, 1e-5);
}

TEST_F(UnpackMediaSequenceCalculatorTest, GetFrameRateFromExample) {
  SetUpCalculator({}, {"IMAGE_FRAME_RATE:frame_rate"});
  runner_->MutableSidePackets()->Tag(kSequenceExampleTag) =
      Adopt(sequence_.release());
  MP_ASSERT_OK(runner_->Run());
  MP_EXPECT_OK(runner_->OutputSidePackets()
                   .Tag(kImageFrameRateTag)
                   .ValidateAsType<double>());
  EXPECT_EQ(runner_->OutputSidePackets().Tag(kImageFrameRateTag).Get<double>(),
            image_frame_rate_);
}

}  // namespace
}  // namespace mediapipe
