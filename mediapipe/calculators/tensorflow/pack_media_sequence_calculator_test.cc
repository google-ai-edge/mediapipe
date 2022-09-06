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

#include <algorithm>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "mediapipe/calculators/image/opencv_image_encoder_calculator.pb.h"
#include "mediapipe/calculators/tensorflow/pack_media_sequence_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/location_opencv.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/util/sequence/media_sequence.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

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
constexpr char kFloatFeatureOtherTag[] = "FLOAT_FEATURE_OTHER";
constexpr char kFloatFeatureTestTag[] = "FLOAT_FEATURE_TEST";
constexpr char kIntFeatureOtherTag[] = "INT_FEATURE_OTHER";
constexpr char kIntFeatureTestTag[] = "INT_FEATURE_TEST";
constexpr char kImagePrefixTag[] = "IMAGE_PREFIX";
constexpr char kSequenceExampleTag[] = "SEQUENCE_EXAMPLE";
constexpr char kImageTag[] = "IMAGE";

class PackMediaSequenceCalculatorTest : public ::testing::Test {
 protected:
  void SetUpCalculator(const std::vector<std::string>& input_streams,
                       const tf::Features& features,
                       const bool output_only_if_all_present,
                       const bool replace_instead_of_append,
                       const bool output_as_zero_timestamp = false) {
    CalculatorGraphConfig::Node config;
    config.set_calculator("PackMediaSequenceCalculator");
    config.add_input_side_packet("SEQUENCE_EXAMPLE:input_sequence");
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
  ASSERT_TRUE(cv::imencode(".jpg", image, bytes, {80}));
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
  ASSERT_TRUE(cv::imencode(".jpg", image, bytes, {80}));
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
    auto vi_ptr = ::absl::make_unique<std::vector<int64>>(2, 2 << i);
    runner_->MutableInputs()
        ->Tag(kIntFeatureTestTag)
        .packets.push_back(Adopt(vi_ptr.release()).At(Timestamp(i)));
    vi_ptr = ::absl::make_unique<std::vector<int64>>(2, 2 << i);
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
                ::testing::ElementsAreArray(std::vector<int64>(2, 2 << i)));
    ASSERT_EQ(i, mpms::GetFeatureTimestampAt("OTHER", output_sequence, i));
    ASSERT_THAT(mpms::GetFeatureIntsAt("OTHER", output_sequence, i),
                ::testing::ElementsAreArray(std::vector<int64>(2, 2 << i)));
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
  ASSERT_TRUE(cv::imencode(".jpg", image, bytes, {80}));
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
  ASSERT_TRUE(cv::imencode(".jpg", image, bytes, {80}));
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
    mediapipe::CreateCvMaskLocation<uint8>(image).ConvertToProto(
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
    mediapipe::CreateCvMaskLocation<uint8>(image).ConvertToProto(
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
    mediapipe::CreateCvMaskLocation<uint8>(image).ConvertToProto(
        detection.mutable_location_data());
    detections->push_back(detection);

    runner_->MutableInputs()
        ->Tag(kBboxPredictedTag)
        .packets.push_back(Adopt(detections.release()).At(Timestamp(i)));
  }
  cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(cv::imencode(".jpg", image, bytes, {80}));
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
    mediapipe::CreateCvMaskLocation<uint8>(image).ConvertToProto(
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

TEST_F(PackMediaSequenceCalculatorTest, MissingStreamOK) {
  SetUpCalculator(
      {"FORWARD_FLOW_ENCODED:flow", "FLOAT_FEATURE_I3D_FLOW:feature"}, {},
      false, false);
  auto input_sequence = ::absl::make_unique<tf::SequenceExample>();
  std::string test_video_id = "test_video_id";
  mpms::SetClipMediaId(test_video_id, input_sequence.get());

  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> bytes;
  ASSERT_TRUE(cv::imencode(".jpg", image, bytes, {80}));
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
  ASSERT_TRUE(cv::imencode(".jpg", image, bytes, {80}));
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
  ASSERT_TRUE(cv::imencode(".jpg", image, bytes, {80}));
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
  ASSERT_TRUE(cv::imencode(".jpg", image, bytes, {80}));
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
