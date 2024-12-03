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

#include <iterator>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "mediapipe/calculators/tensorflow/lapped_tensor_buffer_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

namespace mediapipe {
namespace {

const char kId[] = "id";
const char kRgb[] = "rgb";
const char kAudio[] = "audio";
const char kDesiredSegmentSize[] = "DESIRED_SEGMENT_SIZE";
const char kYt8mId[] = "YT8M_ID";
const char kYt8mSequenceExample[] = "YT8M_SEQUENCE_EXAMPLE";
const char kQuantizedRgbFeature[] = "QUANTIZED_RGB_FEATURE";
const char kQuantizedAudioFeature[] = "QUANTIZED_AUDIO_FEATURE";
const char kSegmentSize[] = "SEGMENT_SIZE";
const char kLappedTensorBufferCalculatorOptions[] =
    "LAPPED_TENSOR_BUFFER_CALCULATOR_OPTIONS";

std::string GetQuantizedFeature(
    const tensorflow::SequenceExample& sequence_example, const std::string& key,
    int index) {
  const auto& bytes_list = sequence_example.feature_lists()
                               .feature_list()
                               .at(key)
                               .feature()
                               .Get(index)
                               .bytes_list()
                               .value();
  ABSL_CHECK_EQ(1, bytes_list.size());
  return bytes_list.Get(0);
}
}  // namespace

// Unpacks YT8M Sequence Example. Note that the audio feature and rgb feature
// output are quantized. DequantizeByteArrayCalculator can do the dequantization
// for you.
//
// Example config:
// node {
//   calculator: "UnpackYt8mSequenceExampleCalculator"
//   input_side_packet: "YT8M_SEQUENCE_EXAMPLE:yt8m_sequence_example"
//   output_stream: "QUANTIZED_RGB_FEATURE:quantized_rgb_feature"
//   output_stream: "QUANTIZED_AUDIO_FEATURE:quantized_audio_feature"
// }
class UnpackYt8mSequenceExampleCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets()
        .Tag(kYt8mSequenceExample)
        .Set<tensorflow::SequenceExample>();
    if (cc->InputSidePackets().HasTag(kDesiredSegmentSize)) {
      cc->InputSidePackets().Tag(kDesiredSegmentSize).Set<int>();
    }
    cc->Outputs().Tag(kQuantizedRgbFeature).Set<std::string>();
    cc->Outputs().Tag(kQuantizedAudioFeature).Set<std::string>();
    if (cc->OutputSidePackets().HasTag(kYt8mId)) {
      cc->OutputSidePackets().Tag(kYt8mId).Set<std::string>();
    }
    if (cc->OutputSidePackets().HasTag(kLappedTensorBufferCalculatorOptions)) {
      cc->OutputSidePackets()
          .Tag(kLappedTensorBufferCalculatorOptions)
          .Set<::mediapipe::LappedTensorBufferCalculatorOptions>();
    }
    if (cc->OutputSidePackets().HasTag(kSegmentSize)) {
      cc->OutputSidePackets().Tag(kSegmentSize).Set<int>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const tensorflow::SequenceExample& sequence_example =
        cc->InputSidePackets()
            .Tag(kYt8mSequenceExample)
            .Get<tensorflow::SequenceExample>();
    const std::string& yt8m_id =
        sequence_example.context().feature().at(kId).bytes_list().value().Get(
            0);
    if (cc->OutputSidePackets().HasTag(kYt8mId)) {
      cc->OutputSidePackets().Tag(kYt8mId).Set(
          MakePacket<std::string>(yt8m_id));
    }

    int rgb_feature_list_length =
        sequence_example.feature_lists().feature_list().at(kRgb).feature_size();
    int audio_feature_list_length = sequence_example.feature_lists()
                                        .feature_list()
                                        .at(kAudio)
                                        .feature_size();

    if (rgb_feature_list_length != audio_feature_list_length) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Data corruption: the length of audio features and rgb features are "
          "not equal. Please check the sequence example that contains yt8m "
          "id: ",
          yt8m_id));
    }
    feature_list_length_ = rgb_feature_list_length;
    if (cc->OutputSidePackets().HasTag(kLappedTensorBufferCalculatorOptions) ||
        cc->OutputSidePackets().HasTag(kSegmentSize)) {
      // If the desired segment size is specified, take the min of the length of
      // the feature list and the desired size to be the output segment size.
      int segment_size = feature_list_length_;
      if (cc->InputSidePackets().HasTag(kDesiredSegmentSize)) {
        int desired_segment_size =
            cc->InputSidePackets().Tag(kDesiredSegmentSize).Get<int>();
        RET_CHECK(desired_segment_size > 0)
            << "The desired segment size must be greater than zero.";
        segment_size = std::min(
            feature_list_length_,
            cc->InputSidePackets().Tag(kDesiredSegmentSize).Get<int>());
      }
      if (cc->OutputSidePackets().HasTag(
              kLappedTensorBufferCalculatorOptions)) {
        auto lapped_tensor_buffer_calculator_options = absl::make_unique<
            ::mediapipe::LappedTensorBufferCalculatorOptions>();
        lapped_tensor_buffer_calculator_options->set_add_batch_dim_to_tensors(
            true);
        lapped_tensor_buffer_calculator_options->set_buffer_size(segment_size);
        lapped_tensor_buffer_calculator_options->set_overlap(segment_size - 1);
        lapped_tensor_buffer_calculator_options->set_timestamp_offset(
            segment_size - 1);
        cc->OutputSidePackets()
            .Tag(kLappedTensorBufferCalculatorOptions)
            .Set(Adopt(lapped_tensor_buffer_calculator_options.release()));
      }
      if (cc->OutputSidePackets().HasTag(kSegmentSize)) {
        cc->OutputSidePackets()
            .Tag(kSegmentSize)
            .Set(MakePacket<int>(segment_size));
      }
    }
    ABSL_LOG(INFO) << "Reading the sequence example that contains yt8m id: "
                   << yt8m_id
                   << ". Feature list length: " << feature_list_length_;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (current_index_ >= feature_list_length_) {
      return mediapipe::tool::StatusStop();
    }
    const tensorflow::SequenceExample& sequence_example =
        cc->InputSidePackets()
            .Tag(kYt8mSequenceExample)
            .Get<tensorflow::SequenceExample>();

    // Uses microsecond as the unit of time. In the YT8M dataset, each feature
    // represents a second.
    const Timestamp timestamp = Timestamp(current_index_ * 1000000);
    cc->Outputs()
        .Tag(kQuantizedRgbFeature)
        .AddPacket(
            MakePacket<std::string>(
                GetQuantizedFeature(sequence_example, kRgb, current_index_))
                .At(timestamp));
    cc->Outputs()
        .Tag(kQuantizedAudioFeature)
        .AddPacket(
            MakePacket<std::string>(
                GetQuantizedFeature(sequence_example, kAudio, current_index_))
                .At(timestamp));
    ++current_index_;
    return absl::OkStatus();
  }

 private:
  int current_index_ = 0;
  int feature_list_length_ = 0;
};

REGISTER_CALCULATOR(UnpackYt8mSequenceExampleCalculator);

}  // namespace mediapipe
