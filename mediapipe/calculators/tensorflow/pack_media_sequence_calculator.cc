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
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/strip.h"
#include "mediapipe/calculators/image/opencv_image_encoder_calculator.pb.h"
#include "mediapipe/calculators/tensorflow/pack_media_sequence_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/location_opencv.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/sequence/media_sequence.h"
#include "mediapipe/util/sequence/media_sequence_util.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

namespace mediapipe {

const char kSequenceExampleTag[] = "SEQUENCE_EXAMPLE";
const char kImageTag[] = "IMAGE";
const char kImageLabelPrefixTag[] = "IMAGE_LABEL_";
const char kClipLabelPrefixTag[] = "CLIP_LABEL_";
const char kFloatContextFeaturePrefixTag[] = "FLOAT_CONTEXT_FEATURE_";
const char kIntsContextFeaturePrefixTag[] = "INTS_CONTEXT_FEATURE_";
const char kBytesContextFeaturePrefixTag[] = "BYTES_CONTEXT_FEATURE_";
const char kFloatFeaturePrefixTag[] = "FLOAT_FEATURE_";
const char kIntFeaturePrefixTag[] = "INT_FEATURE_";
const char kBytesFeaturePrefixTag[] = "BYTES_FEATURE_";
const char kForwardFlowEncodedTag[] = "FORWARD_FLOW_ENCODED";
const char kBBoxTag[] = "BBOX";
const char kKeypointsTag[] = "KEYPOINTS";
const char kSegmentationMaskTag[] = "CLASS_SEGMENTATION";
const char kClipMediaIdTag[] = "CLIP_MEDIA_ID";

namespace tf = ::tensorflow;
namespace mpms = mediapipe::mediasequence;

// Sink calculator to package streams into tf.SequenceExamples.
//
// The calculator takes a tf.SequenceExample as a side input and then adds
// the data from inputs to the SequenceExample with timestamps. Additional
// context features can be supplied verbatim in the calculator's options. The
// SequenceExample will conform to the description in media_sequence.h.
//
// The supported input stream tags are:
// * "IMAGE", which stores the encoded images from the
//   OpenCVImageEncoderCalculator,
// * "IMAGE_LABEL", which stores whole image labels from Detection,
// * "FORWARD_FLOW_ENCODED", which stores the encoded optical flow from the same
//   calculator,
// * "BBOX" which stores bounding boxes from vector<Detections>,
// * streams with the "FLOAT_FEATURE_${NAME}" pattern, which stores the values
//   from vector<float>'s associated with the name ${NAME},
// * "KEYPOINTS" stores a map of 2D keypoints from flat_hash_map<string,
//   vector<pair<float, float>>>,
// * "CLIP_MEDIA_ID", which stores the clip's media ID as a string.
// * "CLIP_LABEL_${NAME}" which stores sparse feature labels, ID and scores in
//   mediapipe::Detection. In the input Detection, the score field is required,
//   and label and label_id are optional but at least one of them should be set.
// "IMAGE_${NAME}", "BBOX_${NAME}", and "KEYPOINTS_${NAME}" will also store
// prefixed versions of each stream, which allows for multiple image streams to
// be included. However, the default names are supported by more tools.
//
// Example config:
// node {
//   calculator: "PackMediaSequenceCalculator"
//   input_side_packet: "SEQUENCE_EXAMPLE:example_input_side_packet"
//   input_stream: "IMAGE:frames"
//   input_stream: "FLOAT_FEATURE_FDENSE:fdense_vf"
//   output_stream: "SEQUENCE_EXAMPLE:example_output_stream"
//   options {
//     [mediapipe.PackMediaSequenceCalculatorOptions.ext]: {
//       context_feature_map {
//         feature {
//           key: "image/frames_per_second"
//           value {
//             float_list {
//               value: 30.0
//             }
//           }
//         }
//       }
//     }
//   }
// }
namespace {
uint8_t ConvertFloatToByte(const float float_value) {
  float clamped_value = std::clamp(0.0f, 1.0f, float_value);
  return static_cast<uint8_t>(clamped_value * 255.0 + .5f);
}
}  // namespace

class PackMediaSequenceCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->InputSidePackets().HasTag(kSequenceExampleTag));
    cc->InputSidePackets().Tag(kSequenceExampleTag).Set<tf::SequenceExample>();
    if (cc->InputSidePackets().HasTag(kClipMediaIdTag)) {
      cc->InputSidePackets().Tag(kClipMediaIdTag).Set<std::string>();
    }

    if (cc->Inputs().HasTag(kForwardFlowEncodedTag)) {
      cc->Inputs()
          .Tag(kForwardFlowEncodedTag)
          .Set<OpenCvImageEncoderCalculatorResults>();
    }
    if (cc->Inputs().HasTag(kSegmentationMaskTag)) {
      cc->Inputs().Tag(kSegmentationMaskTag).Set<std::vector<Detection>>();
    }

    for (const auto& tag : cc->Inputs().GetTags()) {
      if (absl::StartsWith(tag, kImageTag)) {
        if (absl::StartsWith(tag, kImageLabelPrefixTag)) {
          cc->Inputs().Tag(tag).Set<Detection>();
          continue;
        }
        std::string key = "";
        if (tag != kImageTag) {
          int tag_length = sizeof(kImageTag) / sizeof(*kImageTag) - 1;
          if (tag[tag_length] == '_') {
            key = tag.substr(tag_length + 1);
          } else {
            continue;  // Skip keys that don't match "(kImageTag)_?"
          }
        }
        cc->Inputs().Tag(tag).Set<OpenCvImageEncoderCalculatorResults>();
      }
      if (absl::StartsWith(tag, kKeypointsTag)) {
        std::string key = "";
        if (tag != kKeypointsTag) {
          int tag_length = sizeof(kKeypointsTag) / sizeof(*kKeypointsTag) - 1;
          if (tag[tag_length] == '_') {
            key = tag.substr(tag_length + 1);
          } else {
            continue;  // Skip keys that don't match "(kKeypointsTag)_?"
          }
        }
        cc->Inputs()
            .Tag(tag)
            .Set<absl::flat_hash_map<std::string,
                                     std::vector<std::pair<float, float>>>>();
      }
      if (absl::StartsWith(tag, kBBoxTag)) {
        std::string key = "";
        if (tag != kBBoxTag) {
          int tag_length = sizeof(kBBoxTag) / sizeof(*kBBoxTag) - 1;
          if (tag[tag_length] == '_') {
            key = tag.substr(tag_length + 1);
          } else {
            continue;  // Skip keys that don't match "(kBBoxTag)_?"
          }
        }
        cc->Inputs().Tag(tag).Set<std::vector<Detection>>();
      }
      if (absl::StartsWith(tag, kClipLabelPrefixTag)) {
        cc->Inputs().Tag(tag).Set<Detection>();
      }
      if (absl::StartsWith(tag, kFloatContextFeaturePrefixTag)) {
        cc->Inputs().Tag(tag).Set<std::vector<float>>();
      }
      if (absl::StartsWith(tag, kIntsContextFeaturePrefixTag)) {
        cc->Inputs().Tag(tag).Set<std::vector<int64_t>>();
      }
      if (absl::StartsWith(tag, kBytesContextFeaturePrefixTag)) {
        cc->Inputs().Tag(tag).Set<std::vector<std::string>>();
      }
      if (absl::StartsWith(tag, kFloatFeaturePrefixTag)) {
        cc->Inputs().Tag(tag).Set<std::vector<float>>();
      }
      if (absl::StartsWith(tag, kIntFeaturePrefixTag)) {
        cc->Inputs().Tag(tag).Set<std::vector<int64_t>>();
      }
      if (absl::StartsWith(tag, kBytesFeaturePrefixTag)) {
        cc->Inputs().Tag(tag).Set<std::vector<std::string>>();
      }
    }

    RET_CHECK(cc->Outputs().HasTag(kSequenceExampleTag) ||
              cc->OutputSidePackets().HasTag(kSequenceExampleTag))
        << "Neither the output stream nor the output side packet is set to "
           "output the sequence example.";
    if (cc->Outputs().HasTag(kSequenceExampleTag)) {
      cc->Outputs().Tag(kSequenceExampleTag).Set<tf::SequenceExample>();
    }
    if (cc->OutputSidePackets().HasTag(kSequenceExampleTag)) {
      cc->OutputSidePackets()
          .Tag(kSequenceExampleTag)
          .Set<tf::SequenceExample>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    sequence_ = ::absl::make_unique<tf::SequenceExample>(
        cc->InputSidePackets()
            .Tag(kSequenceExampleTag)
            .Get<tf::SequenceExample>());
    if (cc->InputSidePackets().HasTag(kClipMediaIdTag) &&
        !cc->InputSidePackets().Tag(kClipMediaIdTag).IsEmpty()) {
      clip_media_id_ =
          cc->InputSidePackets().Tag(kClipMediaIdTag).Get<std::string>();
    }

    const auto& context_features =
        cc->Options<PackMediaSequenceCalculatorOptions>().context_feature_map();
    for (const auto& feature : context_features.feature()) {
      *mpms::MutableContext(feature.first, sequence_.get()) = feature.second;
    }
    for (const auto& tag : cc->Inputs().GetTags()) {
      features_present_[tag] = false;
    }

    replace_keypoints_ = false;
    if (cc->Options<PackMediaSequenceCalculatorOptions>()
            .replace_data_instead_of_append()) {
      // Clear the existing values under the same key.
      for (const auto& tag : cc->Inputs().GetTags()) {
        if (absl::StartsWith(tag, kImageTag)) {
          if (absl::StartsWith(tag, kImageLabelPrefixTag)) {
            std::string key =
                std::string(absl::StripPrefix(tag, kImageLabelPrefixTag));
            mpms::ClearImageLabelString(key, sequence_.get());
            mpms::ClearImageLabelConfidence(key, sequence_.get());
            if (!key.empty() || mpms::HasImageEncoded(*sequence_)) {
              mpms::ClearImageTimestamp(key, sequence_.get());
            }
            continue;
          }
          std::string key = "";
          if (tag != kImageTag) {
            int tag_length = sizeof(kImageTag) / sizeof(*kImageTag) - 1;
            if (tag[tag_length] == '_') {
              key = tag.substr(tag_length + 1);
            } else {
              continue;  // Skip keys that don't match "(kImageTag)_?"
            }
          }
          mpms::ClearImageEncoded(key, sequence_.get());
          mpms::ClearImageTimestamp(key, sequence_.get());
        }
        if (absl::StartsWith(tag, kBBoxTag)) {
          std::string key = "";
          if (tag != kBBoxTag) {
            int tag_length = sizeof(kBBoxTag) / sizeof(*kBBoxTag) - 1;
            if (tag[tag_length] == '_') {
              key = tag.substr(tag_length + 1);
            } else {
              continue;  // Skip keys that don't match "(kBBoxTag)_?"
            }
          }
          mpms::ClearBBox(key, sequence_.get());
          mpms::ClearBBoxTimestamp(key, sequence_.get());
          mpms::ClearBBoxIsAnnotated(key, sequence_.get());
          mpms::ClearBBoxNumRegions(key, sequence_.get());
          mpms::ClearBBoxLabelString(key, sequence_.get());
          mpms::ClearBBoxLabelIndex(key, sequence_.get());
          mpms::ClearBBoxLabelConfidence(key, sequence_.get());
          mpms::ClearBBoxClassString(key, sequence_.get());
          mpms::ClearBBoxClassIndex(key, sequence_.get());
          mpms::ClearBBoxTrackString(key, sequence_.get());
          mpms::ClearBBoxTrackIndex(key, sequence_.get());
          mpms::ClearUnmodifiedBBoxTimestamp(key, sequence_.get());
        }
        if (absl::StartsWith(tag, kClipLabelPrefixTag)) {
          const std::string& key = tag.substr(
              sizeof(kClipLabelPrefixTag) / sizeof(*kClipLabelPrefixTag) - 1);
          mpms::ClearClipLabelIndex(key, sequence_.get());
          mpms::ClearClipLabelString(key, sequence_.get());
          mpms::ClearClipLabelConfidence(key, sequence_.get());
        }
        if (absl::StartsWith(tag, kFloatContextFeaturePrefixTag)) {
          const std::string& key =
              tag.substr(sizeof(kFloatContextFeaturePrefixTag) /
                             sizeof(*kFloatContextFeaturePrefixTag) -
                         1);
          mpms::ClearContextFeatureFloats(key, sequence_.get());
        }
        if (absl::StartsWith(tag, kIntsContextFeaturePrefixTag)) {
          const std::string& key =
              tag.substr(sizeof(kIntsContextFeaturePrefixTag) /
                             sizeof(*kIntsContextFeaturePrefixTag) -
                         1);
          mpms::ClearContextFeatureInts(key, sequence_.get());
        }
        if (absl::StartsWith(tag, kBytesContextFeaturePrefixTag)) {
          const std::string& key =
              tag.substr(sizeof(kBytesContextFeaturePrefixTag) /
                             sizeof(*kBytesContextFeaturePrefixTag) -
                         1);
          mpms::ClearContextFeatureBytes(key, sequence_.get());
        }
        if (absl::StartsWith(tag, kFloatFeaturePrefixTag)) {
          std::string key = tag.substr(sizeof(kFloatFeaturePrefixTag) /
                                           sizeof(*kFloatFeaturePrefixTag) -
                                       1);
          mpms::ClearFeatureFloats(key, sequence_.get());
          mpms::ClearFeatureTimestamp(key, sequence_.get());
        }
        if (absl::StartsWith(tag, kIntFeaturePrefixTag)) {
          std::string key = tag.substr(
              sizeof(kIntFeaturePrefixTag) / sizeof(*kIntFeaturePrefixTag) - 1);
          mpms::ClearFeatureInts(key, sequence_.get());
          mpms::ClearFeatureTimestamp(key, sequence_.get());
        }
        if (absl::StartsWith(tag, kBytesFeaturePrefixTag)) {
          std::string key = tag.substr(sizeof(kBytesFeaturePrefixTag) /
                                           sizeof(*kBytesFeaturePrefixTag) -
                                       1);
          mpms::ClearFeatureBytes(key, sequence_.get());
          mpms::ClearFeatureTimestamp(key, sequence_.get());
        }
        if (absl::StartsWith(tag, kKeypointsTag)) {
          std::string key =
              tag.substr(sizeof(kKeypointsTag) / sizeof(*kKeypointsTag) - 1);
          replace_keypoints_ = true;
        }
      }
      if (cc->Inputs().HasTag(kForwardFlowEncodedTag)) {
        mpms::ClearForwardFlowEncoded(sequence_.get());
        mpms::ClearForwardFlowTimestamp(sequence_.get());
      }
    }

    return absl::OkStatus();
  }

  absl::Status VerifySequence() {
    std::string error_msg = "Missing features - ";
    bool all_present = true;
    for (const auto& iter : features_present_) {
      if (!iter.second) {
        all_present = false;
        absl::StrAppend(&error_msg, iter.first, ", ");
      }
    }
    if (all_present) {
      return absl::OkStatus();
    } else {
      return ::mediapipe::NotFoundErrorBuilder(MEDIAPIPE_LOC) << error_msg;
    }
  }

  absl::Status VerifySize() {
    const int64_t MAX_PROTO_BYTES = 1073741823;
    std::string id = mpms::HasExampleId(*sequence_)
                         ? mpms::GetExampleId(*sequence_)
                         : "example";
    RET_CHECK_LT(sequence_->ByteSizeLong(), MAX_PROTO_BYTES)
        << "sequence '" << id
        << "' would be too many bytes to serialize after adding features.";
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    auto& options = cc->Options<PackMediaSequenceCalculatorOptions>();
    if (options.reconcile_metadata()) {
      RET_CHECK_OK(mpms::ReconcileMetadata(
          options.reconcile_bbox_annotations(),
          options.reconcile_region_annotations(), sequence_.get()));
    }

    if (options.skip_large_sequences()) {
      RET_CHECK_OK(VerifySize());
    }
    if (options.output_only_if_all_present()) {
      absl::Status status = VerifySequence();
      if (!status.ok()) {
        cc->GetCounter(status.ToString())->Increment();
        return status;
      }
    }

    if (cc->OutputSidePackets().HasTag(kSequenceExampleTag)) {
      cc->OutputSidePackets()
          .Tag(kSequenceExampleTag)
          .Set(MakePacket<tensorflow::SequenceExample>(*sequence_));
    }
    if (cc->Outputs().HasTag(kSequenceExampleTag)) {
      cc->Outputs()
          .Tag(kSequenceExampleTag)
          .Add(sequence_.release(), options.output_as_zero_timestamp()
                                        ? Timestamp(0ll)
                                        : Timestamp::PostStream());
    }
    sequence_.reset();

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    int image_height = -1;
    int image_width = -1;
    // Because the tag order may vary, we need to loop through tags to get
    // image information before processing other tag types.
    for (const auto& tag : cc->Inputs().GetTags()) {
      if (!cc->Inputs().Tag(tag).IsEmpty()) {
        features_present_[tag] = true;
      }
      if (absl::StartsWith(tag, kImageTag) &&
          !cc->Inputs().Tag(tag).IsEmpty()) {
        std::string key = "";
        if (absl::StartsWith(tag, kImageLabelPrefixTag)) {
          std::string key =
              std::string(absl::StripPrefix(tag, kImageLabelPrefixTag));
          const auto& detection = cc->Inputs().Tag(tag).Get<Detection>();
          if (detection.label().empty()) continue;
          RET_CHECK(detection.label_size() == detection.score_size())
              << "Wrong image label data format: " << detection.label_size()
              << " vs " << detection.score_size();
          if (!detection.label_id().empty()) {
            RET_CHECK(detection.label_id_size() == detection.label_size())
                << "Wrong image label ID format: " << detection.label_id_size()
                << " vs " << detection.label_size();
          }
          std::vector<std::string> labels(detection.label().begin(),
                                          detection.label().end());
          std::vector<float> confidences(detection.score().begin(),
                                         detection.score().end());
          std::vector<int32_t> ids(detection.label_id().begin(),
                                   detection.label_id().end());
          if (!key.empty() || mpms::HasImageEncoded(*sequence_)) {
            mpms::AddImageTimestamp(key, cc->InputTimestamp().Value(),
                                    sequence_.get());
          }
          mpms::AddImageLabelString(key, labels, sequence_.get());
          mpms::AddImageLabelConfidence(key, confidences, sequence_.get());
          if (!ids.empty()) mpms::AddImageLabelIndex(key, ids, sequence_.get());
          continue;
        }
        if (tag != kImageTag) {
          int tag_length = sizeof(kImageTag) / sizeof(*kImageTag) - 1;
          if (tag[tag_length] == '_') {
            key = tag.substr(tag_length + 1);
          } else {
            continue;  // Skip keys that don't match "(kImageTag)_?"
          }
        }
        const OpenCvImageEncoderCalculatorResults& image =
            cc->Inputs().Tag(tag).Get<OpenCvImageEncoderCalculatorResults>();
        if (!image.has_encoded_image()) {
          return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                 << "No encoded image";
        }
        image_height = image.height();
        image_width = image.width();
        mpms::AddImageTimestamp(key, cc->InputTimestamp().Value(),
                                sequence_.get());
        mpms::AddImageEncoded(key, image.encoded_image(), sequence_.get());
      }
    }
    for (const auto& tag : cc->Inputs().GetTags()) {
      if (!cc->Inputs().Tag(tag).IsEmpty()) {
        features_present_[tag] = true;
      }
      if (absl::StartsWith(tag, kKeypointsTag) &&
          !cc->Inputs().Tag(tag).IsEmpty()) {
        std::string key = "";
        if (tag != kKeypointsTag) {
          int tag_length = sizeof(kKeypointsTag) / sizeof(*kKeypointsTag) - 1;
          if (tag[tag_length] == '_') {
            key = tag.substr(tag_length + 1);
          } else {
            continue;  // Skip keys that don't match "(kKeypointsTag)_?"
          }
        }
        const auto& keypoints =
            cc->Inputs()
                .Tag(tag)
                .Get<absl::flat_hash_map<
                    std::string, std::vector<std::pair<float, float>>>>();
        for (const auto& pair : keypoints) {
          std::string prefix = mpms::merge_prefix(key, pair.first);
          if (replace_keypoints_) {
            mpms::ClearBBoxPoint(prefix, sequence_.get());
            mpms::ClearBBoxTimestamp(prefix, sequence_.get());
            mpms::ClearBBoxIsAnnotated(prefix, sequence_.get());
            mpms::ClearBBoxNumRegions(prefix, sequence_.get());
            mpms::ClearBBoxLabelString(prefix, sequence_.get());
            mpms::ClearBBoxLabelIndex(prefix, sequence_.get());
            mpms::ClearBBoxLabelConfidence(prefix, sequence_.get());
            mpms::ClearBBoxClassString(prefix, sequence_.get());
            mpms::ClearBBoxClassIndex(prefix, sequence_.get());
            mpms::ClearBBoxTrackString(prefix, sequence_.get());
            mpms::ClearBBoxTrackIndex(prefix, sequence_.get());
            mpms::ClearUnmodifiedBBoxTimestamp(prefix, sequence_.get());
          }
          mpms::AddBBoxTimestamp(prefix, cc->InputTimestamp().Value(),
                                 sequence_.get());
          mpms::AddBBoxPoint(prefix, pair.second, sequence_.get());
        }
        replace_keypoints_ = false;
      }
      if (absl::StartsWith(tag, kClipLabelPrefixTag) &&
          !cc->Inputs().Tag(tag).IsEmpty()) {
        const std::string& key = tag.substr(
            sizeof(kClipLabelPrefixTag) / sizeof(*kClipLabelPrefixTag) - 1);
        const Detection& detection = cc->Inputs().Tag(tag).Get<Detection>();
        bool add_empty_labels =
            cc->Options<PackMediaSequenceCalculatorOptions>()
                .add_empty_labels();
        if (detection.score().empty()) {
          if (add_empty_labels) {
            mpms::SetClipLabelString(key, {}, sequence_.get());
            mpms::SetClipLabelConfidence(key, {}, sequence_.get());
          }
          continue;
        }
        if (detection.label().empty() && detection.label_id().empty()) {
          return absl::InvalidArgumentError(
              "detection.label and detection.label_id can't be both empty");
        }
        // Allow empty label (for indexed feature inputs), but if label is not
        // empty, it should have the same size as the score field.
        if (!detection.label().empty()) {
          if (detection.label().size() != detection.score().size()) {
            return absl::InvalidArgumentError(
                "Different size of detection.label and detection.score");
          }
        }
        // Allow empty label_ids, but if label_ids is not empty, it should have
        // the same size as the score field.
        if (!detection.label_id().empty()) {
          if (detection.label_id().size() != detection.score().size()) {
            return absl::InvalidArgumentError(
                "Different size of detection.label_id and detection.score");
          }
        }
        for (int i = 0; i < detection.score().size(); ++i) {
          if (!detection.label_id().empty()) {
            mpms::AddClipLabelIndex(key, detection.label_id(i),
                                    sequence_.get());
          }
          if (!detection.label().empty()) {
            mpms::AddClipLabelString(key, detection.label(i), sequence_.get());
          }
          mpms::AddClipLabelConfidence(key, detection.score(i),
                                       sequence_.get());
        }
      }
      if (absl::StartsWith(tag, kFloatContextFeaturePrefixTag) &&
          !cc->Inputs().Tag(tag).IsEmpty()) {
        std::string key =
            tag.substr(sizeof(kFloatContextFeaturePrefixTag) /
                           sizeof(*kFloatContextFeaturePrefixTag) -
                       1);
        RET_CHECK_EQ(cc->InputTimestamp(), Timestamp::PostStream());
        for (const auto& value :
             cc->Inputs().Tag(tag).Get<std::vector<float>>()) {
          mpms::AddContextFeatureFloats(key, value, sequence_.get());
        }
      }
      if (absl::StartsWith(tag, kIntsContextFeaturePrefixTag) &&
          !cc->Inputs().Tag(tag).IsEmpty()) {
        const std::string& key =
            tag.substr(sizeof(kIntsContextFeaturePrefixTag) /
                           sizeof(*kIntsContextFeaturePrefixTag) -
                       1);
        // To ensure only one packet is provided for this tag.
        RET_CHECK_EQ(cc->InputTimestamp(), Timestamp::PostStream());
        for (const auto& value :
             cc->Inputs().Tag(tag).Get<std::vector<int64_t>>()) {
          mpms::AddContextFeatureInts(key, value, sequence_.get());
        }
      }
      if (absl::StartsWith(tag, kBytesContextFeaturePrefixTag) &&
          !cc->Inputs().Tag(tag).IsEmpty()) {
        const std::string& key =
            tag.substr(sizeof(kBytesContextFeaturePrefixTag) /
                           sizeof(*kBytesContextFeaturePrefixTag) -
                       1);
        // To ensure only one packet is provided for this tag.
        RET_CHECK_EQ(cc->InputTimestamp(), Timestamp::PostStream());
        for (const auto& value :
             cc->Inputs().Tag(tag).Get<std::vector<std::string>>()) {
          mpms::AddContextFeatureBytes(key, value, sequence_.get());
        }
      }
      if (absl::StartsWith(tag, kFloatFeaturePrefixTag) &&
          !cc->Inputs().Tag(tag).IsEmpty()) {
        std::string key = tag.substr(sizeof(kFloatFeaturePrefixTag) /
                                         sizeof(*kFloatFeaturePrefixTag) -
                                     1);
        mpms::AddFeatureTimestamp(key, cc->InputTimestamp().Value(),
                                  sequence_.get());
        mpms::AddFeatureFloats(key,
                               cc->Inputs().Tag(tag).Get<std::vector<float>>(),
                               sequence_.get());
      }
      if (absl::StartsWith(tag, kIntFeaturePrefixTag) &&
          !cc->Inputs().Tag(tag).IsEmpty()) {
        std::string key = tag.substr(
            sizeof(kIntFeaturePrefixTag) / sizeof(*kIntFeaturePrefixTag) - 1);
        mpms::AddFeatureTimestamp(key, cc->InputTimestamp().Value(),
                                  sequence_.get());
        mpms::AddFeatureInts(key,
                             cc->Inputs().Tag(tag).Get<std::vector<int64_t>>(),
                             sequence_.get());
      }
      if (absl::StartsWith(tag, kBytesFeaturePrefixTag) &&
          !cc->Inputs().Tag(tag).IsEmpty()) {
        std::string key = tag.substr(sizeof(kBytesFeaturePrefixTag) /
                                         sizeof(*kBytesFeaturePrefixTag) -
                                     1);
        mpms::AddFeatureTimestamp(key, cc->InputTimestamp().Value(),
                                  sequence_.get());
        mpms::AddFeatureBytes(
            key, cc->Inputs().Tag(tag).Get<std::vector<std::string>>(),
            sequence_.get());
      }
      if (absl::StartsWith(tag, kBBoxTag) && !cc->Inputs().Tag(tag).IsEmpty()) {
        std::string key = "";
        if (tag != kBBoxTag) {
          int tag_length = sizeof(kBBoxTag) / sizeof(*kBBoxTag) - 1;
          if (tag[tag_length] == '_') {
            key = tag.substr(tag_length + 1);
          } else {
            continue;  // Skip keys that don't match "(kBBoxTag)_?"
          }
        }
        std::vector<Location> predicted_locations;
        std::vector<std::string> predicted_class_strings;
        std::vector<float> predicted_class_confidences;
        std::vector<int> predicted_label_ids;
        for (auto& detection :
             cc->Inputs().Tag(tag).Get<std::vector<Detection>>()) {
          if (detection.location_data().format() ==
                  LocationData::BOUNDING_BOX ||
              detection.location_data().format() ==
                  LocationData::RELATIVE_BOUNDING_BOX) {
            if (mpms::HasImageHeight(*sequence_) &&
                mpms::HasImageWidth(*sequence_)) {
              image_height = mpms::GetImageHeight(*sequence_);
              image_width = mpms::GetImageWidth(*sequence_);
            }
            if (image_height == -1 || image_width == -1) {
              return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                     << "Images must be provided with bounding boxes or the "
                        "image "
                     << "height and width must already be in the example.";
            }
            Location relative_bbox = Location::CreateRelativeBBoxLocation(
                Location(detection.location_data())
                    .ConvertToRelativeBBox(image_width, image_height));
            predicted_locations.push_back(relative_bbox);
            if (detection.label_size() > 0) {
              predicted_class_strings.push_back(detection.label(0));
            }
            if (detection.label_id_size() > 0) {
              predicted_label_ids.push_back(detection.label_id(0));
            }
            if (detection.score_size() > 0) {
              predicted_class_confidences.push_back(detection.score(0));
            }
          }
        }
        if (!predicted_locations.empty()) {
          mpms::AddBBox(key, predicted_locations, sequence_.get());
          mpms::AddBBoxTimestamp(key, cc->InputTimestamp().Value(),
                                 sequence_.get());
          if (!predicted_class_strings.empty()) {
            mpms::AddBBoxLabelString(key, predicted_class_strings,
                                     sequence_.get());
          }
          if (!predicted_label_ids.empty()) {
            mpms::AddBBoxLabelIndex(key, predicted_label_ids, sequence_.get());
          }
          if (!predicted_class_confidences.empty()) {
            mpms::AddBBoxLabelConfidence(key, predicted_class_confidences,
                                         sequence_.get());
          }
        }
      }
    }
    if (cc->Inputs().HasTag(kForwardFlowEncodedTag) &&
        !cc->Inputs().Tag(kForwardFlowEncodedTag).IsEmpty()) {
      const OpenCvImageEncoderCalculatorResults& forward_flow =
          cc->Inputs()
              .Tag(kForwardFlowEncodedTag)
              .Get<OpenCvImageEncoderCalculatorResults>();
      if (!forward_flow.has_encoded_image()) {
        return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
               << "No encoded forward flow";
      }
      mpms::AddForwardFlowTimestamp(cc->InputTimestamp().Value(),
                                    sequence_.get());
      mpms::AddForwardFlowEncoded(forward_flow.encoded_image(),
                                  sequence_.get());
    }
    if (cc->Inputs().HasTag(kSegmentationMaskTag) &&
        !cc->Inputs().Tag(kSegmentationMaskTag).IsEmpty()) {
      bool already_has_mask = false;
      for (auto& detection : cc->Inputs()
                                 .Tag(kSegmentationMaskTag)
                                 .Get<std::vector<Detection>>()) {
        if (detection.location_data().format() == LocationData::MASK) {
          RET_CHECK(!already_has_mask)
              << "We currently only support adding one mask per timestamp. "
              << sequence_->DebugString();
          auto mask_mat_ptr = GetCvMask(Location(detection.location_data()));
          std::vector<uchar> bytes;
          RET_CHECK(cv::imencode(".png", *mask_mat_ptr, bytes, {}));

          std::string encoded_mask(bytes.begin(), bytes.end());
          mpms::AddClassSegmentationEncoded(encoded_mask, sequence_.get());
          mpms::AddClassSegmentationTimestamp(cc->InputTimestamp().Value(),
                                              sequence_.get());
          // SegmentationClassLabelString is a context feature for the entire
          // sequence. The values in the last detection will be saved.
          mpms::SetClassSegmentationClassLabelString({detection.label(0)},
                                                     sequence_.get());
          already_has_mask = true;
        } else {
          return absl::UnimplementedError(
              "Global detections and empty detections are not supported.");
        }
      }
    }
    if (clip_media_id_.has_value()) {
      mpms::SetClipMediaId(*clip_media_id_, sequence_.get());
    }
    return absl::OkStatus();
  }

  std::unique_ptr<tf::SequenceExample> sequence_;
  std::optional<std::string> clip_media_id_ = std::nullopt;
  std::map<std::string, bool> features_present_;
  bool replace_keypoints_;
};
REGISTER_CALCULATOR(PackMediaSequenceCalculator);

}  // namespace mediapipe
