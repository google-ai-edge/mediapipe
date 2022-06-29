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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "mediapipe/calculators/core/packet_resampler_calculator.pb.h"
#include "mediapipe/calculators/tensorflow/unpack_media_sequence_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/audio_decoder.pb.h"
#include "mediapipe/util/sequence/media_sequence.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

namespace mediapipe {

// Streams:
const char kBBoxTag[] = "BBOX";
const char kImageTag[] = "IMAGE";
const char kKeypointsTag[] = "KEYPOINTS";
const char kFloatFeaturePrefixTag[] = "FLOAT_FEATURE_";
const char kForwardFlowImageTag[] = "FORWARD_FLOW_ENCODED";

// Side Packets:
const char kSequenceExampleTag[] = "SEQUENCE_EXAMPLE";
const char kDatasetRootDirTag[] = "DATASET_ROOT";
const char kDataPath[] = "DATA_PATH";
const char kPacketResamplerOptions[] = "RESAMPLER_OPTIONS";
const char kImagesFrameRateTag[] = "IMAGE_FRAME_RATE";
const char kAudioDecoderOptions[] = "AUDIO_DECODER_OPTIONS";

namespace tf = ::tensorflow;
namespace mpms = mediapipe::mediasequence;

// Source calculator to unpack side_packets and streams from tf.SequenceExamples
//
// Often, only side_packets or streams need to be output, but both can be output
// if needed. A tf.SequenceExample always needs to be supplied as an
// input_side_packet. The SequenceExample must be in the format described in
// media_sequence.h. This documentation will first describe the side_packets
// the calculator can output, and then describe the streams.
//
// Side_packets are commonly used to specify which clip to extract data from.
// Seeking into a video does not necessarily provide consistent timestamps when
// resampling to a known rate. To enable consistent timestamps, we unpack the
// metadata into options for the MediaDecoderCalculator and the
// PacketResamplerCalculator. To ensure consistent timestamps, the MediaDecoder
// needs to seek to slightly before the clip starts, so it sees at least one
// packet before the first packet we want to keep. The PacketResamplerCalculator
// then trims down the timestamps. Furthermore, we should always specify that we
// want timestamps from a base timestamp of 0, so we have the same resampled
// frames after a seek that we would have from the start of a video. In summary,
// when decoding image frames, output both the DECODER_OPTIONS and
// RESAMPLER_OPTIONS. In the base_media_decoder_options, specify which streams
// you want. In the base_packet_resampler_options, specify the frame_rate you
// want and base_timestamp = 0. In the options for this calculator, specify
// padding extra_padding_from_media_decoder such that at least one frame arrives
// before the first frame the PacketResamplerCalculator should output.
//
// Optional output_side_packets include (referenced by tag):
//  DATA_PATH: The data_path context feature joined onto the
//    options.dataset_root_directory or input_side_packet of DATASET_ROOT.
//  RESAMPLER_OPTIONS: CalculatorOptions to pass to the
//    PacketResamplerCalculator. The most accurate procedure for sampling a
//    range of frames is to request a padded time range from the
//    MediaDecoderCalculator and then trim it down to the proper time range with
//    the PacketResamplerCalculator.
//  IMAGES_FRAME_RATE: The frame rate of the images in the original video as a
//    double.
//
// Example config:
// node {
//   calculator: "UnpackMediaSequenceCalculator"
//   input_side_packet: "SEQUENCE_EXAMPLE:example_input_side_packet"
//   input_side_packet: "DATASET_ROOT:path_to_dataset_root_directory"
//   output_side_packet: "DATA_PATH:full_path_to_data_element"
//   output_side_packet: "RESAMPLER_OPTIONS:packet_resampler_options"
//   options {
//     [mediapipe.UnpackMediaSequenceCalculatorOptions.ext]: {
//       base_packet_resampler_options {
//         frame_rate: 1.0  # PARAM_FRAME_RATE
//         base_timestamp: 0
//       }
//     }
//   }
// }
//
// The calculator also takes a tf.SequenceExample as a side input and outputs
// the data in streams from the SequenceExample at the proper timestamps. The
// SequenceExample must conform to the description in media_sequence.h.
// Timestamps in the SequenceExample must be in sequential order.
//
// The following output stream tags are supported:
//   IMAGE: encoded images as strings. (IMAGE_${NAME} is supported.)
//   FORWARD_FLOW_ENCODED: encoded FORWARD_FLOW prefix images as strings.
//   FLOAT_FEATURE_${NAME}: the feature named ${NAME} as vector<float>.
//   BBOX: bounding boxes as vector<Location>s. (BBOX_${NAME} is supported.)
//
// Example config:
// node {
//   calculator: "UnpackMediaSequenceCalculator"
//   input_side_packet: "SEQUENCE_EXAMPLE:example_input_side_packet"
//   output_stream: "IMAGE:frames"
//   output_stream: "FLOAT_FEATURE_FDENSE:fdense_vf"
//   output_stream: "BBOX:faces"
// }
class UnpackMediaSequenceCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    const auto& options = cc->Options<UnpackMediaSequenceCalculatorOptions>();
    RET_CHECK(cc->InputSidePackets().HasTag(kSequenceExampleTag));
    cc->InputSidePackets().Tag(kSequenceExampleTag).Set<tf::SequenceExample>();
    // Optional side inputs.
    if (cc->InputSidePackets().HasTag(kDatasetRootDirTag)) {
      cc->InputSidePackets().Tag(kDatasetRootDirTag).Set<std::string>();
    }
    if (cc->OutputSidePackets().HasTag(kDataPath)) {
      cc->OutputSidePackets().Tag(kDataPath).Set<std::string>();
    }
    if (cc->OutputSidePackets().HasTag(kAudioDecoderOptions)) {
      cc->OutputSidePackets()
          .Tag(kAudioDecoderOptions)
          .Set<AudioDecoderOptions>();
    }
    if (cc->OutputSidePackets().HasTag(kImagesFrameRateTag)) {
      cc->OutputSidePackets().Tag(kImagesFrameRateTag).Set<double>();
    }
    if (cc->OutputSidePackets().HasTag(kPacketResamplerOptions)) {
      cc->OutputSidePackets()
          .Tag(kPacketResamplerOptions)
          .Set<CalculatorOptions>();
    }
    if ((options.has_padding_before_label() ||
         options.has_padding_after_label()) &&
        !(cc->OutputSidePackets().HasTag(kAudioDecoderOptions) ||
          cc->OutputSidePackets().HasTag(kPacketResamplerOptions))) {
      return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "If specifying padding, must output " << kPacketResamplerOptions
             << "or" << kAudioDecoderOptions;
    }

    if (cc->Outputs().HasTag(kForwardFlowImageTag)) {
      cc->Outputs().Tag(kForwardFlowImageTag).Set<std::string>();
    }
    for (const auto& tag : cc->Outputs().GetTags()) {
      if (absl::StartsWith(tag, kImageTag)) {
        std::string key = "";
        if (tag != kImageTag) {
          int tag_length = sizeof(kImageTag) / sizeof(*kImageTag) - 1;
          if (tag[tag_length] == '_') {
            key = tag.substr(tag_length + 1);
          } else {
            continue;  // Skip keys that don't match "(kImageTag)_?"
          }
        }
        cc->Outputs().Tag(tag).Set<std::string>();
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
        cc->Outputs().Tag(tag).Set<std::vector<Location>>();
      }
      if (absl::StartsWith(tag, kFloatFeaturePrefixTag)) {
        cc->Outputs().Tag(tag).Set<std::vector<float>>();
      }
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    // Copy the packet to copy the otherwise inaccessible shared ptr.
    example_packet_holder_ = cc->InputSidePackets().Tag(kSequenceExampleTag);
    sequence_ = &example_packet_holder_.Get<tf::SequenceExample>();

    // Collect the timestamps for all streams keyed by the timestamp feature's
    // key. While creating this data structure we also identify the last
    // timestamp and the associated feature. This information is used in process
    // to output batches of packets in order.
    timestamps_.clear();
    int64 last_timestamp_seen = Timestamp::PreStream().Value();
    first_timestamp_seen_ = Timestamp::OneOverPostStream().Value();
    for (const auto& map_kv : sequence_->feature_lists().feature_list()) {
      if (absl::StrContains(map_kv.first, "/timestamp")) {
        LOG(INFO) << "Found feature timestamps: " << map_kv.first
                  << " with size: " << map_kv.second.feature_size();
        int64 recent_timestamp = Timestamp::PreStream().Value();
        for (int i = 0; i < map_kv.second.feature_size(); ++i) {
          int64 next_timestamp =
              mpms::GetInt64sAt(*sequence_, map_kv.first, i).Get(0);
          RET_CHECK_GT(next_timestamp, recent_timestamp)
              << "Timestamps must be sequential. If you're seeing this message "
              << "you may have added images to the same SequenceExample twice. "
              << "Key: " << map_kv.first;
          timestamps_[map_kv.first].push_back(next_timestamp);
          recent_timestamp = next_timestamp;
          if (recent_timestamp < first_timestamp_seen_) {
            first_timestamp_seen_ = recent_timestamp;
          }
        }
        if (recent_timestamp > last_timestamp_seen &&
            recent_timestamp < Timestamp::PostStream().Value()) {
          last_timestamp_key_ = map_kv.first;
          last_timestamp_seen = recent_timestamp;
        }
      }
    }
    if (!timestamps_.empty()) {
      for (const auto& kv : timestamps_) {
        if (!kv.second.empty() &&
            kv.second[0] < Timestamp::PostStream().Value()) {
          // These checks only make sense if any values are not PostStream, but
          // only need to be made once.
          RET_CHECK(!last_timestamp_key_.empty())
              << "Something went wrong because the timestamp key is unset. "
              << "Example: " << sequence_->DebugString();
          RET_CHECK_GT(last_timestamp_seen, Timestamp::PreStream().Value())
              << "Something went wrong because the last timestamp is unset. "
              << "Example: " << sequence_->DebugString();
          RET_CHECK_LT(first_timestamp_seen_,
                       Timestamp::OneOverPostStream().Value())
              << "Something went wrong because the first timestamp is unset. "
              << "Example: " << sequence_->DebugString();
          break;
        }
      }
    }
    current_timestamp_index_ = 0;
    process_poststream_ = false;

    // Determine the data path and output it.
    const auto& options = cc->Options<UnpackMediaSequenceCalculatorOptions>();
    const auto& sequence = cc->InputSidePackets()
                               .Tag(kSequenceExampleTag)
                               .Get<tensorflow::SequenceExample>();
    if (cc->OutputSidePackets().HasTag(kDataPath)) {
      std::string root_directory = "";
      if (cc->InputSidePackets().HasTag(kDatasetRootDirTag)) {
        root_directory =
            cc->InputSidePackets().Tag(kDatasetRootDirTag).Get<std::string>();
      } else if (options.has_dataset_root_directory()) {
        root_directory = options.dataset_root_directory();
      }

      std::string data_path = mpms::GetClipDataPath(sequence);
      if (!root_directory.empty()) {
        if (root_directory[root_directory.size() - 1] == '/') {
          data_path = root_directory + data_path;
        } else {
          data_path = root_directory + "/" + data_path;
        }
      }
      cc->OutputSidePackets().Tag(kDataPath).Set(
          MakePacket<std::string>(data_path));
    }

    // Set the start and end of the clip in the appropriate options protos.
    double start_time = 0;
    double end_time = 0;
    if (cc->OutputSidePackets().HasTag(kAudioDecoderOptions) ||
        cc->OutputSidePackets().HasTag(kPacketResamplerOptions)) {
      if (mpms::HasClipStartTimestamp(sequence)) {
        start_time =
            Timestamp(mpms::GetClipStartTimestamp(sequence)).Seconds() -
            options.padding_before_label();
      }
      if (mpms::HasClipEndTimestamp(sequence)) {
        end_time = Timestamp(mpms::GetClipEndTimestamp(sequence)).Seconds() +
                   options.padding_after_label();
      }
    }
    if (cc->OutputSidePackets().HasTag(kAudioDecoderOptions)) {
      auto audio_decoder_options = absl::make_unique<AudioDecoderOptions>(
          options.base_audio_decoder_options());
      if (mpms::HasClipStartTimestamp(sequence)) {
        if (options.force_decoding_from_start_of_media()) {
          audio_decoder_options->set_start_time(0);
        } else {
          audio_decoder_options->set_start_time(
              start_time - options.extra_padding_from_media_decoder());
        }
      }
      if (mpms::HasClipEndTimestamp(sequence)) {
        audio_decoder_options->set_end_time(
            end_time + options.extra_padding_from_media_decoder());
      }
      LOG(INFO) << "Created AudioDecoderOptions:\n"
                << audio_decoder_options->DebugString();
      cc->OutputSidePackets()
          .Tag(kAudioDecoderOptions)
          .Set(Adopt(audio_decoder_options.release()));
    }
    if (cc->OutputSidePackets().HasTag(kPacketResamplerOptions)) {
      auto resampler_options = absl::make_unique<CalculatorOptions>();
      *(resampler_options->MutableExtension(
          PacketResamplerCalculatorOptions::ext)) =
          options.base_packet_resampler_options();
      if (mpms::HasClipStartTimestamp(sequence)) {
        resampler_options
            ->MutableExtension(PacketResamplerCalculatorOptions::ext)
            ->set_start_time(Timestamp::FromSeconds(start_time).Value());
      }
      if (mpms::HasClipEndTimestamp(sequence)) {
        resampler_options
            ->MutableExtension(PacketResamplerCalculatorOptions::ext)
            ->set_end_time(Timestamp::FromSeconds(end_time).Value());
      }

      LOG(INFO) << "Created PacketResamplerOptions:\n"
                << resampler_options->DebugString();
      cc->OutputSidePackets()
          .Tag(kPacketResamplerOptions)
          .Set(Adopt(resampler_options.release()));
    }

    // Output the remaining side outputs.
    if (cc->OutputSidePackets().HasTag(kImagesFrameRateTag)) {
      cc->OutputSidePackets()
          .Tag(kImagesFrameRateTag)
          .Set(MakePacket<double>(mpms::GetImageFrameRate(sequence)));
    }

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (timestamps_.empty()) {
      // This occurs when we only have metadata to unpack.
      LOG(INFO) << "only unpacking metadata because there are no timestamps.";
      return tool::StatusStop();
    }
    // In Process(), we loop through timestamps on a reference stream and emit
    // all packets on all streams that have a timestamp between the current
    // reference timestep and the previous reference timestep. This ensures that
    // we emit all timestamps in order, but also only emit a limited number in
    // any particular call to Process(). At the every end, we output the
    // poststream packets. If we only have poststream packets,
    // last_timestamp_key_ will be empty.
    int64 start_timestamp = 0;
    int64 end_timestamp = 0;
    if (last_timestamp_key_.empty() || process_poststream_) {
      process_poststream_ = true;
      start_timestamp = Timestamp::PostStream().Value();
      end_timestamp = Timestamp::OneOverPostStream().Value();
    } else {
      start_timestamp =
          timestamps_[last_timestamp_key_][current_timestamp_index_];
      if (current_timestamp_index_ == 0) {
        start_timestamp = first_timestamp_seen_;
      }

      end_timestamp = start_timestamp + 1;  // Base case at end of sequence.
      if (current_timestamp_index_ <
          timestamps_[last_timestamp_key_].size() - 1) {
        end_timestamp =
            timestamps_[last_timestamp_key_][current_timestamp_index_ + 1];
      }
    }

    for (const auto& map_kv : timestamps_) {
      for (int i = 0; i < map_kv.second.size(); ++i) {
        if (map_kv.second[i] >= start_timestamp &&
            map_kv.second[i] < end_timestamp) {
          const Timestamp current_timestamp =
              map_kv.second[i] == Timestamp::PostStream().Value()
                  ? Timestamp::PostStream()
                  : Timestamp(map_kv.second[i]);

          if (absl::StrContains(map_kv.first, mpms::GetImageTimestampKey())) {
            std::vector<std::string> pieces = absl::StrSplit(map_kv.first, '/');
            std::string feature_key = "";
            std::string possible_tag = kImageTag;
            if (pieces[0] != "image") {
              feature_key = pieces[0];
              possible_tag = absl::StrCat(kImageTag, "_", feature_key);
            }
            if (cc->Outputs().HasTag(possible_tag)) {
              cc->Outputs()
                  .Tag(possible_tag)
                  .Add(new std::string(
                           mpms::GetImageEncodedAt(feature_key, *sequence_, i)),
                       current_timestamp);
            }
          }

          if (cc->Outputs().HasTag(kForwardFlowImageTag) &&
              map_kv.first == mpms::GetForwardFlowTimestampKey()) {
            cc->Outputs()
                .Tag(kForwardFlowImageTag)
                .Add(new std::string(
                         mpms::GetForwardFlowEncodedAt(*sequence_, i)),
                     current_timestamp);
          }
          if (absl::StrContains(map_kv.first, mpms::GetBBoxTimestampKey())) {
            std::vector<std::string> pieces = absl::StrSplit(map_kv.first, '/');
            std::string feature_key = "";
            std::string possible_tag = kBBoxTag;
            if (pieces[0] != "region") {
              feature_key = pieces[0];
              possible_tag = absl::StrCat(kBBoxTag, "_", feature_key);
            }
            if (cc->Outputs().HasTag(possible_tag)) {
              const auto& bboxes = mpms::GetBBoxAt(feature_key, *sequence_, i);
              cc->Outputs()
                  .Tag(possible_tag)
                  .Add(new std::vector<Location>(bboxes.begin(), bboxes.end()),
                       current_timestamp);
            }
          }

          if (absl::StrContains(map_kv.first, "feature")) {
            std::vector<std::string> pieces = absl::StrSplit(map_kv.first, '/');
            RET_CHECK_GT(pieces.size(), 1)
                << "Failed to parse the feature substring before / from key "
                << map_kv.first;
            std::string feature_key = pieces[0];
            std::string possible_tag = kFloatFeaturePrefixTag + feature_key;
            if (cc->Outputs().HasTag(possible_tag)) {
              const auto& float_list =
                  mpms::GetFeatureFloatsAt(feature_key, *sequence_, i);
              cc->Outputs()
                  .Tag(possible_tag)
                  .Add(new std::vector<float>(float_list.begin(),
                                              float_list.end()),
                       current_timestamp);
            }
          }
        }
      }
    }

    ++current_timestamp_index_;
    if (current_timestamp_index_ < timestamps_[last_timestamp_key_].size()) {
      return absl::OkStatus();
    } else {
      if (process_poststream_) {
        // Once we've processed the PostStream timestamp we can stop.
        return tool::StatusStop();
      } else {
        // Otherwise, we still need to do one more pass to process it.
        process_poststream_ = true;
        return absl::OkStatus();
      }
    }
  }

  // Hold a copy of the packet to prevent the shared_ptr from dying and then
  // access the SequenceExample with a handy pointer.
  const tf::SequenceExample* sequence_;
  Packet example_packet_holder_;

  // Store a map from the keys for each stream to the timestamps for each
  // key. This allows us to identify which packets to output for each stream
  // for timestamps within a given time window.
  std::map<std::string, std::vector<int64>> timestamps_;
  // Store the stream with the latest timestamp in the SequenceExample.
  std::string last_timestamp_key_;
  // Store the index of the current timestamp. Will be less than
  // timestamps_[last_timestamp_key_].size().
  int current_timestamp_index_;
  // Store the very first timestamp, so we output everything on the first frame.
  int64 first_timestamp_seen_;
  // List of keypoint names.
  std::vector<std::string> keypoint_names_;
  // Default keypoint location when missing.
  float default_keypoint_location_;
  bool process_poststream_;
};
REGISTER_CALCULATOR(UnpackMediaSequenceCalculator);
}  // namespace mediapipe
