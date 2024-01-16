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

#include <stdlib.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/strings/str_split.h"
#include "mediapipe/calculators/video/opencv_video_encoder_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/tool/status_util.h"

namespace mediapipe {

constexpr char kAudioFilePathTag[] = "AUDIO_FILE_PATH";
constexpr char kOutputFilePathTag[] = "OUTPUT_FILE_PATH";
constexpr char kVideoPrestreamTag[] = "VIDEO_PRESTREAM";
constexpr char kVideoTag[] = "VIDEO";

// Encodes the input video stream and produces a media file.
// The media file can be output to the output_file_path specified as a side
// packet. Currently, the calculator only supports one video stream (in
// mediapipe::ImageFrame).
//
// Example config:
// node {
//   calculator: "OpenCvVideoEncoderCalculator"
//   input_stream: "VIDEO:video"
//   input_stream: "VIDEO_PRESTREAM:video_header"
//   input_side_packet: "OUTPUT_FILE_PATH:output_file_path"
//   node_options {
//     [type.googleapis.com/mediapipe.OpenCvVideoEncoderCalculatorOptions]: {
//        codec: "avc1"
//        video_format: "mp4"
//     }
//   }
// }
//
// OpenCV's VideoWriter doesn't encode audio. If an input side packet with tag
// "AUDIO_FILE_PATH" is specified, the calculator will call FFmpeg binary to
// attach the audio file to the video as the last step in Close().
//
// Example config:
// node {
//   calculator: "OpenCvVideoEncoderCalculator"
//   input_stream: "VIDEO:video"
//   input_stream: "VIDEO_PRESTREAM:video_header"
//   input_side_packet: "OUTPUT_FILE_PATH:output_file_path"
//   input_side_packet: "AUDIO_FILE_PATH:audio_path"
//   node_options {
//     [type.googleapis.com/mediapipe.OpenCvVideoEncoderCalculatorOptions]: {
//        codec: "avc1"
//        video_format: "mp4"
//     }
//   }
// }
//
class OpenCvVideoEncoderCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status SetUpVideoWriter(float frame_rate, int width, int height);

  std::string output_file_path_;
  int four_cc_;
  std::unique_ptr<cv::VideoWriter> writer_;
};

absl::Status OpenCvVideoEncoderCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kVideoTag));
  cc->Inputs().Tag(kVideoTag).Set<ImageFrame>();
  if (cc->Inputs().HasTag(kVideoPrestreamTag)) {
    cc->Inputs().Tag(kVideoPrestreamTag).Set<VideoHeader>();
  }
  RET_CHECK(cc->InputSidePackets().HasTag(kOutputFilePathTag));
  cc->InputSidePackets().Tag(kOutputFilePathTag).Set<std::string>();
  if (cc->InputSidePackets().HasTag(kAudioFilePathTag)) {
    cc->InputSidePackets().Tag(kAudioFilePathTag).Set<std::string>();
  }
  return absl::OkStatus();
}

absl::Status OpenCvVideoEncoderCalculator::Open(CalculatorContext* cc) {
  OpenCvVideoEncoderCalculatorOptions options =
      cc->Options<OpenCvVideoEncoderCalculatorOptions>();
  RET_CHECK(options.has_codec() && options.codec().length() == 4)
      << "A 4-character codec code must be specified in "
         "OpenCvVideoEncoderCalculatorOptions";
  const char* codec_array = options.codec().c_str();
  four_cc_ = mediapipe::fourcc(codec_array[0], codec_array[1], codec_array[2],
                               codec_array[3]);
  RET_CHECK(!options.video_format().empty())
      << "Video format must be specified in "
         "OpenCvVideoEncoderCalculatorOptions";
  output_file_path_ =
      cc->InputSidePackets().Tag(kOutputFilePathTag).Get<std::string>();
  std::vector<std::string> splited_file_path =
      absl::StrSplit(output_file_path_, '.');
  RET_CHECK(splited_file_path.size() >= 2 &&
            splited_file_path[splited_file_path.size() - 1] ==
                options.video_format())
      << "The output file path is invalid.";
  // If the video header will be available, the video metadata will be fetched
  // from the video header directly. The calculator will receive the video
  // header packet at timestamp prestream.
  if (cc->Inputs().HasTag(kVideoPrestreamTag)) {
    return absl::OkStatus();
  }
  return SetUpVideoWriter(options.fps(), options.width(), options.height());
}

absl::Status OpenCvVideoEncoderCalculator::Process(CalculatorContext* cc) {
  if (cc->InputTimestamp() == Timestamp::PreStream()) {
    const VideoHeader& video_header =
        cc->Inputs().Tag(kVideoPrestreamTag).Get<VideoHeader>();
    return SetUpVideoWriter(video_header.frame_rate, video_header.width,
                            video_header.height);
  }

  const ImageFrame& image_frame =
      cc->Inputs().Tag(kVideoTag).Value().Get<ImageFrame>();
  ImageFormat::Format format = image_frame.Format();
  cv::Mat frame;
  if (format == ImageFormat::GRAY8) {
    frame = formats::MatView(&image_frame);
    if (frame.empty()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Receive empty frame at timestamp "
             << cc->Inputs().Tag(kVideoTag).Value().Timestamp()
             << " in OpenCvVideoEncoderCalculator::Process()";
    }
  } else {
    cv::Mat tmp_frame = formats::MatView(&image_frame);
    if (tmp_frame.empty()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Receive empty frame at timestamp "
             << cc->Inputs().Tag(kVideoTag).Value().Timestamp()
             << " in OpenCvVideoEncoderCalculator::Process()";
    }
    if (format == ImageFormat::SRGB) {
      cv::cvtColor(tmp_frame, frame, cv::COLOR_RGB2BGR);
    } else if (format == ImageFormat::SRGBA) {
      cv::cvtColor(tmp_frame, frame, cv::COLOR_RGBA2BGR);
    } else {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Unsupported image format: " << format;
    }
  }
  writer_->write(frame);
  return absl::OkStatus();
}

absl::Status OpenCvVideoEncoderCalculator::Close(CalculatorContext* cc) {
  if (writer_ && writer_->isOpened()) {
    writer_->release();
  }
  if (cc->InputSidePackets().HasTag(kAudioFilePathTag)) {
#ifdef HAVE_FFMPEG
    const std::string& audio_file_path =
        cc->InputSidePackets().Tag(kAudioFilePathTag).Get<std::string>();
    if (audio_file_path.empty()) {
      ABSL_LOG(WARNING)
          << "OpenCvVideoEncoderCalculator isn't able to attach the "
             "audio tracks to the generated video because the audio "
             "file path is not specified.";
    } else {
      // A temp output file is needed because FFmpeg can't do in-place editing.
      const std::string temp_file_path = std::tmpnam(nullptr);
      system(absl::StrCat("mv ", output_file_path_, " ", temp_file_path,
                          "&& ffmpeg -nostats -loglevel 0 -i ", temp_file_path,
                          " -i ", audio_file_path,
                          "  -c copy -map 0:v:0 -map 1:a:0 ", output_file_path_,
                          "&& rm ", temp_file_path)
                 .c_str());
    }

#else
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "OpenCVVideoEncoderCalculator can't attach the audio tracks to "
              "the video because FFmpeg is not installed. Please remove "
              "input_side_packet: \"AUDIO_FILE_PATH\" from the node "
              "config.";
#endif
  }
  return absl::OkStatus();
}

absl::Status OpenCvVideoEncoderCalculator::SetUpVideoWriter(float frame_rate,
                                                            int width,
                                                            int height) {
  RET_CHECK(frame_rate > 0 && width > 0 && height > 0)
      << "Invalid video metadata: frame_rate=" << frame_rate
      << ", width=" << width << ", height=" << height;
  writer_ = absl::make_unique<cv::VideoWriter>(
      output_file_path_, four_cc_, frame_rate, cv::Size(width, height));
  if (!writer_->isOpened()) {
    return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Fail to open file at " << output_file_path_;
  }
  return absl::OkStatus();
}

REGISTER_CALCULATOR(OpenCvVideoEncoderCalculator);
}  // namespace mediapipe
