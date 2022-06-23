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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"

namespace mediapipe {

namespace {

constexpr char kSavedAudioPathTag[] = "SAVED_AUDIO_PATH";
constexpr char kVideoPrestreamTag[] = "VIDEO_PRESTREAM";
constexpr char kVideoTag[] = "VIDEO";
constexpr char kInputFilePathTag[] = "INPUT_FILE_PATH";

// cv::VideoCapture set data type to unsigned char by default. Therefore, the
// image format is only related to the number of channles the cv::Mat has.
ImageFormat::Format GetImageFormat(int num_channels) {
  ImageFormat::Format format;
  switch (num_channels) {
    case 1:
      format = ImageFormat::GRAY8;
      break;
    case 3:
      format = ImageFormat::SRGB;
      break;
    case 4:
      format = ImageFormat::SRGBA;
      break;
    default:
      format = ImageFormat::UNKNOWN;
      break;
  }
  return format;
}
}  // namespace

// This Calculator takes no input streams and produces video packets.
// All streams and input side packets are specified using tags and all of them
// are optional.
//
// Output Streams:
//   VIDEO: Output video frames (ImageFrame).
//   VIDEO_PRESTREAM:
//       Optional video header information output at
//       Timestamp::PreStream() for the corresponding stream.
// Input Side Packets:
//   INPUT_FILE_PATH: The input file path.
//
// Example config:
// node {
//   calculator: "OpenCvVideoDecoderCalculator"
//   input_side_packet: "INPUT_FILE_PATH:input_file_path"
//   output_stream: "VIDEO:video_frames"
//   output_stream: "VIDEO_PRESTREAM:video_header"
// }
//
// OpenCV's VideoCapture doesn't decode audio tracks. If the audio tracks need
// to be saved, specify an output side packet with tag "SAVED_AUDIO_PATH".
// The calculator will call FFmpeg binary to save audio tracks as an aac file.
// If the audio tracks can't be extracted by FFmpeg, the output side packet
// will contain an empty string.
//
// Example config:
// node {
//   calculator: "OpenCvVideoDecoderCalculator"
//   input_side_packet: "INPUT_FILE_PATH:input_file_path"
//   output_side_packet: "SAVED_AUDIO_PATH:audio_path"
//   output_stream: "VIDEO:video_frames"
//   output_stream: "VIDEO_PRESTREAM:video_header"
// }
//
class OpenCvVideoDecoderCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Tag(kInputFilePathTag).Set<std::string>();
    cc->Outputs().Tag(kVideoTag).Set<ImageFrame>();
    if (cc->Outputs().HasTag(kVideoPrestreamTag)) {
      cc->Outputs().Tag(kVideoPrestreamTag).Set<VideoHeader>();
    }
    if (cc->OutputSidePackets().HasTag(kSavedAudioPathTag)) {
      cc->OutputSidePackets().Tag(kSavedAudioPathTag).Set<std::string>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    const std::string& input_file_path =
        cc->InputSidePackets().Tag(kInputFilePathTag).Get<std::string>();
    cap_ = absl::make_unique<cv::VideoCapture>(input_file_path);
    if (!cap_->isOpened()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Fail to open video file at " << input_file_path;
    }
    width_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = static_cast<double>(cap_->get(cv::CAP_PROP_FPS));
    frame_count_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_COUNT));
    // Unfortunately, cap_->get(cv::CAP_PROP_FORMAT) always returns CV_8UC1
    // back. To get correct image format, we read the first frame from the video
    // and get the number of channels.
    cv::Mat frame;
    ReadFrame(frame);
    if (frame.empty()) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Fail to read any frames from the video file at "
             << input_file_path;
    }
    format_ = GetImageFormat(frame.channels());
    if (format_ == ImageFormat::UNKNOWN) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Unsupported video format of the video file at "
             << input_file_path;
    }

    if (fps <= 0 || frame_count_ <= 0 || width_ <= 0 || height_ <= 0) {
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "Fail to make video header due to the incorrect metadata from "
                "the video file at "
             << input_file_path;
    }
    auto header = absl::make_unique<VideoHeader>();
    header->format = format_;
    header->width = width_;
    header->height = height_;
    header->frame_rate = fps;
    header->duration = frame_count_ / fps;

    if (cc->Outputs().HasTag(kVideoPrestreamTag)) {
      cc->Outputs()
          .Tag(kVideoPrestreamTag)
          .Add(header.release(), Timestamp::PreStream());
      cc->Outputs().Tag(kVideoPrestreamTag).Close();
    }
    // Rewind to the very first frame.
    cap_->set(cv::CAP_PROP_POS_AVI_RATIO, 0);

    if (cc->OutputSidePackets().HasTag(kSavedAudioPathTag)) {
#ifdef HAVE_FFMPEG
      std::string saved_audio_path = std::tmpnam(nullptr);
      std::string ffmpeg_command =
          absl::StrCat("ffmpeg -nostats -loglevel 0 -i ", input_file_path,
                       " -vn -f adts ", saved_audio_path);
      system(ffmpeg_command.c_str());
      int status_code = system(absl::StrCat("ls ", saved_audio_path).c_str());
      if (status_code == 0) {
        cc->OutputSidePackets()
            .Tag(kSavedAudioPathTag)
            .Set(MakePacket<std::string>(saved_audio_path));
      } else {
        LOG(WARNING) << "FFmpeg can't extract audio from " << input_file_path
                     << " by executing the following command: "
                     << ffmpeg_command;
        cc->OutputSidePackets()
            .Tag(kSavedAudioPathTag)
            .Set(MakePacket<std::string>(std::string()));
      }
#else
      return mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
             << "OpenCVVideoDecoderCalculator can't save the audio file "
                "because FFmpeg is not installed. Please remove "
                "output_side_packet: \"SAVED_AUDIO_PATH\" from the node "
                "config.";
#endif
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    auto image_frame = absl::make_unique<ImageFrame>(format_, width_, height_,
                                                     /*alignment_boundary=*/1);
    // Use microsecond as the unit of time.
    Timestamp timestamp(cap_->get(cv::CAP_PROP_POS_MSEC) * 1000);
    if (format_ == ImageFormat::GRAY8) {
      cv::Mat frame = formats::MatView(image_frame.get());
      ReadFrame(frame);
      if (frame.empty()) {
        return tool::StatusStop();
      }
    } else {
      cv::Mat tmp_frame;
      ReadFrame(tmp_frame);
      if (tmp_frame.empty()) {
        return tool::StatusStop();
      }
      if (format_ == ImageFormat::SRGB) {
        cv::cvtColor(tmp_frame, formats::MatView(image_frame.get()),
                     cv::COLOR_BGR2RGB);
      } else if (format_ == ImageFormat::SRGBA) {
        cv::cvtColor(tmp_frame, formats::MatView(image_frame.get()),
                     cv::COLOR_BGRA2RGBA);
      }
    }
    // If the timestamp of the current frame is not greater than the one of the
    // previous frame, the new frame will be discarded.
    if (prev_timestamp_ < timestamp) {
      cc->Outputs().Tag(kVideoTag).Add(image_frame.release(), timestamp);
      prev_timestamp_ = timestamp;
      decoded_frames_++;
    }

    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    if (cap_ && cap_->isOpened()) {
      cap_->release();
    }
    if (decoded_frames_ != frame_count_) {
      LOG(WARNING) << "Not all the frames are decoded (total frames: "
                   << frame_count_ << " vs decoded frames: " << decoded_frames_
                   << ").";
    }
    return absl::OkStatus();
  }

  // Sometimes an empty frame is returned even though there are more frames.
  void ReadFrame(cv::Mat& frame) {
    cap_->read(frame);
    if (frame.empty()) {
      cap_->read(frame);  // Try again.
    }
  }

 private:
  std::unique_ptr<cv::VideoCapture> cap_;
  int width_;
  int height_;
  int frame_count_;
  int decoded_frames_ = 0;
  ImageFormat::Format format_;
  Timestamp prev_timestamp_ = Timestamp::Unset();
};

REGISTER_CALCULATOR(OpenCvVideoDecoderCalculator);
}  // namespace mediapipe
