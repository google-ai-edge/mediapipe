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

#include "mediapipe/util/audio_decoder.h"

#include <algorithm>
#include <cstdint>  // required by avutil.h
#include <cstdlib>
#include <memory>
#include <string>

#include "Eigen/Core"
#include "absl/base/internal/endian.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "mediapipe/framework/deps/cleanup.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libavutil/mem.h"
#include "libavutil/samplefmt.h"
}

ABSL_FLAG(int64_t, media_decoder_allowed_audio_gap_merge, 5,
          "The time gap forwards or backwards in the audio to ignore.  "
          "Timestamps in media files are restricted by the container format "
          "and stream codec and are invariably not accurate to exact sample "
          "numbers.  If the discrepency between time based on counting "
          "samples and based on the container timestamps grows beyond this "
          "value it will be reset to the value in the audio stream and "
          "counting based on samples will resume.");

namespace mediapipe {

// MPEG PTS max value + 1, used to correct for PTS rollover. Unit is PTS ticks.
const int64 kMpegPtsEpoch = 1LL << 33;
// Maximum PTS change between frames. Larger changes are considered to indicate
// the MPEG PTS has rolled over. Unit is PTS ticks.
const int64 kMpegPtsMaxDelta = kMpegPtsEpoch / 2;

// BasePacketProcessor
namespace {

inline std::string TimestampToString(int64 timestamp) {
  if (timestamp == AV_NOPTS_VALUE) {
    return "NOPTS";
  }
  return absl::StrCat(timestamp);
}

float Uint32ToFloat(uint32 raw_value) {
  float value;
  memcpy(&value, &raw_value, 4);
  return value;
}

std::string AvErrorToString(int error) {
  if (error >= 0) {
    return absl::StrCat("Not an error (", error, ")");
  }

  switch (error) {
    case AVERROR(EINVAL):
      return "AVERROR(EINVAL) - unknown error or invalid data";
    case AVERROR(EIO):
      return "AVERROR(EIO) - I/O error";
    case AVERROR(EDOM):
      return "AVERROR(EDOM) - Number syntax expected in filename.";
    case AVERROR(ENOMEM):
      return "AVERROR(ENOMEM) - not enough memory";
    case AVERROR(EILSEQ):
      return "AVERROR(EILSEQ) - unknown format";
    case AVERROR(ENOSYS):
      return "AVERROR(ENOSYS) - Operation not supported.";
    case AVERROR(ENOENT):
      return "AVERROR(ENOENT) - No such file or directory.";
    case AVERROR(EPIPE):
      return "AVERROR(EPIPE) - End of file.";
    case AVERROR_BSF_NOT_FOUND:
      return "AVERROR_BSF_NOT_FOUND - Bitstream filter not found.";
    case AVERROR_BUG:
      return "AVERROR_BUG - Internal bug, should not have happened.";
    case AVERROR_BUG2:
      return "AVERROR_BUG2 - Internal bug, should not have happened.";
    case AVERROR_BUFFER_TOO_SMALL:
      return "AVERROR_BUFFER_TOO_SMALL - Buffer too small.";
    case AVERROR_DECODER_NOT_FOUND:
      return "AVERROR_DECODER_NOT_FOUND - Decoder not found.";
    case AVERROR_DEMUXER_NOT_FOUND:
      return "AVERROR_DEMUXER_NOT_FOUND - Demuxer not found.";
    case AVERROR_ENCODER_NOT_FOUND:
      return "AVERROR_ENCODER_NOT_FOUND - Encoder not found.";
    case AVERROR_EOF:
      return "AVERROR_EOF - End of file.";
    case AVERROR_EXIT:
      return "AVERROR_EXIT - Immediate exit was requested.";
    case AVERROR_EXTERNAL:
      return "AVERROR_EXTERNAL - Generic error in an external library.";
    case AVERROR_FILTER_NOT_FOUND:
      return "AVERROR_FILTER_NOT_FOUND - Filter not found.";
    case AVERROR_INVALIDDATA:
      return "AVERROR_INVALIDDATA - Invalid data found when processing input.";
    case AVERROR_MUXER_NOT_FOUND:
      return "AVERROR_MUXER_NOT_FOUND - Muxer not found.";
    case AVERROR_OPTION_NOT_FOUND:
      return "AVERROR_OPTION_NOT_FOUND - Option not found.";
    case AVERROR_PATCHWELCOME:
      return "AVERROR_PATCHWELCOME - Not yet implemented in FFmpeg, "
             "patches welcome.";
    case AVERROR_PROTOCOL_NOT_FOUND:
      return "AVERROR_PROTOCOL_NOT_FOUND - Protocol not found.";
    case AVERROR_STREAM_NOT_FOUND:
      return "AVERROR_STREAM_NOT_FOUND - Stream not found.";
    case AVERROR_EXPERIMENTAL:
      return "AVERROR_EXPERIMENTAL - Requested feature is flagged "
             "experimental.";
    case AVERROR_INPUT_CHANGED:
      return "AVERROR_INPUT_CHANGED - Input changed between calls.";
    case AVERROR_OUTPUT_CHANGED:
      return "AVERROR_OUTPUT_CHANGED - Output changed between calls.";
    default:
      // FALLTHRU
      {}
  }

  char buf[AV_ERROR_MAX_STRING_SIZE];
  if (av_strerror(error, buf, sizeof(buf)) == 0) {
    return absl::StrCat("AVERROR(", error, ") - ", buf);
  }

  return absl::StrCat("Unknown AVERROR number ", error);
}

// Send a packet to the decoder.
absl::Status SendPacket(const AVPacket& packet, AVCodecContext* avcodec_ctx) {
  const int error = avcodec_send_packet(avcodec_ctx, &packet);
  if (error != 0 && error != AVERROR_EOF) {
    // Not consider AVERROR_EOF as an error because it can happen when more
    // than 1 flush packet is sent.
    return UnknownError(absl::StrCat("Failed to send packet: error=", error,
                                     " (", AvErrorToString(error),
                                     "). Packet size: ", packet.size));
  }
  return absl::OkStatus();
}

// Receive a decoded frame from the decoder.
absl::Status ReceiveFrame(AVCodecContext* avcodec_ctx, AVFrame* frame,
                          bool* received) {
  const int error = avcodec_receive_frame(avcodec_ctx, frame);
  *received = error == 0;
  if (error != 0 && error != AVERROR_EOF && error != AVERROR(EAGAIN)) {
    // Not consider AVERROR_EOF as an error because it can happen after a
    // flush, and AVERROR(EAGAIN) because it happens when there's no (more)
    // frame to be received from this packet.
    return UnknownError(absl::StrCat(" Failed to receive frame: error=", error,
                                     " (", AvErrorToString(error), ")."));
  }
  return absl::OkStatus();
}

absl::Status LogStatus(const absl::Status& status,
                       const AVCodecContext& avcodec_ctx,
                       const AVPacket& packet, bool always_return_ok_status) {
  if (status.ok()) {
    return status;
  }

  VLOG(3) << "Failed to process packet:"
          << " media_type:"
          << (avcodec_ctx.codec_type == AVMEDIA_TYPE_VIDEO ? "video" : "audio")
          << " codec_id:" << avcodec_ctx.codec_id
          << " frame_number:" << avcodec_ctx.frame_number
          << " pts:" << TimestampToString(packet.pts)
          << " dts:" << TimestampToString(packet.dts) << " size:" << packet.size
          << (packet.flags & AV_PKT_FLAG_KEY ? " Key Frame." : "");

  if (always_return_ok_status) {
    LOG(WARNING) << status.message();
    return absl::OkStatus();
  } else {
    return status;
  }
}

class AVPacketDeleter {
 public:
  void operator()(void* x) const {
    AVPacket* packet = static_cast<AVPacket*>(x);
    if (packet) {
      av_free_packet(packet);
      delete packet;
    }
  }
};

}  // namespace

BasePacketProcessor::BasePacketProcessor()
    : decoded_frame_(av_frame_alloc()),
      source_time_base_{0, 0},
      output_time_base_{1, 1000000},
      source_frame_rate_{0, 0} {}

BasePacketProcessor::~BasePacketProcessor() { Close(); }

bool BasePacketProcessor::HasData() { return !buffer_.empty(); }

absl::Status BasePacketProcessor::GetData(Packet* packet) {
  CHECK(packet);
  CHECK(!buffer_.empty());
  *packet = buffer_.front();
  buffer_.pop_front();

  return absl::OkStatus();
}

absl::Status BasePacketProcessor::Flush() {
  int64 last_num_frames_processed;
  do {
    std::unique_ptr<AVPacket, AVPacketDeleter> av_packet(new AVPacket());
    av_init_packet(av_packet.get());
    av_packet->size = 0;
    av_packet->data = nullptr;
    av_packet->stream_index = id_;

    last_num_frames_processed = num_frames_processed_;
    // ProcessPacket increments num_frames_processed_ if it is able to
    // decode a frame.  Not being able to decode a frame while being
    // flushed signals that the codec is completely done.
    MP_RETURN_IF_ERROR(ProcessPacket(av_packet.get()));
  } while (last_num_frames_processed != num_frames_processed_);

  flushed_ = true;
  return absl::OkStatus();
}

void BasePacketProcessor::Close() {
  if (avcodec_ctx_) {
    if (avcodec_ctx_->codec) {
      avcodec_close(avcodec_ctx_);
      av_free(avcodec_ctx_);
    }
    avcodec_ctx_ = nullptr;
  }
  if (avcodec_opts_) {
    av_dict_free(&avcodec_opts_);
  }
  if (decoded_frame_) {
    av_frame_free(&decoded_frame_);
  }
}

absl::Status BasePacketProcessor::Decode(const AVPacket& packet,
                                         bool ignore_decode_failures) {
  MP_RETURN_IF_ERROR(LogStatus(SendPacket(packet, avcodec_ctx_), *avcodec_ctx_,
                               packet, ignore_decode_failures));
  while (true) {
    bool received;
    MP_RETURN_IF_ERROR(
        LogStatus(ReceiveFrame(avcodec_ctx_, decoded_frame_, &received),
                  *avcodec_ctx_, packet, ignore_decode_failures));
    if (received) {
      // Successfully decoded a frame (i.e., received it from the decoder). Now
      // further process it.
      MP_RETURN_IF_ERROR(ProcessDecodedFrame(packet));
    } else {
      break;
    }
  }
  return absl::OkStatus();
}

int64 BasePacketProcessor::CorrectPtsForRollover(int64 media_pts) {
  const int64 rollover_pts_media_bits = kMpegPtsEpoch - 1;
  // Ensure PTS in range 0 ... kMpegPtsEpoch. This avoids errors from post
  // decode PTS corrections that overflow the epoch range (while still yielding
  // the correct result as long as the corrections do not exceed
  // kMpegPtsMaxDelta).
  media_pts &= rollover_pts_media_bits;
  if (rollover_corrected_last_pts_ == AV_NOPTS_VALUE) {
    // First seen PTS.
    rollover_corrected_last_pts_ = media_pts;
  } else {
    int64 prev_media_pts =
        rollover_corrected_last_pts_ & rollover_pts_media_bits;
    int64 pts_step = media_pts - prev_media_pts;
    if (pts_step > kMpegPtsMaxDelta) {
      pts_step = pts_step - kMpegPtsEpoch;
    } else if (pts_step < -kMpegPtsMaxDelta) {
      pts_step = kMpegPtsEpoch + pts_step;
    }
    rollover_corrected_last_pts_ =
        std::max((int64)0, rollover_corrected_last_pts_ + pts_step);
  }
  return rollover_corrected_last_pts_;
}

// AudioPacketProcessor
namespace {

// Converts a PCM_S16LE-encoded input sample to float between -1 and 1.
inline float PcmEncodedSampleToFloat(const char* data) {
  static const float kMultiplier = 1.f / (1 << 15);
  return absl::little_endian::Load16(data) * kMultiplier;
}

// Converts a PCM_S32LE-encoded input sample to float between -1 and 1.
inline float PcmEncodedSampleInt32ToFloat(const char* data) {
  static constexpr float kMultiplier = 1.f / (1u << 31);
  return absl::little_endian::Load32(data) * kMultiplier;
}

}  // namespace

AudioPacketProcessor::AudioPacketProcessor(const AudioStreamOptions& options)
    : sample_time_base_{0, 0}, options_(options) {
  DCHECK(absl::little_endian::IsLittleEndian());
}

absl::Status AudioPacketProcessor::Open(int id, AVStream* stream) {
  id_ = id;
  avcodec_ = avcodec_find_decoder(stream->codecpar->codec_id);
  if (!avcodec_) {
    return absl::InvalidArgumentError("Failed to find codec");
  }
  avcodec_ctx_ = avcodec_alloc_context3(avcodec_);
  avcodec_parameters_to_context(avcodec_ctx_, stream->codecpar);
  if (avcodec_open2(avcodec_ctx_, avcodec_, &avcodec_opts_) < 0) {
    return UnknownError("avcodec_open() failed.");
  }
  CHECK(avcodec_ctx_->codec);

  source_time_base_ = stream->time_base;
  source_frame_rate_ = stream->r_frame_rate;
  last_frame_time_regression_detected_ = false;

  MP_RETURN_IF_ERROR(ValidateSampleFormat());
  bytes_per_sample_ = av_get_bytes_per_sample(avcodec_ctx_->sample_fmt);
  num_channels_ = avcodec_ctx_->channels;
  sample_rate_ = avcodec_ctx_->sample_rate;

  if (num_channels_ <= 0) {
    return UnknownError("num_channels must be strictly positive.");
  }
  if (sample_rate_ <= 0) {
    return UnknownError("sample_rate must be strictly positive.");
  }

  sample_time_base_ = {1, static_cast<int>(sample_rate_)};

  VLOG(0) << absl::Substitute(
      "Opened audio stream (id: $0, channels: $1, sample rate: $2, time base: "
      "$3/$4).",
      id_, num_channels_, sample_rate_, source_time_base_.num,
      source_time_base_.den);

  return absl::OkStatus();
}

absl::Status AudioPacketProcessor::ValidateSampleFormat() {
  switch (avcodec_ctx_->sample_fmt) {
    case AV_SAMPLE_FMT_S16:
    case AV_SAMPLE_FMT_S16P:
    case AV_SAMPLE_FMT_S32:
    case AV_SAMPLE_FMT_FLT:
    case AV_SAMPLE_FMT_FLTP:
      return absl::OkStatus();
    default:
      return mediapipe::UnimplementedErrorBuilder(MEDIAPIPE_LOC)
             << "sample_fmt = " << avcodec_ctx_->sample_fmt;
  }
}

int64 AudioPacketProcessor::SampleNumberToTimestamp(const int64 sample_number) {
  return av_rescale_q(sample_number, sample_time_base_, source_time_base_);
}

int64 AudioPacketProcessor::TimestampToSampleNumber(const int64 timestamp) {
  return av_rescale_q(timestamp, source_time_base_, sample_time_base_);
}

int64 AudioPacketProcessor::TimestampToMicroseconds(const int64 timestamp) {
  return av_rescale_q(timestamp, source_time_base_, {1, 1000000});
}

int64 AudioPacketProcessor::SampleNumberToMicroseconds(
    const int64 sample_number) {
  return av_rescale_q(sample_number, sample_time_base_, {1, 1000000});
}

absl::Status AudioPacketProcessor::ProcessPacket(AVPacket* packet) {
  CHECK(packet);
  if (flushed_) {
    return UnknownError(
        "ProcessPacket was called, but AudioPacketProcessor is already "
        "finished.");
  }
  RET_CHECK_EQ(packet->stream_index, id_);

  decoded_frame_->nb_samples = 0;
  return Decode(*packet, options_.ignore_decode_failures());
}

absl::Status AudioPacketProcessor::ProcessDecodedFrame(const AVPacket& packet) {
  RET_CHECK_EQ(decoded_frame_->channels, num_channels_);
  int buf_size_bytes = av_samples_get_buffer_size(nullptr, num_channels_,
                                                  decoded_frame_->nb_samples,
                                                  avcodec_ctx_->sample_fmt, 1);
  VLOG(3) << "Audio packet " << avcodec_ctx_->frame_number
          << " pts: " << TimestampToString(packet.pts)
          << " frame.pts:" << TimestampToString(decoded_frame_->pts)
          << " pkt_dts:" << TimestampToString(decoded_frame_->pkt_dts)
          << " dts:" << TimestampToString(packet.dts) << " size:" << packet.size
          << " decoded:" << buf_size_bytes;
  uint8* const* data_ptr = decoded_frame_->data;
  if (!data_ptr[0]) {
    return UnknownError("No data in audio frame.");
  }
  if (decoded_frame_->pts != AV_NOPTS_VALUE) {
    int64 pts = MaybeCorrectPtsForRollover(decoded_frame_->pts);
    if (num_frames_processed_ == 0) {
      expected_sample_number_ = TimestampToSampleNumber(pts);
    }

    const int64 expected_us =
        SampleNumberToMicroseconds(expected_sample_number_);
    const int64 actual_us = TimestampToMicroseconds(pts);
    if (absl::Microseconds(std::abs(expected_us - actual_us)) >
        absl::Seconds(
            absl::GetFlag(FLAGS_media_decoder_allowed_audio_gap_merge))) {
      LOG(ERROR) << "The expected time based on how many samples we have seen ("
                 << expected_us
                 << " microseconds) no longer matches the time based "
                    "on what the audio stream is telling us ("
                 << actual_us
                 << " microseconds).  The difference is more than "
                    "--media_decoder_allowed_audio_gap_merge ("
                 << absl::FormatDuration(absl::Seconds(absl::GetFlag(
                        FLAGS_media_decoder_allowed_audio_gap_merge)))
                 << " microseconds).  Resetting the timestamps to track what "
                    "the audio stream is telling us.";
      expected_sample_number_ = TimestampToSampleNumber(pts);
    }
  }

  MP_RETURN_IF_ERROR(AddAudioDataToBuffer(
      Timestamp(av_rescale_q(expected_sample_number_, sample_time_base_,
                             output_time_base_)),
      data_ptr, buf_size_bytes));

  ++num_frames_processed_;
  return absl::OkStatus();
}

absl::Status AudioPacketProcessor::AddAudioDataToBuffer(
    const Timestamp output_timestamp, uint8* const* raw_audio,
    int buf_size_bytes) {
  if (buf_size_bytes == 0) {
    return absl::OkStatus();
  }

  if (buf_size_bytes % (num_channels_ * bytes_per_sample_) != 0) {
    return UnknownError("Buffer is not an integral number of samples.");
  }

  const int64 num_samples = buf_size_bytes / bytes_per_sample_ / num_channels_;
  VLOG(3) << "Adding " << num_samples << " audio samples in " << num_channels_
          << " channels to output.";
  auto current_frame = absl::make_unique<Matrix>(num_channels_, num_samples);

  const char* sample_ptr = nullptr;
  switch (avcodec_ctx_->sample_fmt) {
    case AV_SAMPLE_FMT_S16:
      sample_ptr = reinterpret_cast<const char*>(raw_audio[0]);
      for (int64 sample_index = 0; sample_index < num_samples; ++sample_index) {
        for (int channel = 0; channel < num_channels_; ++channel) {
          (*current_frame)(channel, sample_index) =
              PcmEncodedSampleToFloat(sample_ptr);
          sample_ptr += bytes_per_sample_;
        }
      }
      break;
    case AV_SAMPLE_FMT_S32:
      sample_ptr = reinterpret_cast<const char*>(raw_audio[0]);
      for (int64 sample_index = 0; sample_index < num_samples; ++sample_index) {
        for (int channel = 0; channel < num_channels_; ++channel) {
          (*current_frame)(channel, sample_index) =
              PcmEncodedSampleInt32ToFloat(sample_ptr);
          sample_ptr += bytes_per_sample_;
        }
      }
      break;
    case AV_SAMPLE_FMT_FLT:
      sample_ptr = reinterpret_cast<const char*>(raw_audio[0]);
      for (int64 sample_index = 0; sample_index < num_samples; ++sample_index) {
        for (int channel = 0; channel < num_channels_; ++channel) {
          (*current_frame)(channel, sample_index) =
              Uint32ToFloat(absl::little_endian::Load32(sample_ptr));
          sample_ptr += bytes_per_sample_;
        }
      }
      break;
    case AV_SAMPLE_FMT_S16P:
      for (int channel = 0; channel < num_channels_; ++channel) {
        sample_ptr = reinterpret_cast<const char*>(raw_audio[channel]);
        for (int64 sample_index = 0; sample_index < num_samples;
             ++sample_index) {
          (*current_frame)(channel, sample_index) =
              PcmEncodedSampleToFloat(sample_ptr);
          sample_ptr += bytes_per_sample_;
        }
      }
      break;
    case AV_SAMPLE_FMT_FLTP:
      for (int channel = 0; channel < num_channels_; ++channel) {
        sample_ptr = reinterpret_cast<const char*>(raw_audio[channel]);
        for (int64 sample_index = 0; sample_index < num_samples;
             ++sample_index) {
          (*current_frame)(channel, sample_index) =
              Uint32ToFloat(absl::little_endian::Load32(sample_ptr));
          sample_ptr += bytes_per_sample_;
        }
      }
      break;
    default:
      return mediapipe::UnimplementedErrorBuilder(MEDIAPIPE_LOC)
             << "sample_fmt = " << avcodec_ctx_->sample_fmt;
  }

  if (options_.output_regressing_timestamps() ||
      last_timestamp_ == Timestamp::Unset() ||
      output_timestamp > last_timestamp_) {
    buffer_.push_back(Adopt(current_frame.release()).At(output_timestamp));
    last_timestamp_ = output_timestamp;
    if (last_frame_time_regression_detected_) {
      last_frame_time_regression_detected_ = false;
      LOG(INFO) << "Processor " << this << " resumed audio packet processing.";
    }
  } else if (!last_frame_time_regression_detected_) {
    last_frame_time_regression_detected_ = true;
    LOG(ERROR) << "Processor " << this
               << " is dropping an audio packet because the timestamps "
                  "regressed.  Was "
               << last_timestamp_ << " but got " << output_timestamp;
  }
  expected_sample_number_ += num_samples;

  return absl::OkStatus();
}

absl::Status AudioPacketProcessor::FillHeader(TimeSeriesHeader* header) const {
  CHECK(header);
  header->set_sample_rate(sample_rate_);
  header->set_num_channels(num_channels_);
  return absl::OkStatus();
}

int64 AudioPacketProcessor::MaybeCorrectPtsForRollover(int64 media_pts) {
  return options_.correct_pts_for_rollover() ? CorrectPtsForRollover(media_pts)
                                             : media_pts;
}

// AudioDecoder
AudioDecoder::AudioDecoder() { av_register_all(); }

AudioDecoder::~AudioDecoder() {
  absl::Status status = Close();
  if (!status.ok()) {
    LOG(ERROR) << "Encountered error while closing media file: "
               << status.message();
  }
}

absl::Status AudioDecoder::Initialize(
    const std::string& input_file,
    const mediapipe::AudioDecoderOptions options) {
  if (options.audio_stream().empty()) {
    return absl::InvalidArgumentError(
        "At least one audio_stream must be defined in AudioDecoderOptions");
  }
  std::map<int, int> stream_index_to_audio_options_index;
  int options_index = 0;
  for (const auto& audio_stream : options.audio_stream()) {
    InsertIfNotPresent(&stream_index_to_audio_options_index,
                       audio_stream.stream_index(), options_index);
    ++options_index;
  }

  Cleanup<std::function<void()>> decoder_closer([this]() {
    absl::Status status = Close();
    if (!status.ok()) {
      LOG(ERROR) << "Encountered error while closing media file: "
                 << status.message();
    }
  });

  avformat_ctx_ = avformat_alloc_context();
  if (avformat_open_input(&avformat_ctx_, input_file.c_str(), NULL, NULL) < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not open file: ", input_file));
  }

  if (avformat_find_stream_info(avformat_ctx_, NULL) < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Could not find stream information of file: ", input_file));
  }

  std::map<int, int> audio_options_index_to_stream_id;
  for (int current_audio_index = 0, stream_id = 0;
       stream_id < avformat_ctx_->nb_streams; ++stream_id) {
    AVStream* stream = avformat_ctx_->streams[stream_id];
    AVCodecParameters* dec_param = stream->codecpar;
    switch (dec_param->codec_type) {
      case AVMEDIA_TYPE_AUDIO: {
        const int* options_index_ptr = FindOrNull(
            stream_index_to_audio_options_index, current_audio_index);
        if (options_index_ptr) {
          std::unique_ptr<AudioPacketProcessor> processor =
              absl::make_unique<AudioPacketProcessor>(
                  options.audio_stream(*options_index_ptr));
          if (!ContainsKey(audio_processor_, stream_id)) {
            LOG(INFO) << "Created audio processor " << processor.get()
                      << " for file \"" << input_file << "\"";
          } else {
            LOG(ERROR) << "Stream " << stream_id
                       << " already mapped to audio processor "
                       << audio_processor_[stream_id].get();
          }

          MP_RETURN_IF_ERROR(processor->Open(stream_id, stream));
          audio_processor_.emplace(stream_id, std::move(processor));
          CHECK(InsertIfNotPresent(
              &stream_index_to_stream_id_,
              options.audio_stream(*options_index_ptr).stream_index(),
              stream_id));
          CHECK(InsertIfNotPresent(&stream_id_to_audio_options_index_,
                                   stream_id, *options_index_ptr));
          CHECK(InsertIfNotPresent(&audio_options_index_to_stream_id,
                                   *options_index_ptr, stream_id));
        }
        ++current_audio_index;
        break;
      }
      default: {
        // Ignore other stream types.
      }
    }
  }
  for (int i = 0; i < options.audio_stream_size(); ++i) {
    RET_CHECK(ContainsKey(audio_options_index_to_stream_id, i) ||
              options.audio_stream(i).allow_missing())
        << absl::StrCat("Could not find audio stream with index ", i,
                        " in file ", input_file);
  }

  if (options.has_start_time()) {
    start_time_ = Timestamp::FromSeconds(options.start_time());
  }
  if (options.has_end_time()) {
    end_time_ = Timestamp::FromSeconds(options.end_time());
  }
  is_first_packet_.resize(avformat_ctx_->nb_streams, true);

  decoder_closer.release();
  return absl::OkStatus();
}

absl::Status AudioDecoder::GetData(int* options_index, Packet* data) {
  while (true) {
    for (auto& item : audio_processor_) {
      while (item.second && item.second->HasData()) {
        bool is_first_packet = is_first_packet_[item.first];
        is_first_packet_[item.first] = false;
        *options_index =
            FindOrDie(stream_id_to_audio_options_index_, item.first);
        absl::Status status = item.second->GetData(data);
        // Ignore packets which are out of the requested timestamp range.
        if (start_time_ != Timestamp::Unset()) {
          if (is_first_packet && data->Timestamp() > start_time_) {
            LOG(ERROR) << "First packet in audio stream " << *options_index
                       << " has timestamp " << data->Timestamp()
                       << " which is after start time of " << start_time_
                       << ".";
          }
          if (data->Timestamp() < start_time_) {
            VLOG(1) << "Skipping audio frame with timestamp "
                    << data->Timestamp() << " before start time "
                    << start_time_;
            *data = Packet();
            continue;
          }
        }
        if (end_time_ != Timestamp::Unset() && data->Timestamp() > end_time_) {
          VLOG(1) << "Skipping audio frame with timestamp " << data->Timestamp()
                  << " after end time " << end_time_;
          // We are past the last timestamp we care about, close the
          // packet processor.  We cannot remove the element from
          // audio_processor_ right now, because we need to continue
          // iterating through it.
          item.second->Close();
          item.second.reset(nullptr);
          *data = Packet();
          continue;
        }
        return status;
      }
    }
    if (flushed_) {
      MP_RETURN_IF_ERROR(Close());
      return tool::StatusStop();
    }
    MP_RETURN_IF_ERROR(ProcessPacket());
  }
  return absl::OkStatus();
}

absl::Status AudioDecoder::Close() {
  for (auto& item : audio_processor_) {
    if (item.second) {
      item.second->Close();
      item.second.reset(nullptr);
    }
  }
  // Free the context.
  if (avformat_ctx_) {
    avformat_close_input(&avformat_ctx_);
  }
  return absl::OkStatus();
}

absl::Status AudioDecoder::FillAudioHeader(
    const AudioStreamOptions& stream_option, TimeSeriesHeader* header) const {
  const std::unique_ptr<AudioPacketProcessor>* processor_ptr_ = FindOrNull(
      audio_processor_,
      FindOrDie(stream_index_to_stream_id_, stream_option.stream_index()));

  RET_CHECK(processor_ptr_ && *processor_ptr_) << "audio stream is not open.";
  MP_RETURN_IF_ERROR((*processor_ptr_)->FillHeader(header));
  return absl::OkStatus();
}

absl::Status AudioDecoder::ProcessPacket() {
  std::unique_ptr<AVPacket, AVPacketDeleter> av_packet(new AVPacket());
  av_init_packet(av_packet.get());
  av_packet->size = 0;
  av_packet->data = nullptr;
  int ret = av_read_frame(avformat_ctx_, av_packet.get());
  if (ret >= 0) {
    CHECK(av_packet->data) << "AVPacket does not include any data but "
                              "av_read_frame was successful.";
    const int stream_id = av_packet->stream_index;
    auto audio_iterator = audio_processor_.find(stream_id);
    if (audio_iterator != audio_processor_.end()) {
      // This stream_id is belongs to an audio stream we care about.
      if (audio_iterator->second) {
        MP_RETURN_IF_ERROR(
            audio_iterator->second->ProcessPacket(av_packet.get()));
      } else {
        VLOG(3) << "processor for stream " << stream_id << " is nullptr.";
      }
    } else {
      VLOG(3) << "Ignoring packet for stream " << stream_id;
    }
    return absl::OkStatus();
  }
  VLOG(1) << "Demuxing returned error (or EOF): " << AvErrorToString(ret);
  if (ret == AVERROR(EAGAIN)) {
    // EAGAIN is used to signify that the av_packet should be skipped
    // (maybe the demuxer is trying to re-sync).  This definitely
    // occurs in the FLV and MpegT demuxers.
    return absl::OkStatus();
  }

  // Unrecoverable demuxing error with details in avformat_ctx_->pb->error.
  int demuxing_error =
      avformat_ctx_->pb ? avformat_ctx_->pb->error : 0 /* no error */;
  if (ret == AVERROR_EOF && !demuxing_error) {
    VLOG(1) << "Reached EOF.";
    return Flush();
  }

  RET_CHECK(!demuxing_error) << absl::Substitute(
      "Failed to read a frame: retval = $0 ($1), avformat_ctx_->pb->error = "
      "$2 ($3)",
      ret, AvErrorToString(ret), demuxing_error,
      AvErrorToString(demuxing_error));

  if (is_first_packet_[av_packet->stream_index]) {
    RET_CHECK_FAIL() << "Couldn't even read the first frame; maybe a partial "
                        "file with only metadata?";
  }

  // Unrecoverable demuxing error without details.
  RET_CHECK_FAIL() << absl::Substitute(
      "Failed to read a frame: retval = $0 ($1)", ret, AvErrorToString(ret));
}

absl::Status AudioDecoder::Flush() {
  std::vector<absl::Status> statuses;
  for (auto& item : audio_processor_) {
    if (item.second) {
      statuses.push_back(item.second->Flush());
    }
  }
  flushed_ = true;
  return tool::CombinedStatus("Error while flushing codecs: ", statuses);
}

}  // namespace mediapipe
