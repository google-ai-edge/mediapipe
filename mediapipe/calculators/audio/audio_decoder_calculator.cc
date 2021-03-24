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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/audio_decoder.h"
#include "mediapipe/util/audio_decoder.pb.h"

namespace mediapipe {

// The AudioDecoderCalculator decodes an audio stream of the media file. It
// produces two output streams contain audio packets and the header infomation.
//
// Output Streams:
//   AUDIO: Output audio frames (Matrix).
//   AUDIO_HEADER:
//       Optional audio header information output
// Input Side Packets:
//   INPUT_FILE_PATH: The input file path.
//
// Example config:
// node {
//   calculator: "AudioDecoderCalculator"
//   input_side_packet: "INPUT_FILE_PATH:input_file_path"
//   output_stream: "AUDIO:audio"
//   output_stream: "AUDIO_HEADER:audio_header"
//   node_options {
//     [type.googleapis.com/mediapipe.AudioDecoderOptions]: {
//        audio_stream { stream_index: 0 }
//        start_time: 0
//        end_time: 1
//   }
// }
//
// TODO: support decoding multiple streams.
class AudioDecoderCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  std::unique_ptr<AudioDecoder> decoder_;
};

absl::Status AudioDecoderCalculator::GetContract(CalculatorContract* cc) {
  cc->InputSidePackets().Tag("INPUT_FILE_PATH").Set<std::string>();
  if (cc->InputSidePackets().HasTag("OPTIONS")) {
    cc->InputSidePackets().Tag("OPTIONS").Set<mediapipe::AudioDecoderOptions>();
  }
  cc->Outputs().Tag("AUDIO").Set<Matrix>();
  if (cc->Outputs().HasTag("AUDIO_HEADER")) {
    cc->Outputs().Tag("AUDIO_HEADER").SetNone();
  }
  return absl::OkStatus();
}

absl::Status AudioDecoderCalculator::Open(CalculatorContext* cc) {
  const std::string& input_file_path =
      cc->InputSidePackets().Tag("INPUT_FILE_PATH").Get<std::string>();
  const auto& decoder_options =
      tool::RetrieveOptions(cc->Options<mediapipe::AudioDecoderOptions>(),
                            cc->InputSidePackets(), "OPTIONS");
  decoder_ = absl::make_unique<AudioDecoder>();
  MP_RETURN_IF_ERROR(decoder_->Initialize(input_file_path, decoder_options));
  std::unique_ptr<mediapipe::TimeSeriesHeader> header =
      absl::make_unique<mediapipe::TimeSeriesHeader>();
  if (decoder_->FillAudioHeader(decoder_options.audio_stream(0), header.get())
          .ok()) {
    // Only pass on a header if the decoder could actually produce one.
    // otherwise, the header will be empty.
    cc->Outputs().Tag("AUDIO_HEADER").SetHeader(Adopt(header.release()));
  }
  cc->Outputs().Tag("AUDIO_HEADER").Close();
  return absl::OkStatus();
}

absl::Status AudioDecoderCalculator::Process(CalculatorContext* cc) {
  Packet data;
  int options_index = -1;
  auto status = decoder_->GetData(&options_index, &data);
  if (status.ok()) {
    cc->Outputs().Tag("AUDIO").AddPacket(data);
  }
  return status;
}

absl::Status AudioDecoderCalculator::Close(CalculatorContext* cc) {
  return decoder_->Close();
}

REGISTER_CALCULATOR(AudioDecoderCalculator);

}  // namespace mediapipe
