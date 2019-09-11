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
//
// MediaPipe Calculator wrapper around audio/dsp/mfcc/
// classes MelFilterbank (magnitude spectrograms warped to the Mel
// approximation of the auditory frequency scale) and Mfcc (Mel Frequency
// Cepstral Coefficients, the decorrelated transform of log-Mel-spectrum
// commonly used as acoustic features in speech and other audio tasks.
// Both calculators expect as input the SQUARED_MAGNITUDE-domain outputs
// from the MediaPipe SpectrogramCalculator object.
#include <memory>
#include <vector>

#include "Eigen/Core"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "audio/dsp/mfcc/mel_filterbank.h"
#include "audio/dsp/mfcc/mfcc.h"
#include "mediapipe/calculators/audio/mfcc_mel_calculators.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/time_series_util.h"

namespace mediapipe {

namespace {

// Portable version of TimeSeriesHeader's DebugString.
std::string PortableDebugString(const TimeSeriesHeader& header) {
  std::string unsubstituted_header_debug_str = R"(
    sample_rate: $0
    num_channels: $1
    num_samples: $2
    packet_rate: $3
    audio_sample_rate: $4
  )";
  return absl::Substitute(unsubstituted_header_debug_str, header.sample_rate(),
                          header.num_channels(), header.num_samples(),
                          header.packet_rate(), header.audio_sample_rate());
}

}  // namespace

// Abstract base class for Calculators that transform feature vectors on a
// frame-by-frame basis.
// Subclasses must override pure virtual methods ConfigureTransform and
// TransformFrame.
// Input and output MediaPipe packets are matrices with one column per frame,
// and one row per feature dimension.  Each input packet results in an
// output packet with the same number of columns (but differing numbers of
// rows corresponding to the new feature space).
class FramewiseTransformCalculatorBase : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<Matrix>(
        // Sequence of Matrices, each column describing a particular time frame,
        // each row a feature dimension, with TimeSeriesHeader.
    );
    cc->Outputs().Index(0).Set<Matrix>(
        // Sequence of Matrices, each column describing a particular time frame,
        // each row a feature dimension, with TimeSeriesHeader.
    );
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

  int num_output_channels(void) { return num_output_channels_; }

  void set_num_output_channels(int num_output_channels) {
    num_output_channels_ = num_output_channels;
  }

 private:
  // Takes header and options, and sets up state including calling
  // set_num_output_channels() on the base object.
  virtual ::mediapipe::Status ConfigureTransform(const TimeSeriesHeader& header,
                                                 CalculatorContext* cc) = 0;

  // Takes a vector<double> corresponding to an input frame, and
  // perform the specific transformation to produce an output frame.
  virtual void TransformFrame(const std::vector<double>& input,
                              std::vector<double>* output) const = 0;

 private:
  int num_output_channels_;
};

::mediapipe::Status FramewiseTransformCalculatorBase::Open(
    CalculatorContext* cc) {
  TimeSeriesHeader input_header;
  MP_RETURN_IF_ERROR(time_series_util::FillTimeSeriesHeaderIfValid(
      cc->Inputs().Index(0).Header(), &input_header));

  ::mediapipe::Status status = ConfigureTransform(input_header, cc);

  auto output_header = new TimeSeriesHeader(input_header);
  output_header->set_num_channels(num_output_channels_);
  cc->Outputs().Index(0).SetHeader(Adopt(output_header));

  return status;
}

::mediapipe::Status FramewiseTransformCalculatorBase::Process(
    CalculatorContext* cc) {
  const Matrix& input = cc->Inputs().Index(0).Get<Matrix>();
  const int num_frames = input.cols();
  std::unique_ptr<Matrix> output(new Matrix(num_output_channels_, num_frames));
  // The main work here is converting each column of the float Matrix
  // into a vector of doubles, which is what our target functions from
  // dsp_core consume, and doing the reverse with their output.
  std::vector<double> input_frame(input.rows());
  std::vector<double> output_frame(num_output_channels_);

  for (int frame = 0; frame < num_frames; ++frame) {
    // Copy input from Eigen::Matrix column to vector<float>.
    Eigen::Map<Eigen::MatrixXd> input_frame_map(&input_frame[0],
                                                input_frame.size(), 1);
    input_frame_map = input.col(frame).cast<double>();

    // Perform the actual transformation.
    TransformFrame(input_frame, &output_frame);

    // Copy output from vector<float> to Eigen::Vector.
    CHECK_EQ(output_frame.size(), num_output_channels_);
    Eigen::Map<const Eigen::MatrixXd> output_frame_map(&output_frame[0],
                                                       output_frame.size(), 1);
    output->col(frame) = output_frame_map.cast<float>();
  }
  cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}

// Calculator wrapper around the dsp/mfcc/mfcc.cc routine.
// Take frames of squared-magnitude spectra from the SpectrogramCalculator
// and convert them into Mel Frequency Cepstral Coefficients.
//
// Example config:
// node {
//   calculator: "MfccCalculator"
//   input_stream: "spectrogram_frames_stream"
//   output_stream: "mfcc_frames_stream"
//   options {
//     [mediapipe.MfccCalculatorOptions.ext] {
//       mel_spectrum_params {
//         channel_count: 20
//         min_frequency_hertz: 125.0
//         max_frequency_hertz: 3800.0
//       }
//       mfcc_count: 13
//     }
//   }
// }
class MfccCalculator : public FramewiseTransformCalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    return FramewiseTransformCalculatorBase::GetContract(cc);
  }

 private:
  ::mediapipe::Status ConfigureTransform(const TimeSeriesHeader& header,
                                         CalculatorContext* cc) override {
    MfccCalculatorOptions mfcc_options = cc->Options<MfccCalculatorOptions>();
    mfcc_.reset(new audio_dsp::Mfcc());
    int input_length = header.num_channels();
    // Set up the parameters to the Mfcc object.
    set_num_output_channels(mfcc_options.mfcc_count());
    mfcc_->set_dct_coefficient_count(num_output_channels());
    mfcc_->set_upper_frequency_limit(
        mfcc_options.mel_spectrum_params().max_frequency_hertz());
    mfcc_->set_lower_frequency_limit(
        mfcc_options.mel_spectrum_params().min_frequency_hertz());
    mfcc_->set_filterbank_channel_count(
        mfcc_options.mel_spectrum_params().channel_count());
    // An upstream calculator (such as SpectrogramCalculator) must store
    // the sample rate of its input audio waveform in the TimeSeries Header.
    // audio_dsp::MelFilterBank needs to know this to
    // correctly interpret the spectrogram bins.
    if (!header.has_audio_sample_rate()) {
      return ::mediapipe::InvalidArgumentError(
          absl::StrCat("No audio_sample_rate in input TimeSeriesHeader ",
                       PortableDebugString(header)));
    }
    // Now we can initialize the Mfcc object.
    bool initialized =
        mfcc_->Initialize(input_length, header.audio_sample_rate());

    if (initialized) {
      return ::mediapipe::OkStatus();
    } else {
      return ::mediapipe::Status(mediapipe::StatusCode::kInternal,
                                 "Mfcc::Initialize returned uninitialized");
    }
  }

  void TransformFrame(const std::vector<double>& input,
                      std::vector<double>* output) const override {
    mfcc_->Compute(input, output);
  }

 private:
  std::unique_ptr<audio_dsp::Mfcc> mfcc_;
};
REGISTER_CALCULATOR(MfccCalculator);

// Calculator wrapper around the dsp/mfcc/mel_filterbank.cc routine.
// Take frames of squared-magnitude spectra from the SpectrogramCalculator
// and convert them into Mel-warped (linear-magnitude) spectra.
// Note: This code computes a mel-frequency filterbank, using a simple
// algorithm that gives bad results (some mel channels that are always zero)
// if you ask for too many channels.
class MelSpectrumCalculator : public FramewiseTransformCalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    return FramewiseTransformCalculatorBase::GetContract(cc);
  }

 private:
  ::mediapipe::Status ConfigureTransform(const TimeSeriesHeader& header,
                                         CalculatorContext* cc) override {
    MelSpectrumCalculatorOptions mel_spectrum_options =
        cc->Options<MelSpectrumCalculatorOptions>();
    mel_filterbank_.reset(new audio_dsp::MelFilterbank());
    int input_length = header.num_channels();
    set_num_output_channels(mel_spectrum_options.channel_count());
    // An upstream calculator (such as SpectrogramCalculator) must store
    // the sample rate of its input audio waveform in the TimeSeries Header.
    // audio_dsp::MelFilterBank needs to know this to
    // correctly interpret the spectrogram bins.
    if (!header.has_audio_sample_rate()) {
      return ::mediapipe::InvalidArgumentError(
          absl::StrCat("No audio_sample_rate in input TimeSeriesHeader ",
                       PortableDebugString(header)));
    }
    bool initialized = mel_filterbank_->Initialize(
        input_length, header.audio_sample_rate(), num_output_channels(),
        mel_spectrum_options.min_frequency_hertz(),
        mel_spectrum_options.max_frequency_hertz());

    if (initialized) {
      return ::mediapipe::OkStatus();
    } else {
      return ::mediapipe::Status(mediapipe::StatusCode::kInternal,
                                 "mfcc::Initialize returned uninitialized");
    }
  }

  void TransformFrame(const std::vector<double>& input,
                      std::vector<double>* output) const override {
    mel_filterbank_->Compute(input, output);
  }

 private:
  std::unique_ptr<audio_dsp::MelFilterbank> mel_filterbank_;
};
REGISTER_CALCULATOR(MelSpectrumCalculator);

}  // namespace mediapipe
