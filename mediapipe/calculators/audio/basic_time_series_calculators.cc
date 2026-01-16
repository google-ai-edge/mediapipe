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
// Basic Calculators that operate on TimeSeries streams.
#include "mediapipe/calculators/audio/basic_time_series_calculators.h"

#include <cmath>
#include <memory>

#include "Eigen/Core"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/time_series_util.h"

namespace mediapipe {
namespace {
static bool SafeMultiply(int x, int y, int* result) {
  static_assert(sizeof(int64_t) >= 2 * sizeof(int),
                "Unable to detect overflow after multiplication");
  const int64_t big = static_cast<int64_t>(x) * static_cast<int64_t>(y);
  if (big > static_cast<int64_t>(INT_MIN) &&
      big < static_cast<int64_t>(INT_MAX)) {
    if (result != nullptr) *result = static_cast<int>(big);
    return true;
  } else {
    return false;
  }
}
}  // namespace

absl::Status BasicTimeSeriesCalculatorBase::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<Matrix>(
      // Input stream with TimeSeriesHeader.
  );
  cc->Outputs().Index(0).Set<Matrix>(
      // Output stream with TimeSeriesHeader.
  );
  return absl::OkStatus();
}

absl::Status BasicTimeSeriesCalculatorBase::Open(CalculatorContext* cc) {
  TimeSeriesHeader input_header;
  MP_RETURN_IF_ERROR(time_series_util::FillTimeSeriesHeaderIfValid(
      cc->Inputs().Index(0).Header(), &input_header));

  auto output_header = new TimeSeriesHeader(input_header);
  MP_RETURN_IF_ERROR(MutateHeader(output_header));
  cc->Outputs().Index(0).SetHeader(Adopt(output_header));

  cc->SetOffset(0);

  return absl::OkStatus();
}

absl::Status BasicTimeSeriesCalculatorBase::Process(CalculatorContext* cc) {
  const Matrix& input = cc->Inputs().Index(0).Get<Matrix>();
  MP_RETURN_IF_ERROR(time_series_util::IsMatrixShapeConsistentWithHeader(
      input, cc->Inputs().Index(0).Header().Get<TimeSeriesHeader>()));

  std::unique_ptr<Matrix> output(new Matrix(ProcessMatrix(input)));
  MP_RETURN_IF_ERROR(time_series_util::IsMatrixShapeConsistentWithHeader(
      *output, cc->Outputs().Index(0).Header().Get<TimeSeriesHeader>()));

  cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

absl::Status BasicTimeSeriesCalculatorBase::MutateHeader(
    TimeSeriesHeader* output_header) {
  return absl::OkStatus();
}

// Calculator to sum an input time series across channels.  This is
// useful for e.g. computing 'summary SAI' pitchogram features.
//
// Options proto: None.
class SumTimeSeriesAcrossChannelsCalculator
    : public BasicTimeSeriesCalculatorBase {
 protected:
  absl::Status MutateHeader(TimeSeriesHeader* output_header) final {
    output_header->set_num_channels(1);
    return absl::OkStatus();
  }

  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    return input_matrix.colwise().sum();
  }
};
REGISTER_CALCULATOR(SumTimeSeriesAcrossChannelsCalculator);

// Calculator to average an input time series across channels.  This is
// useful for e.g. converting stereo or multi-channel files to mono.
//
// Options proto: None.
class AverageTimeSeriesAcrossChannelsCalculator
    : public BasicTimeSeriesCalculatorBase {
 protected:
  absl::Status MutateHeader(TimeSeriesHeader* output_header) final {
    output_header->set_num_channels(1);
    return absl::OkStatus();
  }

  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    return input_matrix.colwise().mean();
  }
};
REGISTER_CALCULATOR(AverageTimeSeriesAcrossChannelsCalculator);

// Calculator to convert a (temporal) summary SAI stream (a single-channel
// stream output by SumTimeSeriesAcrossChannelsCalculator) into pitchogram
// frames by transposing the input packets, swapping the time and channel axes.
//
// Options proto: None.
class SummarySaiToPitchogramCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  absl::Status MutateHeader(TimeSeriesHeader* output_header) final {
    if (output_header->num_channels() != 1) {
      return tool::StatusInvalid(
          absl::StrCat("Expected single-channel input, got ",
                       output_header->num_channels()));
    }
    output_header->set_num_channels(output_header->num_samples());
    output_header->set_num_samples(1);
    output_header->set_sample_rate(output_header->packet_rate());
    return absl::OkStatus();
  }

  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    return input_matrix.transpose();
  }
};
REGISTER_CALCULATOR(SummarySaiToPitchogramCalculator);

// Calculator to reverse the order of channels in TimeSeries packets.
// This is useful for e.g. interfacing with the speech pipeline which uses the
// opposite convention to the hearing filterbanks.
//
// Options proto: None.
class ReverseChannelOrderCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    return input_matrix.colwise().reverse();
  }
};
REGISTER_CALCULATOR(ReverseChannelOrderCalculator);

// Calculator to flatten all samples in a TimeSeries packet down into
// a single 'sample' vector.  This is useful for e.g. stacking several
// frames of features into a single feature vector.
//
// Options proto: None.
class FlattenPacketCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  absl::Status MutateHeader(TimeSeriesHeader* output_header) final {
    const int num_input_channels = output_header->num_channels();
    const int num_input_samples = output_header->num_samples();
    RET_CHECK(num_input_channels >= 0)
        << "FlattenPacketCalculator: num_input_channels < 0";
    RET_CHECK(num_input_samples >= 0)
        << "FlattenPacketCalculator: num_input_samples < 0";
    int output_num_channels;
    RET_CHECK(SafeMultiply(num_input_channels, num_input_samples,
                           &output_num_channels))
        << "FlattenPacketCalculator: Multiplication failed.";
    output_header->set_num_channels(output_num_channels);
    output_header->set_num_samples(1);
    output_header->set_sample_rate(output_header->packet_rate());
    return absl::OkStatus();
  }

  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    // Flatten by interleaving channels so that full samples are
    // stacked on top of each other instead of interleaving samples
    // from the same channel.
    Matrix output(input_matrix.size(), 1);
    for (int sample = 0; sample < input_matrix.cols(); ++sample) {
      output.middleRows(sample * input_matrix.rows(), input_matrix.rows()) =
          input_matrix.col(sample);
    }
    return output;
  }
};
REGISTER_CALCULATOR(FlattenPacketCalculator);

// Calculator to subtract the within-packet mean for each channel from each
// corresponding channel.
//
// Options proto: None.
class SubtractMeanCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    Matrix mean = input_matrix.rowwise().mean();
    return input_matrix - mean.replicate(1, input_matrix.cols());
  }
};
REGISTER_CALCULATOR(SubtractMeanCalculator);

// Calculator to subtract the mean over all values (across all times and
// channels) in a Packet from the values in that Packet.
//
// Options proto: None.
class SubtractMeanAcrossChannelsCalculator
    : public BasicTimeSeriesCalculatorBase {
 protected:
  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    auto mean = input_matrix.mean();
    return (input_matrix.array() - mean).matrix();
  }
};
REGISTER_CALCULATOR(SubtractMeanAcrossChannelsCalculator);

// Calculator to divide all values in a Packet by the average value across all
// times and channels in the packet. This is useful for normalizing
// nonnegative quantities like power, but might cause unexpected results if used
// with Packets that can contain negative numbers.
//
// If mean is exactly zero, the output will be a matrix of all ones, because
// that's what happens in other cases where all values are equal.
//
// Options proto: None.
class DivideByMeanAcrossChannelsCalculator
    : public BasicTimeSeriesCalculatorBase {
 protected:
  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    auto mean = input_matrix.mean();

    if (mean != 0) {
      return input_matrix / mean;

      // When used with nonnegative matrices, the mean will only be zero if the
      // entire matrix is exactly zero. If mean is exactly zero, the output will
      // be a matrix of all ones, because that's what happens in other cases
      // where
      // all values are equal.
    } else {
      return Matrix::Ones(input_matrix.rows(), input_matrix.cols());
    }
  }
};
REGISTER_CALCULATOR(DivideByMeanAcrossChannelsCalculator);

// Calculator to calculate the mean for each channel.
//
// Options proto: None.
class MeanCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  absl::Status MutateHeader(TimeSeriesHeader* output_header) final {
    output_header->set_num_samples(1);
    output_header->set_sample_rate(output_header->packet_rate());
    return absl::OkStatus();
  }

  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    return input_matrix.rowwise().mean();
  }
};
REGISTER_CALCULATOR(MeanCalculator);

// Calculator to calculate the uncorrected sample standard deviation in each
// channel, independently for each Packet.  I.e. divide by the number of samples
// in the Packet, not (<number of samples> - 1).
//
// Options proto: None.
class StandardDeviationCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  absl::Status MutateHeader(TimeSeriesHeader* output_header) final {
    output_header->set_num_samples(1);
    output_header->set_sample_rate(output_header->packet_rate());
    return absl::OkStatus();
  }

  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    Eigen::VectorXf mean = input_matrix.rowwise().mean();
    return (input_matrix.colwise() - mean).rowwise().norm() /
           sqrt(input_matrix.cols());
  }
};
REGISTER_CALCULATOR(StandardDeviationCalculator);

// Calculator to calculate the covariance matrix. If the input matrix
// has N channels, the output matrix will be an N by N symmetric
// matrix.
//
// Options proto: None.
class CovarianceCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  absl::Status MutateHeader(TimeSeriesHeader* output_header) final {
    output_header->set_num_samples(output_header->num_channels());
    return absl::OkStatus();
  }

  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    auto mean = input_matrix.rowwise().mean();
    auto zero_mean_input =
        input_matrix - mean.replicate(1, input_matrix.cols());
    return (zero_mean_input * zero_mean_input.transpose()) /
           input_matrix.cols();
  }
};
REGISTER_CALCULATOR(CovarianceCalculator);

// Calculator to get the per column L2 norm of an input time series.
//
// Options proto: None.
class L2NormCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  absl::Status MutateHeader(TimeSeriesHeader* output_header) final {
    output_header->set_num_channels(1);
    return absl::OkStatus();
  }

  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    return input_matrix.colwise().norm();
  }
};
REGISTER_CALCULATOR(L2NormCalculator);

// Calculator to convert each column of a matrix to a unit vector.
//
// Options proto: None.
class L2NormalizeColumnCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    return input_matrix.colwise().normalized();
  }
};
REGISTER_CALCULATOR(L2NormalizeColumnCalculator);

// Calculator to apply L2 normalization to the input matrix.
//
// Returns the matrix as is if the RMS is <= 1E-8.
// Options proto: None.
class L2NormalizeCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    constexpr double kEpsilon = 1e-8;
    double rms = std::sqrt(input_matrix.array().square().mean());
    if (rms <= kEpsilon) {
      return input_matrix;
    }
    return input_matrix / rms;
  }
};
REGISTER_CALCULATOR(L2NormalizeCalculator);

// Calculator to apply Peak normalization to the input matrix.
//
// Returns the matrix as is if the peak is <= 1E-8.
// Options proto: None.
class PeakNormalizeCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    constexpr double kEpsilon = 1e-8;
    double max_pcm = input_matrix.cwiseAbs().maxCoeff();
    if (max_pcm <= kEpsilon) {
      return input_matrix;
    }
    return input_matrix / max_pcm;
  }
};
REGISTER_CALCULATOR(PeakNormalizeCalculator);

// Calculator to compute the elementwise square of an input time series.
//
// Options proto: None.
class ElementwiseSquareCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    return input_matrix.array().square();
  }
};
REGISTER_CALCULATOR(ElementwiseSquareCalculator);

// Calculator that outputs first floor(num_samples / 2) of the samples.
//
// Options proto: None.
class FirstHalfSlicerCalculator : public BasicTimeSeriesCalculatorBase {
 protected:
  absl::Status MutateHeader(TimeSeriesHeader* output_header) final {
    const int num_input_samples = output_header->num_samples();
    RET_CHECK(num_input_samples >= 0)
        << "FirstHalfSlicerCalculator: num_input_samples < 0";
    output_header->set_num_samples(num_input_samples / 2);
    return absl::OkStatus();
  }

  Matrix ProcessMatrix(const Matrix& input_matrix) final {
    return input_matrix.block(0, 0, input_matrix.rows(),
                              input_matrix.cols() / 2);
  }
};
REGISTER_CALCULATOR(FirstHalfSlicerCalculator);

}  // namespace mediapipe
