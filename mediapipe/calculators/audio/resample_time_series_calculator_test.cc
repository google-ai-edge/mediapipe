#include "mediapipe/calculators/audio/resample_time_series_calculator.h"

#include <math.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "audio/dsp/resampler_q.h"
#include "mediapipe/calculators/audio/resample_time_series_calculator.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/util/time_series_test_util.h"

namespace mediapipe {
namespace {

const int kInitialTimestampOffsetMilliseconds = 4;

class ResampleTimeSeriesCalculatorTest
    : public TimeSeriesCalculatorTest<ResampleTimeSeriesCalculatorOptions> {
 protected:
  void SetUp() override {
    calculator_name_ = "ResampleTimeSeriesCalculator";
    input_sample_rate_ = 4000.0;
    num_input_channels_ = 3;
  }

  // Expects two vectors whose lengths are almost the same and whose
  // elements are equal (for indices that are present in both).
  //
  // This is useful because the resampler doesn't make precise
  // guarantees about its output size.
  void ExpectVectorMostlyFloatEq(absl::Span<const float> expected,
                                 absl::Span<const float> actual) {
    // Lengths should be close, but don't have to be equal.
    ASSERT_NEAR(expected.size(), actual.size(), 1);
    for (int i = 0; i < std::min(expected.size(), actual.size()); ++i) {
      EXPECT_NEAR(expected[i], actual[i], 5e-7f) << " where i=" << i << ".";
    }
  }

  // Caller takes ownership of the returned value.
  Matrix* NewTestFrame(int num_channels, int num_samples, int timestamp) {
    return new Matrix(Matrix::Random(num_channels, num_samples));
  }

  // Initializes and runs the test graph.
  absl::Status Run(double output_sample_rate) {
    options_.set_target_sample_rate(output_sample_rate);
    InitializeGraph();

    FillInputHeader();
    concatenated_input_samples_.resize(num_input_channels_, 0);
    num_input_samples_ = 0;
    for (int i = 0; i < 5; ++i) {
      int packet_size = (i + 1) * 10;
      int timestamp = kInitialTimestampOffsetMilliseconds +
                      num_input_samples_ / input_sample_rate_ *
                          Timestamp::kTimestampUnitsPerSecond;
      Matrix* data_frame =
          NewTestFrame(num_input_channels_, packet_size, timestamp);

      // Keep a reference copy of the input.
      //
      // conservativeResize() is needed here to preserve the existing
      // data.  Eigen's resize() resizes without preserving data.
      concatenated_input_samples_.conservativeResize(
          num_input_channels_, num_input_samples_ + packet_size);
      concatenated_input_samples_.rightCols(packet_size) = *data_frame;
      num_input_samples_ += packet_size;

      AppendInputPacket(data_frame, timestamp);
    }

    return RunGraph();
  }

  void CheckOutputLength(double output_sample_rate) {
    double factor = output_sample_rate / input_sample_rate_;

    int num_output_samples = 0;
    for (const Packet& packet : output().packets) {
      num_output_samples += packet.Get<Matrix>().cols();
    }

    // The exact number of expected samples may vary based on the implementation
    // of the resampler since the exact value is not an integer.
    const double expected_num_output_samples = num_input_samples_ * factor;
    EXPECT_LE(ceil(expected_num_output_samples), num_output_samples);
    EXPECT_GE(ceil(expected_num_output_samples) + 11, num_output_samples);
  }

  // Checks that output timestamps are consistent with the
  // output_sample_rate and output packet sizes.
  void CheckOutputPacketTimestamps(double output_sample_rate) {
    int num_output_samples = 0;
    for (const Packet& packet : output().packets) {
      const int expected_timestamp = kInitialTimestampOffsetMilliseconds +
                                     num_output_samples / output_sample_rate *
                                         Timestamp::kTimestampUnitsPerSecond;
      EXPECT_NEAR(expected_timestamp, packet.Timestamp().Value(), 1);
      num_output_samples += packet.Get<Matrix>().cols();
    }
  }

  // Checks that output values from the calculator (which resamples
  // packet-by-packet) are consistent with resampling the entire
  // signal at once.
  void CheckOutputValues(
      double output_sample_rate,
      std::unique_ptr<ResampleTimeSeriesCalculatorImpl::ResamplerWrapper>
          verification_resampler = nullptr) {
    if (!verification_resampler) {
      verification_resampler =
          ResampleTimeSeriesCalculatorImpl::TestAccess::ResamplerFromOptions(
              input_sample_rate_, output_sample_rate, num_input_channels_,
              options_);
    }

    Matrix expected_resampled;
    verification_resampler->Resample(concatenated_input_samples_,
                                     &expected_resampled, false);
    Matrix flushed;
    verification_resampler->Resample({}, &flushed, true);
    expected_resampled.conservativeResize(
        num_input_channels_, expected_resampled.cols() + flushed.cols());
    expected_resampled.rightCols(flushed.cols()) = flushed;

    for (int i = 0; i < num_input_channels_; ++i) {
      std::vector<float> expected_resampled_i(expected_resampled.row(i).begin(),
                                              expected_resampled.row(i).end());
      std::vector<float> actual_resampled;
      for (const Packet& packet : output().packets) {
        auto output_frame_row = packet.Get<Matrix>().row(i);
        actual_resampled.insert(actual_resampled.end(),
                                output_frame_row.begin(),
                                output_frame_row.end());
      }

      ExpectVectorMostlyFloatEq(expected_resampled_i, actual_resampled);
    }
  }

  void CheckOutputHeaders(double output_sample_rate) {
    const TimeSeriesHeader& output_header =
        output().header.Get<TimeSeriesHeader>();
    TimeSeriesHeader expected_header;
    expected_header.set_sample_rate(output_sample_rate);
    expected_header.set_num_channels(num_input_channels_);
    EXPECT_THAT(output_header, mediapipe::EqualsProto(expected_header));
  }

  void CheckOutput(double output_sample_rate) {
    CheckOutputLength(output_sample_rate);
    CheckOutputPacketTimestamps(output_sample_rate);
    CheckOutputValues(output_sample_rate);
    CheckOutputHeaders(output_sample_rate);
  }

  void CheckOutputUnchanged() {
    for (int i = 0; i < num_input_channels_; ++i) {
      std::vector<float> expected_resampled_data;
      for (int j = 0; j < num_input_samples_; ++j) {
        expected_resampled_data.push_back(concatenated_input_samples_(i, j));
      }
      std::vector<float> actual_resampled_data;
      for (const Packet& packet : output().packets) {
        Matrix output_frame_row = packet.Get<Matrix>().row(i);
        actual_resampled_data.insert(
            actual_resampled_data.end(), &output_frame_row(0),
            &output_frame_row(0) + output_frame_row.cols());
      }
      ExpectVectorMostlyFloatEq(expected_resampled_data, actual_resampled_data);
    }
  }

  Matrix concatenated_input_samples_;
};

TEST_F(ResampleTimeSeriesCalculatorTest, Upsample) {
  const double kUpsampleRate = input_sample_rate_ * 1.9;
  MP_ASSERT_OK(Run(kUpsampleRate));
  CheckOutput(kUpsampleRate);
}

TEST_F(ResampleTimeSeriesCalculatorTest, Downsample) {
  const double kDownsampleRate = input_sample_rate_ / 1.9;
  MP_ASSERT_OK(Run(kDownsampleRate));
  CheckOutput(kDownsampleRate);
}

TEST_F(ResampleTimeSeriesCalculatorTest, UsesRationalFactorResampler) {
  options_.set_resampler_type(
      ResampleTimeSeriesCalculatorOptions::RESAMPLER_RATIONAL_FACTOR);
  // Pick an upsample rate so the resample ratio is 2.
  const double kUpsampleRate = input_sample_rate_ * 2;
  MP_ASSERT_OK(Run(kUpsampleRate));
  CheckOutput(kUpsampleRate);
}

TEST_F(ResampleTimeSeriesCalculatorTest, PassthroughIfSampleRateUnchanged) {
  const double kUpsampleRate = input_sample_rate_;
  MP_ASSERT_OK(Run(kUpsampleRate));
  CheckOutputUnchanged();
}

TEST_F(ResampleTimeSeriesCalculatorTest, FailsOnBadTargetRate) {
  ASSERT_FALSE(Run(-999.9).ok());  // Invalid output sample rate.
}

TEST_F(ResampleTimeSeriesCalculatorTest, DoesNotDieOnEmptyInput) {
  options_.set_target_sample_rate(input_sample_rate_);
  InitializeGraph();
  FillInputHeader();
  MP_ASSERT_OK(RunGraph());
  EXPECT_TRUE(output().packets.empty());
}

TEST_F(ResampleTimeSeriesCalculatorTest, CustomQResamplerKernel) {
  const float kOutputSampleRate = input_sample_rate_ * 0.7;
  const float kRadiusFactor = 11.0;
  const float kCutoffProportion = 0.85;
  options_.set_resampler_type(
      ResampleTimeSeriesCalculatorOptions::RESAMPLER_RATIONAL_FACTOR);
  auto resampler_options = options_.mutable_resampler_rational_factor_options();
  resampler_options->set_radius_factor(kRadiusFactor);
  resampler_options->set_cutoff_proportion(kCutoffProportion);
  MP_ASSERT_OK(Run(kOutputSampleRate));

  audio_dsp::QResamplerParams params;
  params.filter_radius_factor = kRadiusFactor;
  params.cutoff_proportion = kCutoffProportion;
  CheckOutputValues(
      kOutputSampleRate,
      std::make_unique<ResampleTimeSeriesCalculatorImpl::QResamplerWrapper>(
          input_sample_rate_, kOutputSampleRate, num_input_channels_, params));
}

TEST_F(ResampleTimeSeriesCalculatorTest, CustomLegacyKernel) {
  const float kOutputSampleRate = input_sample_rate_ * 0.7;
  const float kRadiusFactor = 11.0;
  const float kCutoffProportion = 0.85;
  // Convert to equivalent legacy parameters.
  const float kRadius =
      kRadiusFactor *
      std::max<float>(1.0f, input_sample_rate_ / kOutputSampleRate);
  const float kCutoff = 0.5f * kCutoffProportion *
                        std::min<float>(input_sample_rate_, kOutputSampleRate);

  options_.set_resampler_type(
      ResampleTimeSeriesCalculatorOptions::RESAMPLER_RATIONAL_FACTOR);
  auto resampler_options = options_.mutable_resampler_rational_factor_options();
  resampler_options->set_radius(kRadius);
  resampler_options->set_cutoff(kCutoff);
  MP_ASSERT_OK(Run(kOutputSampleRate));

  audio_dsp::QResamplerParams params;
  params.filter_radius_factor = kRadiusFactor;
  params.cutoff_proportion = kCutoffProportion;

  CheckOutputValues(
      kOutputSampleRate,
      std::make_unique<ResampleTimeSeriesCalculatorImpl::QResamplerWrapper>(
          input_sample_rate_, kOutputSampleRate, num_input_channels_, params));
}

}  // anonymous namespace
}  // namespace mediapipe
