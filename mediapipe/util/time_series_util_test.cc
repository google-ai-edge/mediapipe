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

#include "mediapipe/util/time_series_util.h"

#include "Eigen/Core"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace time_series_util {
namespace {

TEST(TimeSeriesUtilTest, LogWarningIfTimestampIsInconsistent) {
  // "Special" timestamps aren't considered consistent.
  EXPECT_FALSE(LogWarningIfTimestampIsInconsistent(Timestamp::Unset(),
                                                   Timestamp(0), 0, 1));

  EXPECT_TRUE(LogWarningIfTimestampIsInconsistent(
      Timestamp(2 * Timestamp::kTimestampUnitsPerSecond),
      Timestamp(1 * Timestamp::kTimestampUnitsPerSecond), 10000, 10000.0));
  EXPECT_FALSE(LogWarningIfTimestampIsInconsistent(
      Timestamp(2 * Timestamp::kTimestampUnitsPerSecond),
      Timestamp(1 * Timestamp::kTimestampUnitsPerSecond), 10001, 10000.0));
}

TEST(TimeSeriesUtilTest, FillTimeSeriesHeaderIfValid) {
  {
    Packet empty_packet;
    TimeSeriesHeader header;
    EXPECT_FALSE(FillTimeSeriesHeaderIfValid(empty_packet, &header).ok());
  }

  {
    std::unique_ptr<TimeSeriesHeader> valid_header(new TimeSeriesHeader);
    valid_header->set_sample_rate(1234.5);
    valid_header->set_num_channels(3);
    Packet valid_packet = Adopt(valid_header.release());
    TimeSeriesHeader packet_header;
    MP_EXPECT_OK(FillTimeSeriesHeaderIfValid(valid_packet, &packet_header));
    EXPECT_EQ(packet_header.sample_rate(), 1234.5);
    EXPECT_EQ(packet_header.num_channels(), 3);
  }

  {
    std::unique_ptr<TimeSeriesHeader> missing_num_channels_header(
        new TimeSeriesHeader);
    missing_num_channels_header->set_sample_rate(1234.5);
    Packet packet_with_missing_num_channel =
        Adopt(missing_num_channels_header.release());
    TimeSeriesHeader header;
    EXPECT_FALSE(
        FillTimeSeriesHeaderIfValid(packet_with_missing_num_channel, &header)
            .ok());
  }

  {
    std::unique_ptr<TimeSeriesHeader> missing_sample_rate_header(
        new TimeSeriesHeader);
    missing_sample_rate_header->set_num_channels(3);
    Packet packet_with_missing_sample_rate =
        Adopt(missing_sample_rate_header.release());
    TimeSeriesHeader header;
    EXPECT_FALSE(
        FillTimeSeriesHeaderIfValid(packet_with_missing_sample_rate, &header)
            .ok());
  }
}

TEST(TimeSeriesUtilTest, FillMultiStreamTimeSeriesHeaderIfValid) {
  {
    Packet empty_packet;
    MultiStreamTimeSeriesHeader header;
    EXPECT_FALSE(
        FillMultiStreamTimeSeriesHeaderIfValid(empty_packet, &header).ok());
  }

  {
    Packet packet_with_empty_header = Adopt(new MultiStreamTimeSeriesHeader());
    MultiStreamTimeSeriesHeader header;
    EXPECT_FALSE(FillMultiStreamTimeSeriesHeaderIfValid(
                     packet_with_empty_header, &header)
                     .ok());
  }

  {
    std::unique_ptr<MultiStreamTimeSeriesHeader> valid_header(
        new MultiStreamTimeSeriesHeader);
    valid_header->mutable_time_series_header()->set_sample_rate(1234.5);
    valid_header->mutable_time_series_header()->set_num_channels(3);
    Packet valid_packet = Adopt(valid_header.release());
    MultiStreamTimeSeriesHeader packet_header;
    MP_EXPECT_OK(
        FillMultiStreamTimeSeriesHeaderIfValid(valid_packet, &packet_header));
    EXPECT_EQ(packet_header.time_series_header().sample_rate(), 1234.5);
    EXPECT_EQ(packet_header.time_series_header().num_channels(), 3);
  }

  {
    TimeSeriesHeader missing_num_channels_header;
    std::unique_ptr<MultiStreamTimeSeriesHeader>
        header_with_invalid_time_series_header(new MultiStreamTimeSeriesHeader);
    header_with_invalid_time_series_header->mutable_time_series_header()
        ->set_sample_rate(1234.5);
    Packet packet_with_invalid_time_series_header =
        Adopt(header_with_invalid_time_series_header.release());
    MultiStreamTimeSeriesHeader header;
    EXPECT_FALSE(FillMultiStreamTimeSeriesHeaderIfValid(
                     packet_with_invalid_time_series_header, &header)
                     .ok());
  }
}

TEST(IsMatrixShapeConsistentWithHeaderTest, BasicOperation) {
  TimeSeriesHeader header;
  header.set_num_samples(2);
  header.set_num_channels(3);

  EXPECT_TRUE(
      IsMatrixShapeConsistentWithHeader(Matrix::Zero(3, 2), header).ok());
  EXPECT_FALSE(
      IsMatrixShapeConsistentWithHeader(Matrix::Zero(0, 0), header).ok());
  // Transposed Matrix.
  EXPECT_FALSE(
      IsMatrixShapeConsistentWithHeader(Matrix::Zero(2, 3), header).ok());
}

TEST(IsMatrixShapeConsistentWithHeaderTest,
     EmptyHeaderConsistentWithAnyMatrix) {
  TimeSeriesHeader empty_header;
  EXPECT_TRUE(
      IsMatrixShapeConsistentWithHeader(Matrix::Zero(0, 0), empty_header).ok());
  EXPECT_TRUE(
      IsMatrixShapeConsistentWithHeader(Matrix::Zero(3, 2), empty_header).ok());
}

TEST(IsMatrixShapeConsistentWithHeaderTest,
     NumChannelsUnsetConsistentWithAnyNumRows) {
  TimeSeriesHeader header;
  header.set_num_channels(2);
  for (int num_cols : {1, 2, 5, 9}) {
    EXPECT_TRUE(
        IsMatrixShapeConsistentWithHeader(Matrix::Zero(2, num_cols), header)
            .ok());
  }
}

TEST(IsMatrixShapeConsistentWithHeaderTest,
     NumSamplesUnsetConsistentWithAnyNumColumns) {
  TimeSeriesHeader header;
  header.set_num_samples(2);
  for (int num_rows : {1, 2, 5, 9}) {
    EXPECT_TRUE(
        IsMatrixShapeConsistentWithHeader(Matrix::Zero(num_rows, 2), header)
            .ok());
  }
}

TEST(TimeSeriesUtilTest, SecondsToSamples) {
  // If the time is an integer multiple of the sampling period, we
  // should get an exact result.
  double sample_rate = 10.0;
  double integer_multiple_time = 5;
  EXPECT_EQ(integer_multiple_time * sample_rate,
            SecondsToSamples(integer_multiple_time, sample_rate));

  // Otherwise we should be within one sample.
  double arbitrary_time = 5.01;
  EXPECT_NEAR(arbitrary_time * sample_rate,
              SecondsToSamples(arbitrary_time, sample_rate), 1);
}

TEST(TimeSeriesUtilTest, SamplesToSeconds) {
  double sample_rate = 32.5;
  int64_t num_samples = 128;
  EXPECT_EQ(num_samples / sample_rate,
            SamplesToSeconds(num_samples, sample_rate));
}

}  // namespace
}  // namespace time_series_util
}  // namespace mediapipe
