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

#ifndef MEDIAPIPE_UTIL_TIME_SERIES_TEST_UTIL_H_
#define MEDIAPIPE_UTIL_TIME_SERIES_TEST_UTIL_H_

#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/time_series_util.h"

namespace mediapipe {

// Base class for testing Calculators that operate on TimeSeries inputs.
// Subclasses that do not need a special options proto should inherit from
// the subclass BasicTimeSeriesCalculatorTestBase.
//
// This class handles calculators that accept one or more input streams
// specified either by indices or by tags and produce one or more output
// streams, again either specified by indices or tags.
// The default is to use one input stream and one output stream, specified by
// index. To use more streams by index, set num_input_streams_ or
// num_output_streams for the number of input or output streams, respectively.
// These have to be set before calling InitializeGraph(). To use one or more
// streams by tag, set input_stream_tags_ or output_stream_tags_ before calling
// InitializeGraph(), for example:
//   input_stream_tags_ = {"MATRIX", "FRAMES"};
//   output_stream_tags_ = {"MATRIX"};
//   InitializeGraph();
// These options are exclusive since mediapipe requires calculators to use
// either indices or tags, but not both.
template <typename OptionsClass>
class TimeSeriesCalculatorTest : public ::testing::Test {
 protected:
  // Sentinal value which can be used to tell methods like
  // PopulateHeader to ignore certain fields.
  static constexpr int kUnset = 0;

  TimeSeriesCalculatorTest()
      : num_side_packets_(0),
        num_input_streams_(1),
        num_output_streams_(1),
        input_packet_rate_(kUnset),
        num_input_samples_(kUnset),
        audio_sample_rate_(kUnset) {}

  // Makes the input stream names used in CalculatorRunner runner_.
  // If tags are used, that is, input_stream_tags_ is not empty, it returns
  // names of the form:
  // <tag[0]>:<base_name>_<LOWERCASE(tag[0])>,
  // <tag[1]>:<base_name>_<LOWERCASE(tag[1])>, etc.
  // For the index format, returns names of the form <base_name>_0,
  // <base_name>_1, etc.
  std::vector<std::string> MakeInputStreamNames(const std::string& base_name) {
    if (!input_stream_tags_.empty()) {
      return MakeNames(base_name, input_stream_tags_);
    } else {
      return MakeNames(base_name, num_input_streams_);
    }
    return std::vector<std::string>();
  }

  // Same as MakeInputStreamNames, but for output streams.
  std::vector<std::string> MakeOutputStreamNames(const std::string& base_name) {
    if (!output_stream_tags_.empty()) {
      return MakeNames(base_name, output_stream_tags_);
    } else {
      return MakeNames(base_name, num_output_streams_);
    }
    return std::vector<std::string>();
  }

  // Makes names used in CalculatorRunner runner_ that use the tag format. Tags
  // must be capitalized. Returns names of the form
  // <tag[0]>:<base_name>_<LOWERCASE(tag[0])>,
  // <tag[1]>:<base_name>_<LOWERCASE(tag[1])>, etc.
  std::vector<std::string> MakeNames(const std::string& base_name,
                                     const std::vector<std::string>& tags) {
    std::vector<std::string> base_names;
    std::vector<std::string> ids;
    for (const std::string& tag : tags) {
      const std::string tagged_base_name = absl::StrCat(tag, ":", base_name);
      base_names.push_back(tagged_base_name);

      std::string id;
      id.reserve(tag.size());
      for (std::size_t i = 0; i < tag.size(); ++i) {
        id += std::tolower(tag[i]);
      }
      ids.push_back(id);
    }
    const std::vector<std::string> names = MakeNames(base_names, ids);
    return names;
  }

  // Makes names used in CalculatorRunner runner_ that use the index format.
  // Total is the number of names to create. Returns names of the form
  // <base_name>_0, <base_name>_1, ..., <base_name>_<total - 1>.
  std::vector<std::string> MakeNames(const std::string& base_name,
                                     const int total) {
    std::vector<std::string> base_names;
    std::vector<std::string> ids;
    for (int i = 0; i < total; ++i) {
      const std::string id = absl::StrCat(i);
      ids.push_back(id);
      base_names.push_back(base_name);
    }
    const std::vector<std::string> names = MakeNames(base_names, ids);
    return names;
  }

  // Makes names used in CalculatorRunner runner_. Returns names of the form
  // <base_names[0]>_<ids[0]>, <base_names[1]>_<ids[1]>, etc.
  std::vector<std::string> MakeNames(const std::vector<std::string>& base_names,
                                     const std::vector<std::string>& ids) {
    CHECK_EQ(base_names.size(), ids.size());
    std::vector<std::string> names;
    for (int i = 0; i < base_names.size(); ++i) {
      const std::string name_template = R"($0_$1)";
      const std::string& base_name = base_names[i];
      const std::string& id = ids[i];
      const std::string name = absl::Substitute(name_template, base_name, id);
      names.push_back(name);
    }
    return names;
  }

  // Makes the CalculatorGraphConfig used to initialize CalculatorRunner
  // runner_. If no options are needed, pass the empty string for options.
  CalculatorGraphConfig::Node MakeNodeConfig(const std::string& calculator_name,
                                             const int num_side_packets,
                                             const CalculatorOptions& options) {
    CalculatorGraphConfig::Node node_config;
    node_config.set_calculator(calculator_name);
    CalculatorOptions* node_config_options = node_config.mutable_options();
    *node_config_options = options;

    const std::string input_stream_base_name = "input_stream";
    const std::vector<std::string> input_stream_names =
        MakeInputStreamNames(input_stream_base_name);
    for (const std::string& input_stream_name : input_stream_names) {
      node_config.add_input_stream(input_stream_name);
    }

    const std::string input_side_packet_base_name = "input_side_packet";
    const std::vector<std::string> input_side_packet_names =
        MakeNames(input_side_packet_base_name, num_side_packets);
    for (const std::string& input_side_packet_name : input_side_packet_names) {
      node_config.add_input_side_packet(input_side_packet_name);
    }

    const std::string output_stream_base_name = "output_stream";
    const std::vector<std::string> output_stream_names =
        MakeOutputStreamNames(output_stream_base_name);
    for (const std::string& output_stream_name : output_stream_names) {
      node_config.add_output_stream(output_stream_name);
    }
    return node_config;
  }

  void InitializeGraph(const CalculatorOptions& options) {
    if (num_external_inputs_ != -1) {
      LOG(WARNING) << "Use num_side_packets_ instead of num_external_inputs_.";
      num_side_packets_ = num_external_inputs_;
    }

    if (!input_stream_tags_.empty()) {
      num_input_streams_ = input_stream_tags_.size();
    }

    if (!output_stream_tags_.empty()) {
      num_output_streams_ = output_stream_tags_.size();
    }

    const CalculatorGraphConfig::Node node_config =
        MakeNodeConfig(calculator_name_, num_side_packets_, options);
    runner_.reset(new CalculatorRunner(node_config));
  }

  void InitializeGraph() {
    CalculatorOptions options;
    FillOptionsExtension(&options);
    InitializeGraph(options);
  }

  // Provide an alternative to InitializeGraph for tests that want the
  // options not to be set.
  void InitializeGraphWithoutOptions() {
    CalculatorOptions options;  // Left empty.
    InitializeGraph(options);
  }

  // This is broken out into a separate function to facilitate the
  // NoOptions specialization defined below.
  void FillOptionsExtension(CalculatorOptions* options) {
    options->MutableExtension(OptionsClass::ext)->MergeFrom(options_);
  }

  void PopulateHeader(TimeSeriesHeader* header) {
    header->set_num_channels(num_input_channels_);
    header->set_sample_rate(input_sample_rate_);
    if (num_input_samples_ != kUnset) {
      header->set_num_samples(num_input_samples_);
    }
    if (input_packet_rate_ != kUnset) {
      header->set_packet_rate(input_packet_rate_);
    }
    if (audio_sample_rate_ != kUnset) {
      header->set_audio_sample_rate(audio_sample_rate_);
    }
  }

  std::unique_ptr<TimeSeriesHeader> CreateInputHeader() {
    std::unique_ptr<TimeSeriesHeader> header(new TimeSeriesHeader);
    PopulateHeader(header.get());
    return header;
  }

  void FillInputHeader(const size_t input_index = 0) {
    runner_->MutableInputs()->Index(input_index).header =
        Adopt(CreateInputHeader().release());
  }

  void FillInputHeader(const std::string& input_tag) {
    runner_->MutableInputs()->Tag(input_tag).header =
        Adopt(CreateInputHeader().release());
  }

  template <typename TimeSeriesHeaderExtensionClass>
  std::unique_ptr<TimeSeriesHeaderExtensionClass>
  CreateInputHeaderWithExtension(
      const TimeSeriesHeaderExtensionClass& extension) {
    auto header = CreateInputHeader();
    time_series_util::SetExtensionInHeader(extension, header.get());
    return header;
  }

  template <typename TimeSeriesHeaderExtensionClass>
  void FillInputHeaderWithExtension(
      const TimeSeriesHeaderExtensionClass& extension,
      const size_t input_index = 0) {
    auto header = CreateInputHeaderWithExtension(extension);
    runner_->MutableInputs()->Index(input_index).header =
        Adopt(header.release());
  }

  template <typename TimeSeriesHeaderExtensionClass>
  void FillInputHeaderWithExtension(
      const TimeSeriesHeaderExtensionClass& extension,
      const std::string& input_tag) {
    auto header = CreateInputHeaderWithExtension(extension);
    runner_->MutableInputs()->Tag(input_tag).header = Adopt(header.release());
  }

  // Takes ownership of payload.
  template <typename T>
  void AppendInputPacket(const T* payload, const Timestamp timestamp,
                         const size_t input_index = 0) {
    runner_->MutableInputs()
        ->Index(input_index)
        .packets.push_back(Adopt(payload).At(timestamp));
  }

  // Overload to allow explicit conversion from int64 to Timestamp
  template <typename T>
  void AppendInputPacket(const T* payload, const int64 timestamp,
                         const size_t input_index = 0) {
    AppendInputPacket(payload, Timestamp(timestamp), input_index);
  }

  template <typename T>
  void AppendInputPacket(const T* payload, const Timestamp timestamp,
                         const std::string& input_tag) {
    runner_->MutableInputs()->Tag(input_tag).packets.push_back(
        Adopt(payload).At(timestamp));
  }

  template <typename T>
  void AppendInputPacket(const T* payload, const int64 timestamp,
                         const std::string& input_tag) {
    AppendInputPacket(payload, Timestamp(timestamp), input_tag);
  }

  absl::Status RunGraph() { return runner_->Run(); }

  bool HasInputHeader(const size_t input_index = 0) const {
    return input(input_index)
        .header.template ValidateAsType<TimeSeriesHeader>()
        .ok();
  }

  bool HasOutputHeader() const {
    return output().header.template ValidateAsType<TimeSeriesHeader>().ok();
  }

  template <typename Proto>
  void ExpectOutputHeaderEquals(const Proto& expected,
                                const size_t output_index = 0) const {
    EXPECT_THAT(output(output_index).header.template Get<TimeSeriesHeader>(),
                mediapipe::EqualsProto(expected));
  }

  void ExpectOutputHeaderEqualsInputHeader(
      const size_t input_index = 0, const size_t output_index = 0) const {
    EXPECT_THAT(
        output(output_index).header.template Get<TimeSeriesHeader>(),
        mediapipe::EqualsProto(
            input(input_index).header.template Get<TimeSeriesHeader>()));
  }

  void ExpectOutputHeaderEqualsInputHeader(
      const std::string& input_tag, const size_t output_index = 0) const {
    EXPECT_THAT(output(output_index).header.template Get<TimeSeriesHeader>(),
                mediapipe::EqualsProto(
                    input(input_tag).header.template Get<TimeSeriesHeader>()));
  }

  void ExpectOutputHeaderEqualsInputHeader(
      const size_t input_index, const std::string& output_tag) const {
    EXPECT_THAT(
        output(output_tag).header.template Get<TimeSeriesHeader>(),
        mediapipe::EqualsProto(
            input(input_index).header.template Get<TimeSeriesHeader>()));
  }

  void ExpectOutputHeaderEqualsInputHeader(
      const std::string& input_tag, const std::string& output_tag) const {
    EXPECT_THAT(output(output_tag).header.template Get<TimeSeriesHeader>(),
                mediapipe::EqualsProto(
                    input(input_tag).header.template Get<TimeSeriesHeader>()));
  }

  void ExpectApproximatelyEqual(const Matrix& expected,
                                const Matrix& actual) const {
    const float kPrecision = 1e-6;
    EXPECT_TRUE(actual.isApprox(expected, kPrecision))
        << "Expected: " << expected << ", but got: " << actual;
  }

  const CalculatorRunner::StreamContents& input(
      const size_t input_index = 0) const {
    return runner_->MutableInputs()->Index(input_index);
  }

  const CalculatorRunner::StreamContents& input(
      const std::string& input_tag) const {
    return runner_->MutableInputs()->Tag(input_tag);
  }

  const CalculatorRunner::StreamContents& output(
      const size_t output_index = 0) const {
    return runner_->Outputs().Index(output_index);
  }

  const CalculatorRunner::StreamContents& output(
      const std::string& output_tag) const {
    return runner_->Outputs().Tag(output_tag);
  }

  // Caller takes ownership of the return value.
  static Matrix* NewRandomMatrix(int num_channels, int num_samples) {
    // TODO: Fix a consistent lack of random seed setting in tests.
    auto matrix = new Matrix;
    matrix->setRandom(num_channels, num_samples);
    return matrix;
  }

  std::string calculator_name_;
  OptionsClass options_;
  int num_side_packets_;
  int num_input_streams_;
  std::vector<std::string> input_stream_tags_;
  int num_output_streams_;
  std::vector<std::string> output_stream_tags_;
  // TODO For backwards compatibility, remove after all clients
  // are updated.
  int num_external_inputs_ = -1;
  int num_input_channels_;
  double input_sample_rate_;
  // If this is non-zero, it will be used to set the packet_rate field
  // of the header proto.
  double input_packet_rate_;
  // If this is non-zero, it will be used to set the num_samples field
  // of the header proto.
  int num_input_samples_;
  // If this is non-zero, it will be used to set the audio_sample_rate field
  // of the header proto.
  double audio_sample_rate_;

  std::unique_ptr<CalculatorRunner> runner_;
};

template <typename OptionsClass>
class MultiStreamTimeSeriesCalculatorTest
    : public TimeSeriesCalculatorTest<OptionsClass> {
 protected:
  void FillInputHeader() {
    std::unique_ptr<MultiStreamTimeSeriesHeader> header(
        new MultiStreamTimeSeriesHeader);
    PopulateHeader(header.get());
    this->runner_->MutableInputs()->Index(0).header = Adopt(header.release());
  }

  template <typename TimeSeriesHeaderExtensionClass>
  void FillInputHeaderWithExtension(
      const TimeSeriesHeaderExtensionClass& extension) {
    std::unique_ptr<MultiStreamTimeSeriesHeader> header(
        new MultiStreamTimeSeriesHeader);
    PopulateHeader(header.get());
    time_series_util::SetExtensionInHeader(
        extension, header->mutable_time_series_header());
    this->runner_->MutableInputs()->Index(0).header = Adopt(header.release());
  }

  // Takes ownership of input_vector.
  void AppendInputPacket(const std::vector<Matrix>* input_vector,
                         const Timestamp timestamp) {
    this->runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input_vector).At(timestamp));
  }

  // Overload to allow explicit conversion from int64 to Timestamp
  void AppendInputPacket(const std::vector<Matrix>* input_vector,
                         const int64 timestamp) {
    AppendInputPacket(input_vector, Timestamp(timestamp));
  }

  template <typename StringOrProto>
  void ExpectOutputHeaderEquals(const StringOrProto& expected) const {
    EXPECT_THAT(
        this->output().header.template Get<MultiStreamTimeSeriesHeader>(),
        mediapipe::EqualsProto(expected));
  }

  void ExpectOutputHeaderEqualsInputHeader() const {
    ExpectOutputHeaderEquals(
        this->input().header.template Get<MultiStreamTimeSeriesHeader>());
  }

  int num_input_streams_;

 private:
  void PopulateHeader(MultiStreamTimeSeriesHeader* header) {
    TimeSeriesCalculatorTest<OptionsClass>::PopulateHeader(
        header->mutable_time_series_header());
    header->set_num_streams(num_input_streams_);
  }
};

struct NoOptions {};

template <>
void TimeSeriesCalculatorTest<NoOptions>::FillOptionsExtension(
    CalculatorOptions* options) {}

// Base class for testing basic time series calculators, which are calculators
// that take no options.
class BasicTimeSeriesCalculatorTestBase
    : public TimeSeriesCalculatorTest<NoOptions> {
 protected:
  TimeSeriesHeader ParseTextFormat(const std::string& text_format) {
    TimeSeriesHeader header =
        ParseTextProtoOrDie<TimeSeriesHeader>(text_format);
    return header;
  }

  void Test(const TimeSeriesHeader& input_header,
            const std::vector<Matrix>& input_packets,
            const TimeSeriesHeader& expected_output_header,
            const std::vector<Matrix>& expected_output_packets) {
    InitializeGraph();
    runner_->MutableInputs()->Index(0).header =
        Adopt(new TimeSeriesHeader(input_header));
    for (int i = 0; i < input_packets.size(); ++i) {
      const Timestamp timestamp(i * Timestamp::kTimestampUnitsPerSecond);
      AppendInputPacket(new Matrix(input_packets[i]), timestamp);
    }

    MP_ASSERT_OK(RunGraph());

    ExpectOutputHeaderEquals(expected_output_header);
    EXPECT_EQ(input().packets.size(), output().packets.size());
    ASSERT_EQ(output().packets.size(), expected_output_packets.size());
    for (int i = 0; i < output().packets.size(); ++i) {
      EXPECT_EQ(input().packets[i].Timestamp(),
                output().packets[i].Timestamp());
      ExpectApproximatelyEqual(expected_output_packets[i],
                               output().packets[i].Get<Matrix>());
    }
  }
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_TIME_SERIES_TEST_UTIL_H_
