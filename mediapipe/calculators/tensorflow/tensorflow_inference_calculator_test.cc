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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "mediapipe/calculators/tensorflow/tensorflow_inference_calculator.pb.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session_from_frozen_graph_generator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT
#include "mediapipe/framework/tool/validate_type.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "testing/base/public/gunit.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif  // defined(__APPLE__)

namespace mediapipe {

using ::testing::AllOf;
using ::testing::HasSubstr;

namespace tf = ::tensorflow;

namespace {

constexpr char kMultipliedTag[] = "MULTIPLIED";
constexpr char kBTag[] = "B";
constexpr char kSessionTag[] = "SESSION";

std::string GetGraphDefPath() {
#ifdef __APPLE__
  char path[1024];
  CFURLRef bundle_url = CFBundleCopyBundleURL(CFBundleGetMainBundle());
  CFURLGetFileSystemRepresentation(
      bundle_url, true, reinterpret_cast<UInt8*>(path), sizeof(path));
  CFRelease(bundle_url);
  return mediapipe::file::JoinPath(path, "testdata/frozen_graph_def.pb");
#elif defined(__ANDROID__)
  char path[1024];
  getcwd(path, sizeof(path));
  return mediapipe::file::JoinPath(path,
                                   "mediapipe/calculators/tensorflow/"
                                   "testdata/frozen_graph_def.pb");
#else
  return mediapipe::file::JoinPath(
      ::testing::SrcDir(),
      // This should match the path of the output files
      // of the genrule() that generates test model files.
      "mediapipe/calculators/tensorflow/testdata/", "frozen_graph_def.pb");
#endif  // defined(__APPLE__)
}
}  // namespace

class TensorflowInferenceCalculatorTest : public ::testing::Test {
 protected:
  // Add the input side packet.
  void AddSessionInputSidePacket() {
    PacketGeneratorOptions extendable_options;
    TensorFlowSessionFromFrozenGraphGeneratorOptions* generator_options;
    generator_options = extendable_options.MutableExtension(
        TensorFlowSessionFromFrozenGraphGeneratorOptions::ext);
    generator_options->set_graph_proto_path(GetGraphDefPath());
    (*generator_options->mutable_tag_to_tensor_names())["MULTIPLIED"] =
        "multiplied:0";
    (*generator_options->mutable_tag_to_tensor_names())["A"] = "a:0";
    (*generator_options->mutable_tag_to_tensor_names())["B"] = "b:0";
    (*generator_options->mutable_tag_to_tensor_names())["EXPENSIVE"] =
        "expensive:0";

    PacketSet input_side_packets({});
    PacketSet output_side_packets({"SESSION"});
    MEDIAPIPE_CHECK_OK(tool::RunGenerateAndValidateTypes(
        "TensorFlowSessionFromFrozenGraphGenerator", extendable_options,
        input_side_packets, &output_side_packets));
    runner_->MutableSidePackets()->Tag(kSessionTag) =
        output_side_packets.Tag(kSessionTag);
  }

  Packet CreateTensorPacket(const std::vector<int32_t>& input, int64_t time) {
    tf::TensorShape tensor_shape;
    tensor_shape.AddDim(input.size());
    auto tensor = absl::make_unique<tf::Tensor>(tf::DT_INT32, tensor_shape);
    for (int i = 0; i < input.size(); ++i) {
      tensor->vec<int32_t>()(i) = input[i];
    }
    return Adopt(tensor.release()).At(Timestamp(time));
  }

  // Create tensor from Vector and add as a Packet to the provided tag as input.
  void AddVectorToInputsAsTensor(const std::vector<int32_t>& input,
                                 const std::string& tag, int64_t time) {
    runner_->MutableInputs()->Tag(tag).packets.push_back(
        CreateTensorPacket(input, time));
  }

  // Create tensor from Vector and add as a Packet to the provided tag as input.
  void AddVectorToInputsAsPacket(const std::vector<Packet>& packets,
                                 const std::string& tag) {
    ABSL_CHECK(!packets.empty())
        << "Please specify at least some data in the packet";
    auto packets_ptr = absl::make_unique<std::vector<Packet>>(packets);
    runner_->MutableInputs()->Tag(tag).packets.push_back(
        Adopt(packets_ptr.release()).At(packets.begin()->Timestamp()));
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(TensorflowInferenceCalculatorTest, GetConstants) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_in");
  config.add_output_stream("B:tensor_out");
  config.add_output_stream("MULTIPLIED:tensor_multiplied");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(1);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(false);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({0, 0, 0}, "A", 0);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_b =
      runner_->Outputs().Tag(kBTag).packets;
  ASSERT_EQ(output_packets_b.size(), 1);
  const tf::Tensor& tensor_b = output_packets_b[0].Get<tf::Tensor>();
  tf::TensorShape expected_shape({1, 3});
  auto expected_tensor = tf::test::AsTensor<int32_t>({3, 2, 1}, expected_shape);
  tf::test::ExpectTensorEqual<int32_t>(expected_tensor, tensor_b);

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(1, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  expected_tensor = tf::test::AsTensor<int32_t>({0, 0, 0}, expected_shape);
  tf::test::ExpectTensorEqual<int32_t>(expected_tensor, tensor_mult);

  EXPECT_EQ(1, runner_
                   ->GetCounter(
                       "TensorFlowInferenceCalculator-TotalProcessedTimestamps")
                   ->Get());
}

TEST_F(TensorflowInferenceCalculatorTest, GetComputed) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(1);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(false);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({2, 2, 2}, "A", 0);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 0);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(1, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  tf::TensorShape expected_shape({3});
  auto expected_tensor =
      tf::test::AsTensor<int32_t>({6, 8, 10}, expected_shape);
  tf::test::ExpectTensorEqual<int32_t>(expected_tensor, tensor_mult);

  // Add only one of the two expected tensors at the next timestamp, expect
  // useful failure message.
  AddVectorToInputsAsTensor({1, 2, 3}, "A", 1);
  auto run_status = runner_->Run();
  ASSERT_FALSE(run_status.ok());
  EXPECT_THAT(run_status.ToString(),
              HasSubstr("TensorFlowInferenceCalculator"));
  EXPECT_THAT(run_status.ToString(), HasSubstr("Tag B"));
}

TEST_F(TensorflowInferenceCalculatorTest, GetComputed_MaxInFlight) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  config.set_max_in_flight(2);
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(1);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(false);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({2, 2, 2}, "A", 0);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 0);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(1, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  tf::TensorShape expected_shape({3});
  auto expected_tensor =
      tf::test::AsTensor<int32_t>({6, 8, 10}, expected_shape);
  tf::test::ExpectTensorEqual<int32_t>(expected_tensor, tensor_mult);

  // Add only one of the two expected tensors at the next timestamp, expect
  // useful failure message.
  AddVectorToInputsAsTensor({1, 2, 3}, "A", 1);
  auto run_status = runner_->Run();
  ASSERT_FALSE(run_status.ok());
  EXPECT_THAT(run_status.ToString(),
              HasSubstr("TensorFlowInferenceCalculator"));
  EXPECT_THAT(run_status.ToString(), HasSubstr("Tag B"));
}

TEST_F(TensorflowInferenceCalculatorTest, BadTag) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("BAD:tensor_in");  // This one is bad.
  config.add_output_stream("B:tensor_out");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(1);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  absl::Status status = runner_->Run();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      AllOf(HasSubstr("Can't find tag 'BAD' in signature"),
            HasSubstr("instead found tags A, B, EXPENSIVE, MULTIPLIED")));
}

TEST_F(TensorflowInferenceCalculatorTest, GetMultiBatchComputed) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(1);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({2, 2, 2}, "A", 0);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 0);
  AddVectorToInputsAsTensor({3, 3, 3}, "A", 1);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 1);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(2, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  auto expected_tensor = tf::test::AsTensor<int32_t>({6, 8, 10});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult, expected_tensor);
  const tf::Tensor& tensor_mult1 = output_packets_mult[1].Get<tf::Tensor>();
  auto expected_tensor1 = tf::test::AsTensor<int32_t>({9, 12, 15});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult1, expected_tensor1);

  EXPECT_EQ(2, runner_
                   ->GetCounter(
                       "TensorFlowInferenceCalculator-TotalProcessedTimestamps")
                   ->Get());
}

TEST_F(TensorflowInferenceCalculatorTest, GetMultiBatchComputed_MaxInFlight) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  config.set_max_in_flight(2);
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(1);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({2, 2, 2}, "A", 0);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 0);
  AddVectorToInputsAsTensor({3, 3, 3}, "A", 1);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 1);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(2, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  auto expected_tensor = tf::test::AsTensor<int32_t>({6, 8, 10});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult, expected_tensor);
  const tf::Tensor& tensor_mult1 = output_packets_mult[1].Get<tf::Tensor>();
  auto expected_tensor1 = tf::test::AsTensor<int32_t>({9, 12, 15});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult1, expected_tensor1);

  EXPECT_EQ(2, runner_
                   ->GetCounter(
                       "TensorFlowInferenceCalculator-TotalProcessedTimestamps")
                   ->Get());
}

TEST_F(TensorflowInferenceCalculatorTest,
       GetMultiBatchComputed_MoreThanMaxInFlight) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  config.set_max_in_flight(2);
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(1);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({2, 2, 2}, "A", 0);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 0);
  AddVectorToInputsAsTensor({3, 3, 3}, "A", 1);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 1);
  AddVectorToInputsAsTensor({4, 4, 4}, "A", 2);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 2);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(3, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  auto expected_tensor = tf::test::AsTensor<int32_t>({6, 8, 10});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult, expected_tensor);
  const tf::Tensor& tensor_mult1 = output_packets_mult[1].Get<tf::Tensor>();
  auto expected_tensor1 = tf::test::AsTensor<int32_t>({9, 12, 15});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult1, expected_tensor1);
  const tf::Tensor& tensor_mult2 = output_packets_mult[2].Get<tf::Tensor>();
  auto expected_tensor2 = tf::test::AsTensor<int32_t>({12, 16, 20});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult2, expected_tensor2);

  EXPECT_EQ(3, runner_
                   ->GetCounter(
                       "TensorFlowInferenceCalculator-TotalProcessedTimestamps")
                   ->Get());
}

TEST_F(TensorflowInferenceCalculatorTest, GetSingleBatchComputed) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(2);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(true);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({2, 2, 2}, "A", 0);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 0);
  AddVectorToInputsAsTensor({3, 3, 3}, "A", 1);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 1);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(2, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  auto expected_tensor = tf::test::AsTensor<int32_t>({6, 8, 10});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult, expected_tensor);
  const tf::Tensor& tensor_mult1 = output_packets_mult[1].Get<tf::Tensor>();
  auto expected_tensor1 = tf::test::AsTensor<int32_t>({9, 12, 15});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult1, expected_tensor1);

  EXPECT_EQ(2, runner_
                   ->GetCounter(
                       "TensorFlowInferenceCalculator-TotalProcessedTimestamps")
                   ->Get());
}

TEST_F(TensorflowInferenceCalculatorTest, GetCloseBatchComputed) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(3);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(true);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({2, 2, 2}, "A", 0);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 0);
  AddVectorToInputsAsTensor({3, 3, 3}, "A", 1);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 1);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(2, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  auto expected_tensor = tf::test::AsTensor<int32_t>({6, 8, 10});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult, expected_tensor);
  const tf::Tensor& tensor_mult1 = output_packets_mult[1].Get<tf::Tensor>();
  auto expected_tensor1 = tf::test::AsTensor<int32_t>({9, 12, 15});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult1, expected_tensor1);

  EXPECT_EQ(2, runner_
                   ->GetCounter(
                       "TensorFlowInferenceCalculator-TotalProcessedTimestamps")
                   ->Get());
}

TEST_F(TensorflowInferenceCalculatorTest, GetCloseBatchComputedNoPadding) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(3);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_pad_to_batch_size(false);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(true);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({2, 2, 2}, "A", 0);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 0);
  AddVectorToInputsAsTensor({3, 3, 3}, "A", 1);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 1);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(2, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  auto expected_tensor = tf::test::AsTensor<int32_t>({6, 8, 10});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult, expected_tensor);
  const tf::Tensor& tensor_mult1 = output_packets_mult[1].Get<tf::Tensor>();
  auto expected_tensor1 = tf::test::AsTensor<int32_t>({9, 12, 15});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult1, expected_tensor1);

  EXPECT_EQ(2, runner_
                   ->GetCounter(
                       "TensorFlowInferenceCalculator-TotalProcessedTimestamps")
                   ->Get());
}

TEST_F(TensorflowInferenceCalculatorTest, GetBatchComputed_MaxInFlight) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  config.set_max_in_flight(2);
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(2);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(true);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batched_input(true);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsPacket(
      {CreateTensorPacket({2, 2, 2}, 0), CreateTensorPacket({3, 3, 3}, 1)},
      "A");
  AddVectorToInputsAsPacket(
      {CreateTensorPacket({3, 4, 5}, 0), CreateTensorPacket({3, 4, 5}, 1)},
      "B");
  AddVectorToInputsAsPacket(
      {CreateTensorPacket({4, 4, 4}, 2), CreateTensorPacket({5, 5, 5}, 3)},
      "A");
  AddVectorToInputsAsPacket(
      {CreateTensorPacket({3, 4, 5}, 2), CreateTensorPacket({3, 4, 5}, 3)},
      "B");
  AddVectorToInputsAsPacket({CreateTensorPacket({6, 6, 6}, 4)}, "A");
  AddVectorToInputsAsPacket({CreateTensorPacket({3, 4, 5}, 4)}, "B");
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(5, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  auto expected_tensor = tf::test::AsTensor<int32_t>({6, 8, 10});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult, expected_tensor);
  const tf::Tensor& tensor_mult1 = output_packets_mult[1].Get<tf::Tensor>();
  auto expected_tensor1 = tf::test::AsTensor<int32_t>({9, 12, 15});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult1, expected_tensor1);
  const tf::Tensor& tensor_mult2 = output_packets_mult[2].Get<tf::Tensor>();
  auto expected_tensor2 = tf::test::AsTensor<int32_t>({12, 16, 20});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult2, expected_tensor2);
  const tf::Tensor& tensor_mult3 = output_packets_mult[3].Get<tf::Tensor>();
  auto expected_tensor3 = tf::test::AsTensor<int32_t>({15, 20, 25});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult3, expected_tensor3);
  const tf::Tensor& tensor_mult4 = output_packets_mult[4].Get<tf::Tensor>();
  auto expected_tensor4 = tf::test::AsTensor<int32_t>({18, 24, 30});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult4, expected_tensor4);

  EXPECT_EQ(5, runner_
                   ->GetCounter(
                       "TensorFlowInferenceCalculator-TotalProcessedTimestamps")
                   ->Get());
}

TEST_F(TensorflowInferenceCalculatorTest, TestRecurrentStates) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(1);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(true);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->add_recurrent_tag_pair("A:MULTIPLIED");
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 0);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 1);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(2, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  ABSL_LOG(INFO) << "timestamp: " << 0;
  auto expected_tensor = tf::test::AsTensor<int32_t>({3, 8, 15});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult, expected_tensor);
  const tf::Tensor& tensor_mult1 = output_packets_mult[1].Get<tf::Tensor>();
  auto expected_tensor1 = tf::test::AsTensor<int32_t>({9, 32, 75});
  ABSL_LOG(INFO) << "timestamp: " << 1;
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult1, expected_tensor1);

  EXPECT_EQ(2, runner_
                   ->GetCounter(
                       "TensorFlowInferenceCalculator-TotalProcessedTimestamps")
                   ->Get());
}
TEST_F(TensorflowInferenceCalculatorTest, TestRecurrentStateOverride) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(1);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(true);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->add_recurrent_tag_pair("A:MULTIPLIED");
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({1, 1, 1}, "A", 0);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 0);
  AddVectorToInputsAsTensor({1, 1, 1}, "A", 1);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 1);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(2, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  ABSL_LOG(INFO) << "timestamp: " << 0;
  auto expected_tensor = tf::test::AsTensor<int32_t>({3, 4, 5});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult, expected_tensor);
  const tf::Tensor& tensor_mult1 = output_packets_mult[1].Get<tf::Tensor>();
  auto expected_tensor1 = tf::test::AsTensor<int32_t>({3, 4, 5});
  ABSL_LOG(INFO) << "timestamp: " << 1;
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult1, expected_tensor1);

  EXPECT_EQ(2, runner_
                   ->GetCounter(
                       "TensorFlowInferenceCalculator-TotalProcessedTimestamps")
                   ->Get());
}

// TODO: Investigate this test failure.
TEST_F(TensorflowInferenceCalculatorTest, DISABLED_CheckTiming) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_in");
  config.add_output_stream("EXPENSIVE:tensor_expensive");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(1);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(false);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({0, 0, 0}, "A", 0);
  MP_ASSERT_OK(runner_->Run());

  EXPECT_EQ(1, runner_
                   ->GetCounter(
                       "TensorFlowInferenceCalculator-TotalProcessedTimestamps")
                   ->Get());
  // We only test the timing counter here because we are requesting an
  // expensive tensor output. Because the precision on android is
  // sometimes closer to milliseconds, we need to request a large tensor
  // to be sure this will be greater than zero.
  EXPECT_GT(runner_->GetCounter("TensorFlowInferenceCalculator-TotalTimeUsecs")
                ->Get(),
            0);
}

TEST_F(TensorflowInferenceCalculatorTest, MissingInputFeature) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(2);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(true);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_skip_on_missing_features(false);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({2, 2, 2}, "A", 0);
  ASSERT_FALSE(runner_->Run().ok());
}

TEST_F(TensorflowInferenceCalculatorTest, MissingInputFeature_Skip) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(2);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(true);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_skip_on_missing_features(true);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({2, 2, 2}, "A", 0);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(0, output_packets_mult.size());
}

TEST_F(TensorflowInferenceCalculatorTest,
       MissingInputFeature_SkipCheckInternalState) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(2);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(true);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_skip_on_missing_features(true);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsTensor({2, 2, 2}, "A", 0);
  AddVectorToInputsAsTensor({3, 3, 3}, "A", 1);
  AddVectorToInputsAsTensor({3, 4, 5}, "B", 1);
  MP_ASSERT_OK(runner_->Run());

  const std::vector<Packet>& output_packets_mult =
      runner_->Outputs().Tag(kMultipliedTag).packets;
  ASSERT_EQ(1, output_packets_mult.size());
  const tf::Tensor& tensor_mult = output_packets_mult[0].Get<tf::Tensor>();
  auto expected_tensor = tf::test::AsTensor<int32_t>({9, 12, 15});
  tf::test::ExpectTensorEqual<int32_t>(tensor_mult, expected_tensor);

  EXPECT_EQ(1, runner_
                   ->GetCounter(
                       "TensorFlowInferenceCalculator-TotalProcessedTimestamps")
                   ->Get());
}

TEST_F(TensorflowInferenceCalculatorTest, BatchedInputTooBigBatch) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorFlowInferenceCalculator");
  config.add_input_stream("A:tensor_a");
  config.add_input_stream("B:tensor_b");
  config.add_output_stream("MULTIPLIED:tensor_o1");
  config.add_input_side_packet("SESSION:session");
  config.set_max_in_flight(2);
  CalculatorOptions options;
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batch_size(2);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_add_batch_dim_to_tensors(true);
  options.MutableExtension(TensorFlowInferenceCalculatorOptions::ext)
      ->set_batched_input(true);
  *config.mutable_options() = options;

  runner_ = absl::make_unique<CalculatorRunner>(config);
  AddSessionInputSidePacket();
  AddVectorToInputsAsPacket(
      {CreateTensorPacket({2, 2, 2}, 0), CreateTensorPacket({3, 3, 3}, 1),
       CreateTensorPacket({4, 4, 4}, 2)},
      "A");
  AddVectorToInputsAsPacket(
      {CreateTensorPacket({3, 4, 5}, 0), CreateTensorPacket({3, 4, 5}, 1),
       CreateTensorPacket({3, 4, 5}, 2)},
      "B");

  auto status = runner_->Run();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "has more packets than batch capacity. batch_size: 2 packets: 3"));
}

}  // namespace mediapipe
