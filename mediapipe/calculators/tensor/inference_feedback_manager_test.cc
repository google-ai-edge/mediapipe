// Copyright 2024 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/inference_feedback_manager.h"

#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/util.h"

namespace mediapipe {
namespace api2 {
namespace {

// feedback_tensor_test_model.iflite model passes through stateless/non-feedback
// tensors and increments "stateful" tensors by one during inference:
//
// class Wrapper(tf.Module):
//   @tf.function(input_signature=
//      (tf.TensorSpec(shape=[1,1],
//                     dtype=tf.float32,
//                     name="regular_input0"),
//       tf.TensorSpec(shape=[1,1],
//                     dtype=tf.float32,
//                     name="feedback_float_input"),
//       tf.TensorSpec(shape=[1,2],
//                     dtype=tf.float32,
//                     name="regular_input1"),
//       tf.TensorSpec(shape=[1,1], dtype=tf.int32,
//       name="feedback_int_input")))
//   def model(self, ...):
//     return {"regular_output0": regular_input0,
//             "feedback_incremented_float_output": feedback_float_input + 1,
//             "regular_output1": regular_input1,
//             "feedback_incremented_int_output": feedback_int_input + 1}
//
constexpr char kFeedbackTestModelPath[] =
    "mediapipe/calculators/tensor/testdata/"
    "feedback_tensor_test_model.tflite";

// feedback_tensor_with_state_copy_model.tflite model passes through
// stateless/non-feedback tensors and increments "stateful" tensors by one. It
// also copies the stateful tensor to a second tensor to enable to observe its
// state.
//
// class Wrapper(tf.Module):
//   @tf.function(input_signature=(tf.TensorSpec(shape=[1,1],
//                                               dtype=tf.int32,
//                                               name="regular_int_input"),
//                                 tf.TensorSpec(shape=[1,1],
//                                               dtype=tf.int32,
//                                               name="feedback_int_input")))
//   def model(self, regular_int_input, feedback_int_input):
//     return {"regular_int_output": regular_int_input,
//             "feedback_incremented_int_output": feedback_int_input + 1,
//             "feedback_incremented_int_copy": feedback_int_input + 0}
constexpr char kFeedbackTestWithStateCopyModelPath[] =
    "mediapipe/calculators/tensor/testdata/"
    "feedback_tensor_with_state_copy_model.tflite";

using ::mediapipe::Packet;
using ::mediapipe::tool::AddVectorSink;
using ::testing::HasSubstr;
using ::tflite::InterpreterBuilder;
using TfLiteDelegatePtr =
    std::unique_ptr<TfLiteOpaqueDelegate,
                    std::function<void(TfLiteOpaqueDelegate*)>>;

static Tensor CreateSingleIntTensor(int value) {
  std::vector<int> dims = {1, 1};
  Tensor tensor(Tensor::ElementType::kInt32, Tensor::Shape(dims),
                /*memory_manager=*/nullptr, tflite::kDefaultTensorAlignment);
  auto write_view = tensor.GetCpuWriteView();
  *write_view.buffer<int>() = value;
  return tensor;
}

class InferenceFeedbackManagerTest : public ::testing::Test {
 protected:
  void InitModelAndInterpreter(const std::string& model_path) {
    MP_ASSERT_OK_AND_ASSIGN(model_,
                            TfLiteModelLoader::LoadFromPath(model_path));
    op_resolver_ = std::make_unique<tflite::ops::builtin::BuiltinOpResolver>(
        tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates());
    InterpreterBuilder builder(*model_.Get(), *op_resolver_);
    auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_opts.num_threads = 1;
    delegate_ = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                                  &TfLiteXNNPackDelegateDelete);
    builder.AddDelegate(delegate_.get());
    builder.SetNumThreads(1);
    ASSERT_EQ(builder(&interpreter_), kTfLiteOk);
    ASSERT_NE(interpreter_, nullptr);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  }

  // Helper methods to access tensors of feedback_tensor_test_model.tflite

  // ~~~~~~~~~~ INPUTS ~~~~~~~~~~
  // 0 :  regular_input0 :  [1 1] :  F32
  // 1 :  feedback_float_input :  [1 1] :  F32
  // 2 :  feedback_int_input :  [1 1] :  I32
  // 3 :  regular_input1 :  [1 2] :  F32
  // ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
  // 0 :  feedback_incremented_float_output :  [1 1] :  F32
  // 1 :  regular_output1 :  [1 2] :  F32
  // 2 :  feedback_incremented_int_output :  [1 1] :  I32
  // 3 :  regular_output0 :  [1 1] :  F32
  void PopulateRegularSingleFloatInputTensor(float value) {
    std::vector<float> input_buffer = {value};
    CopyTensorBufferToInterpreter(input_buffer, /*input_tensor_index=*/0);
  }

  void PopulateFeedbackFloatInputTensor(float value) {
    std::vector<float> input_buffer = {value};
    CopyTensorBufferToInterpreter(input_buffer, /*input_tensor_index=*/1);
  }

  void PopulateFeedbackIntInputTensor(int value) {
    std::vector<int> input_buffer = {value};
    CopyTensorBufferToInterpreter(input_buffer, /*input_tensor_index=*/2);
  }

  void PopulateRegularTwoFloatInputTensor(std::vector<float> input_buffer) {
    CopyTensorBufferToInterpreter(input_buffer, /*input_tensor_index=*/3);
  }

  void RunInference() { ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk); }

  float GetFeedbackFloatOutput() {
    std::vector<float> output_buffer = CopyTensorBufferFromInterpreter<float>(
        /*output_tensor_index=*/0, /*num_elements=*/1);
    return output_buffer[0];
  }

  std::vector<float> GetRegularTwoFloatsOutput() {
    return CopyTensorBufferFromInterpreter<float>(/*output_tensor_index=*/1,
                                                  /*num_elements=*/2);
  }
  int GetFeedbackIntOutput() {
    std::vector<int> output_buffer = CopyTensorBufferFromInterpreter<int>(
        /*output_tensor_index=*/2, /*num_elements=*/1);
    return output_buffer[0];
  }

  float GetRegularSingleFloatOutput() {
    std::vector<float> output_buffer = CopyTensorBufferFromInterpreter<float>(
        /*output_tensor_index=*/3, /*num_elements=*/1);
    return output_buffer[0];
  }

  api2::Packet<TfLiteModelPtr> model_;
  std::unique_ptr<tflite::OpResolver> op_resolver_;
  TfLiteDelegatePtr delegate_;
  std::unique_ptr<tflite::Interpreter> interpreter_;

 private:
  template <typename T>
  void CopyTensorBufferToInterpreter(const std::vector<T>& input_buffer,
                                     int input_tensor_index) {
    EXPECT_LT(input_tensor_index, interpreter_->inputs().size());
    T* local_tensor_buffer =
        interpreter_->typed_input_tensor<T>(input_tensor_index);
    std::memcpy(local_tensor_buffer,
                static_cast<const void*>(input_buffer.data()),
                input_buffer.size() * sizeof(T));
  }

  template <typename T>
  std::vector<T> CopyTensorBufferFromInterpreter(int output_tensor_index,
                                                 int num_elements) {
    EXPECT_LT(output_tensor_index, interpreter_->outputs().size());
    std::vector<T> result(num_elements);
    T* local_tensor_buffer =
        interpreter_->typed_output_tensor<T>(output_tensor_index);
    std::memcpy(static_cast<void*>(result.data()), local_tensor_buffer,
                result.size() * sizeof(T));
    return result;
  }
};

TEST_F(InferenceFeedbackManagerTest, ModelShouldIncreaseStatefulTensorsByOne) {
  // Test the test model.
  InitModelAndInterpreter(kFeedbackTestModelPath);
  // Initialize "stateful" input tensors with zero values.
  PopulateFeedbackFloatInputTensor(0);
  PopulateRegularTwoFloatInputTensor({1.0f, 2.0f});
  PopulateFeedbackIntInputTensor(0);
  PopulateRegularSingleFloatInputTensor(3.14f);
  RunInference();
  EXPECT_EQ(GetFeedbackIntOutput(), 1);
  EXPECT_FLOAT_EQ(GetFeedbackFloatOutput(), 1.0f);
  EXPECT_FLOAT_EQ(GetRegularSingleFloatOutput(), 3.14f);
  EXPECT_THAT(GetRegularTwoFloatsOutput(), testing::ElementsAre(1.0f, 2.0f));
}

TEST_F(InferenceFeedbackManagerTest,
       ShouldInitializeFeedbackTensorInputWithZeros) {
  InitModelAndInterpreter(kFeedbackTestModelPath);

  constexpr int kFeedbackFloatInputIndex = 1;
  constexpr int kFeedbackIntInputIndex = 2;
  EXPECT_STREQ(interpreter_->GetInputName(kFeedbackFloatInputIndex),
               "serving_default_feedback_float_input:0");
  EXPECT_STREQ(interpreter_->GetInputName(kFeedbackIntInputIndex),
               "serving_default_feedback_int_input:0");
  // Initialize "stateful" input tensors with non-zero values.
  *interpreter_->typed_input_tensor<float>(kFeedbackFloatInputIndex) = 123.0f;
  *interpreter_->typed_input_tensor<int>(kFeedbackIntInputIndex) = 123;

  InferenceFeedbackManager feedback_manager;
  InferenceCalculatorOptions::InputOutputConfig config =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_float_output",
          to_input_tensor_name: "feedback_float_input",
        }
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_int_output",
          to_input_tensor_name: "feedback_int_input",
        }
      )pb");

  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  MP_ASSERT_OK(feedback_manager.Init(config, input_output_tensor_names,
                                     interpreter_.get()));

  EXPECT_EQ(*interpreter_->typed_input_tensor<float>(kFeedbackFloatInputIndex),
            0.0f);
  EXPECT_EQ(*interpreter_->typed_input_tensor<int>(kFeedbackIntInputIndex), 0);
}

TEST_F(InferenceFeedbackManagerTest,
       ShouldAllowToQueryForFeedbackTensorIndices) {
  InitModelAndInterpreter(kFeedbackTestModelPath);

  // ~~~~~~~~~~ INPUTS ~~~~~~~~~~
  // 0 :  regular_input0 :  [1 1] :  F32
  // 1 :  feedback_float_input :  [1 1] :  F32
  // 2 :  feedback_int_input :  [1 1] :  I32
  // 3 :  regular_input1 :  [1 2] :  F32
  // ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
  // 0 :  feedback_incremented_float_output :  [1 1] :  F32
  // 1 :  regular_output1 :  [1 2] :  F32
  // 2 :  feedback_incremented_int_output :  [1 1] :  I32
  // 3 :  regular_output0 :  [1 1] :  F32

  // Confirm input signatures.
  constexpr int kRegularInput0Index = 0;
  constexpr int kFeedbackFloatInputIndex = 1;
  constexpr int kFeedbackIntInputIndex = 2;
  constexpr int kRegularInput1Index = 3;

  // Confirm output signatures.
  constexpr int kFeedbackIncrementedFloatOutputIndex = 0;
  constexpr int kRegularOutput0Index = 1;
  constexpr int kFeedbackIncrementedIntOutputIndex = 2;
  constexpr int kRegularOutput1Index = 3;

  InferenceFeedbackManager feedback_manager;
  InferenceCalculatorOptions::InputOutputConfig config =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_float_output",
          to_input_tensor_name: "feedback_float_input",
        }
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_int_output",
          to_input_tensor_name: "feedback_int_input",
        }
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  MP_ASSERT_OK(feedback_manager.Init(config, input_output_tensor_names,
                                     interpreter_.get()));

  EXPECT_TRUE(
      feedback_manager.IsFeedbackInputTensorAtIndex(kFeedbackFloatInputIndex));
  EXPECT_TRUE(
      feedback_manager.IsFeedbackInputTensorAtIndex(kFeedbackIntInputIndex));
  EXPECT_FALSE(
      feedback_manager.IsFeedbackInputTensorAtIndex(kRegularInput0Index));
  EXPECT_FALSE(
      feedback_manager.IsFeedbackInputTensorAtIndex(kRegularInput1Index));

  EXPECT_TRUE(feedback_manager.IsFeedbackOutputTensorAtIndex(
      kFeedbackIncrementedIntOutputIndex));
  EXPECT_TRUE(feedback_manager.IsFeedbackOutputTensorAtIndex(
      kFeedbackIncrementedFloatOutputIndex));
  EXPECT_FALSE(
      feedback_manager.IsFeedbackOutputTensorAtIndex(kRegularOutput0Index));
  EXPECT_FALSE(
      feedback_manager.IsFeedbackOutputTensorAtIndex(kRegularOutput1Index));
}

TEST_F(InferenceFeedbackManagerTest, ShouldMapInputTensorToModelTensorIndices) {
  InitModelAndInterpreter(kFeedbackTestModelPath);

  // First two input tensors are stateful / feedback tensors.
  constexpr int kRegularInput0Index = 0;
  constexpr int kRegularInput1Index = 3;

  InferenceFeedbackManager feedback_manager;
  InferenceCalculatorOptions::InputOutputConfig config =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_float_output",
          to_input_tensor_name: "feedback_float_input",
        }
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_int_output",
          to_input_tensor_name: "feedback_int_input",
        }
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  MP_ASSERT_OK(feedback_manager.Init(config, input_output_tensor_names,
                                     interpreter_.get()));

  // Feedback tensors are skipped in InferenceRunner input. Therefore the first
  // two InferenceRunner input tensors must point to the first two non-feedback
  // model input tensors.
  MP_ASSERT_OK_AND_ASSIGN(const int inference_runner_index0,
                          feedback_manager.MapInputTensorToModelIndex(0));
  MP_ASSERT_OK_AND_ASSIGN(const int inference_runner_index1,
                          feedback_manager.MapInputTensorToModelIndex(1));
  EXPECT_EQ(inference_runner_index0, kRegularInput0Index);
  EXPECT_EQ(inference_runner_index1, kRegularInput1Index);
}

TEST_F(InferenceFeedbackManagerTest, ShouldDetectLinksWithDifferentTypes) {
  InitModelAndInterpreter(kFeedbackTestModelPath);

  constexpr int kFeedbackFloatInputIndex = 1;
  EXPECT_STREQ(interpreter_->GetInputName(kFeedbackFloatInputIndex),
               "serving_default_feedback_float_input:0");

  InferenceFeedbackManager feedback_manager;
  InferenceCalculatorOptions::InputOutputConfig config =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_int_output",
          to_input_tensor_name: "feedback_float_input",
        }
      )pb");

  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  EXPECT_THAT(feedback_manager.Init(config, input_output_tensor_names,
                                    interpreter_.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Feedback tensors must have the same spec")));
}

TEST_F(InferenceFeedbackManagerTest, ShouldDetectDynamicInputTensors) {
  InitModelAndInterpreter(kFeedbackTestModelPath);

  InferenceFeedbackManager feedback_manager;
  InferenceCalculatorOptions::InputOutputConfig config =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_float_output",
          to_input_tensor_name: "feedback_float_input",
        }
      )pb");

  // Mark feedback tensor as dynamic by setting one dimension to -1.
  interpreter_->tensor(interpreter_->inputs()[1])->dims->data[0] = -1;

  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  EXPECT_THAT(
      feedback_manager.Init(config, input_output_tensor_names,
                            interpreter_.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Feedback input tensors must not be dynamic")));
}

TEST_F(InferenceFeedbackManagerTest, ShouldDetectDynamicOutputTensors) {
  InitModelAndInterpreter(kFeedbackTestModelPath);

  InferenceFeedbackManager feedback_manager;
  InferenceCalculatorOptions::InputOutputConfig config =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_float_output",
          to_input_tensor_name: "feedback_float_input",
        }
      )pb");

  // Mark feedback tensor as dynamic by setting one dimension to -1.
  interpreter_->tensor(interpreter_->outputs()[0])->dims->data[0] = -1;

  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  EXPECT_THAT(
      feedback_manager.Init(config, input_output_tensor_names,
                            interpreter_.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Feedback output tensors must not be dynamic")));
}

TEST_F(InferenceFeedbackManagerTest, ShouldDetectLinksWithDifferentDimensions) {
  InitModelAndInterpreter(kFeedbackTestModelPath);

  constexpr int kFeedbackFloatInputIndex = 1;
  EXPECT_STREQ(interpreter_->GetInputName(kFeedbackFloatInputIndex),
               "serving_default_feedback_float_input:0");

  InferenceFeedbackManager feedback_manager;
  InferenceCalculatorOptions::InputOutputConfig config =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        feedback_tensor_links {
          from_output_tensor_name: "regular_output1",    # has dimension [1 2]
          to_input_tensor_name: "feedback_float_input",  # has dimension [1 1]
        }
      )pb");

  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  EXPECT_THAT(feedback_manager.Init(config, input_output_tensor_names,
                                    interpreter_.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Feedback tensors must have the same spec")));
}

TEST_F(InferenceFeedbackManagerTest,
       ShouldDetectMismatchBetweenStatefulAndRegularTensors) {
  InitModelAndInterpreter(kFeedbackTestModelPath);

  InferenceFeedbackManager feedback_manager;
  InferenceCalculatorOptions::InputOutputConfig config =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_names_map { tensor_names: "feedback_float_input" }
        feedback_tensor_links {
          from_output_tensor_name: "regular_output1",    # has dimension [1 2]
          to_input_tensor_name: "feedback_float_input",  # has dimension [1 1]
        }
      )pb");

  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  EXPECT_THAT(feedback_manager.Init(config, input_output_tensor_names,
                                    interpreter_.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Feedback input tensor [feedback_float_input] "
                                 "cannot be used for input/output mapping")));
}

TEST_F(InferenceFeedbackManagerTest, ShouldSwapFeedbackTensors) {
  InitModelAndInterpreter(kFeedbackTestModelPath);
  InferenceFeedbackManager feedback_manager;
  InferenceCalculatorOptions::InputOutputConfig config =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_float_output",
          to_input_tensor_name: "feedback_float_input",
        }
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_int_output",
          to_input_tensor_name: "feedback_int_input",
        }
      )pb");
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  MP_ASSERT_OK(feedback_manager.Init(config, input_output_tensor_names,
                                     interpreter_.get()));
  // Initialize "stateful" input tensors with zero values.
  PopulateFeedbackFloatInputTensor(0);
  PopulateFeedbackIntInputTensor(0);
  PopulateRegularSingleFloatInputTensor(3.14f);
  PopulateRegularTwoFloatInputTensor({1.0f, 2.0f});
  RunInference();
  EXPECT_EQ(GetFeedbackIntOutput(), 1);
  EXPECT_FLOAT_EQ(GetFeedbackFloatOutput(), 1.0f);
  EXPECT_FLOAT_EQ(GetRegularSingleFloatOutput(), 3.14f);
  EXPECT_THAT(GetRegularTwoFloatsOutput(), testing::ElementsAre(1.0f, 2.0f));

  feedback_manager.SwapFeedbackTensors();

  RunInference();
  EXPECT_EQ(GetFeedbackIntOutput(), 2);
  EXPECT_FLOAT_EQ(GetFeedbackFloatOutput(), 2.0f);
  EXPECT_FLOAT_EQ(GetRegularSingleFloatOutput(), 3.14f);
  EXPECT_THAT(GetRegularTwoFloatsOutput(), testing::ElementsAre(1.0f, 2.0f));
}

TEST_F(InferenceFeedbackManagerTest, ShouldRunE2ESmokeTest) {
  CalculatorGraph graph;
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::StrReplaceAll(
          R"pb(
            input_stream: "regular_int_input"
            input_stream: "feedback_int_input"
            output_stream: "regular_int_output"
            output_stream: "feedback_incremented_int_copy"
            node {
              calculator: "InferenceCalculator"
              # ~~~~~~~~~~ INPUTS ~~~~~~~~~~
              # 0 :  feedback_int_input :  [1 1] :  I32
              # 1 :  regular_int_input :  [1 1] :  I32
              # ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
              # 0 :  feedback_incremented_int_copy :  [1 1] :  I32
              # 1 :  regular_int_output :  [1 1] :  I32
              # 2 :  feedback_incremented_int_output :  [1 1] :  I32
              #      (copy of feedback_incremented_int_output)
              input_stream: "TENSOR:0:regular_int_input"
              output_stream: "TENSOR:0:feedback_incremented_int_copy"
              output_stream: "TENSOR:1:regular_int_output"
              options {
                [mediapipe.InferenceCalculatorOptions.ext] {
                  model_path: "$model"
                  delegate {}  # empty delegate message enables CPU inference.
                  input_output_config {
                    input_tensor_names_map { tensor_names: "regular_int_input" }
                    output_tensor_names_map {
                      tensor_names: "feedback_incremented_int_copy"
                      tensor_names: "regular_int_output"
                    }
                    feedback_tensor_links {
                      from_output_tensor_name: "feedback_incremented_int_output"
                      to_input_tensor_name: "feedback_int_input"
                    }
                  }
                }
              }
            }
          )pb",
          {{"$model", kFeedbackTestWithStateCopyModelPath}}));

  std::vector<Packet> regular_int_output;
  AddVectorSink("regular_int_output", &graph_config, &regular_int_output);
  std::vector<Packet> feedback_incremented_int_copy;
  AddVectorSink("feedback_incremented_int_copy", &graph_config,
                &feedback_incremented_int_copy);

  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  const std::vector<int> kRegularInputTensorValues = {100, 200, 300};
  // Simulate 3 inference steps.
  for (int n = 0; n < 3; ++n) {
    Tensor input_tensor = CreateSingleIntTensor(kRegularInputTensorValues[n]);
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "regular_int_input",
        mediapipe::MakePacket<Tensor>(std::move(input_tensor))
            .At(Timestamp(n))));
  }
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_EQ(regular_int_output.size(), 3);
  EXPECT_EQ(feedback_incremented_int_copy.size(), 3);
  for (int i = 0; i < regular_int_output.size(); ++i) {
    const auto regular_read_view =
        regular_int_output[i].Get<Tensor>().GetCpuReadView();
    EXPECT_EQ(regular_read_view.buffer<int>()[0], kRegularInputTensorValues[i]);

    // Stateful tensor are initialized with zero and incremented by one in
    // every iteration.
    const auto feedback_read_view =
        feedback_incremented_int_copy[i].Get<Tensor>().GetCpuReadView();
    EXPECT_EQ(feedback_read_view.buffer<int>()[0], i);
  }
}

TEST_F(InferenceFeedbackManagerTest, ShouldRunE2EWithZeroIoCopiesSmokeTest) {
  CalculatorGraph graph;
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::StrReplaceAll(
          R"pb(
            input_stream: "regular_int_input"
            input_stream: "feedback_int_input"
            output_stream: "regular_int_output"
            output_stream: "feedback_incremented_int_copy"
            node {
              calculator: "InferenceCalculator"
              # ~~~~~~~~~~ INPUTS ~~~~~~~~~~
              # 0 :  feedback_int_input :  [1 1] :  I32
              # 1 :  regular_int_input :  [1 1] :  I32
              # ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
              # 0 :  feedback_incremented_int_copy :  [1 1] :  I32
              # 1 :  regular_int_output :  [1 1] :  I32
              # 2 :  feedback_incremented_int_output :  [1 1] :  I32
              #      (copy of feedback_incremented_int_output)
              input_stream: "TENSOR:0:regular_int_input"
              output_stream: "TENSOR:0:feedback_incremented_int_copy"
              output_stream: "TENSOR:1:regular_int_output"
              options {
                [mediapipe.InferenceCalculatorOptions.ext] {
                  model_path: "$model"
                  delegate {
                    xnnpack {
                      enable_zero_copy_tensor_io: true,
                    }
                  }
                  input_output_config {
                    input_tensor_names_map { tensor_names: "regular_int_input" }
                    output_tensor_names_map {
                      tensor_names: "feedback_incremented_int_copy"
                      tensor_names: "regular_int_output"
                    }
                    feedback_tensor_links {
                      from_output_tensor_name: "feedback_incremented_int_output"
                      to_input_tensor_name: "feedback_int_input"
                    }
                  }
                }
              }
            }
          )pb",
          {{"$model", kFeedbackTestWithStateCopyModelPath}}));

  std::vector<Packet> regular_int_output;
  AddVectorSink("regular_int_output", &graph_config, &regular_int_output);
  std::vector<Packet> feedback_incremented_int_copy;
  AddVectorSink("feedback_incremented_int_copy", &graph_config,
                &feedback_incremented_int_copy);

  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  const std::vector<int> kRegularInputTensorValues = {100, 200, 300};
  // Simulate 3 inference steps.
  for (int n = 0; n < 3; ++n) {
    Tensor input_tensor = CreateSingleIntTensor(kRegularInputTensorValues[n]);
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "regular_int_input",
        mediapipe::MakePacket<Tensor>(std::move(input_tensor))
            .At(Timestamp(n))));
  }
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(regular_int_output.size(), 3);
  ASSERT_EQ(regular_int_output.size(), feedback_incremented_int_copy.size());
  for (int i = 0; i < regular_int_output.size(); ++i) {
    const auto regular_read_view =
        regular_int_output[i].Get<Tensor>().GetCpuReadView();
    EXPECT_EQ(regular_read_view.buffer<int>()[0], kRegularInputTensorValues[i]);

    // Stateful tensor are initialized with zero and incremented by one in
    // every iteration.
    const auto feedback_read_view =
        feedback_incremented_int_copy[i].Get<Tensor>().GetCpuReadView();
    EXPECT_EQ(feedback_read_view.buffer<int>()[0], i);
  }
}

TEST_F(InferenceFeedbackManagerTest,
       ShouldRunE2EWithWithoutFeedbackManagerConfigSmokeTest) {
  CalculatorGraph graph;
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::StrReplaceAll(
          R"pb(
            input_stream: "regular_int_input"
            input_stream: "feedback_int_input"
            output_stream: "regular_int_output"
            output_stream: "feedback_incremented_int_output"
            output_stream: "feedback_incremented_int_copy"
            node {
              calculator: "InferenceCalculator"
              # ~~~~~~~~~~ INPUTS ~~~~~~~~~~
              # 0 :  feedback_int_input :  [1 1] :  I32
              # 1 :  regular_int_input :  [1 1] :  I32
              # ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
              # 0 :  feedback_incremented_int_copy :  [1 1] :  I32
              # 1 :  regular_int_output :  [1 1] :  I32
              # 2 :  feedback_incremented_int_output :  [1 1] :  I32
              #      (copy of feedback_incremented_int_output)
              input_stream: "TENSOR:0:feedback_int_input"
              input_stream: "TENSOR:1:regular_int_input"
              output_stream: "TENSOR:0:feedback_incremented_int_copy"
              output_stream: "TENSOR:1:regular_int_output"
              output_stream: "TENSOR:2:feedback_incremented_int_output"
              options {
                [mediapipe.InferenceCalculatorOptions.ext] {
                  model_path: "$model"
                  delegate {}  # empty delegate message enables CPU inference.
                }
              }
            }
          )pb",
          {{"$model", kFeedbackTestWithStateCopyModelPath}}));

  std::vector<Packet> regular_int_output;
  AddVectorSink("regular_int_output", &graph_config, &regular_int_output);
  std::vector<Packet> feedback_incremented_int_output;
  AddVectorSink("feedback_incremented_int_output", &graph_config,
                &feedback_incremented_int_output);
  std::vector<Packet> feedback_incremented_int_copy;
  AddVectorSink("feedback_incremented_int_copy", &graph_config,
                &feedback_incremented_int_copy);

  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  const std::vector<int> kRegularInputTensorValues = {100, 200, 300};
  const std::vector<int> kFeedbackInputTensorValues = {111, 222, 333};
  // Simulate 3 inference steps.
  for (int n = 0; n < 3; ++n) {
    Tensor regular_int_input_tensor =
        CreateSingleIntTensor(kRegularInputTensorValues[n]);
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "regular_int_input",
        mediapipe::MakePacket<Tensor>(std::move(regular_int_input_tensor))
            .At(Timestamp(n))));
    Tensor feedback_int_input_tensor =
        CreateSingleIntTensor(kFeedbackInputTensorValues[n]);
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "feedback_int_input",
        mediapipe::MakePacket<Tensor>(std::move(feedback_int_input_tensor))
            .At(Timestamp(n))));
  }

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_EQ(regular_int_output.size(), 3);
  EXPECT_EQ(feedback_incremented_int_copy.size(), 3);
  EXPECT_EQ(feedback_incremented_int_output.size(), 3);
  for (int i = 0; i < regular_int_output.size(); ++i) {
    const auto regular_read_view =
        regular_int_output[i].Get<Tensor>().GetCpuReadView();
    EXPECT_EQ(regular_read_view.buffer<int>()[0], kRegularInputTensorValues[i]);

    // Stateful tensor are initialized with zero and incremented by one in
    // every iteration.
    {
      const auto read_view =
          feedback_incremented_int_output[i].Get<Tensor>().GetCpuReadView();
      EXPECT_EQ(read_view.buffer<int>()[0], kFeedbackInputTensorValues[i] + 1);
    }
    {
      const auto read_view =
          feedback_incremented_int_copy[i].Get<Tensor>().GetCpuReadView();
      EXPECT_EQ(read_view.buffer<int>()[0], kFeedbackInputTensorValues[i]);
    }
  }
}

}  // namespace
}  // namespace api2
}  // namespace mediapipe
