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
#include "mediapipe/calculators/tensor/inference_runner_litert.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "devtools/build/runtime/get_runfiles_dir.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#import <Metal/Metal.h>

#include "mediapipe/gpu/MPPMetalHelper.h"
#endif  // defined(__APPLE__)

namespace mediapipe {
namespace {

using ::litert::testing::GetTestFilePath;
using ::testing::FloatEq;
using ::testing::Pointwise;

std::string GetFeedbackTensorModelPath() {
  return devtools_build::GetRunfilesDir() + "/mediapipe/calculators/tensor/";
}

TEST(InferenceRunnterLitertTest, FailedWithNoAccelerator) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(
                              *resources, GetTestFilePath(kModelFileName),
                              /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;

  // Error: No accelerator is specified
  EXPECT_THAT(InferenceRunnerLiteRt::Create(std::move(model), options,
                                            /*input_output_config=*/nullptr,
                                            /*memory_manager=*/nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(InferenceRunnterLitertTest,
     ShouldRunSuccessfullyWithAlignedInputTensorsWithCpuAndGpu) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(
                              *resources, GetTestFilePath(kModelFileName),
                              /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  options.mutable_cpu();
  options.mutable_gpu();

  MP_ASSERT_OK_AND_ASSIGN(auto runner, InferenceRunnerLiteRt::Create(
                                           std::move(model), options,
                                           /*input_output_config=*/nullptr,
                                           /*memory_manager=*/nullptr));

  std::vector<Tensor> input_tensors;
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInput0Size}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[0].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput0Tensor, kTestInput0Size * sizeof(float));
  }
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInput1Size}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[1].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput1Tensor, kTestInput1Size * sizeof(float));
  }

  MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                          runner->Run(/*CalculatorContext=*/nullptr,
                                      MakeTensorSpan(input_tensors)));
  {
    auto read_view = output_tensors[0].GetCpuReadView();
    const float* data = read_view.buffer<float>();
    for (int i = 0; i < kTestOutputSize; ++i) {
      EXPECT_FLOAT_EQ(data[i], kTestOutputTensor[i]);
    }
  }
}

TEST(InferenceRunnterLitertTest, ShouldRunSuccessfullyWithAlignedInputTensors) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(
                              *resources, GetTestFilePath(kModelFileName),
                              /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  options.mutable_cpu();

  MP_ASSERT_OK_AND_ASSIGN(auto runner, InferenceRunnerLiteRt::Create(
                                           std::move(model), options,
                                           /*input_output_config=*/nullptr,
                                           /*memory_manager=*/nullptr));

  std::vector<Tensor> input_tensors;
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInput0Size}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[0].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput0Tensor, kTestInput0Size * sizeof(float));
  }
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInput1Size}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[1].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput1Tensor, kTestInput1Size * sizeof(float));
  }

  MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                          runner->Run(/*CalculatorContext=*/nullptr,
                                      MakeTensorSpan(input_tensors)));
  {
    auto read_view = output_tensors[0].GetCpuReadView();
    const float* data = read_view.buffer<float>();
    for (int i = 0; i < kTestOutputSize; ++i) {
      EXPECT_FLOAT_EQ(data[i], kTestOutputTensor[i]);
    }
  }
}

TEST(InferenceRunnterLitertTest,
     ShouldRunSuccessfullyWithUnalignedInputTensors) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(
                              *resources, GetTestFilePath(kModelFileName),
                              /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  options.mutable_cpu();

  MP_ASSERT_OK_AND_ASSIGN(auto runner, InferenceRunnerLiteRt::Create(
                                           std::move(model), options,
                                           /*input_output_config=*/nullptr,
                                           /*memory_manager=*/nullptr));

  std::vector<Tensor> input_tensors;
  input_tensors.push_back(Tensor(Tensor::ElementType::kFloat32,
                                 Tensor::Shape({1, kTestInput0Size}),
                                 /*memory_manager=*/nullptr,
                                 /*memory_alignment=*/0));
  {
    auto write_view = input_tensors[0].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput0Tensor, kTestInput0Size * sizeof(float));
  }
  input_tensors.push_back(Tensor(Tensor::ElementType::kFloat32,
                                 Tensor::Shape({1, kTestInput1Size}),
                                 /*memory_manager=*/nullptr,
                                 /*memory_alignment=*/0));
  {
    auto write_view = input_tensors[1].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput1Tensor, kTestInput1Size * sizeof(float));
  }

  MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                          runner->Run(/*CalculatorContext=*/nullptr,
                                      MakeTensorSpan(input_tensors)));
  {
    auto read_view = output_tensors[0].GetCpuReadView();
    const float* data = read_view.buffer<float>();
    for (int i = 0; i < kTestOutputSize; ++i) {
      EXPECT_FLOAT_EQ(data[i], kTestOutputTensor[i]);
    }
  }
}

TEST(InferenceRunnterLitertTest, ShouldRunSuccessfullyWithDynamicModel) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(
      auto model, TfLiteModelLoader::LoadFromPath(
                      *resources, GetTestFilePath(kDynamicModelFileName),
                      /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  options.mutable_cpu();

  MP_ASSERT_OK_AND_ASSIGN(auto runner, InferenceRunnerLiteRt::Create(
                                           std::move(model), options,
                                           /*input_output_config=*/nullptr,
                                           /*memory_manager=*/nullptr));

  std::vector<Tensor> input_tensors;
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, 2, 3}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[0].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput0Tensor, kTestInput0Size * sizeof(float));
  }
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, 2, 3}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[1].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput1Tensor, kTestInput1Size * sizeof(float));
  }

  MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                          runner->Run(/*CalculatorContext=*/nullptr,
                                      MakeTensorSpan(input_tensors)));
  {
    auto read_view = output_tensors[0].GetCpuReadView();
    const float* data = read_view.buffer<float>();
    for (int i = 0; i < kTestOutputSize; ++i) {
      EXPECT_FLOAT_EQ(data[i], kTestOutputTensor[i]);
    }
  }
}

TEST(InferenceRunnterLitertTest, ShouldReleaseModelPacketWhenFullyAccelerated) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(
                              *resources, GetTestFilePath(kModelFileName),
                              /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  options.mutable_cpu();
  options.mutable_gpu();

  MP_ASSERT_OK_AND_ASSIGN(auto runner, InferenceRunnerLiteRt::Create(
                                           std::move(model), options,
                                           /*input_output_config=*/nullptr,
                                           /*memory_manager=*/nullptr));

  EXPECT_TRUE(runner->IsModelPacketEmpty());
}

TEST(InferenceRunnterLitertTest,
     ShouldRunSuccessfullyWithAlignedInputTensorsWithNpu) {
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");

  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(
                              *resources, GetTestFilePath("one_mul.tflite"),
                              /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  options.mutable_npu()->set_compiler_plugin_library_path(litert_libs_path);
  options.mutable_npu()->set_dispatch_library_path(litert_libs_path);

  MP_ASSERT_OK_AND_ASSIGN(auto runner, InferenceRunnerLiteRt::Create(
                                           std::move(model), options,
                                           /*input_output_config=*/nullptr,
                                           /*memory_manager=*/nullptr));

  std::vector<Tensor> input_tensors;

  const int kInputSize = 4;
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_EQ(input_data.size(), kInputSize);
  const Tensor::Shape kInputShape = {2, 2};
  ASSERT_EQ(kInputShape.num_elements(), kInputSize);

  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, kInputShape,
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[0].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    ASSERT_EQ(input_tensors[0].bytes(), kInputSize * sizeof(float));
    memcpy(data, input_data.data(), kInputSize * sizeof(float));
  }
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, kInputShape,
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[1].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    ASSERT_EQ(input_tensors[1].bytes(), kInputSize * sizeof(float));
    memcpy(data, input_data.data(), kInputSize * sizeof(float));
  }

  MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                          runner->Run(/*CalculatorContext=*/nullptr,
                                      MakeTensorSpan(input_tensors)));
  ASSERT_EQ(output_tensors.size(), 1);
  {
    std::vector<float> expected_output = {1.0f, 4.0f, 9.0f, 16.0f};
    auto read_view = output_tensors[0].GetCpuReadView();
    const float* data = read_view.buffer<float>();
    ASSERT_EQ(output_tensors[0].bytes(), kInputSize * sizeof(float));
    absl::Span<const float> actual_output(data, kInputSize);
    EXPECT_THAT(actual_output, Pointwise(FloatEq(), expected_output));
  }
}

TEST(InferenceRunnterLitertTest, ShouldRunSuccessfullyWithDarwinnOptions) {
  const auto litert_libs_path =
      litert::testing::GetLiteRtPath("vendors/examples");

  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(
                              *resources, GetTestFilePath("one_mul.tflite"),
                              /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  auto* npu_opts = options.mutable_npu();
  npu_opts->set_compiler_plugin_library_path(litert_libs_path);
  npu_opts->set_dispatch_library_path(litert_libs_path);
  auto* darwinn_opts = npu_opts->mutable_darwinn();
  darwinn_opts->set_inference_priority(1);

  MP_ASSERT_OK_AND_ASSIGN(auto runner, InferenceRunnerLiteRt::Create(
                                           std::move(model), options,
                                           /*input_output_config=*/nullptr,
                                           /*memory_manager=*/nullptr));

  std::vector<Tensor> input_tensors;

  const int kInputSize = 4;
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_EQ(input_data.size(), kInputSize);
  const Tensor::Shape kInputShape = {2, 2};
  ASSERT_EQ(kInputShape.num_elements(), kInputSize);

  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, kInputShape,
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[0].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    ASSERT_EQ(input_tensors[0].bytes(), kInputSize * sizeof(float));
    memcpy(data, input_data.data(), kInputSize * sizeof(float));
  }
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, kInputShape,
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[1].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    ASSERT_EQ(input_tensors[1].bytes(), kInputSize * sizeof(float));
    memcpy(data, input_data.data(), kInputSize * sizeof(float));
  }

  MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                          runner->Run(/*CalculatorContext=*/nullptr,
                                      MakeTensorSpan(input_tensors)));
  ASSERT_EQ(output_tensors.size(), 1);
  {
    std::vector<float> expected_output = {1.0f, 4.0f, 9.0f, 16.0f};
    auto read_view = output_tensors[0].GetCpuReadView();
    const float* data = read_view.buffer<float>();
    ASSERT_EQ(output_tensors[0].bytes(), kInputSize * sizeof(float));
    absl::Span<const float> actual_output(data, kInputSize);
    EXPECT_THAT(actual_output, Pointwise(FloatEq(), expected_output));
  }
}

TEST(InferenceRunnterLitertTest,
     ShouldRunSuccessfullyWithFeedbackTensorScalar) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  const std::string kFeedbackTensor1x1xint32Path =
      GetFeedbackTensorModelPath() + "FeedbackTensorModel1x1Int32.tflite";
  MP_ASSERT_OK_AND_ASSIGN(
      auto model,
      TfLiteModelLoader::LoadFromPath(*resources, kFeedbackTensor1x1xint32Path,
                                      /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  options.mutable_cpu();

  InferenceCalculatorOptions::InputOutputConfig input_output_config =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_names_map { tensor_names: "regular_int_input" }
        output_tensor_names_map {
          tensor_names: "regular_int_plus_feedback_int_output"
          tensor_names: "regular_int_incremented_output"
        }
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_int_output"
          to_input_tensor_name: "feedback_int_input"
        }
      )pb");

  constexpr const int32_t kTestInput[] = {1};
  constexpr const size_t kTestInputSize =
      sizeof(kTestInput) / sizeof(kTestInput[0]);

  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          InferenceRunnerLiteRt::Create(
                              std::move(model), options,
                              /*input_output_config=*/&input_output_config,
                              /*memory_manager=*/nullptr));
  {
    std::vector<Tensor> input_tensors;
    input_tensors.push_back(
        Tensor(Tensor::ElementType::kInt32, Tensor::Shape({1, kTestInputSize}),
               /*memory_manager=*/nullptr,
               /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
    {
      auto write_view = input_tensors[0].GetCpuWriteView();
      int32_t* data = write_view.buffer<int32_t>();
      memcpy(data, kTestInput, kTestInputSize * sizeof(int32_t));
    }

    MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                            runner->Run(/*CalculatorContext=*/nullptr,
                                        MakeTensorSpan(input_tensors)));
    EXPECT_EQ(output_tensors.size(), 2);
    {
      auto read_view0 = output_tensors[0].GetCpuReadView();
      const int32_t* data0 = read_view0.buffer<int32_t>();
      for (int i = 0; i < kTestInputSize; ++i) {
        EXPECT_EQ(data0[i], 2);
      }
      auto read_view1 = output_tensors[1].GetCpuReadView();
      const int32_t* data1 = read_view1.buffer<int32_t>();
      for (int i = 0; i < kTestInputSize; ++i) {
        EXPECT_EQ(data1[i], 2);
      }
    }
  }
  {
    constexpr const int32_t kSecondTestInput[] = {3};
    constexpr const size_t kSecondTestInputSize = 1;
    std::vector<Tensor> input_tensors;
    input_tensors.push_back(Tensor(
        Tensor::ElementType::kInt32, Tensor::Shape({1, kSecondTestInputSize}),
        /*memory_manager=*/nullptr,
        /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
    {
      auto write_view = input_tensors[0].GetCpuWriteView();
      int32_t* data = write_view.buffer<int32_t>();
      memcpy(data, kSecondTestInput, kSecondTestInputSize * sizeof(int32_t));
    }

    MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                            runner->Run(/*CalculatorContext=*/nullptr,
                                        MakeTensorSpan(input_tensors)));
    EXPECT_EQ(output_tensors.size(), 2);
    {
      auto read_view0 = output_tensors[0].GetCpuReadView();
      const int32_t* data0 = read_view0.buffer<int32_t>();
      for (int i = 0; i < kTestInputSize; ++i) {
        EXPECT_EQ(data0[i], 4);
      }
      auto read_view1 = output_tensors[1].GetCpuReadView();
      const int32_t* data1 = read_view1.buffer<int32_t>();
      for (int i = 0; i < kTestInputSize; ++i) {
        EXPECT_EQ(data1[i], 5);
      }
    }

    // Subsequent runs, the regular output remains the same, the feedback
    // output is incremented by 1 each time.
    for (int j = 0; j < 10; ++j) {
      MP_ASSERT_OK_AND_ASSIGN(output_tensors,
                              runner->Run(/*CalculatorContext=*/nullptr,
                                          MakeTensorSpan(input_tensors)));
      EXPECT_EQ(output_tensors.size(), 2);
      {
        auto read_view0 = output_tensors[0].GetCpuReadView();
        const int32_t* data0 = read_view0.buffer<int32_t>();
        for (int i = 0; i < kTestInputSize; ++i) {
          EXPECT_EQ(data0[i], 4);
        }
        auto read_view1 = output_tensors[1].GetCpuReadView();
        const int32_t* data1 = read_view1.buffer<int32_t>();
        for (int i = 0; i < kTestInputSize; ++i) {
          EXPECT_EQ(data1[i], 6 + j);
        }
      }
    }
  }
}

TEST(InferenceRunnterLitertTest, ShouldRunSuccessfullyWithFeedbackTensor) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  const std::string kFeedbackTensor1x2xint32Path =
      GetFeedbackTensorModelPath() + "FeedbackTensorModel1x2Int32.tflite";
  MP_ASSERT_OK_AND_ASSIGN(
      auto model,
      TfLiteModelLoader::LoadFromPath(*resources, kFeedbackTensor1x2xint32Path,
                                      /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  options.mutable_cpu();

  InferenceCalculatorOptions::InputOutputConfig input_output_config =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_names_map { tensor_names: "regular_int_input" }
        output_tensor_names_map {
          tensor_names: "regular_int_plus_feedback_int_output"
          tensor_names: "regular_int_incremented_output"
        }
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_int_output"
          to_input_tensor_name: "feedback_int_input"
        }
      )pb");

  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          InferenceRunnerLiteRt::Create(
                              std::move(model), options,
                              /*input_output_config=*/&input_output_config,
                              /*memory_manager=*/nullptr));

  constexpr const int32_t kTestInput[] = {1, 2};
  constexpr const size_t kTestInputSize =
      sizeof(kTestInput) / sizeof(kTestInput[0]);

  std::vector<Tensor> input_tensors;
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kInt32, Tensor::Shape({1, kTestInputSize}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[0].GetCpuWriteView();
    int32_t* data = write_view.buffer<int32_t>();
    memcpy(data, kTestInput, kTestInputSize * sizeof(int32_t));
  }

  MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                          runner->Run(/*CalculatorContext=*/nullptr,
                                      MakeTensorSpan(input_tensors)));
  EXPECT_EQ(output_tensors.size(), 2);
  {
    auto read_view0 = output_tensors[0].GetCpuReadView();
    const int32_t* data0 = read_view0.buffer<int32_t>();
    EXPECT_EQ(data0[0], 2);
    EXPECT_EQ(data0[1], 3);

    auto read_view1 = output_tensors[1].GetCpuReadView();
    const int32_t* data1 = read_view1.buffer<int32_t>();
    EXPECT_EQ(data1[0], 2);
    EXPECT_EQ(data1[1], 3);
  }
}

TEST(InferenceRunnterLitertTest, ShouldRunSuccessfullyWithFeedbackTensorFloat) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  const std::string kFeedbackTensor1x2xfloatPath =
      GetFeedbackTensorModelPath() + "FeedbackTensorModel1x2Float.tflite";
  MP_ASSERT_OK_AND_ASSIGN(
      auto model,
      TfLiteModelLoader::LoadFromPath(*resources, kFeedbackTensor1x2xfloatPath,
                                      /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  options.mutable_cpu();

  InferenceCalculatorOptions::InputOutputConfig input_output_config =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_names_map { tensor_names: "regular_float_input" }
        output_tensor_names_map {
          tensor_names: "regular_float_plus_feedback_float_output"
          tensor_names: "regular_float_incremented_output"
        }
        feedback_tensor_links {
          from_output_tensor_name: "feedback_incremented_float_output"
          to_input_tensor_name: "feedback_float_input"
        }
      )pb");

  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          InferenceRunnerLiteRt::Create(
                              std::move(model), options,
                              /*input_output_config=*/&input_output_config,
                              /*memory_manager=*/nullptr));

  constexpr const float kTestInput[] = {1.0f, 2.0f};
  constexpr const size_t kTestInputSize =
      sizeof(kTestInput) / sizeof(kTestInput[0]);

  std::vector<Tensor> input_tensors;
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInputSize}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[0].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput, kTestInputSize * sizeof(float));
  }

  {
    MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                            runner->Run(/*CalculatorContext=*/nullptr,
                                        MakeTensorSpan(input_tensors)));
    EXPECT_EQ(output_tensors.size(), 2);
    {
      auto read_view0 = output_tensors[0].GetCpuReadView();
      const float* data0 = read_view0.buffer<float>();
      EXPECT_FLOAT_EQ(data0[0], 2.0f);
      EXPECT_FLOAT_EQ(data0[1], 3.0f);

      auto read_view1 = output_tensors[1].GetCpuReadView();
      const float* data1 = read_view1.buffer<float>();
      EXPECT_FLOAT_EQ(data1[0], 2.0f);
      EXPECT_FLOAT_EQ(data1[1], 3.0f);
    }
  }
  // Second run, the regular output remains the same, the feedback output
  // is incremented by 1.
  {
    MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                            runner->Run(/*CalculatorContext=*/nullptr,
                                        MakeTensorSpan(input_tensors)));
    EXPECT_EQ(output_tensors.size(), 2);
    {
      auto read_view0 = output_tensors[0].GetCpuReadView();
      const float* data0 = read_view0.buffer<float>();
      EXPECT_FLOAT_EQ(data0[0], 2.0f);
      EXPECT_FLOAT_EQ(data0[1], 3.0f);

      auto read_view1 = output_tensors[1].GetCpuReadView();
      const float* data1 = read_view1.buffer<float>();
      EXPECT_FLOAT_EQ(data1[0], 3.0f);
      EXPECT_FLOAT_EQ(data1[1], 4.0f);
    }
  }
}

TEST(InferenceRunnterLitertTest,
     ShouldRunSuccessfullyWithHintFullyDelegatedToSingleDelegate) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(
                              *resources, GetTestFilePath(kModelFileName),
                              /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  options.mutable_cpu();
  options.mutable_gpu()->set_hint_fully_delegated_to_single_delegate(true);

  MP_ASSERT_OK_AND_ASSIGN(auto runner, InferenceRunnerLiteRt::Create(
                                           std::move(model), options,
                                           /*input_output_config=*/nullptr,
                                           /*memory_manager=*/nullptr));

  std::vector<Tensor> input_tensors;
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInput0Size}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[0].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput0Tensor, kTestInput0Size * sizeof(float));
  }
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInput1Size}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[1].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput1Tensor, kTestInput1Size * sizeof(float));
  }

  MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                          runner->Run(/*CalculatorContext=*/nullptr,
                                      MakeTensorSpan(input_tensors)));
  {
    auto read_view = output_tensors[0].GetCpuReadView();
    const float* data = read_view.buffer<float>();
    for (int i = 0; i < kTestOutputSize; ++i) {
      EXPECT_FLOAT_EQ(data[i], kTestOutputTensor[i]);
    }
  }
}

TEST(InferenceRunnterLitertTest, ShouldRunSuccessfullyWithWaitType) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(
                              *resources, GetTestFilePath(kModelFileName),
                              /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  options.mutable_cpu();
  options.mutable_gpu()->set_wait_type(
      InferenceCalculatorOptions::Delegate::LiteRt::Gpu::WAIT_TYPE_ACTIVE);

  MP_ASSERT_OK_AND_ASSIGN(auto runner, InferenceRunnerLiteRt::Create(
                                           std::move(model), options,
                                           /*input_output_config=*/nullptr,
                                           /*memory_manager=*/nullptr));

  std::vector<Tensor> input_tensors;
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInput0Size}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[0].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput0Tensor, kTestInput0Size * sizeof(float));
  }
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInput1Size}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[1].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput1Tensor, kTestInput1Size * sizeof(float));
  }

  MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                          runner->Run(/*CalculatorContext=*/nullptr,
                                      MakeTensorSpan(input_tensors)));
  {
    auto read_view = output_tensors[0].GetCpuReadView();
    const float* data = read_view.buffer<float>();
    for (int i = 0; i < kTestOutputSize; ++i) {
      EXPECT_FLOAT_EQ(data[i], kTestOutputTensor[i]);
    }
  }
}

TEST(InferenceRunnterLitertTest, ShouldRunSuccessfullyWithMetalHelper) {
#if defined(__APPLE__)
  std::unique_ptr<Resources> resources = CreateDefaultResources();

  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(
                              *resources, GetTestFilePath(kModelFileName),
                              /*try_mmap_model=*/false));

  InferenceCalculatorOptions::Delegate::LiteRt options;
  options.mutable_gpu();

  MP_ASSERT_OK_AND_ASSIGN(auto gpu_resources, GpuResources::Create());
  MPPMetalHelper* metal_helper =
      [[MPPMetalHelper alloc] initWithGpuResources:gpu_resources.get()];

  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          InferenceRunnerLiteRt::Create(
                              std::move(model), options,
                              /*input_output_config=*/nullptr,
                              /*memory_manager=*/nullptr,
                              /*gl_context=*/nullptr,
                              /*litert_options=*/absl::nullopt,
                              /*metal_helper=*/(__bridge void*)metal_helper));

  std::vector<Tensor> input_tensors;
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInput0Size}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[0].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput0Tensor, kTestInput0Size * sizeof(float));
  }
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInput1Size}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/litert::kHostMemoryBufferAlignment));
  {
    auto write_view = input_tensors[1].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput1Tensor, kTestInput1Size * sizeof(float));
  }

  MP_ASSERT_OK_AND_ASSIGN(auto output_tensors,
                          runner->Run(/*CalculatorContext=*/nullptr,
                                      MakeTensorSpan(input_tensors)));
  {
    auto read_view = output_tensors[0].GetCpuReadView();
    const float* data = read_view.buffer<float>();
    for (int i = 0; i < kTestOutputSize; ++i) {
      EXPECT_FLOAT_EQ(data[i], kTestOutputTensor[i]);
    }
  }
#else
  GTEST_SKIP() << "Skipping test because Metal is not enabled.";
#endif  // defined(__APPLE__)
}

}  // namespace
}  // namespace mediapipe
