#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "litert/c/litert_platform_support.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/tflite_test_data_embed.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/framework/resources_service.h"
#include "mediapipe/framework/tool/sink.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

// Select model for testing on different target devices.
ABSL_FLAG(std::string, model_filename, kModelFileName,
          "Model filename to use for testing.");

namespace mediapipe::api2 {
namespace {

using ::mediapipe::Packet;
using ::mediapipe::TfLiteModelLoader;
using ::mediapipe::api2::builder::Stream;

class ModelResources : public Resources {
 public:
  absl::StatusOr<std::unique_ptr<Resource>> Get(
      absl::string_view resource_id, const Options& options) const override {
    std::string model;
    const std::string& model_filename = absl::GetFlag(FLAGS_model_filename);
    const FileToc* const graph_toc = tflite_test_data_embed_create();
    for (const FileToc* p = graph_toc; p->name != nullptr; ++p) {
      if (strcmp(p->name, model_filename.c_str()) == 0) {
        model = std::string(p->data, p->size);
        break;
      }
    }
    RET_CHECK(!model.empty()) << "Failed to load model: " << model_filename;
    return MakeStringResource(std::move(model));
  }
};

std::vector<Tensor> CreateInputTensors() {
  std::vector<Tensor> input_tensors;
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInput0Size}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/LITERT_HOST_MEMORY_BUFFER_ALIGNMENT));
  {
    auto write_view = input_tensors[0].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput0Tensor, kTestInput0Size * sizeof(float));
  }
  input_tensors.push_back(
      Tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, kTestInput1Size}),
             /*memory_manager=*/nullptr,
             /*memory_alignment=*/LITERT_HOST_MEMORY_BUFFER_ALIGNMENT));
  {
    auto write_view = input_tensors[1].GetCpuWriteView();
    float* data = write_view.buffer<float>();
    memcpy(data, kTestInput1Tensor, kTestInput1Size * sizeof(float));
  }
  return input_tensors;
}

void ConfirmOutputTensor(const Tensor& tensor) {
  auto read_view = tensor.GetCpuReadView();
  const float* data = read_view.buffer<float>();
  EXPECT_EQ(tensor.shape().num_elements(), kTestOutputSize);
  for (int i = 0; i < kTestOutputSize; ++i) {
    EXPECT_FLOAT_EQ(data[i], kTestOutputTensor[i]);
  }
}

CalculatorGraphConfig BuildTestGraph(
    const mediapipe::InferenceCalculatorOptions& options) {
  mediapipe::api2::builder::Graph graph_builder;
  Stream<Tensor> input0 = graph_builder.In(0).SetName("input0").Cast<Tensor>();
  Stream<Tensor> input1 = graph_builder.In(1).SetName("input1").Cast<Tensor>();
  auto& inference_calculator = graph_builder.AddNode("InferenceCalculator");
  input0 >> inference_calculator.In("TENSOR")[0];
  input1 >> inference_calculator.In("TENSOR")[1];
  inference_calculator.GetOptions<InferenceCalculatorOptions>() = options;
  inference_calculator.Out("TENSORS").SetName("output") >> graph_builder.Out(0);
  return graph_builder.GetConfig();
}

absl::StatusOr<std::vector<Packet>> SetUpGraphAndRun(
    const mediapipe::InferenceCalculatorOptions& options,
    std::vector<Tensor> input_tensors) {
  auto graph_config = BuildTestGraph(options);
  std::vector<Packet> result_packets;
  tool::AddVectorSink("output", &graph_config, &result_packets);

  CalculatorGraph graph;
  std::shared_ptr<Resources> resources = std::make_unique<ModelResources>();
  MP_RETURN_IF_ERROR(
      graph.SetServiceObject(kResourcesService, std::move(resources)));
  MP_RETURN_IF_ERROR(graph.Initialize(graph_config));
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  for (int i = 0; i < input_tensors.size(); ++i) {
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        absl::StrCat("input", i),
        mediapipe::MakePacket<Tensor>(std::move(input_tensors[i]))
            .At(mediapipe::Timestamp(1))));
  }
  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return std::move(result_packets);
}

TEST(InferenceCalculatorLiteRtTest, ShouldPerformLiteRtInference) {
  const std::string& model_filename = absl::GetFlag(FLAGS_model_filename);
  mediapipe::InferenceCalculatorOptions options;
  options.set_model_path(model_filename);
  auto* litert_delegate = options.mutable_delegate()->mutable_litert();
  litert_delegate->mutable_cpu();

  MP_ASSERT_OK_AND_ASSIGN(auto result_packets,
                          SetUpGraphAndRun(options, CreateInputTensors()));
  ASSERT_EQ(result_packets.size(), 1);
  const auto& output_tensor = result_packets[0].Get<std::vector<Tensor>>();
  ConfirmOutputTensor(output_tensor[0]);
}

TEST(InferenceCalculatorLiteRtTest,
     ShouldPerformLiteRtInferenceWithSlowConsistentArithmetic) {
  const std::string& model_filename = absl::GetFlag(FLAGS_model_filename);
  mediapipe::InferenceCalculatorOptions options;
  options.set_model_path(model_filename);
  auto* litert_delegate = options.mutable_delegate()->mutable_litert();
  litert_delegate->mutable_cpu()->mutable_xnnpack()->set_flags(
      TFLITE_XNNPACK_DELEGATE_FLAG_SLOW_CONSISTENT_ARITHMETIC);

  MP_ASSERT_OK_AND_ASSIGN(auto result_packets,
                          SetUpGraphAndRun(options, CreateInputTensors()));
  ASSERT_EQ(result_packets.size(), 1);
  const auto& output_tensor = result_packets[0].Get<std::vector<Tensor>>();
  ConfirmOutputTensor(output_tensor[0]);
}

TEST(InferenceCalculatorLiteRtTest,
     ShouldPerformLiteRtInferenceWithEnableSubgraphReshaping) {
  const std::string& model_filename = absl::GetFlag(FLAGS_model_filename);
  mediapipe::InferenceCalculatorOptions options;
  options.set_model_path(model_filename);
  auto* litert_delegate = options.mutable_delegate()->mutable_litert();
  litert_delegate->mutable_cpu()->mutable_xnnpack()->set_flags(
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING);

  MP_ASSERT_OK_AND_ASSIGN(auto result_packets,
                          SetUpGraphAndRun(options, CreateInputTensors()));
  ASSERT_EQ(result_packets.size(), 1);
  const auto& output_tensor = result_packets[0].Get<std::vector<Tensor>>();
  ConfirmOutputTensor(output_tensor[0]);
}

TEST(InferenceCalculatorLiteRtTest,
     ShouldPerformLiteRtInferenceWithUseLatestOperators) {
  const std::string& model_filename = absl::GetFlag(FLAGS_model_filename);
  mediapipe::InferenceCalculatorOptions options;
  options.set_model_path(model_filename);
  auto* litert_delegate = options.mutable_delegate()->mutable_litert();
  litert_delegate->mutable_cpu()->mutable_xnnpack()->set_flags(
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS);

  MP_ASSERT_OK_AND_ASSIGN(auto result_packets,
                          SetUpGraphAndRun(options, CreateInputTensors()));
  ASSERT_EQ(result_packets.size(), 1);
  const auto& output_tensor = result_packets[0].Get<std::vector<Tensor>>();
  ConfirmOutputTensor(output_tensor[0]);
}

TEST(InferenceCalculatorLiteRtTest, ShouldPerformLiteRtMetalInference) {
  if (!LiteRtHasMetalSupport()) {
    GTEST_SKIP() << "OpenGL buffers are not supported on this platform";
  }

  const std::string& model_filename = absl::GetFlag(FLAGS_model_filename);
  mediapipe::InferenceCalculatorOptions options;
  options.set_model_path(model_filename);
  auto* litert_delegate = options.mutable_delegate()->mutable_litert();
  litert_delegate->mutable_gpu();

  MP_ASSERT_OK_AND_ASSIGN(auto result_packets,
                          SetUpGraphAndRun(options, CreateInputTensors()));

  ASSERT_EQ(result_packets.size(), 1);
  const auto& output_tensor = result_packets[0].Get<std::vector<Tensor>>();
  ConfirmOutputTensor(output_tensor[0]);
}

bool HasMediaPipeAhwbSupport() {
#if MEDIAPIPE_TENSOR_USE_AHWB
  return true;
#else
  return false;
#endif
}

TEST(InferenceCalculatorLiteRtTest, ShouldPerformLiteRtInferenceWithNpuAsync) {
  if (!HasMediaPipeAhwbSupport()) {
    GTEST_SKIP()
        << "Skipping test because MediaPipe AHWB support is not enabled.";
  }
  const std::string& model_filename = absl::GetFlag(FLAGS_model_filename);
  mediapipe::InferenceCalculatorOptions options;
  options.set_model_path(model_filename);
  auto* litert_delegate = options.mutable_delegate()->mutable_litert();
  litert_delegate->mutable_npu();
  litert_delegate->mutable_npu()->set_dispatch_library_path("/data/local/tmp/");
  litert_delegate->mutable_npu()->set_compiler_plugin_library_path(
      "/data/local/tmp/");
  litert_delegate->set_run_async(true);

  MP_ASSERT_OK_AND_ASSIGN(auto result_packets,
                          SetUpGraphAndRun(options, CreateInputTensors()));
  ASSERT_EQ(result_packets.size(), 1);
  const auto& output_tensor = result_packets[0].Get<std::vector<Tensor>>();
  ConfirmOutputTensor(output_tensor[0]);
}

TEST(InferenceCalculatorLiteRtTest, LiteRtInferenceWithGpu) {
  if (!LiteRtHasOpenClSupport()) {
    GTEST_SKIP()
        << "Skipping test because OpenCL and/or OpenGL is not supported.";
  }
  const std::string& model_filename = absl::GetFlag(FLAGS_model_filename);
  mediapipe::InferenceCalculatorOptions options;
  options.set_model_path(model_filename);
  auto* litert_delegate = options.mutable_delegate()->mutable_litert();
  litert_delegate->mutable_gpu();

  MP_ASSERT_OK_AND_ASSIGN(auto result_packets,
                          SetUpGraphAndRun(options, CreateInputTensors()));
  ASSERT_EQ(result_packets.size(), 1);
  const auto& output_tensor = result_packets[0].Get<std::vector<Tensor>>();
  ConfirmOutputTensor(output_tensor[0]);
}

TEST(InferenceCalculatorLiteRtTest, LiteRtInferenceWithMetalGpu) {
  if (!LiteRtHasMetalSupport()) {
    GTEST_SKIP() << "Skipping test because Metal is not supported.";
  }
  const std::string& model_filename = absl::GetFlag(FLAGS_model_filename);
  mediapipe::InferenceCalculatorOptions options;
  options.set_model_path(model_filename);
  auto litert_delegate = options.mutable_delegate()->mutable_litert();
  litert_delegate->mutable_gpu();

  MP_ASSERT_OK_AND_ASSIGN(auto result_packets,
                          SetUpGraphAndRun(options, CreateInputTensors()));
  ASSERT_EQ(result_packets.size(), 1);
  const auto& output_tensor = result_packets[0].Get<std::vector<Tensor>>();
  ConfirmOutputTensor(output_tensor[0]);
}

TEST(InferenceCalculatorLiteRtTest, LiteRtInferenceWithMetalGpuZeroCopy) {
  if (!LiteRtHasMetalSupport()) {
    GTEST_SKIP() << "Skipping test because Metal is not supported.";
  }
  const std::string& model_filename = absl::GetFlag(FLAGS_model_filename);
  mediapipe::InferenceCalculatorOptions options;
  options.set_model_path(model_filename);
  auto litert_delegate = options.mutable_delegate()->mutable_litert();
  litert_delegate->mutable_gpu();

  MP_ASSERT_OK_AND_ASSIGN(auto result_packets,
                          SetUpGraphAndRun(options, CreateInputTensors()));
  ASSERT_EQ(result_packets.size(), 1);
  const auto& output_tensors = result_packets[0].Get<std::vector<Tensor>>();
  EXPECT_TRUE(output_tensors[0].ready_as_metal_buffer());
  EXPECT_FALSE(output_tensors[0].ready_on_cpu());
  ConfirmOutputTensor(output_tensors[0]);
}

}  // namespace
}  // namespace mediapipe::api2
