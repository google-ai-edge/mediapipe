#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "research/aimatter/api/utils/embedded_files.h"

namespace mediapipe::api2 {
namespace {

using ::mediapipe::Packet;
using ::mediapipe::TfLiteModelLoader;
using ::mediapipe::api2::builder::Stream;
using ::research::aimatter::api::EmbeddedFiles;
using ::testing::FloatNear;
using ::testing::Pointwise;

constexpr const char kFloat32ModelFile[] = "embedded:1x3_square_float32.tflite";

std::string TestSrcDir() {
  const char* from_env = getenv("TEST_TMPDIR");
  if (from_env) {
    return from_env;
  } else {
    // Local Android run without TEST_TMPDIR environment.
    return "/data/local/tmp";
  }
}

absl::StatusOr<std::string> CreateFileFromEmbeddedData(
    const absl::string_view filename) {
  auto file_span = EmbeddedFiles::GetOrDie(filename);
  const std::string filedata(file_span.data(), file_span.size());
  const std::string test_dir = file::JoinPath(TestSrcDir(), "data");
  std::string local_file_path = file::JoinPath(test_dir, filename);
  if (!file::Exists(test_dir).ok()) {
    MP_RETURN_IF_ERROR(file::RecursivelyCreateDir(test_dir));
  }
  MP_RETURN_IF_ERROR(file::SetContents(local_file_path, filedata));
  return local_file_path;
}

Tensor CreateInputTensors(std::vector<float> values) {
  std::vector<int> dims({1});
  dims.push_back(values.size());
  Tensor tensor(Tensor::ElementType::kFloat32, dims);
  {
    auto input_tensor_view = tensor.GetCpuWriteView();
    float* const input_buffer = input_tensor_view.buffer<float>();
    EXPECT_EQ(values.size(), tensor.shape().num_elements());
    std::memcpy(input_buffer, values.data(), values.size() * sizeof(float));
  }
  return tensor;
}

std::vector<float> GetOutputTensorValues(const Tensor& tensor) {
  std::vector<float> values;
  values.reserve(tensor.shape().num_elements());
  {
    auto output_tensor_view = tensor.GetCpuReadView();
    const float* const output_buffer = output_tensor_view.buffer<float>();
    for (int i = 0; i < tensor.shape().num_elements(); ++i) {
      values.push_back(output_buffer[i]);
    }
  }
  return values;
}

CalculatorGraphConfig BuildTestGraph(
    absl::string_view model_path,
    const mediapipe::InferenceCalculatorOptions::Delegate& delegate_config) {
  mediapipe::api2::builder::Graph graph_builder;
  Stream<Tensor> input =
      graph_builder.In("TENSOR").SetName("input").Cast<Tensor>();
  auto& inference_calculator = graph_builder.AddNode("InferenceCalculator");
  input >> inference_calculator.In("TENSOR")[0];
  InferenceCalculatorOptions inference_options;
  inference_options.set_model_path(model_path);
  *inference_options.mutable_delegate() = delegate_config;
  inference_calculator.GetOptions<InferenceCalculatorOptions>() =
      inference_options;
  inference_calculator.Out("TENSORS").SetName("output") >> graph_builder.Out(0);
  return graph_builder.GetConfig();
}

absl::StatusOr<std::vector<Packet>> SetUpGraphAndRun(
    const mediapipe::InferenceCalculatorOptions::Delegate& delegate_config,
    absl::string_view model_path, Tensor&& input_tensor) {
  auto graph_config = BuildTestGraph(model_path, delegate_config);
  CalculatorGraph graph;
  std::vector<Packet> result_packets;
  tool::AddVectorSink("output", &graph_config, &result_packets);
  MP_RETURN_IF_ERROR(graph.Initialize(graph_config));
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      "input", mediapipe::MakePacket<Tensor>(std::move(input_tensor))
                   .At(mediapipe::Timestamp(1))));
  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return std::move(result_packets);
}

TEST(InferenceCalculatorQnnTest, ShouldExecuteQnnInference) {
  MP_ASSERT_OK_AND_ASSIGN(const std::string model_path,
                          CreateFileFromEmbeddedData(kFloat32ModelFile));
  MP_ASSERT_OK_AND_ASSIGN(auto model,
                          TfLiteModelLoader::LoadFromPath(model_path));
  mediapipe::InferenceCalculatorOptions::Delegate delegate_config;
  delegate_config.mutable_qnn()->set_backend(
      InferenceCalculatorOptions::Delegate::Qnn::HTP);

  Tensor input_tensor = CreateInputTensors({1.0f, 2.0f, 3.0f});
  std::vector<float> expected_output_values = {1.0f, 4.0f, 9.0f};

  MP_ASSERT_OK_AND_ASSIGN(
      auto result_packets,
      SetUpGraphAndRun(delegate_config, model_path, std::move(input_tensor)));
  ASSERT_EQ(result_packets.size(), 1);
  const auto& output_tensor = result_packets[0].Get<std::vector<Tensor>>();
  ASSERT_EQ(output_tensor.size(), 1);
  EXPECT_THAT(GetOutputTensorValues(output_tensor[0]),
              Pointwise(FloatNear(0.01), expected_output_values));
}

}  // namespace
}  // namespace mediapipe::api2
