#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_state.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gpu_service.h"

namespace mediapipe {
namespace api2 {
namespace {

using ::mediapipe::MakePacket;
using ::mediapipe::Packet;
using ::mediapipe::Tensor;
using ::mediapipe::api2::builder::Stream;
using ::testing::HasSubstr;

constexpr const char kInt32ModelFile[] =
    "mediapipe/calculators/tensor/testdata/"
    "1x3_square_int32.tflite";

using TestService = GraphService<int>;
inline constexpr TestService kTestService(
    "test_service", GraphServiceBase::kAllowDefaultInitialization);

absl::StatusOr<Tensor> Create1x3IntTensor(std::vector<int> values) {
  std::vector<int> dims = {1, 3};
  Tensor tensor(Tensor::ElementType::kInt32, Tensor::Shape(dims));
  auto write_view = tensor.GetCpuWriteView();
  RET_CHECK_EQ(values.size(), 3);
  int* tensor_ptr = write_view.buffer<int>();
  for (int i = 0; i < 3; ++i) {
    tensor_ptr[i] = values[i];
  }
  return tensor;
}

class NestedGraphCalculator : public Node {
 public:
  static constexpr Input<Tensor>::Multiple kInput{"TENSORS"};
  static constexpr Output<Packet>::Multiple kOutput{"TENSORS"};
  MEDIAPIPE_NODE_CONTRACT(kInput, kOutput);

  absl::Status Process(CalculatorContext* cc) override {
    const auto& input_tensor = kInput(cc)[0];

    // Create a calculator-nested graph from the current CalculatorContext.
    CalculatorGraph graph(cc);

    CalculatorGraphConfig graph_config =
        ParseTextProtoOrDie<CalculatorGraphConfig>(absl::StrReplaceAll(
            R"pb(
              input_stream: "input"
              output_stream: "output"

              executor { name: "" type: "ApplicationThreadExecutor" }

              node {
                calculator: "InferenceCalculator"
                input_stream: "TENSOR:0:input"
                output_stream: "TENSOR:0:output"
                options {
                  [mediapipe.InferenceCalculatorOptions.ext] {
                    model_path: "$model"
                    delegate {}  # empty delegate message enables CPU inference.
                  }
                }
              }
            )pb",
            {{"$model", kInt32ModelFile}}));

    std::vector<Packet> output_packets;
    tool::AddVectorSink("output", &graph_config, &output_packets);

    MP_EXPECT_OK(graph.Initialize(graph_config));
    std::map<std::string, Packet> side_packets;
    MP_EXPECT_OK(graph.StartRun(side_packets));
    MP_EXPECT_OK(graph.AddPacketToInputStream("input", input_tensor));
    MP_EXPECT_OK(graph.CloseAllInputStreams());
    MP_EXPECT_OK(graph.WaitUntilDone());
    EXPECT_EQ(output_packets.size(), 1);
    kOutput(cc)[0].Send(std::move(output_packets[0]));
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(NestedGraphCalculator);

TEST(NestedCalculatorGraphTest, ExecutedNestedGraphWithInferenceCalculator) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        input_stream: "input"
        output_stream: "output"
        node {
          calculator: "NestedGraphCalculator"
          input_stream: "TENSORS:input"
          output_stream: "TENSORS:output"
        })pb");

  CalculatorGraph graph;

  // Start graph and configure a sink.
  std::vector<Packet> result_packets;
  tool::AddVectorSink("output", &graph_config, &result_packets);

  MP_ASSERT_OK(graph.Initialize(graph_config));

  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK_AND_ASSIGN(Tensor input_tensor, Create1x3IntTensor({1, 2, 3}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", mediapipe::MakePacket<Tensor>(std::move(input_tensor))
                   .At(mediapipe::Timestamp(1))));

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_EQ(result_packets.size(), 1);
  const Tensor& result_tensor = result_packets[0].Get<Packet>().Get<Tensor>();
  EXPECT_EQ(result_tensor.shape().num_elements(), 3);
  {
    const auto view = result_tensor.GetCpuReadView();
    const int* data = view.buffer<int>();
    EXPECT_EQ(data[0], 1 * 1);
    EXPECT_EQ(data[1], 2 * 2);
    EXPECT_EQ(data[2], 3 * 3);
  }
}

template <bool UseService>
class ServiceRequestCalculator : public Node {
 public:
  static constexpr Input<int> kInput{"TICK"};
  MEDIAPIPE_NODE_CONTRACT(kInput);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    if (UseService) {
      cc->UseService(kTestService);
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    auto service = cc->Service(kTestService);
    EXPECT_EQ(service.IsAvailable(), UseService);
    return absl::OkStatus();
  }
};

using ServiceRequestCalculatorWithUseServiceRequest =
    ServiceRequestCalculator<true>;
using ServiceRequestCalculatorWithoutUseServiceRequest =
    ServiceRequestCalculator<false>;
REGISTER_CALCULATOR(ServiceRequestCalculatorWithUseServiceRequest);
REGISTER_CALCULATOR(ServiceRequestCalculatorWithoutUseServiceRequest);

class NestedGraphServiceTestCalculator : public Node {
 public:
  static constexpr Input<int> kTestValue{"TEST_VALUE"};
  static constexpr Output<absl::Status> kStartupError{"STARTUP_ERROR"};
  MEDIAPIPE_NODE_CONTRACT(kTestValue, kStartupError);

  absl::Status Process(CalculatorContext* cc) override {
    const auto& test_value = kTestValue(cc).Get();

    CalculatorGraph graph(cc);

    if (test_value > 0) {
      // Check that the service is available on the current graph.
      const auto service = graph.GetServiceObject(kTestService);
      EXPECT_NE(service, nullptr);
      EXPECT_EQ(*service, test_value);
    }

    // Test that inherited services must be still request on calculators on the
    // sub graph.
    auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
        R"pb(
          input_stream: "tick"
          node {
            calculator: "ServiceRequestCalculatorWithUseServiceRequest"
            input_stream: "TICK:tick"
          }
          node {
            calculator: "ServiceRequestCalculatorWithoutUseServiceRequest"
            input_stream: "TICK:tick"
          }
        )pb");

    MP_RETURN_IF_ERROR(graph.Initialize(graph_config));
    const auto status = graph.StartRun({});
    kStartupError(cc).Send(status);
    if (!status.ok()) {
      // Exit early if the graph failed to start.
      return absl::OkStatus();
    }
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream("tick", kTestValue(cc)));
    MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
    MP_RETURN_IF_ERROR(graph.WaitUntilDone());
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(NestedGraphServiceTestCalculator);

TEST(NestedCalculatorGraphTest, TestNestedGraphServiceInheriting) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        input_stream: "test_value"
        output_stream: "startup_error"
        node {
          calculator: "NestedGraphServiceTestCalculator"
          input_stream: "TEST_VALUE:test_value"
          output_stream: "STARTUP_ERROR:startup_error"
        })pb");
  CalculatorGraph graph;

  constexpr int kTestValue = 123;
  MP_EXPECT_OK(
      graph.SetServiceObject(kTestService, std::make_shared<int>(kTestValue)));
  std::vector<Packet> result_status;
  tool::AddVectorSink("startup_error", &graph_config, &result_status);
  MP_ASSERT_OK(graph.Initialize(graph_config));

  MP_ASSERT_OK(graph.StartRun({}));

  const mediapipe::Timestamp timestamp(1);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "test_value", mediapipe::MakePacket<int>(kTestValue).At(timestamp)));
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  MP_EXPECT_OK(result_status[0].Get<absl::Status>());
}

TEST(NestedCalculatorGraphTest, NestedGraphsCannotRegisterNewServices) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        input_stream: "test_value"
        output_stream: "startup_error"
        node {
          calculator: "NestedGraphServiceTestCalculator"
          input_stream: "TEST_VALUE:test_value"
          output_stream: "STARTUP_ERROR:startup_error"
        })pb");
  CalculatorGraph graph;
  std::vector<Packet> result_status;
  tool::AddVectorSink("startup_error", &graph_config, &result_status);
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "test_value",
      mediapipe::MakePacket<int>(
          -1  // disable service check in  NestedGraphServiceTestCalculator
          )
          .At(mediapipe::Timestamp(1))));
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_THAT(
      result_status[0].Get<absl::Status>(),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Service default initialization is disallowed.")));
}

class GpuServiceRequestingCalculator : public Node {
 public:
  static constexpr Input<int> kTick{"TICK"};
  MEDIAPIPE_NODE_CONTRACT(kTick);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    cc->UseService(kGpuService);
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    auto service = cc->Service(kGpuService);
    EXPECT_TRUE(service.IsAvailable());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(GpuServiceRequestingCalculator);

class NestedGraphWithGpuServiceRequestingCalculator : public Node {
 public:
  static constexpr Input<int> kTick{"TICK"};
  MEDIAPIPE_NODE_CONTRACT(kTick);

  absl::Status Open(CalculatorContext* cc) override {
    CalculatorGraph graph(cc);

    const auto service = graph.GetServiceObject(kGpuService);
    EXPECT_EQ(service, nullptr);

    // Test that inherited services must be still request on calculators on the
    // sub graph.
    auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
        R"pb(
          input_stream: "tick"
          node {
            calculator: "GpuServiceRequestingCalculator"
            input_stream: "TICK:tick"
          }
        )pb");

    MP_RETURN_IF_ERROR(graph.Initialize(graph_config));
    EXPECT_THAT(
        graph.StartRun({}),
        StatusIs(absl::StatusCode::kInternal,
                 HasSubstr("Service default initialization is disallowed.")));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(NestedGraphWithGpuServiceRequestingCalculator);

TEST(NestedCalculatorGraphTest, NestedGraphWithGpuServiceRequestShouldFail) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        input_stream: "input"
        input_stream: "tick"
        node {
          calculator: "NestedGraphWithGpuServiceRequestingCalculator"
          input_stream: "TICK:tick"
        })pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "tick", mediapipe::MakePacket<int>(0).At(mediapipe::Timestamp(1))));
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

}  // namespace
}  // namespace api2
}  // namespace mediapipe
