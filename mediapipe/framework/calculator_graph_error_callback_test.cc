#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Node;
using ::mediapipe::api2::Output;
using ::testing::HasSubstr;

namespace {

constexpr absl::string_view kErrorMsgFromProcess =
    "Error from Calculator::Process.";

class ProcessFnErrorCalculator : public Node {
 public:
  static constexpr Input<int> kIn{"IN"};
  static constexpr Output<int> kOut{"OUT"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Process(CalculatorContext* cc) override {
    return absl::InternalError(kErrorMsgFromProcess);
  }
};
MEDIAPIPE_REGISTER_NODE(ProcessFnErrorCalculator);

TEST(CalculatorGraphAsyncErrorsTest, ErrorCallbackReceivesProcessErrors) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: 'input'
    node {
      calculator: "ProcessFnErrorCalculator"
      input_stream: 'IN:input'
      output_stream: 'OUT:output'
    }
  )pb");

  CalculatorGraph graph;

  bool is_error_received = false;
  absl::Status output_error;
  absl::Mutex m;
  auto error_callback_fn = [&graph, &m, &output_error,
                            &is_error_received](absl::Status error) {
    EXPECT_TRUE(graph.HasError());

    absl::MutexLock lock(&m);
    output_error = std::move(error);
    is_error_received = true;
  };

  MP_ASSERT_OK(graph.SetErrorCallback(error_callback_fn));
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", MakePacket<int>(0).At(Timestamp(10))));

  {
    absl::MutexLock lock(&m);
    ASSERT_TRUE(m.AwaitWithTimeout(absl::Condition(&is_error_received),
                                   absl::Seconds(1)));
  }
  EXPECT_THAT(output_error, StatusIs(absl::StatusCode::kInternal,
                                     HasSubstr(kErrorMsgFromProcess)));

  EXPECT_THAT(graph.WaitUntilIdle(), StatusIs(absl::StatusCode::kInternal,
                                              HasSubstr(kErrorMsgFromProcess)));
}

constexpr absl::string_view kErrorMsgFromOpen = "Error from Calculator::Open.";

class OpenFnErrorCalculator : public Node {
 public:
  static constexpr Input<int> kIn{"IN"};
  static constexpr Output<int> kOut{"OUT"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  absl::Status Open(CalculatorContext* cc) override {
    return absl::InternalError(kErrorMsgFromOpen);
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};
MEDIAPIPE_REGISTER_NODE(OpenFnErrorCalculator);

TEST(CalculatorGraphAsyncErrorsTest, ErrorCallbackReceivesOpenErrors) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: 'input'
    node {
      calculator: "OpenFnErrorCalculator"
      input_stream: 'IN:input'
      output_stream: 'OUT:output'
    }
  )pb");

  CalculatorGraph graph;

  bool is_error_received = false;
  absl::Status output_error;
  absl::Mutex m;
  auto error_callback_fn = [&graph, &m, &output_error,
                            &is_error_received](absl::Status error) {
    EXPECT_TRUE(graph.HasError());

    absl::MutexLock lock(&m);
    output_error = std::move(error);
    is_error_received = true;
  };

  MP_ASSERT_OK(graph.SetErrorCallback(error_callback_fn));
  MP_ASSERT_OK(graph.Initialize(graph_config, {}));
  MP_ASSERT_OK(graph.StartRun({}));

  {
    absl::MutexLock lock(&m);
    ASSERT_TRUE(m.AwaitWithTimeout(absl::Condition(&is_error_received),
                                   absl::Seconds(1)));
  }
  EXPECT_THAT(output_error, StatusIs(absl::StatusCode::kInternal,
                                     HasSubstr(kErrorMsgFromOpen)));

  EXPECT_THAT(graph.WaitUntilIdle(), StatusIs(absl::StatusCode::kInternal,
                                              HasSubstr(kErrorMsgFromOpen)));
}

TEST(CalculatorGraphAsyncErrorsTest, ErrorCallbackMustBeSetBeforeInit) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: 'input'
    node {
      calculator: "OpenFnErrorCalculator"
      input_stream: 'IN:input'
      output_stream: 'OUT:output'
    }
  )pb");

  CalculatorGraph graph;
  ABSL_CHECK_OK(graph.Initialize(graph_config, {}));
  EXPECT_THAT(graph.SetErrorCallback({}),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace mediapipe
