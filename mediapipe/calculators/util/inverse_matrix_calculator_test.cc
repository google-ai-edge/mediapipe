#include "mediapipe/calculators/util/inverse_matrix_calculator.h"

#include <array>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

void RunTest(const std::array<float, 16>& matrix,
             const std::array<float, 16>& expected_inverse_matrix) {
  auto graph_config = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(
      R"pb(
        input_stream: "matrix"
        node {
          calculator: "InverseMatrixCalculator"
          input_stream: "MATRIX:matrix"
          output_stream: "MATRIX:inverse_matrix"
        }
      )pb");

  std::vector<Packet> output_packets;
  tool::AddVectorSink("inverse_matrix", &graph_config, &output_packets);

  // Run the graph.
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "matrix",
      MakePacket<std::array<float, 16>>(std::move(matrix)).At(Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_THAT(output_packets, testing::SizeIs(1));

  const auto& inverse_matrix = output_packets[0].Get<std::array<float, 16>>();

  EXPECT_THAT(
      inverse_matrix,
      testing::Pointwise(testing::FloatEq(),
                         absl::MakeSpan(expected_inverse_matrix.data(),
                                        expected_inverse_matrix.size())));

  // Fully close graph at end, otherwise calculator+tensors are destroyed
  // after calling WaitUntilDone().
  MP_ASSERT_OK(graph.CloseInputStream("matrix"));
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST(InverseMatrixCalculatorTest, Identity) {
  // clang-format off
  std::array<float, 16> matrix = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
  };
  std::array<float, 16> expected_inverse_matrix = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
  };
  // clang-format on
  RunTest(matrix, expected_inverse_matrix);
}

TEST(InverseMatrixCalculatorTest, Translation) {
  // clang-format off
  std::array<float, 16> matrix = {
    1.0f, 0.0f, 0.0f,  2.0f,
    0.0f, 1.0f, 0.0f, -5.0f,
    0.0f, 0.0f, 1.0f,  0.0f,
    0.0f, 0.0f, 0.0f,  1.0f,
  };
  std::array<float, 16> expected_inverse_matrix = {
    1.0f, 0.0f, 0.0f, -2.0f,
    0.0f, 1.0f, 0.0f,  5.0f,
    0.0f, 0.0f, 1.0f,  0.0f,
    0.0f, 0.0f, 0.0f,  1.0f,
  };
  // clang-format on
  RunTest(matrix, expected_inverse_matrix);
}

TEST(InverseMatrixCalculatorTest, Scale) {
  // clang-format off
  std::array<float, 16> matrix = {
    5.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 2.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
  };
  std::array<float, 16> expected_inverse_matrix = {
    0.2f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.5f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
  };
  // clang-format on
  RunTest(matrix, expected_inverse_matrix);
}

TEST(InverseMatrixCalculatorTest, Rotation90) {
  // clang-format off
  std::array<float, 16> matrix = {
    0.0f, -1.0f, 0.0f, 0.0f,
    1.0f,  0.0f, 0.0f, 0.0f,
    0.0f,  0.0f, 1.0f, 0.0f,
    0.0f,  0.0f, 0.0f, 1.0f,
  };
  std::array<float, 16> expected_inverse_matrix = {
     0.0f, 1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f, 0.0f,
     0.0f, 0.0f, 1.0f, 0.0f,
     0.0f, 0.0f, 0.0f, 1.0f,
  };
  // clang-format on
  RunTest(matrix, expected_inverse_matrix);
}

TEST(InverseMatrixCalculatorTest, CheckPrecision) {
  // clang-format off
  std::array<float, 16> matrix = {
    0.00001f,  0.0f,      0.0f, 0.0f,
    0.0f,      0.00001f,  0.0f, 0.0f,
    0.0f,      0.0f,      1.0f, 0.0f,
    0.0f,      0.0f,      0.0f, 1.0f,
  };

  std::array<float, 16> expected_inverse_matrix = {
    100000.0f, 0.0f,      0.0f, 0.0f,
    0.0f,      100000.0f, 0.0f, 0.0f,
    0.0f,      0.0f,      1.0f, 0.0f,
    0.0f,      0.0f,      0.0f, 1.0f,
  };
  // clang-format on

  RunTest(matrix, expected_inverse_matrix);
}

}  // namespace
}  // namespace mediapipe
