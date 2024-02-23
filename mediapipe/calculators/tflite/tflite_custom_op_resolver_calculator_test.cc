
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "tensorflow/lite/kernels/register.h"

namespace mediapipe {

TEST(TfLiteCustomOpResolverCalculatorTest, FindResamplerOp) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          R"pb(
            node {
              calculator: "TfLiteCustomOpResolverCalculator"
              output_side_packet: "op_resolver"
            }
          )pb");
  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  MP_ASSERT_OK_AND_ASSIGN(auto resolver_packet,
                          graph.GetOutputSidePacket("op_resolver"));

  auto resolver =
      resolver_packet.Get<tflite::ops::builtin::BuiltinOpResolver>();

  EXPECT_THAT(resolver.FindOp("Resampler", /*version=*/1), testing::NotNull());
}

}  // namespace mediapipe
