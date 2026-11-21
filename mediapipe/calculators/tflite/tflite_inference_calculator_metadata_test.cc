// Tests that TfLiteInferenceCalculator emits MODEL_METADATA side packet.
#include "mediapipe/calculators/tflite/tflite_model_metadata.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

// A minimal float32 TFLite model with input [1,4] -> output [1,6].
// Replace with a real tiny model path in your environment.
constexpr char kTinyModelPath[] =
    "mediapipe/calculators/tflite/testdata/tiny_detect.tflite";

// DISABLED: requires a real TFLite model at
// mediapipe/calculators/tflite/testdata/tiny_detect.tflite which does not yet
// exist. Add the model and rename to EmitsModelMetadataSidePacket to enable.
TEST(TfLiteInferenceCalculatorMetadataTest,
     DISABLED_EmitsModelMetadataSidePacket) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        input_stream: "tensors"
        output_stream: "out_tensors"
        output_side_packet: "metadata"
        node {
          calculator: "TfLiteInferenceCalculator"
          input_stream:  "TENSORS:tensors"
          output_stream: "TENSORS:out_tensors"
          output_side_packet: "MODEL_METADATA:metadata"
          options {
            [mediapipe.TfLiteInferenceCalculatorOptions.ext] {
              model_path: "mediapipe/calculators/tflite/testdata/tiny_detect.tflite"
            }
          }
        }
      )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  MP_ASSERT_OK_AND_ASSIGN(Packet metadata_packet,
                          graph.GetOutputSidePacket("metadata"));
  ASSERT_FALSE(metadata_packet.IsEmpty());

  const auto& meta = metadata_packet.Get<TfLiteModelMetadata>();
  EXPECT_GE(meta.inputs_size(), 1);
  EXPECT_GE(meta.outputs_size(), 1);
  EXPECT_GE(meta.inputs(0).shape_size(), 1);
}

}  // namespace
}  // namespace mediapipe
