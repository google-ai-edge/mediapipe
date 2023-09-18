#include "mediapipe/framework/api2/stream/threshold.h"

#include "mediapipe/calculators/util/thresholding_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"

namespace mediapipe::api2::builder {

Stream<bool> IsOverThreshold(Stream<float> value, double threshold,
                             mediapipe::api2::builder::Graph& graph) {
  auto& node = graph.AddNode("ThresholdingCalculator");
  auto& node_opts = node.GetOptions<mediapipe::ThresholdingCalculatorOptions>();
  node_opts.set_threshold(threshold);
  value.ConnectTo(node.In("FLOAT"));
  return node.Out("FLAG").Cast<bool>();
}

}  // namespace mediapipe::api2::builder
