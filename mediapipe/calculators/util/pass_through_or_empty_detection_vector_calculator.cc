#include "mediapipe/calculators/util/pass_through_or_empty_detection_vector_calculator.h"

#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/formats/detection.pb.h"

namespace mediapipe {

class PassThroughOrEmptyDetectionVectorCalculatorImpl
    : public mediapipe::api2::NodeImpl<
          PassThroughOrEmptyDetectionVectorCalculator> {
 public:
  absl::Status Process(CalculatorContext* cc) override {
    if (kInputVector(cc).IsEmpty()) {
      kOutputVector(cc).Send(std::vector<mediapipe::Detection>{});
      return absl::OkStatus();
    }
    kOutputVector(cc).Send(kInputVector(cc));
    return absl::OkStatus();
  }
};
MEDIAPIPE_NODE_IMPLEMENTATION(PassThroughOrEmptyDetectionVectorCalculatorImpl);

}  // namespace mediapipe
