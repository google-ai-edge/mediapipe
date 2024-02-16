#ifndef MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_RUNNER_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_RUNNER_H_

#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe {

// Common interface to implement inference runners in MediaPipe.
class InferenceRunner {
 public:
  virtual ~InferenceRunner() = default;
  virtual absl::StatusOr<std::vector<Tensor>> Run(
      CalculatorContext* cc, const TensorSpan& tensor_span) = 0;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_RUNNER_H_
