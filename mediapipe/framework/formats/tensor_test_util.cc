#include "mediapipe/framework/formats/tensor_test_util.h"

#include <ostream>

#include "absl/types/span.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

class TensorNearMatcher {
 public:
  using is_gtest_matcher = void;

  TensorNearMatcher(double precision, const Tensor& expected_tensor)
      : precision_(precision), expected_tensor_(expected_tensor) {}

  bool MatchAndExplain(const Tensor& tensor,
                       testing::MatchResultListener* result_listener) const {
    if (tensor.shape().dims != expected_tensor_.shape().dims) {
      *result_listener << "Tensor shape mismatch, actual: "
                       << ::testing::PrintToString(tensor.shape().dims)
                       << ", expected: "
                       << ::testing::PrintToString(
                              expected_tensor_.shape().dims);
      return false;
    }
    if (tensor.element_type() != expected_tensor_.element_type()) {
      *result_listener << "Tensor element type mismatch, actual:"
                       << ::testing::PrintToString(tensor.element_type())
                       << ", expected: "
                       << ::testing::PrintToString(
                              expected_tensor_.element_type());
      return false;
    }

    auto view = tensor.GetCpuReadView();
    auto expected_view = expected_tensor_.GetCpuReadView();
    const float* buffer = view.template buffer<float>();
    const float* expected_buffer = expected_view.template buffer<float>();
    absl::Span<const float> buffer_span(buffer, tensor.shape().num_elements());
    absl::Span<const float> expected_buffer_span(
        expected_buffer, expected_tensor_.shape().num_elements());
    return testing::ExplainMatchResult(
        Pointwise(FloatNear(precision_), expected_buffer_span), buffer_span,
        result_listener);
  }

  void DescribeTo(std::ostream* os) const {
    *os << "is close to " << ::testing::PrintToString(expected_tensor_);
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "is not close to " << ::testing::PrintToString(expected_tensor_);
  }

 private:
  double precision_;
  const Tensor& expected_tensor_;
};

}  // namespace

testing::Matcher<const Tensor&> TensorNear(double precision,
                                           const Tensor& expected_tensor) {
  return TensorNearMatcher(precision, expected_tensor);
}

}  // namespace mediapipe
