#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_TEST_UTIL_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_TEST_UTIL_H_

#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {

// Checks that the two tensors have the same element type and shape, and that
// all values match within the given precision.
// Usage: EXPECT_THAT(tensor1, TensorNear(1e-6, tensor2));
testing::Matcher<const Tensor&> TensorNear(double precision,
                                           const Tensor& expected_tensor);

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_TEST_UTIL_H_
