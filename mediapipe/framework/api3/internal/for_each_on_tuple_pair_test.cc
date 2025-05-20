#include "mediapipe/framework/api3/internal/for_each_on_tuple_pair.h"

#include <cstdint>
#include <tuple>

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::api3 {
namespace {

TEST(ForEachOnTuplePairTest, WorksForTwoTuplesSameSize) {
  std::tuple<uint8_t, float> a = {10, 5.5f};
  std::tuple<int, double> b = {-5, -4.5};

  float sum = 0.0;
  ForEachOnTuplePair(a, b,
                     [&sum](auto el_a, auto el_b) { sum += el_a + el_b; });
  EXPECT_FLOAT_EQ(sum, 6.0f);
}

}  // namespace
}  // namespace mediapipe::api3
