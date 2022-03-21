#include "mediapipe/framework/api2/tag.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace api2 {
namespace {

template <typename A, typename B>
constexpr bool same_type(A, B) {
  return false;
}

template <typename A>
constexpr bool same_type(A, A) {
  return true;
}

auto kFOO = MPP_TAG("FOO");
auto kFOO2 = MPP_TAG("FOO");
auto kBAR = MPP_TAG("BAR");

TEST(TagTest, String) {
  EXPECT_EQ(kFOO.str(), "FOO");
  EXPECT_EQ(kBAR.str(), "BAR");
}

// Separate invocations of MPP_TAG with the same string produce objects of the
// same type.
TEST(TagTest, SameType) { EXPECT_TRUE(same_type(kFOO, kFOO2)); }

// Different tags have different types.
TEST(TagTest, DifferentType) { EXPECT_FALSE(same_type(kFOO, kBAR)); }

TEST(TagTest, Equal) {
  EXPECT_EQ(kFOO, kFOO2);
  EXPECT_NE(kFOO, kBAR);
}

TEST(TagTest, IsTag) {
  EXPECT_TRUE(is_tag(kFOO));
  EXPECT_FALSE(is_tag("FOO"));
}

}  // namespace
}  // namespace api2
}  // namespace mediapipe
