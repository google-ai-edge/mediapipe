#include "mediapipe/framework/api2/port.h"

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace api2 {
namespace {

TEST(PortTest, IntInput) {
  static constexpr auto port = Input<int>("FOO");
  EXPECT_EQ(port.type_id(), typeid(int).hash_code());
}

TEST(PortTest, OptionalInput) {
  static constexpr auto port = Input<float>::Optional("BAR");
  EXPECT_TRUE(port.IsOptional());
}

TEST(PortTest, Tag) {
  static constexpr auto port = Input<int>("FOO");
  EXPECT_EQ(std::string(port.Tag()), "FOO");
}

}  // namespace
}  // namespace api2
}  // namespace mediapipe
