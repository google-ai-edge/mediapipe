#include "mediapipe/framework/api3/node_test.h"

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::api3 {
namespace {

TEST(NodeTest, NameIsRequiredAndCanBeSpecified) {
  EXPECT_EQ(FooNode::GetRegistrationName(), "Foo");
  EXPECT_EQ(BarNode::GetRegistrationName(), "Bar");
  EXPECT_EQ(BarANode::GetRegistrationName(), "BarA");
  EXPECT_EQ(BarBNode::GetRegistrationName(), "BarB");
}

}  // namespace
}  // namespace mediapipe::api3
