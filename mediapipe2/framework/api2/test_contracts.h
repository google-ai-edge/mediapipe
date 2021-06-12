#ifndef MEDIAPIPE_FRAMEWORK_API2_TEST_CONTRACTS_H_
#define MEDIAPIPE_FRAMEWORK_API2_TEST_CONTRACTS_H_

#include "mediapipe/framework/api2/node.h"

namespace mediapipe {
namespace api2 {
namespace test {

struct Foo : public NodeIntf {
  static constexpr Input<int> kBase{"BASE"};
  static constexpr Input<float>::Optional kScale{"SCALE"};
  static constexpr Output<float> kOut{"OUT"};
  static constexpr SideInput<float>::Optional kBias{"BIAS"};

  MEDIAPIPE_NODE_INTERFACE(Foo, kBase, kScale, kOut, kBias);
};

struct Foo2 : public NodeIntf {
  // clang-format off
  static constexpr auto kPorts = std::make_tuple(
      Input<int>{"BASE"},
      Input<float>::Optional{"SCALE"},
      Output<float>{"OUT"},
      SideInput<float>::Optional{"BIAS"}
  );
  // clang-format on
  MEDIAPIPE_NODE_INTERFACE(Foo2, kPorts);
};

struct Bar : public NodeIntf {
  static constexpr Input<AnyType> kIn{"IN"};
  // Should all outputs be treated as optional by default?
  static constexpr Output<SameType<kIn>>::Optional kOut{"OUT"};

  MEDIAPIPE_NODE_INTERFACE(Bar, kIn, kOut);
};

struct Baz : public NodeIntf {
  static constexpr Input<AnyType>::Multiple kData{"DATA"};
  // Should all outputs be treated as optional by default?
  static constexpr Output<SameType<kData>>::Multiple kDataOut{"DATA"};

  MEDIAPIPE_NODE_INTERFACE(Baz, kData, kDataOut);
};

struct IntForwarder : public NodeIntf {
  static constexpr Input<int> kIn{"IN"};
  static constexpr Output<int> kOut{"OUT"};

  MEDIAPIPE_NODE_INTERFACE(IntForwarder, kIn, kOut);
};

struct FloatAdder : public NodeIntf {
  static constexpr Input<float>::Multiple kIn{"IN"};
  static constexpr Output<float> kOut{"OUT"};

  MEDIAPIPE_NODE_INTERFACE(FloatAdder, kIn, kOut);
};

struct ToFloat : public NodeIntf {
  static constexpr Input<OneOf<float, int>> kIn{"IN"};
  static constexpr Output<float> kOut{"OUT"};

  MEDIAPIPE_NODE_INTERFACE(ToFloat, kIn, kOut);
};

struct FooBar : public NodeIntf {
  static constexpr Input<int> kIn{"IN"};
  static constexpr Output<float> kOut{"OUT"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);
};

struct FooBar1 : public FooBar {
  static constexpr char kCalculatorName[] = "FooBar";
};

struct FooBar2 : public FooBar {
  static constexpr char kCalculatorName[] = "FooBar2";
};

}  // namespace test
}  // namespace api2
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_API2_TEST_CONTRACTS_H_
