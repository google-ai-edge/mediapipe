#include "mediapipe/framework/api2/port.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace api2 {
namespace {

constexpr absl::string_view kInputTag{"INPUT"};
constexpr absl::string_view kOutputTag{"OUTPUT"};

TEST(PortTest, IntInput) {
  static constexpr auto port = Input<int>("FOO");
  EXPECT_EQ(port.type_id(), kTypeId<int>);
}

TEST(PortTest, OptionalInput) {
  static constexpr auto port = Input<float>::Optional("BAR");
  EXPECT_TRUE(port.IsOptional());
}

TEST(PortTest, Tag) {
  static constexpr auto port = Input<int>("FOO");
  EXPECT_EQ(std::string(port.Tag()), "FOO");
}

struct DeletedCopyType {
  DeletedCopyType(const DeletedCopyType&) = delete;
  DeletedCopyType& operator=(const DeletedCopyType&) = delete;
};

TEST(PortTest, DeletedCopyConstructorInput) {
  static constexpr Input<DeletedCopyType> kInputPort{"INPUT"};
  EXPECT_EQ(std::string(kInputPort.Tag()), "INPUT");

  static constexpr Output<DeletedCopyType> kOutputPort{"OUTPUT"};
  EXPECT_EQ(std::string(kOutputPort.Tag()), "OUTPUT");

  static constexpr SideInput<DeletedCopyType> kSideInputPort{"SIDE_INPUT"};
  EXPECT_EQ(std::string(kSideInputPort.Tag()), "SIDE_INPUT");

  static constexpr SideOutput<DeletedCopyType> kSideOutputPort{"SIDE_OUTPUT"};
  EXPECT_EQ(std::string(kSideOutputPort.Tag()), "SIDE_OUTPUT");
}

TEST(PortTest, DeletedCopyConstructorStringView) {
  static constexpr Input<DeletedCopyType> kInputPort(kInputTag);
  EXPECT_EQ(std::string(kInputPort.Tag()), kInputTag);

  static constexpr Output<DeletedCopyType> kOutputPort(kOutputTag);
  EXPECT_EQ(std::string(kOutputPort.Tag()), kOutputTag);
}

class AbstractBase {
 public:
  virtual ~AbstractBase() = default;
  virtual absl::string_view name() const = 0;
};

TEST(PortTest, Abstract) {
  static constexpr Input<AbstractBase> kInputPort{"INPUT"};
  EXPECT_EQ(std::string(kInputPort.Tag()), "INPUT");
}

struct TestObject {};

class MultiSideOutputCalculator : public mediapipe::api2::Node {
 public:
  static constexpr SideOutput<TestObject>::Multiple kSideOutput{"SIDE_OUTPUT"};

  MEDIAPIPE_NODE_INTERFACE(MultiSideOutputCalculator, kSideOutput);

  absl::Status Open(CalculatorContext* cc) override {
    for (int i = 0; i < kSideOutput(cc).Count(); ++i) {
      kSideOutput(cc)[i].Set(MakePacket<TestObject>(TestObject()));
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};
MEDIAPIPE_REGISTER_NODE(MultiSideOutputCalculator);

TEST(PortTest, MultiSideOutputPortsWorkInBuilderAndCalculator) {
  builder::Graph graph;

  auto& node = graph.AddNode<MultiSideOutputCalculator>();
  constexpr int kNumOutputSidePackets = 10;
  for (int i = 0; i < kNumOutputSidePackets; ++i) {
    builder::SidePacket<TestObject> side_out =
        node[MultiSideOutputCalculator::kSideOutput][i];
    side_out.SetName(absl::StrCat("side", i));
  }

  CalculatorGraph calculator_graph;
  MP_ASSERT_OK(calculator_graph.Initialize(graph.GetConfig()));
  MP_ASSERT_OK(calculator_graph.Run());

  for (int i = 0; i < kNumOutputSidePackets; ++i) {
    MP_ASSERT_OK_AND_ASSIGN(
        mediapipe::Packet packet,
        calculator_graph.GetOutputSidePacket(absl::StrCat("side", i)));
    EXPECT_FALSE(packet.IsEmpty());
    MP_EXPECT_OK(packet.ValidateAsType<TestObject>());
  }
}

}  // namespace
}  // namespace api2
}  // namespace mediapipe
