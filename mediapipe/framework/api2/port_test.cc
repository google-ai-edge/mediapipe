#include "mediapipe/framework/api2/port.h"

#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/gtest.h"

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

}  // namespace
}  // namespace api2
}  // namespace mediapipe
