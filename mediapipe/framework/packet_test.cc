// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/packet.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/packet_test.pb.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/type_map.h"

namespace mediapipe {
namespace {

class MyClassBase {
 public:
  virtual ~MyClassBase() {}
  virtual int value() const = 0;
  virtual void set_value(int value) = 0;
};

class MyClass : public MyClassBase {
 public:
  MyClass() : value_(0), exist_(nullptr) {}
  MyClass(const MyClass&) = delete;
  MyClass& operator=(const MyClass&) = delete;
  // Creates an object and sets the value of *exist to true. It will set
  // *exist=false upon destruction, which allows the testing of Packet's
  // reference counting mechanism.
  explicit MyClass(bool* exist) : value_(0), exist_(exist) { *exist_ = true; }
  ~MyClass() override {
    if (exist_) *exist_ = false;
  }
  int value() const override { return value_; }
  void set_value(int value) override { value_ = value; }

 private:
  int value_;
  bool* exist_;
};

TEST(PacketTest, WorksAsExpected) {
  bool exist;
  MyClass* my_class = new MyClass(&exist);
  my_class->set_value(22);
  ASSERT_EQ(exist, true);
  Packet packet = Adopt(my_class);
  Packet packet2 = packet;
  EXPECT_EQ(packet2.Get<MyClass>().value(), 22);
  // Checks that the Packet points to the exact object that was adopted.
  EXPECT_EQ(&packet2.Get<MyClass>(), my_class);
  packet = Packet();
  // Checks that the underlying object stays alive as long as at least one
  // Packet points to it.
  EXPECT_EQ(exist, true);
  packet2 = Packet();
  // Checks that once no Packet points to the adopted object, the latter gets
  // deleted.
  EXPECT_EQ(exist, false);
}

TEST(PacketTest, UsesLvalueAndRvalueReferencePacketAtFunctions) {
  Packet packet1 = Adopt(new int(0));
  // Uses rvalue reference overload of Packet::At after std::move().
  Packet packet2 = std::move(packet1).At(Timestamp(100));
  // Expects that packet1 becomes empty and packet2 gets packet1's data
  // with the given timestamp.
  EXPECT_TRUE(packet1.IsEmpty());  // NOLINT used after std::move().
  ASSERT_FALSE(packet2.IsEmpty());
  MP_ASSERT_OK(packet2.ValidateAsType<int>());
  EXPECT_EQ(0, packet2.Get<int>());
  EXPECT_EQ(Timestamp(100), packet2.Timestamp());

  Packet packet3 = Adopt(new int(1));
  // Uses const lvalue reference overload of Packet::At.
  Packet packet4 = packet3.At(Timestamp(200));
  // Expects that packet3 and packet4 share the same data. And, packet4
  // has the given timestamp.
  ASSERT_FALSE(packet3.IsEmpty());
  ASSERT_FALSE(packet4.IsEmpty());
  MP_ASSERT_OK(packet3.ValidateAsType<int>());
  MP_ASSERT_OK(packet4.ValidateAsType<int>());
  EXPECT_EQ(1, packet3.Get<int>());
  EXPECT_EQ(1, packet4.Get<int>());
  EXPECT_EQ(Timestamp(), packet3.Timestamp());
  EXPECT_EQ(Timestamp(200), packet4.Timestamp());
}

TEST(PacketTest, HandlesUniquePtr) {
  // Several ways of doing the same thing.
  for (const Packet& packet :
       {AdoptAsUniquePtr(static_cast<MyClassBase*>(new MyClass)),
        AdoptAsUniquePtr<MyClassBase>(new MyClass),
        Adopt(new std::unique_ptr<MyClassBase>(new MyClass))}) {
    MP_EXPECT_OK(packet.ValidateAsType<std::unique_ptr<MyClassBase>>());
  }
  bool exists = false;
  Packet packet = AdoptAsUniquePtr<MyClassBase>(new MyClass(&exists));
  EXPECT_TRUE(exists);
  Packet other = packet;
  EXPECT_TRUE(GetFromUniquePtr<MyClassBase>(packet) ==
              GetFromUniquePtr<MyClassBase>(other));
  // Call a non-const method on the contents of one Packet, verify that it is
  // reflected in the other.
  GetFromUniquePtr<MyClassBase>(packet)->set_value(42);
  EXPECT_EQ(42, GetFromUniquePtr<MyClassBase>(other)->value());
  packet = Packet();
  EXPECT_TRUE(exists);
  other = Packet();
  // Last copy of MyClass has been destroyed.
  EXPECT_FALSE(exists);
}

TEST(PacketTest, HandlesAbstractClasses) {
  std::unique_ptr<MyClassBase> data(new MyClass());
  data->set_value(42);
  Packet packet = Adopt(data.release());
  EXPECT_EQ(42, packet.Get<MyClassBase>().value());
}

struct RegisteredPairStruct {
  int first;
  float second;
};
struct UnregisteredPairStruct {
  std::string first;
  bool second;
};
MEDIAPIPE_REGISTER_TYPE(::mediapipe::RegisteredPairStruct,
                        "::mediapipe::RegisteredPairStruct", nullptr, nullptr);
MEDIAPIPE_REGISTER_TYPE(int, "int", nullptr, nullptr);
MEDIAPIPE_REGISTER_TYPE(float, "float", nullptr, nullptr);
constexpr bool kHaveUnregisteredTypeNames = MEDIAPIPE_HAS_RTTI;

TEST(PacketTest, TypeRegistrationDebugString) {
  // Test registered type.
  RegisteredPairStruct s{1, 3.5};
  Packet packet = MakePacket<RegisteredPairStruct>(s);
  EXPECT_EQ(packet.DebugString(),
            "mediapipe::Packet with timestamp: Timestamp::Unset() and type: "
            "::mediapipe::RegisteredPairStruct");

  // Unregistered type.
  UnregisteredPairStruct u{"s", true};
  Packet packet2 = MakePacket<UnregisteredPairStruct>(u);
  std::string expected_type_name =
      (kHaveUnregisteredTypeNames)
          ? "mediapipe::(anonymous namespace)::UnregisteredPairStruct"
          : "<unknown>";
  EXPECT_EQ(packet2.DebugString(),
            "mediapipe::Packet with timestamp: Timestamp::Unset() and type: " +
                expected_type_name);
}

TEST(PacketTest, ReturnGenericProtobufMessage) {
  std::unique_ptr<::mediapipe::PacketTestProto> proto_ptr(
      new ::mediapipe::PacketTestProto);
  proto_ptr->add_x(123);
  Packet packet = Adopt(static_cast<proto_ns::Message*>(proto_ptr.release()));
  EXPECT_EQ(123, dynamic_cast<const ::mediapipe::PacketTestProto&>(
                     packet.Get<proto_ns::Message>())
                     .x(0));
}

TEST(PacketTest, TryWrongProtobufMessageSubType) {
  std::unique_ptr<::mediapipe::PacketTestProto> proto_ptr(
      new ::mediapipe::PacketTestProto);
  proto_ptr->add_x(123);
  Packet packet = Adopt(proto_ptr.release());
  EXPECT_FALSE(packet.ValidateAsType<::mediapipe::SimpleProto>().ok());
  EXPECT_TRUE(packet.ValidateAsType<::mediapipe::PacketTestProto>().ok());
}

TEST(PacketTest, GetProtoBase) {
  std::unique_ptr<::mediapipe::PacketTestProto> proto_ptr(
      new ::mediapipe::PacketTestProto);
  proto_ptr->add_x(123);
  Packet packet = Adopt(proto_ptr.release());
  ::mediapipe::PacketTestProto proto_copy;
  proto_copy.CheckTypeAndMergeFrom(packet.GetProtoMessageLite());
  EXPECT_EQ(123, proto_copy.x(0));
  // If not a protocol buffer type, crashes.
  Packet packet2 = MakePacket<int>(3);
  EXPECT_DEATH(packet2.GetProtoMessageLite(),
               "cannot be converted to MessageLite");
}

TEST(PacketTest, ValidateAsProtoMessageLite) {
  auto proto_ptr = absl::make_unique<::mediapipe::PacketTestProto>();
  proto_ptr->add_x(123);
  Packet packet = Adopt(proto_ptr.release());
  MP_EXPECT_OK(packet.ValidateAsProtoMessageLite());
  Packet packet2 = MakePacket<int>(3);
  ::mediapipe::Status status = packet2.ValidateAsProtoMessageLite();
  EXPECT_EQ(status.code(), ::mediapipe::StatusCode::kInvalidArgument);
}

TEST(PacketTest, SyncedPacket) {
  Packet synced_packet = AdoptAsSyncedPacket(new int(100));
  Packet value_packet =
      synced_packet.Get<std::unique_ptr<SyncedPacket>>()->Get();
  EXPECT_EQ(100, value_packet.Get<int>());
  // update the value.
  Packet new_value_packet = Adopt(new int(999));
  synced_packet.Get<std::unique_ptr<SyncedPacket>>()->UpdatePacket(
      new_value_packet);
  Packet packet_new = synced_packet.Get<std::unique_ptr<SyncedPacket>>()->Get();
  EXPECT_EQ(999, packet_new.Get<int>());
}

TEST(PacketTest, MakePacketOfIntArray) {
  Packet int_packet = MakePacket<int>(123);
  EXPECT_EQ(123, int_packet.Get<int>());

  Packet array_packet = MakePacket<int[3]>(32, 64, 128);
  const auto& array_ref = array_packet.Get<int[3]>();
  EXPECT_EQ(32, array_ref[0]);
  EXPECT_EQ(64, array_ref[1]);
  EXPECT_EQ(128, array_ref[2]);
}

TEST(PacketTest, MakePacketOfIntVector) {
  std::vector<int> vector({1, 2, 3});
  Packet vector_packet1 = MakePacket<std::vector<int>>(vector);
  Packet vector_packet2 =
      MakePacket<std::vector<int>>(std::initializer_list<int>({1, 2, 3}));
  EXPECT_EQ(vector_packet1.Get<std::vector<int>>(),
            vector_packet2.Get<std::vector<int>>());
}

TEST(PacketTest, TestPacketMoveConstructor) {
  std::vector<Packet>* packet_vector_ptr = new std::vector<Packet>();
  packet_vector_ptr->push_back(MakePacket<float>(42));
  packet_vector_ptr->push_back(MakePacket<std::string>("test"));
  Packet packet = Adopt(packet_vector_ptr).At(Timestamp(100));
  {
    Packet copied_packet(packet);  // NOLINT explicit unneeded copy.
    // Original packet still keeps a reference to holder_ after a copy.
    ASSERT_FALSE(packet.IsEmpty());
    std::vector<Packet> packet_vector_output1 =
        packet.Get<std::vector<Packet>>();
    ASSERT_EQ(2, packet_vector_output1.size());
    EXPECT_EQ(42, packet_vector_output1[0].Get<float>());
    EXPECT_EQ("test", packet_vector_output1[1].Get<std::string>());
    EXPECT_EQ(Timestamp(100), packet.Timestamp());
    std::vector<Packet> packet_vector_output2 =
        copied_packet.Get<std::vector<Packet>>();
    ASSERT_EQ(2, packet_vector_output2.size());
    EXPECT_EQ(42, packet_vector_output2[0].Get<float>());
    EXPECT_EQ("test", packet_vector_output2[1].Get<std::string>());
    EXPECT_EQ(Timestamp(100), copied_packet.Timestamp());
  }
  Packet moved_packet(std::move(packet));
  // Original packet should become empty after a move.
  EXPECT_TRUE(packet.IsEmpty());  // NOLINT used after std::move().
  std::vector<Packet> packet_vector_output3 =
      moved_packet.Get<std::vector<Packet>>();
  ASSERT_EQ(2, packet_vector_output3.size());
  EXPECT_EQ(42, packet_vector_output3[0].Get<float>());
  EXPECT_EQ("test", packet_vector_output3[1].Get<std::string>());
  EXPECT_EQ(Timestamp(100), moved_packet.Timestamp());
}

TEST(PacketTest, TestPacketConsume) {
  Packet packet1 = MakePacket<int>(33);
  Packet packet_copy = packet1;
  ::mediapipe::StatusOr<std::unique_ptr<int>> result1 =
      packet_copy.Consume<int>();
  // Both packet1 and packet_copy own the data, Consume() should return error.
  ::mediapipe::Status status1 = result1.status();
  EXPECT_EQ(status1.code(), ::mediapipe::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status1.message(),
              testing::HasSubstr("isn't the sole owner of the holder"));
  ASSERT_FALSE(packet1.IsEmpty());
  EXPECT_EQ(33, packet1.Get<int>());
  ASSERT_FALSE(packet_copy.IsEmpty());
  EXPECT_EQ(33, packet_copy.Get<int>());

  Packet packet2 = MakePacket<int>(33);
  // Types don't match (int vs float).
  ::mediapipe::StatusOr<std::unique_ptr<float>> result2 =
      packet2.Consume<float>();
  EXPECT_THAT(
      result2.status().message(),
      testing::AllOf(testing::HasSubstr("int"), testing::HasSubstr("float")));
  ASSERT_FALSE(packet2.IsEmpty());
  EXPECT_EQ(33, packet2.Get<int>());

  // packet3 is the sole owner of the data.
  Packet packet3 = MakePacket<int>(42);
  ::mediapipe::StatusOr<std::unique_ptr<int>> result3 = packet3.Consume<int>();
  // After Consume(), packet3 should be empty and result3 owns the data.
  EXPECT_TRUE(result3.ok());
  ASSERT_NE(nullptr, result3.ValueOrDie());
  EXPECT_EQ(42, *result3.ValueOrDie());
  EXPECT_TRUE(packet3.IsEmpty());
}

TEST(PacketTest, TestPacketConsumeOrCopy) {
  Packet packet1 = MakePacket<int>(33);
  Packet packet_copy = packet1;
  bool was_copied1 = false;
  ::mediapipe::StatusOr<std::unique_ptr<int>> result1 =
      packet_copy.ConsumeOrCopy<int>(&was_copied1);
  // Both packet1 and packet_copy own the data, ConsumeOrCopy() returns a copy
  // of the data and sets packet_copy to empty.
  EXPECT_TRUE(result1.ok());
  EXPECT_TRUE(was_copied1);
  ASSERT_NE(nullptr, result1.ValueOrDie());
  EXPECT_EQ(33, *result1.ValueOrDie());
  EXPECT_TRUE(packet_copy.IsEmpty());
  // ConsumeOrCopy() doesn't affect packet1.
  ASSERT_FALSE(packet1.IsEmpty());
  EXPECT_EQ(33, packet1.Get<int>());

  Packet packet2 = MakePacket<int>(33);
  // Types don't match (int vs float).
  ::mediapipe::StatusOr<std::unique_ptr<float>> result2 =
      packet2.ConsumeOrCopy<float>();
  EXPECT_THAT(
      result2.status().message(),
      testing::AllOf(testing::HasSubstr("int"), testing::HasSubstr("float")));
  ASSERT_FALSE(packet2.IsEmpty());
  EXPECT_EQ(33, packet2.Get<int>());

  Packet packet3 = MakePacket<int>(42);
  bool was_copied3 = false;
  // packet3 is the sole owner of the data. ConsumeOrCopy() transfers the
  // ownership to result3 and makes packet3 empty.
  ::mediapipe::StatusOr<std::unique_ptr<int>> result3 =
      packet3.ConsumeOrCopy<int>(&was_copied3);
  EXPECT_FALSE(was_copied3);
  EXPECT_TRUE(result3.ok());
  ASSERT_NE(nullptr, result3.ValueOrDie());
  EXPECT_EQ(42, *result3.ValueOrDie());
  EXPECT_TRUE(packet3.IsEmpty());
}

TEST(PacketTest, TestConsumeForeignHolder) {
  std::unique_ptr<int> data(new int(33));
  Packet packet = PointToForeign(data.get());
  ::mediapipe::StatusOr<std::unique_ptr<int>> result = packet.Consume<int>();
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), ::mediapipe::StatusCode::kInternal);
  EXPECT_EQ(result.status().message(),
            "Foreign holder can't release data ptr without ownership.");
  ASSERT_FALSE(packet.IsEmpty());
  EXPECT_EQ(33, packet.Get<int>());
}

TEST(PacketTest, TestForeignHolderConsumeOrCopy) {
  std::unique_ptr<int> data1(new int(42));
  Packet packet1 = PointToForeign(data1.get());
  Packet packet_copy = packet1;
  bool was_copied1 = false;
  ::mediapipe::StatusOr<std::unique_ptr<int>> result1 =
      packet_copy.ConsumeOrCopy<int>(&was_copied1);
  // After ConsumeOrCopy(), result1 gets the copy of packet_copy's data and
  // packet_copy is set to empty.
  EXPECT_TRUE(packet_copy.IsEmpty());
  EXPECT_TRUE(was_copied1);
  EXPECT_TRUE(result1.ok());
  ASSERT_NE(nullptr, result1.ValueOrDie());
  EXPECT_EQ(42, *result1.ValueOrDie());
  // ConsumeOrCopy() doesn't affect packet1.
  ASSERT_FALSE(packet1.IsEmpty());
  EXPECT_EQ(42, packet1.Get<int>());

  std::unique_ptr<int> data2(new int(33));
  Packet packet2 = PointToForeign(data2.get());
  bool was_copied2 = false;
  ::mediapipe::StatusOr<std::unique_ptr<int>> result2 =
      packet2.ConsumeOrCopy<int>(&was_copied2);
  // After ConsumeOrCopy(), result2 gets the copy of packet2's data and packet2
  // is set to empty.
  EXPECT_TRUE(packet2.IsEmpty());
  EXPECT_TRUE(was_copied2);
  EXPECT_TRUE(result2.ok());
  ASSERT_NE(nullptr, result2.ValueOrDie());
  EXPECT_EQ(33, *result2.ValueOrDie());
}

TEST(PacketTest, TestConsumeBoundedArray) {
  Packet packet1 = MakePacket<int[3]>(10, 20, 30);
  Packet packet_copy = packet1;
  ::mediapipe::StatusOr<std::unique_ptr<int[3]>> result1 =
      packet_copy.Consume<int[3]>();
  // Both packet1 and packet_copy own the data, Consume() should return error.
  ::mediapipe::Status status1 = result1.status();
  EXPECT_EQ(status1.code(), ::mediapipe::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status1.message(),
              testing::HasSubstr("isn't the sole owner of the holder"));
  ASSERT_FALSE(packet1.IsEmpty());
  const auto& value1 = packet1.Get<int[3]>();
  EXPECT_EQ(10, value1[0]);
  EXPECT_EQ(20, value1[1]);
  EXPECT_EQ(30, value1[2]);
  ASSERT_FALSE(packet_copy.IsEmpty());
  const auto& value2 = packet_copy.Get<int[3]>();
  EXPECT_EQ(10, value2[0]);
  EXPECT_EQ(20, value2[1]);
  EXPECT_EQ(30, value2[2]);

  Packet packet2 = MakePacket<int[3]>(40, 50, 60);
  // After Consume(), packet2 should be empty and result2 owns the data.
  ::mediapipe::StatusOr<std::unique_ptr<int[3]>> result2 =
      packet2.Consume<int[3]>();
  ASSERT_NE(nullptr, result2.ValueOrDie());
  auto value3 = result2.ValueOrDie().get();
  EXPECT_EQ(40, (*value3)[0]);
  EXPECT_EQ(50, (*value3)[1]);
  EXPECT_EQ(60, (*value3)[2]);
  EXPECT_TRUE(packet2.IsEmpty());
}

TEST(PacketTest, TestConsumeOrCopyBoundedArray) {
  Packet packet1 = MakePacket<int[3]>(10, 20, 30);
  Packet packet_copy = packet1;
  bool was_copied1 = false;
  ::mediapipe::StatusOr<std::unique_ptr<int[3]>> result1 =
      packet_copy.ConsumeOrCopy<int[3]>(&was_copied1);
  // Both packet1 and packet_copy own the data, ConsumeOrCopy() returns a copy
  // of the data and sets packet_copy to empty.
  EXPECT_TRUE(result1.ok());
  EXPECT_TRUE(was_copied1);
  ASSERT_NE(nullptr, result1.ValueOrDie());
  auto value1 = result1.ValueOrDie().get();
  EXPECT_EQ(10, (*value1)[0]);
  EXPECT_EQ(20, (*value1)[1]);
  EXPECT_EQ(30, (*value1)[2]);
  EXPECT_TRUE(packet_copy.IsEmpty());
  // ConsumeOrCopy() doesn't affect packet1.
  const auto& value2 = packet1.Get<int[3]>();
  EXPECT_EQ(10, value2[0]);
  EXPECT_EQ(20, value2[1]);
  EXPECT_EQ(30, value2[2]);
  ASSERT_FALSE(packet1.IsEmpty());

  Packet packet2 = MakePacket<int[3]>(40, 50, 60);
  bool was_copied2 = false;
  // packet2 is the sole owner of the data. ConsumeOrCopy() transfers the
  // ownership to result2 and makes packet2 empty.
  ::mediapipe::StatusOr<std::unique_ptr<int[3]>> result2 =
      packet2.ConsumeOrCopy<int[3]>(&was_copied2);
  EXPECT_TRUE(result2.ok());
  EXPECT_FALSE(was_copied2);
  ASSERT_NE(nullptr, result2.ValueOrDie());
  auto value3 = result2.ValueOrDie().get();
  EXPECT_EQ(40, (*value3)[0]);
  EXPECT_EQ(50, (*value3)[1]);
  EXPECT_EQ(60, (*value3)[2]);
  EXPECT_TRUE(packet2.IsEmpty());
}

TEST(PacketTest, MessageHolderRegistration) {
  using testing::Contains;
  Packet packet = MakePacket<mediapipe::SimpleProto>();
  ASSERT_EQ(mediapipe::SimpleProto{}.GetTypeName(), "mediapipe.SimpleProto");
  EXPECT_THAT(packet_internal::MessageHolderRegistry::GetRegisteredNames(),
              Contains("mediapipe.SimpleProto"));
}

TEST(PacketTest, PacketFromSerializedProto) {
  mediapipe::SimpleProto original;
  original.add_value("foo");
  std::string serialized = original.SerializeAsString();

  StatusOr<Packet> maybe_packet = packet_internal::PacketFromDynamicProto(
      "mediapipe.SimpleProto", serialized);
  MP_ASSERT_OK(maybe_packet);
  Packet packet = maybe_packet.ValueOrDie();
  MP_EXPECT_OK(packet.ValidateAsType<::mediapipe::SimpleProto>());
  EXPECT_FALSE(packet.ValidateAsType<::mediapipe::PacketTestProto>().ok());
}

}  // namespace
}  // namespace mediapipe
