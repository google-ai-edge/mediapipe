#include <memory>

#include "mediapipe/framework/api2/packet.h"

namespace mediapipe {
namespace api2 {
namespace {

float SanityCheck() {
  Packet<float> p = MakePacket<float>(1.0);
  return *p;
}

#if defined(TEST_NO_ASSIGN_WRONG_PACKET_TYPE)
int AssignWrongPacketType() {
  Packet<int> p = MakePacket<float>(1.0);
  return *p;
}
#elif defined(TEST_NO_ASSIGN_GENERIC_TO_SPECIFIC)
int AssignWrongPacketType() {
  Packet<> p = MakePacket<float>(1.0);
  Packet<int> p2 = p;
  return *p2;
}
#elif defined(TEST_SHARE)
auto AssignWrongPacketType() {
  Packet<int> p = MakePacket<int>(1.0);
  return p.Share<int>();  // Error! Should be p.Share();
}
#elif defined(TEST_ONEOF)
bool AssignWrongPacketType() {
  Packet<OneOf<float, int> > p = MakePacket<double>(1.0);  // Error!
  return p.IsEmpty();
}
#elif defined(TEST_ONEOF_SHARE)
bool ShareWrongPacketType() {
  Packet<OneOf<float, int> > p = MakePacket<float>(1.0);
  auto p2 = p.Share<double>();
  return p2.ok();
}
#endif

}  // namespace
}  // namespace api2
}  // namespace mediapipe
