#include "mediapipe/framework/api2/packet.h"

namespace api2 {
namespace {

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
#endif

}  // namespace
};  // namespace api2
