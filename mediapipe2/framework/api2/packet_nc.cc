#include "mediapipe/framework/api2/packet.h"

namespace api2 {
namespace {

#if defined(TEST_NO_ASSIGN_WRONG_PACKET_TYPE)
void AssignWrongPacketType() { Packet<int> p = MakePacket<float>(1.0); }
#elif defined(TEST_NO_ASSIGN_GENERIC_TO_SPECIFIC)
void AssignWrongPacketType() {
  Packet<> p = MakePacket<float>(1.0);
  Packet<int> p2 = p;
}
#endif

}  // namespace
};  // namespace api2
