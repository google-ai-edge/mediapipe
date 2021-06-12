
#include "mediapipe/framework/tool/options_util.h"

#include "mediapipe/framework/port/proto_ns.h"

namespace mediapipe {
namespace tool {

// TODO: Return registered protobuf Descriptors when available.
const proto_ns::Descriptor* GetProtobufDescriptor(
    const std::string& type_name) {
  return proto_ns::DescriptorPool::generated_pool()->FindMessageTypeByName(
      type_name);
}

}  // namespace tool
}  // namespace mediapipe
