#ifndef MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_REGISTRY_H_
#define MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_REGISTRY_H_

#include "mediapipe/framework/deps/registration.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"

namespace mediapipe {
namespace tool {

// A static registry that stores descriptors for protobufs used in MediaPipe
// calculator options. Lite-proto builds do not normally include descriptors.
// These registered descriptors allow individual protobuf fields to be
// referenced and specified separately within CalculatorGraphConfigs.
class OptionsRegistry {
 public:
  // Registers the protobuf descriptors for a MessageLite.
  static RegistrationToken Register(const proto_ns::FileDescriptorSet& files);

  // Finds the descriptor for a protobuf.
  static const proto_ns::Descriptor* GetProtobufDescriptor(
      const std::string& type_name);

  // Returns all known proto2 extensions to a type.
  static void FindAllExtensions(
      const proto_ns::Descriptor& extendee,
      std::vector<const proto_ns::FieldDescriptor*>* result);

 private:
  // Stores the descriptors for each options protobuf type.
  static proto_ns::DescriptorPool* options_descriptor_pool();

  // Registers the descriptors for each options protobuf type.
  template <class MessageT>
  static const RegistrationToken registration_token;
};

}  // namespace tool
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_REGISTRY_H_
