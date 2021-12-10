#ifndef MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_REGISTRY_H_
#define MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_REGISTRY_H_

#include "absl/container/flat_hash_map.h"
#include "mediapipe/framework/deps/registration.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"

namespace mediapipe {
namespace tool {

class Descriptor;
class FieldDescriptor;

// A static registry that stores descriptors for protobufs used in MediaPipe
// calculator options. Lite-proto builds do not normally include descriptors.
// These registered descriptors allow individual protobuf fields to be
// referenced and specified separately within CalculatorGraphConfigs.
class OptionsRegistry {
 public:
  // Registers the protobuf descriptors for a MessageLite.
  static RegistrationToken Register(const proto_ns::FileDescriptorSet& files);

  // Finds the descriptor for a protobuf.
  static const Descriptor* GetProtobufDescriptor(const std::string& type_name);

  // Returns all known proto2 extensions to a type.
  static void FindAllExtensions(absl::string_view extendee,
                                std::vector<const FieldDescriptor*>* result);

 private:
  // Registers protobuf descriptors a MessageLite and nested types.
  static void Register(const proto_ns::DescriptorProto& message_type,
                       const std::string& parent_name);

  static absl::flat_hash_map<std::string, Descriptor>& descriptors();
  static absl::flat_hash_map<std::string, std::vector<FieldDescriptor>>&
  extensions();
  static absl::Mutex& mutex();

  // Registers the descriptors for each options protobuf type.
  template <class MessageT>
  static const RegistrationToken registration_token;
};

// A custom implementation proto_ns::Descriptor.  This implementation
// avoids a code size problem introduced by proto_ns::FieldDescriptor.
class Descriptor {
 public:
  Descriptor() {}
  Descriptor(const proto_ns::DescriptorProto& proto,
             const std::string& full_name);
  const std::string& full_name() const;
  const FieldDescriptor* FindFieldByName(const std::string& name) const;

 private:
  std::string full_name_;
  absl::flat_hash_map<std::string, FieldDescriptor> fields_;
};

// A custom implementation proto_ns::FieldDescriptor.  This implementation
// avoids a code size problem introduced by proto_ns::FieldDescriptor.
class FieldDescriptor {
 public:
  FieldDescriptor() {}
  FieldDescriptor(const proto_ns::FieldDescriptorProto& proto);
  const std::string& name() const;
  int number() const;
  proto_ns::FieldDescriptorProto::Type type() const;
  const Descriptor* message_type() const;

 private:
  std::string name_;
  std::string message_type_;
  proto_ns::FieldDescriptorProto::Type type_;
  int number_;
};

}  // namespace tool
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TOOL_OPTIONS_REGISTRY_H_
