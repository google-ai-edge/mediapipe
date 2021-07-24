#include "mediapipe/framework/tool/options_registry.h"

namespace mediapipe {
namespace tool {

proto_ns::DescriptorPool* OptionsRegistry::options_descriptor_pool() {
  static proto_ns::DescriptorPool* result = new proto_ns::DescriptorPool();
  return result;
}

RegistrationToken OptionsRegistry::Register(
    const proto_ns::FileDescriptorSet& files) {
  for (auto& file : files.file()) {
    options_descriptor_pool()->BuildFile(file);
  }
  return RegistrationToken([]() {});
}

const proto_ns::Descriptor* OptionsRegistry::GetProtobufDescriptor(
    const std::string& type_name) {
  const proto_ns::Descriptor* result =
      proto_ns::DescriptorPool::generated_pool()->FindMessageTypeByName(
          type_name);
  if (!result) {
    result = options_descriptor_pool()->FindMessageTypeByName(type_name);
  }
  return result;
}

void OptionsRegistry::FindAllExtensions(
    const proto_ns::Descriptor& extendee,
    std::vector<const proto_ns::FieldDescriptor*>* result) {
  using proto_ns::DescriptorPool;
  std::vector<const proto_ns::FieldDescriptor*> extensions;
  DescriptorPool::generated_pool()->FindAllExtensions(&extendee, &extensions);
  options_descriptor_pool()->FindAllExtensions(&extendee, &extensions);
  absl::flat_hash_set<int> numbers;
  for (const proto_ns::FieldDescriptor* extension : extensions) {
    bool inserted = numbers.insert(extension->number()).second;
    if (inserted) {
      result->push_back(extension);
    }
  }
}

}  // namespace tool
}  // namespace mediapipe
