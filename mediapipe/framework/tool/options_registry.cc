#include "mediapipe/framework/tool/options_registry.h"

#include "absl/synchronization/mutex.h"

namespace mediapipe {
namespace tool {

namespace {

// Returns a canonical message type name, with any leading "." removed.
std::string CanonicalTypeName(const std::string& type_name) {
  return (type_name.rfind('.', 0) == 0) ? type_name.substr(1) : type_name;
}

}  // namespace

RegistrationToken OptionsRegistry::Register(
    const proto_ns::FileDescriptorSet& files) {
  absl::MutexLock lock(&mutex());
  for (auto& file : files.file()) {
    for (auto& message_type : file.message_type()) {
      Register(message_type, file.package());
    }
  }
  return RegistrationToken([]() {});
}

void OptionsRegistry::Register(const proto_ns::DescriptorProto& message_type,
                               const std::string& parent_name) {
  auto full_name = absl::StrCat(parent_name, ".", message_type.name());
  descriptors()[full_name] = Descriptor(message_type, full_name);
  for (auto& nested : message_type.nested_type()) {
    Register(nested, full_name);
  }
  for (auto& extension : message_type.extension()) {
    extensions()[CanonicalTypeName(extension.extendee())].push_back(
        FieldDescriptor(extension));
  }
}

const Descriptor* OptionsRegistry::GetProtobufDescriptor(
    const std::string& type_name) {
  absl::ReaderMutexLock lock(&mutex());
  auto it = descriptors().find(CanonicalTypeName(type_name));
  return (it == descriptors().end()) ? nullptr : &it->second;
}

void OptionsRegistry::FindAllExtensions(
    absl::string_view extendee, std::vector<const FieldDescriptor*>* result) {
  absl::ReaderMutexLock lock(&mutex());
  result->clear();
  if (extensions().count(extendee) > 0) {
    for (const FieldDescriptor& field : extensions().at(extendee)) {
      result->push_back(&field);
    }
  }
}

absl::flat_hash_map<std::string, Descriptor>& OptionsRegistry::descriptors() {
  static auto* descriptors = new absl::flat_hash_map<std::string, Descriptor>();
  return *descriptors;
}

absl::flat_hash_map<std::string, std::vector<FieldDescriptor>>&
OptionsRegistry::extensions() {
  static auto* extensions =
      new absl::flat_hash_map<std::string, std::vector<FieldDescriptor>>();
  return *extensions;
}

absl::Mutex& OptionsRegistry::mutex() {
  static auto* mutex = new absl::Mutex();
  return *mutex;
}

Descriptor::Descriptor(const proto_ns::DescriptorProto& proto,
                       const std::string& full_name)
    : full_name_(full_name) {
  for (auto& field : proto.field()) {
    fields_[field.name()] = FieldDescriptor(field);
  }
}

const std::string& Descriptor::full_name() const { return full_name_; }

const FieldDescriptor* Descriptor::FindFieldByName(
    const std::string& name) const {
  auto it = fields_.find(name);
  return (it != fields_.end()) ? &it->second : nullptr;
}

FieldDescriptor::FieldDescriptor(const proto_ns::FieldDescriptorProto& proto) {
  name_ = proto.name();
  message_type_ = CanonicalTypeName(proto.type_name());
  type_ = proto.type();
  number_ = proto.number();
}

const std::string& FieldDescriptor::name() const { return name_; }

int FieldDescriptor::number() const { return number_; }

proto_ns::FieldDescriptorProto::Type FieldDescriptor::type() const {
  return type_;
}

const Descriptor* FieldDescriptor::message_type() const {
  return OptionsRegistry::GetProtobufDescriptor(message_type_);
}

}  // namespace tool
}  // namespace mediapipe
