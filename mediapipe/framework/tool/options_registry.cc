#include "mediapipe/framework/tool/options_registry.h"

#include <string>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/tool/proto_util_lite.h"

namespace mediapipe {
namespace tool {

namespace {

// Returns a canonical message type name, with any leading "." removed.
std::string CanonicalTypeName(const std::string& type_name) {
  return (absl::StartsWith(type_name, ".")) ? type_name.substr(1) : type_name;
}

// Returns the values from a protobuf field as typed FieldData.
absl::StatusOr<std::vector<FieldData>> GetFieldValues(
    const FieldData& message_data, std::string field_name) {
  std::string type_name =
      ProtoUtilLite::ParseTypeUrl(message_data.message_value().type_url());
  const Descriptor* descriptor =
      OptionsRegistry::GetProtobufDescriptor(type_name);
  RET_CHECK_NE(descriptor, nullptr);
  const FieldDescriptor* field = descriptor->FindFieldByName(field_name);
  if (field == nullptr) {
    return std::vector<FieldData>();
  }
  ProtoUtilLite::ProtoPath proto_path = {{field->number(), 0}};
  ProtoUtilLite::FieldValue mesage_bytes = message_data.message_value().value();
  int count;
  MP_RETURN_IF_ERROR(ProtoUtilLite::GetFieldCount(mesage_bytes, proto_path,
                                                  field->type(), &count));
  std::vector<std::string> field_values;
  MP_RETURN_IF_ERROR(ProtoUtilLite::GetFieldRange(
      mesage_bytes, proto_path, count, field->type(), &field_values));
  std::vector<FieldData> result;
  for (int i = 0; i < field_values.size(); ++i) {
    FieldData r;
    std::string message_type =
        field->message_type() ? field->message_type()->full_name() : "";
    MP_RETURN_IF_ERROR(ProtoUtilLite::ReadValue(field_values[i], field->type(),
                                                message_type, &r));
    result.push_back(std::move(r));
  }
  return result;
}

// Returns a single value from a protobuf string field.
std::string GetFieldString(const FieldData& message_data,
                           std::string field_name) {
  auto values = GetFieldValues(message_data, field_name);
  if (!values->empty()) {
    return values->front().string_value();
  }
  return "";
}

// Registers the descriptors for the descriptor protobufs.  These four
// descriptors are required to deserialize descriptors for other protobufs.
// This implementation avoids a code size problem introduced by
// proto_ns::DescriptorProto.
void RegisterDescriptorProtos(
    absl::flat_hash_map<std::string, Descriptor>& result) {
  std::vector<Descriptor> descriptors = {
      {"google::protobuf.FileDescriptorSet",
       {
           {"file", 1, FieldType::TYPE_MESSAGE,
            "google::protobuf.FileDescriptorProto"},
       }},
      {"google::protobuf.FileDescriptorProto",
       {
           {"package", 2, FieldType::TYPE_STRING, ""},
           {"message_type", 4, FieldType::TYPE_MESSAGE,
            "google::protobuf.DescriptorProto"},
       }},
      {"google::protobuf.DescriptorProto",
       {
           {"name", 1, FieldType::TYPE_STRING, ""},
           {"field", 2, FieldType::TYPE_MESSAGE,
            "google::protobuf.FieldDescriptorProto"},
           {"extension", 6, FieldType::TYPE_MESSAGE,
            "google::protobuf.FieldDescriptorProto"},
           {"nested_type", 3, FieldType::TYPE_MESSAGE,
            "google::protobuf.DescriptorProto"},
       }},
      {"google::protobuf.FieldDescriptorProto",
       {
           {"name", 1, FieldType::TYPE_STRING, ""},
           {"number", 3, FieldType::TYPE_INT32, ""},
           {"type", 5, FieldType::TYPE_ENUM, ""},
           {"type_name", 6, FieldType::TYPE_STRING, ""},
           {"extendee", 2, FieldType::TYPE_STRING, ""},
       }},
  };
  for (const auto& descriptor : descriptors) {
    result[descriptor.full_name()] = descriptor;
  }
}

}  // namespace

RegistrationToken OptionsRegistry::Register(
    const FieldData& file_descriptor_set) {
  auto files = GetFieldValues(file_descriptor_set, "file");
  for (auto& file : *files) {
    std::string package_name = GetFieldString(file, "package");
    auto message_types = GetFieldValues(file, "message_type");
    for (auto& message_type : *message_types) {
      Register(message_type, package_name);
    }
  }
  return RegistrationToken([]() {});
}

void OptionsRegistry::Register(const FieldData& message_type,
                               const std::string& parent_name) {
  std::string name = GetFieldString(message_type, "name");
  std::string full_name = absl::StrCat(parent_name, ".", name);
  Descriptor descriptor(full_name, message_type);
  {
    absl::MutexLock lock(&mutex());
    descriptors()[full_name] = descriptor;
  }
  auto nested_types = GetFieldValues(message_type, "nested_type");
  for (auto& nested : *nested_types) {
    Register(nested, full_name);
  }
  auto exts = GetFieldValues(message_type, "extension");
  for (auto& extension : *exts) {
    FieldDescriptor field(extension);
    std::string extendee = GetFieldString(extension, "extendee");
    {
      absl::MutexLock lock(&mutex());
      extensions()[CanonicalTypeName(extendee)].push_back(field);
    }
  }
}

const Descriptor* OptionsRegistry::GetProtobufDescriptor(
    const std::string& type_name) {
  if (descriptors().count("google::protobuf.DescriptorProto") == 0) {
    RegisterDescriptorProtos(descriptors());
  }
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

Descriptor::Descriptor(const std::string& full_name,
                       const FieldData& descriptor_proto)
    : full_name_(full_name) {
  auto fields = GetFieldValues(descriptor_proto, "field");
  for (const auto& field : *fields) {
    FieldDescriptor f(field);
    fields_[f.name()] = f;
  }
}

Descriptor::Descriptor(const std::string& full_name,
                       const std::vector<FieldDescriptor>& fields)
    : full_name_(full_name) {
  for (const auto& field : fields) {
    fields_[field.name()] = field;
  }
}

const std::string& Descriptor::full_name() const { return full_name_; }

const FieldDescriptor* Descriptor::FindFieldByName(
    const std::string& name) const {
  auto it = fields_.find(name);
  return (it != fields_.end()) ? &it->second : nullptr;
}

FieldDescriptor::FieldDescriptor(const FieldData& field_proto) {
  name_ = GetFieldString(field_proto, "name");
  number_ = GetFieldValues(field_proto, "number")->front().int32_value();
  type_ = (FieldType)GetFieldValues(field_proto, "type")->front().enum_value();
  message_type_ = CanonicalTypeName(GetFieldString(field_proto, "type_name"));
}

FieldDescriptor::FieldDescriptor(std::string name, int number, FieldType type,
                                 std::string message_type)
    : name_(name), number_(number), type_(type), message_type_(message_type) {}

const std::string& FieldDescriptor::name() const { return name_; }

int FieldDescriptor::number() const { return number_; }

FieldType FieldDescriptor::type() const { return type_; }

const Descriptor* FieldDescriptor::message_type() const {
  return OptionsRegistry::GetProtobufDescriptor(message_type_);
}

}  // namespace tool
}  // namespace mediapipe
