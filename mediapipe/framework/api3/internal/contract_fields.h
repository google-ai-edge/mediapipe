#ifndef MEDIAPIPE_FRAMEWORK_API3_INTERNAL_CONTRACT_FIELDS_H_
#define MEDIAPIPE_FRAMEWORK_API3_INTERNAL_CONTRACT_FIELDS_H_

namespace mediapipe::api3 {

// Used internally by the framework to distinguish templated (side) inputs,
// (side) outputs and options.

struct InputStreamField {};
struct OutputStreamField {};
struct InputSidePacketField {};
struct OutputSidePacketField {};
struct RepeatedField {};
struct OptionalField {};
struct OptionsField {};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_FRAMEWORK_API3_INTERNAL_CONTRACT_FIELDS_H_
