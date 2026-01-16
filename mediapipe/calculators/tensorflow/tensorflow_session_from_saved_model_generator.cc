// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>

#include "absl/status/status.h"

#if !defined(__ANDROID__)
#include "mediapipe/framework/port/file_helpers.h"
#endif
#include "absl/log/absl_log.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session_from_saved_model_generator.pb.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/packet_generator.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

namespace mediapipe {

namespace {

constexpr char kSessionTag[] = "SESSION";

static constexpr char kStringSavedModelPath[] = "STRING_SAVED_MODEL_PATH";

static constexpr char kStringSignatureName[] = "STRING_SIGNATURE_NAME";

// Given the path to a directory containing multiple tensorflow saved models
// in subdirectories, replaces path with the alphabetically last subdirectory.
absl::Status GetLatestDirectory(std::string* path) {
#if defined(__ANDROID__)
  return absl::UnimplementedError(
      "GetLatestDirectory is not implemented on Android");
#else
  std::vector<std::string> saved_models;
  RET_CHECK_OK(file::MatchInTopSubdirectories(
      *path, tensorflow::kSavedModelFilenamePb, &saved_models));
  RET_CHECK_GT(saved_models.size(), 0)
      << "No exported bundles found in " << path;
  ::std::sort(saved_models.begin(), saved_models.end());
  *path = std::string(file::Dirname(saved_models.back()));
  return absl::OkStatus();
#endif
}

// If options.convert_signature_to_tags() is set, will convert letters to
// uppercase and replace /, -, and .'s with _'s. This enables the standard
// SavedModel classification, regression, and prediction signatures to be used
// as uppercase INPUTS and OUTPUTS tags for streams and supports other common
// patterns.
const std::string MaybeConvertSignatureToTag(
    const std::string& name,
    const TensorFlowSessionFromSavedModelGeneratorOptions& options) {
  if (options.convert_signature_to_tags()) {
    std::string output;
    output.resize(name.length());
    std::transform(name.begin(), name.end(), output.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    output = absl::StrReplaceAll(
        output, {{"/", "_"}, {"-", "_"}, {".", "_"}, {":", "_"}});
    ABSL_LOG(INFO) << "Renamed TAG from: " << name << " to " << output;
    return output;
  } else {
    return name;
  }
}

}  // namespace

// TensorFlowSessionFromSavedModelGenerator is a MediaPipe packet generator
// that loads a trained TensorFlow model exported via SavedModel's exporter and
// returns a Packet containing a unique_ptr to a mediapipe::TensorFlowSession,
// which in turn contains a TensorFlow Session ready for execution and a map
// between tags and tensor names.
class TensorFlowSessionFromSavedModelGenerator : public PacketGenerator {
 public:
  static absl::Status FillExpectations(
      const PacketGeneratorOptions& extendable_options,
      PacketTypeSet* input_side_packets, PacketTypeSet* output_side_packets) {
    const TensorFlowSessionFromSavedModelGeneratorOptions& options =
        extendable_options.GetExtension(
            TensorFlowSessionFromSavedModelGeneratorOptions::ext);
    const bool has_exactly_one_model =
        options.saved_model_path().empty() ==
        input_side_packets->HasTag(kStringSavedModelPath);
    RET_CHECK(has_exactly_one_model)
        << "Must have exactly one of saved model filepath in options or "
           "input_side_packets STRING_MODEL_FILE_PATH";
    // Path of savedmodel.
    if (input_side_packets->HasTag(kStringSavedModelPath)) {
      input_side_packets->Tag(kStringSavedModelPath).Set<std::string>();
    }
    // Set Signature_def.
    if (input_side_packets->HasTag(kStringSignatureName)) {
      input_side_packets->Tag(kStringSignatureName).Set<std::string>();
    }
    // A TensorFlow model loaded and ready for use along with tensor
    output_side_packets->Tag(kSessionTag).Set<TensorFlowSession>();
    return absl::OkStatus();
  }

  static absl::Status Generate(const PacketGeneratorOptions& extendable_options,
                               const PacketSet& input_side_packets,
                               PacketSet* output_side_packets) {
    const TensorFlowSessionFromSavedModelGeneratorOptions& options =
        extendable_options.GetExtension(
            TensorFlowSessionFromSavedModelGeneratorOptions::ext);
    std::string path =
        input_side_packets.HasTag(kStringSavedModelPath)
            ? input_side_packets.Tag(kStringSavedModelPath).Get<std::string>()
            : options.saved_model_path();
    if (options.load_latest_model()) {
      RET_CHECK_OK(GetLatestDirectory(&path));
    }

    // Set user specified tags properly.
    // If no tags specified will use tensorflow::kSavedModelTagServe by default.
    std::unordered_set<std::string> tags_set;
    for (const std::string& tag : options.saved_model_tag()) {
      tags_set.insert(tag);
    }
    if (tags_set.empty()) {
      tags_set.insert(tensorflow::kSavedModelTagServe);
    }

    tensorflow::RunOptions run_options;
    tensorflow::SessionOptions session_options;
    session_options.config = options.session_config();
    auto saved_model = absl::make_unique<tensorflow::SavedModelBundle>();
    absl::Status status = tensorflow::LoadSavedModel(
        session_options, run_options, path, tags_set, saved_model.get());
    if (!status.ok()) {
      return absl::Status(static_cast<absl::StatusCode>(status.code()),
                          status.ToString());
    }
    auto session = absl::make_unique<TensorFlowSession>();
    session->session = std::move(saved_model->session);

    // Use input side packet to overwrite signature name in options.
    std::string signature_name =
        input_side_packets.HasTag(kStringSignatureName)
            ? input_side_packets.Tag(kStringSignatureName).Get<std::string>()
            : options.signature_name();
    RET_CHECK(!signature_name.empty());
    const auto& signature_def_map = saved_model->meta_graph_def.signature_def();
    if (signature_def_map.find(signature_name) == signature_def_map.end()) {
      return absl::NotFoundError(absl::StrFormat(
          "Signature name '%s' does not exist in the loaded signature def",
          signature_name));
    }
    const auto& signature_def = signature_def_map.at(signature_name);
    for (const auto& input_signature : signature_def.inputs()) {
      session->tag_to_tensor_map[MaybeConvertSignatureToTag(
          input_signature.first, options)] = input_signature.second.name();
    }
    for (const auto& output_signature : signature_def.outputs()) {
      session->tag_to_tensor_map[MaybeConvertSignatureToTag(
          output_signature.first, options)] = output_signature.second.name();
    }

    output_side_packets->Tag(kSessionTag) = Adopt(session.release());
    return absl::OkStatus();
  }
};
REGISTER_PACKET_GENERATOR(TensorFlowSessionFromSavedModelGenerator);

}  // namespace mediapipe
