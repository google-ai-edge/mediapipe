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

#if !defined(__ANDROID__)
#include "mediapipe/framework/port/file_helpers.h"
#endif
#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session.h"
#include "mediapipe/calculators/tensorflow/tensorflow_session_from_saved_model_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

namespace mediapipe {

namespace {
static constexpr char kStringSavedModelPath[] = "STRING_SAVED_MODEL_PATH";

// Given the path to a directory containing multiple tensorflow saved models
// in subdirectories, replaces path with the alphabetically last subdirectory.
::mediapipe::Status GetLatestDirectory(std::string* path) {
#if defined(__ANDROID__)
  return ::mediapipe::UnimplementedError(
      "GetLatestDirectory is not implemented on Android");
#else
  std::vector<std::string> saved_models;
  RET_CHECK_OK(file::MatchInTopSubdirectories(
      *path, tensorflow::kSavedModelFilenamePb, &saved_models));
  RET_CHECK_GT(saved_models.size(), 0)
      << "No exported bundles found in " << path;
  ::std::sort(saved_models.begin(), saved_models.end());
  *path = std::string(file::Dirname(saved_models.back()));
  return ::mediapipe::OkStatus();
#endif
}

// If options.convert_signature_to_tags() is set, will convert letters to
// uppercase and replace /'s and -'s with _'s. This enables the standard
// SavedModel classification, regression, and prediction signatures to be used
// as uppercase INPUTS and OUTPUTS tags for streams and supports other common
// patterns.
const std::string MaybeConvertSignatureToTag(
    const std::string& name,
    const TensorFlowSessionFromSavedModelCalculatorOptions& options) {
  if (options.convert_signature_to_tags()) {
    std::string output;
    output.resize(name.length());
    std::transform(name.begin(), name.end(), output.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    output = absl::StrReplaceAll(output, {{"/", "_"}});
    output = absl::StrReplaceAll(output, {{"-", "_"}});
    return output;
  } else {
    return name;
  }
}

}  // namespace

// TensorFlowSessionFromSavedModelCalculator is a MediaPipe packet calculator
// that loads a trained TensorFlow model exported via SavedModel's exporter (see
// go/savedmodel) and returns a Packet containing a unique_ptr to a
// mediapipe::TensorFlowSession, which in turn contains a TensorFlow Session
// ready for execution and a map between tags and tensor names.
//
// Example usage:
// node {
//   calculator: "TensorFlowSessionFromSavedModelCalculator"
//   output_side_packet: "SESSION:vod_session"
//   options {
//     [mediapipe.TensorFlowSessionFromSavedModelCalculatorOptions.ext]: {
//       signature_name: "serving_default"
//       saved_model_path: "path/to/model"
//     }
//   }
// }
class TensorFlowSessionFromSavedModelCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    const auto& options =
        cc->Options<TensorFlowSessionFromSavedModelCalculatorOptions>();
    const bool has_exactly_one_model =
        options.saved_model_path().empty() ==
        cc->InputSidePackets().HasTag(kStringSavedModelPath);
    RET_CHECK(has_exactly_one_model)
        << "Must have exactly one of saved model filepath in options or "
           "input_side_packets STRING_MODEL_FILE_PATH";
    // Path of savedmodel.
    if (cc->InputSidePackets().HasTag(kStringSavedModelPath)) {
      cc->InputSidePackets().Tag(kStringSavedModelPath).Set<std::string>();
    }
    // A TensorFlow model loaded and ready for use along with tensor
    cc->OutputSidePackets().Tag("SESSION").Set<TensorFlowSession>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    const auto& options =
        cc->Options<TensorFlowSessionFromSavedModelCalculatorOptions>();
    std::string path = cc->InputSidePackets().HasTag(kStringSavedModelPath)
                           ? cc->InputSidePackets()
                                 .Tag(kStringSavedModelPath)
                                 .Get<std::string>()
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
    ::tensorflow::Status status = tensorflow::LoadSavedModel(
        session_options, run_options, path, tags_set, saved_model.get());
    if (!status.ok()) {
      return ::mediapipe::Status(
          static_cast<::mediapipe::StatusCode>(status.code()),
          status.ToString());
    }

    auto session = absl::make_unique<TensorFlowSession>();
    session->session = std::move(saved_model->session);

    RET_CHECK(!options.signature_name().empty());
    const auto& signature_def_map = saved_model->meta_graph_def.signature_def();
    const auto& signature_def = signature_def_map.at(options.signature_name());
    for (const auto& input_signature : signature_def.inputs()) {
      session->tag_to_tensor_map[MaybeConvertSignatureToTag(
          input_signature.first, options)] = input_signature.second.name();
    }
    for (const auto& output_signature : signature_def.outputs()) {
      session->tag_to_tensor_map[MaybeConvertSignatureToTag(
          output_signature.first, options)] = output_signature.second.name();
    }

    cc->OutputSidePackets().Tag("SESSION").Set(Adopt(session.release()));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }
};

REGISTER_CALCULATOR(TensorFlowSessionFromSavedModelCalculator);

}  // namespace mediapipe
