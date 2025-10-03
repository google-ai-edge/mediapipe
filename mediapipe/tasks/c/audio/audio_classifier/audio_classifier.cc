/* Copyright 2025 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/c/audio/audio_classifier/audio_classifier.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "Eigen/Core"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/c/audio/core/common.h"
#include "mediapipe/tasks/c/audio/core/running_mode_converter.h"
#include "mediapipe/tasks/c/components/containers/classification_result.h"
#include "mediapipe/tasks/c/components/containers/classification_result_converter.h"
#include "mediapipe/tasks/c/components/processors/classifier_options_converter.h"
#include "mediapipe/tasks/c/core/base_options_converter.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"
#include "mediapipe/tasks/cc/audio/audio_classifier/audio_classifier.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"

// C API wrapper for the MediaPipe AudioClassifier.
struct MpAudioClassifierInternal {
  std::unique_ptr<::mediapipe::tasks::audio::audio_classifier::AudioClassifier>
      classifier;
};

namespace mediapipe::tasks::c::audio::audio_classifier {

namespace {

using ::mediapipe::tasks::audio::audio_classifier::AudioClassifier;
using ::mediapipe::tasks::audio::audio_classifier::AudioClassifierOptions;
using ::mediapipe::tasks::c::audio::core::CppConvertToRunningMode;
using ::mediapipe::tasks::c::components::containers::
    CppCloseClassificationResult;
using ::mediapipe::tasks::c::components::containers::
    CppConvertToClassificationResult;
using ::mediapipe::tasks::c::components::processors::
    CppConvertToClassifierOptions;
using ::mediapipe::tasks::c::core::CppConvertToBaseOptions;
using ::mediapipe::tasks::c::core::ToMpStatus;
using ::mediapipe::tasks::components::containers::ClassificationResult;

// A static result callback function that calls the provided C result callback.
void CppResultCallback(
    absl::StatusOr<ClassificationResult> result,
    MpAudioClassifierOptions::result_callback_fn c_callback) {
  if (!result.ok()) {
    c_callback(ToMpStatus(result.status()), nullptr);
    return;
  }

  auto classification_result = std::make_unique<::ClassificationResult>();
  CppConvertToClassificationResult(*result, classification_result.get());

  // The C++ async callback receives one classification result at a time, so we
  // create a new result with a single result and call the callback.
  MpAudioClassifierResult c_result = {
      .results = classification_result.release(), .results_count = 1};
  c_callback(kMpOk, &c_result);
  auto status = MpAudioClassifierCloseResult(&c_result);
  if (status != kMpOk) {
    ABSL_LOG(ERROR) << "Failed to close classification result: " << status;
  }
}

mediapipe::Matrix CppConvertToMatrix(const MpAudioData* audio_data) {
  int num_channels = audio_data->num_channels;
  int num_samples_per_channel = audio_data->audio_data_size / num_channels;
  // Convert the buffer to a row-major matrix where rows represent samples and
  // columns represent channels and then transpose to a mediapipe::Matrix with
  // channels as rows and samples as columns.
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      interleaved_map(audio_data->audio_data, num_samples_per_channel,
                      num_channels);
  return interleaved_map.transpose();
}

absl::StatusOr<std::unique_ptr<AudioClassifier>> CppCreateAudioClassifier(
    const MpAudioClassifierOptions& options) {
  auto cpp_options = std::make_unique<AudioClassifierOptions>();
  CppConvertToBaseOptions(options.base_options, &cpp_options->base_options);
  CppConvertToClassifierOptions(options.classifier_options,
                                &cpp_options->classifier_options);
  MP_ASSIGN_OR_RETURN(cpp_options->running_mode,
                      CppConvertToRunningMode(options.running_mode));

  if (options.result_callback) {
    cpp_options->result_callback =
        [c_callback = options.result_callback](
            absl::StatusOr<ClassificationResult> result) {
          CppResultCallback(std::move(result), c_callback);
        };
  }
  return AudioClassifier::Create(std::move(cpp_options));
}

}  // namespace

MpStatus MpAudioClassifierCreate(struct MpAudioClassifierOptions* options,
                                 MpAudioClassifierPtr* classifier_out) {
  auto classifier = CppCreateAudioClassifier(*options);
  if (!classifier.ok()) {
    ABSL_LOG(ERROR) << "Failed to create AudioClassifier: "
                    << classifier.status();
    return ToMpStatus(classifier.status());
  }
  *classifier_out = new MpAudioClassifierInternal(std::move(*classifier));
  return kMpOk;
}

MpStatus MpAudioClassifierClassify(MpAudioClassifierPtr classifier,
                                   const MpAudioData* audio_data,
                                   MpAudioClassifierResult* result_out) {
  auto* cpp_classifier = classifier->classifier.get();
  mediapipe::Matrix audio_matrix = CppConvertToMatrix(audio_data);
  auto cpp_result =
      cpp_classifier->Classify(audio_matrix, audio_data->sample_rate);

  if (!cpp_result.ok()) {
    ABSL_LOG(ERROR) << "Classification failed: " << cpp_result.status();
    return ToMpStatus(cpp_result.status());
  } else if (cpp_result->empty()) {
    result_out->results = nullptr;
    result_out->results_count = 0;
    return kMpOk;
  }

  auto c_classifications =
      std::make_unique<::ClassificationResult[]>(cpp_result->size());
  result_out->results_count = cpp_result->size();
  for (int i = 0; i < result_out->results_count; ++i) {
    CppConvertToClassificationResult(cpp_result->at(i),
                                     &(c_classifications.get()[i]));
  }
  result_out->results = c_classifications.release();
  return kMpOk;
}

MpStatus MpAudioClassifierCloseResult(MpAudioClassifierResult* result) {
  if (result->results) {
    for (int i = 0; i < result->results_count; ++i) {
      CppCloseClassificationResult(&result->results[i]);
    }
    delete[] result->results;
  }
  result->results = nullptr;
  result->results_count = 0;
  return kMpOk;
}

MpStatus MpAudioClassifierClassifyAsync(MpAudioClassifierPtr classifier,
                                        const MpAudioData* audio_data,
                                        int64_t timestamp_ms) {
  auto* cpp_classifier = classifier->classifier.get();
  mediapipe::Matrix audio_matrix = CppConvertToMatrix(audio_data);

  auto cpp_status = cpp_classifier->ClassifyAsync(
      audio_matrix, audio_data->sample_rate, timestamp_ms);
  if (!cpp_status.ok()) {
    ABSL_LOG(ERROR) << "Classification failed: " << cpp_status;
    return ToMpStatus(cpp_status);
  }
  return kMpOk;
}

MpStatus MpAudioClassifierClose(MpAudioClassifierPtr classifier) {
  auto* cpp_classifier = classifier->classifier.get();
  auto cpp_status = cpp_classifier->Close();
  if (!cpp_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to close AudioClassifier: " << cpp_status;
    return ToMpStatus(cpp_status);
  }
  delete classifier;
  return kMpOk;
}

}  // namespace mediapipe::tasks::c::audio::audio_classifier

extern "C" {

MP_EXPORT MpStatus
MpAudioClassifierCreate(struct MpAudioClassifierOptions* options,
                        MpAudioClassifierPtr* classifier_out) {
  return mediapipe::tasks::c::audio::audio_classifier::MpAudioClassifierCreate(
      options, classifier_out);
}

MP_EXPORT MpStatus MpAudioClassifierClassify(
    MpAudioClassifierPtr classifier, const MpAudioData* audio_data,
    MpAudioClassifierResult* result_out) {
  return mediapipe::tasks::c::audio::audio_classifier::
      MpAudioClassifierClassify(classifier, audio_data, result_out);
}

MP_EXPORT MpStatus MpAudioClassifierClassifyAsync(
    MpAudioClassifierPtr classifier, const MpAudioData* audio_data,
    int64_t timestamp_ms) {
  return mediapipe::tasks::c::audio::audio_classifier::
      MpAudioClassifierClassifyAsync(classifier, audio_data, timestamp_ms);
}

MP_EXPORT MpStatus
MpAudioClassifierCloseResult(MpAudioClassifierResult* result) {
  return mediapipe::tasks::c::audio::audio_classifier::
      MpAudioClassifierCloseResult(result);
}

MP_EXPORT MpStatus MpAudioClassifierClose(MpAudioClassifierPtr classifier) {
  return mediapipe::tasks::c::audio::audio_classifier::MpAudioClassifierClose(
      classifier);
}

}  // extern "C"
