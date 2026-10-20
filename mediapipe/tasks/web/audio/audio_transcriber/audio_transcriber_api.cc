/* Copyright 2026 The MediaPipe Authors.

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

// Emscripten Embind bindings for MediaPipe AudioTranscriber on the Web,
// implemented as a thin API shim over the MediaPipe AudioTranscriber C API.
#ifdef __EMSCRIPTEN__

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/c/audio/audio_transcriber/audio_transcriber.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "runtime/util/logging.h"  // from @litert_lm

namespace mediapipe {
namespace {

EM_JS(void, AudioTranscriberThrowJsError, (const char* message),
      { throw new Error(UTF8ToString(message)); });

void ReportError(const std::string& msg) {
  AudioTranscriberThrowJsError(msg.c_str());
}

void DisableLiteRtLmLogging() {
  setenv("LITERT_DISABLE_OPENCL", "1", /*overwrite=*/1);
  setenv("LITERT_DISABLE_OPENGL", "1", /*overwrite=*/1);
  litert::lm::SetMinLogSeverity(litert::lm::LogSeverity::kSilent);
}

void ConsumeError(absl::string_view prefix, char* error_msg) {
  const char* detail = (error_msg && error_msg[0] != '\0')
                           ? error_msg
                           : "Unknown error (or WASM out-of-memory)";
  std::string msg = absl::StrCat(prefix, ": ", detail);
  if (error_msg) {
    MpErrorFree(error_msg);
  }
  ReportError(msg);
}

std::vector<float> ExtractFloatArrayFromJs(
    const emscripten::val& float_array_val) {
  if (float_array_val.isUndefined() || float_array_val.isNull()) {
    return {};
  }
  if (float_array_val.isArray()) {
    return emscripten::vecFromJSArray<float>(float_array_val);
  }
  emscripten::val len_val = float_array_val["length"];
  if (len_val.isUndefined() || len_val.isNull()) {
    return {};
  }
  const size_t length = len_val.as<size_t>();
  if (length == 0) {
    return {};
  }
  std::vector<float> float_samples(length);
  emscripten::val memory_view = emscripten::val(
      emscripten::typed_memory_view(length, float_samples.data()));
  memory_view.call<void>("set", float_array_val);
  return float_samples;
}

MpAudioTranscriberModelArchitecture ToModelArchitecture(int arch) {
  switch (arch) {
    case 1:
      return MP_AUDIO_TRANSCRIBER_MODEL_ARCHITECTURE_OMNI;
    default:
      return MP_AUDIO_TRANSCRIBER_MODEL_ARCHITECTURE_AUTO;
  }
}

emscripten::val ToJsTranscriberResult(const MpAudioTranscriberResult& res) {
  emscripten::val js_result = emscripten::val::object();
  emscripten::val segments_arr = emscripten::val::array();
  std::string full_text;
  for (uint32_t i = 0; i < res.segments_count; ++i) {
    const MpTranscriptionSegment& seg = res.segments[i];
    std::string transcript = seg.transcript ? seg.transcript : "";
    if (!transcript.empty()) {
      if (!full_text.empty() && full_text.back() != ' ' &&
          transcript.front() != ' ') {
        full_text.push_back(' ');
      }
      full_text.append(transcript);
    }
    emscripten::val js_seg = emscripten::val::object();
    js_seg.set("transcript", transcript);
    js_seg.set("isFinal", seg.is_final);
    js_seg.set("fromTimestampMs", static_cast<double>(seg.from_timestamp_ms));
    js_seg.set("toTimestampMs", static_cast<double>(seg.to_timestamp_ms));
    segments_arr.set(i, js_seg);
  }
  js_result.set("segments", segments_arr);
  js_result.set("text", full_text);
  return js_result;
}

intptr_t CreateTranscriberWithOptions(
    const MpAudioTranscriberOptions& options) {
  MpAudioTranscriberPtr transcriber = nullptr;
  char* error_msg = nullptr;
  MpStatus status =
      MpAudioTranscriberCreate(&options, &transcriber, &error_msg);
  if (status != kMpOk || !transcriber) {
    ConsumeError("Failed to create AudioTranscriber", error_msg);
    return 0;
  }
  return reinterpret_cast<intptr_t>(transcriber);
}

}  // namespace

intptr_t CreateAudioTranscriber(std::string model_path, int delegate,
                                int sample_rate_hz, bool enable_interim_results,
                                int model_architecture, std::string prompt,
                                std::string cache_dir) {
  DisableLiteRtLmLogging();
  MpAudioTranscriberOptions options = {};
  options.base_options.model_asset_path = model_path.c_str();
  options.base_options.delegate =
      (delegate < 0) ? MP_DELEGATE_CPU : MP_DELEGATE_GPU;
  options.sample_rate_hz = sample_rate_hz > 0 ? sample_rate_hz : 16000;
  options.enable_interim_results = enable_interim_results;
  options.model_architecture = ToModelArchitecture(model_architecture);
  options.prompt = prompt.empty() ? nullptr : prompt.c_str();
  options.cache_dir = cache_dir.empty() ? nullptr : cache_dir.c_str();
  return CreateTranscriberWithOptions(options);
}

intptr_t CreateAudioTranscriberFromBuffer(double model_ptr, double model_size,
                                          int delegate, int sample_rate_hz,
                                          bool enable_interim_results,
                                          int model_architecture,
                                          std::string prompt,
                                          std::string cache_dir) {
  DisableLiteRtLmLogging();
  MpAudioTranscriberOptions options = {};
  if (model_ptr > 0 && model_size > 0) {
    options.base_options.model_asset_buffer =
        reinterpret_cast<const char*>(static_cast<uintptr_t>(model_ptr));
    options.base_options.model_asset_buffer_count =
        static_cast<unsigned int>(model_size);
  }
  options.base_options.delegate =
      (delegate < 0) ? MP_DELEGATE_CPU : MP_DELEGATE_GPU;
  options.sample_rate_hz = sample_rate_hz > 0 ? sample_rate_hz : 16000;
  options.enable_interim_results = enable_interim_results;
  options.model_architecture = ToModelArchitecture(model_architecture);
  options.prompt = prompt.empty() ? nullptr : prompt.c_str();
  options.cache_dir = cache_dir.empty() ? nullptr : cache_dir.c_str();
  return CreateTranscriberWithOptions(options);
}

void DeleteAudioTranscriber(intptr_t transcriber_ptr) {
  DisableLiteRtLmLogging();
  if (auto* transcriber =
          reinterpret_cast<MpAudioTranscriberPtr>(transcriber_ptr)) {
    char* error_msg = nullptr;
    MpAudioTranscriberClose(transcriber, &error_msg);
    if (error_msg) {
      MpErrorFree(error_msg);
    }
  }
}

emscripten::val TranscribeAudio(intptr_t transcriber_ptr,
                                emscripten::val audio_samples_val,
                                bool end_of_stream) {
  DisableLiteRtLmLogging();
  auto* transcriber = reinterpret_cast<MpAudioTranscriberPtr>(transcriber_ptr);
  if (!transcriber) {
    ReportError("AudioTranscriber has not been created!");
    return emscripten::val::null();
  }
  std::vector<float> samples = ExtractFloatArrayFromJs(audio_samples_val);
  MpAudioTranscriberResult res = {};
  char* error_msg = nullptr;
  if (MpAudioTranscriberTranscribe(
          transcriber, samples.empty() ? nullptr : samples.data(),
          samples.size(), end_of_stream, &res, &error_msg) != kMpOk) {
    MpAudioTranscriberCloseResult(&res);
    ConsumeError("Failed to transcribe audio", error_msg);
    return emscripten::val::null();
  }
  emscripten::val js_res = ToJsTranscriberResult(res);
  MpAudioTranscriberCloseResult(&res);
  return js_res;
}

emscripten::val FinishAudioTranscriberStream(intptr_t transcriber_ptr) {
  DisableLiteRtLmLogging();
  auto* transcriber = reinterpret_cast<MpAudioTranscriberPtr>(transcriber_ptr);
  if (!transcriber) {
    ReportError("AudioTranscriber has not been created!");
    return emscripten::val::null();
  }
  MpAudioTranscriberResult res = {};
  char* error_msg = nullptr;
  if (MpAudioTranscriberFinishStream(transcriber, &res, &error_msg) != kMpOk) {
    MpAudioTranscriberCloseResult(&res);
    ConsumeError("Failed to finish AudioTranscriber stream", error_msg);
    return emscripten::val::null();
  }
  emscripten::val js_res = ToJsTranscriberResult(res);
  MpAudioTranscriberCloseResult(&res);
  return js_res;
}

bool PrewarmAudioTranscriber(intptr_t transcriber_ptr) {
  DisableLiteRtLmLogging();
  auto* transcriber = reinterpret_cast<MpAudioTranscriberPtr>(transcriber_ptr);
  if (!transcriber) {
    ReportError("AudioTranscriber has not been created!");
    return false;
  }
  char* error_msg = nullptr;
  if (MpAudioTranscriberPrewarm(transcriber, &error_msg) != kMpOk) {
    ConsumeError("Failed to prewarm AudioTranscriber", error_msg);
    return false;
  }
  return true;
}

EMSCRIPTEN_BINDINGS(mediapipe_audio_transcriber_wasm) {
  emscripten::function("createAudioTranscriber", &CreateAudioTranscriber,
                       emscripten::async());
  emscripten::function("createAudioTranscriberFromBuffer",
                       &CreateAudioTranscriberFromBuffer, emscripten::async());
  emscripten::function("deleteAudioTranscriber", &DeleteAudioTranscriber);
  emscripten::function("transcribeAudio", &TranscribeAudio,
                       emscripten::async());
  emscripten::function("finishAudioTranscriberStream",
                       &FinishAudioTranscriberStream, emscripten::async());
  emscripten::function("prewarmAudioTranscriber", &PrewarmAudioTranscriber,
                       emscripten::async());
}

}  // namespace mediapipe

#endif  // __EMSCRIPTEN__
