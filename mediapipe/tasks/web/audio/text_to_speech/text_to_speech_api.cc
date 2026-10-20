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

// Emscripten Embind bindings for MediaPipe TextToSpeech on the Web,
// implemented as a thin API shim over the MediaPipe TextToSpeech C API.
#ifdef __EMSCRIPTEN__

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/c/audio/text_to_speech/text_to_speech.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "runtime/util/logging.h"  // from @litert_lm

namespace mediapipe {
namespace {

EM_JS(void, TextToSpeechThrowJsError, (const char* message),
      { throw new Error(UTF8ToString(message)); });

void ReportError(const std::string& msg) {
  TextToSpeechThrowJsError(msg.c_str());
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
    MpTextToSpeechFreeError(error_msg);
  }
  ReportError(msg);
}

emscripten::val ToJsFloat32Array(const float* samples, size_t num_samples) {
  if (!samples || num_samples == 0) {
    return emscripten::val::global("Float32Array").new_(0);
  }
  return emscripten::val::global("Float32Array")
      .new_(emscripten::typed_memory_view(num_samples, samples));
}

intptr_t CreateTtsWithOptions(const MpTextToSpeechOptions& options) {
  MpTextToSpeechPtr tts = nullptr;
  char* error_msg = nullptr;
  MpStatus status = MpTextToSpeechCreate(&options, &tts, &error_msg);
  if (status != kMpOk || !tts) {
    ConsumeError("Failed to create TextToSpeech", error_msg);
    return 0;
  }
  return reinterpret_cast<intptr_t>(tts);
}

}  // namespace

intptr_t CreateTextToSpeech(std::string model_path, int delegate,
                            std::string default_voice_id,
                            std::string default_locale, float speaking_rate,
                            int target_bucket, std::string cache_directory) {
  DisableLiteRtLmLogging();
  MpTextToSpeechOptions options = {};
  MpTextToSpeechOptionsInitDefault(&options);
  options.base_options.model_asset_path = model_path.c_str();
  options.base_options.delegate =
      (delegate < 0) ? MP_DELEGATE_CPU : MP_DELEGATE_GPU;
  options.default_voice_id =
      default_voice_id.empty() ? nullptr : default_voice_id.c_str();
  options.default_locale =
      default_locale.empty() ? nullptr : default_locale.c_str();
  options.speaking_rate = speaking_rate > 0.0f ? speaking_rate : 1.0f;
  options.target_bucket = target_bucket > 0 ? target_bucket : 0;
  options.cache_directory =
      cache_directory.empty() ? nullptr : cache_directory.c_str();
  return CreateTtsWithOptions(options);
}

intptr_t CreateTextToSpeechFromBuffer(double model_ptr, double model_size,
                                      int delegate,
                                      std::string default_voice_id,
                                      std::string default_locale,
                                      float speaking_rate, int target_bucket,
                                      std::string cache_directory) {
  DisableLiteRtLmLogging();
  MpTextToSpeechOptions options = {};
  MpTextToSpeechOptionsInitDefault(&options);
  if (model_ptr > 0 && model_size > 0) {
    options.base_options.model_asset_buffer =
        reinterpret_cast<const char*>(static_cast<uintptr_t>(model_ptr));
    options.base_options.model_asset_buffer_count =
        static_cast<unsigned int>(model_size);
  }
  options.base_options.delegate =
      (delegate < 0) ? MP_DELEGATE_CPU : MP_DELEGATE_GPU;
  options.default_voice_id =
      default_voice_id.empty() ? nullptr : default_voice_id.c_str();
  options.default_locale =
      default_locale.empty() ? nullptr : default_locale.c_str();
  options.speaking_rate = speaking_rate > 0.0f ? speaking_rate : 1.0f;
  options.target_bucket = target_bucket > 0 ? target_bucket : 0;
  options.cache_directory =
      cache_directory.empty() ? nullptr : cache_directory.c_str();
  return CreateTtsWithOptions(options);
}

void DeleteTextToSpeech(intptr_t tts_ptr) {
  DisableLiteRtLmLogging();
  if (auto* tts = reinterpret_cast<MpTextToSpeechPtr>(tts_ptr)) {
    MpTextToSpeechDestroy(tts);
  }
}

emscripten::val ListTextToSpeechVoices(intptr_t tts_ptr) {
  DisableLiteRtLmLogging();
  auto* tts = reinterpret_cast<MpTextToSpeechPtr>(tts_ptr);
  if (!tts) {
    ReportError("TextToSpeech has not been created!");
    return emscripten::val::null();
  }
  MpTtsVoiceList voice_list = {};
  char* error_msg = nullptr;
  if (MpTextToSpeechListVoices(tts, &voice_list, &error_msg) != kMpOk) {
    MpTextToSpeechVoiceListFree(&voice_list);
    ConsumeError("Failed to list voices", error_msg);
    return emscripten::val::null();
  }
  emscripten::val js_voices = emscripten::val::array();
  for (size_t i = 0; i < voice_list.num_voices; ++i) {
    const MpTtsVoiceInfo& v = voice_list.voices[i];
    emscripten::val js_v = emscripten::val::object();
    js_v.set("voiceId", std::string(v.voice_id ? v.voice_id : ""));
    js_v.set("displayName", std::string(v.display_name ? v.display_name : ""));
    js_v.set("locale", std::string(v.locale ? v.locale : ""));
    js_v.set("sampleRateHz", v.sample_rate_hz);
    js_v.set("isDownloaded", v.is_downloaded);
    js_voices.set(i, js_v);
  }
  MpTextToSpeechVoiceListFree(&voice_list);
  return js_voices;
}

emscripten::val SynthesizeTextToSpeech(intptr_t tts_ptr, std::string text,
                                       std::string voice_id,
                                       std::string locale) {
  DisableLiteRtLmLogging();
  auto* tts = reinterpret_cast<MpTextToSpeechPtr>(tts_ptr);
  if (!tts) {
    ReportError("TextToSpeech has not been created!");
    return emscripten::val::null();
  }
  MpTtsSynthesisOptions synth_opts = {};
  MpTtsSynthesisOptionsInitDefault(&synth_opts);
  synth_opts.voice_id = voice_id.empty() ? nullptr : voice_id.c_str();
  synth_opts.locale = locale.empty() ? nullptr : locale.c_str();

  MpTtsAudioResult res = {};
  char* error_msg = nullptr;
  if (MpTextToSpeechSynthesize(tts, text.c_str(), &synth_opts, &res,
                               &error_msg) != kMpOk) {
    MpTextToSpeechFreeAudioResult(&res);
    ConsumeError("Failed to synthesize speech", error_msg);
    return emscripten::val::null();
  }
  emscripten::val js_res = emscripten::val::object();
  js_res.set("samples", ToJsFloat32Array(res.samples, res.num_samples));
  js_res.set("sampleRateHz", res.sample_rate_hz);
  MpTextToSpeechFreeAudioResult(&res);
  return js_res;
}

intptr_t StartTextToSpeechStream(intptr_t tts_ptr, std::string initial_text,
                                 std::string voice_id, std::string locale,
                                 bool finish_input) {
  DisableLiteRtLmLogging();
  auto* tts = reinterpret_cast<MpTextToSpeechPtr>(tts_ptr);
  if (!tts) {
    ReportError("TextToSpeech has not been created!");
    return 0;
  }
  MpTtsSynthesisOptions synth_opts = {};
  MpTtsSynthesisOptionsInitDefault(&synth_opts);
  synth_opts.voice_id = voice_id.empty() ? nullptr : voice_id.c_str();
  synth_opts.locale = locale.empty() ? nullptr : locale.c_str();

  MpTtsStreamPtr stream = nullptr;
  char* error_msg = nullptr;
  if (MpTextToSpeechStartStream(tts, initial_text.c_str(), &synth_opts,
                                finish_input, &stream, &error_msg) != kMpOk ||
      !stream) {
    ConsumeError("Failed to start TextToSpeech stream", error_msg);
    return 0;
  }
  return reinterpret_cast<intptr_t>(stream);
}

emscripten::val ReadNextTextToSpeechChunk(intptr_t stream_ptr) {
  DisableLiteRtLmLogging();
  auto* stream = reinterpret_cast<MpTtsStreamPtr>(stream_ptr);
  if (!stream) {
    ReportError("TextToSpeechStream has not been created!");
    return emscripten::val::null();
  }
  MpTtsAudioResult chunk = {};
  int32_t chunk_index = 0;
  bool is_final = false;
  char* error_msg = nullptr;
  MpStatus status =
      MpTtsStreamReadNext(stream, &chunk, &chunk_index, &is_final, &error_msg);
  if (status == kMpOutOfRange) {
    MpTextToSpeechFreeAudioResult(&chunk);
    return emscripten::val::null();
  }
  if (status != kMpOk) {
    MpTextToSpeechFreeAudioResult(&chunk);
    ConsumeError("Failed to read next TextToSpeech chunk", error_msg);
    return emscripten::val::null();
  }
  emscripten::val js_chunk = emscripten::val::object();
  js_chunk.set("samples", ToJsFloat32Array(chunk.samples, chunk.num_samples));
  js_chunk.set("sampleRateHz", chunk.sample_rate_hz);
  js_chunk.set("chunkIndex", chunk_index);
  js_chunk.set("isFinal", is_final);
  MpTextToSpeechFreeAudioResult(&chunk);
  return js_chunk;
}

bool PushTextToSpeechStream(intptr_t stream_ptr, std::string text_delta) {
  DisableLiteRtLmLogging();
  auto* stream = reinterpret_cast<MpTtsStreamPtr>(stream_ptr);
  if (!stream) {
    ReportError("TextToSpeechStream has not been created!");
    return false;
  }
  char* error_msg = nullptr;
  if (MpTtsStreamPushText(stream, text_delta.c_str(), &error_msg) != kMpOk) {
    ConsumeError("Failed to push text to TextToSpeech stream", error_msg);
    return false;
  }
  return true;
}

bool FinishTextToSpeechStreamInput(intptr_t stream_ptr) {
  DisableLiteRtLmLogging();
  auto* stream = reinterpret_cast<MpTtsStreamPtr>(stream_ptr);
  if (!stream) {
    ReportError("TextToSpeechStream has not been created!");
    return false;
  }
  char* error_msg = nullptr;
  if (MpTtsStreamFinishInput(stream, &error_msg) != kMpOk) {
    ConsumeError("Failed to finish TextToSpeech stream input", error_msg);
    return false;
  }
  return true;
}

void CancelTextToSpeechStream(intptr_t stream_ptr) {
  if (auto* stream = reinterpret_cast<MpTtsStreamPtr>(stream_ptr)) {
    MpTtsStreamCancel(stream);
  }
}

void DeleteTextToSpeechStream(intptr_t stream_ptr) {
  if (auto* stream = reinterpret_cast<MpTtsStreamPtr>(stream_ptr)) {
    MpTtsStreamDestroy(stream);
  }
}

void CancelTextToSpeech(intptr_t tts_ptr) {
  if (auto* tts = reinterpret_cast<MpTextToSpeechPtr>(tts_ptr)) {
    MpTextToSpeechCancel(tts);
  }
}

EMSCRIPTEN_BINDINGS(mediapipe_text_to_speech_wasm) {
  emscripten::function("createTextToSpeech", &CreateTextToSpeech,
                       emscripten::async());
  emscripten::function("createTextToSpeechFromBuffer",
                       &CreateTextToSpeechFromBuffer, emscripten::async());
  emscripten::function("deleteTextToSpeech", &DeleteTextToSpeech);
  emscripten::function("listTextToSpeechVoices", &ListTextToSpeechVoices,
                       emscripten::async());
  emscripten::function("synthesizeTextToSpeech", &SynthesizeTextToSpeech,
                       emscripten::async());
  emscripten::function("startTextToSpeechStream", &StartTextToSpeechStream,
                       emscripten::async());
  emscripten::function("readNextTextToSpeechChunk", &ReadNextTextToSpeechChunk,
                       emscripten::async());
  emscripten::function("pushTextToSpeechStream", &PushTextToSpeechStream,
                       emscripten::async());
  emscripten::function("finishTextToSpeechStreamInput",
                       &FinishTextToSpeechStreamInput, emscripten::async());
  emscripten::function("cancelTextToSpeechStream", &CancelTextToSpeechStream);
  emscripten::function("deleteTextToSpeechStream", &DeleteTextToSpeechStream);
  emscripten::function("cancelTextToSpeech", &CancelTextToSpeech);
}

}  // namespace mediapipe

#endif  // __EMSCRIPTEN__
