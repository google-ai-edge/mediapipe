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

#ifndef MEDIAPIPE_TASKS_CC_AUDIO_BENCHMARKS_EVAL_BENCHMARK_UTILS_H_
#define MEDIAPIPE_TASKS_CC_AUDIO_BENCHMARKS_EVAL_BENCHMARK_UTILS_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/declare.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mediapipe/tasks/cc/audio/audio_transcriber/audio_transcriber.h"
#include "mediapipe/tasks/cc/audio/text_to_speech/text_to_speech.h"
#include "odml/audio/transcription/speech_processor.h"  // from @odml
#include "omni/omni_engine.h"                           // from @litert_lm

ABSL_DECLARE_FLAG(std::string, cache_dir);

namespace mediapipe::tasks::audio::benchmarks {

inline constexpr absl::string_view kDefaultCacheDir = "/tmp/litert_omni_models";
inline constexpr absl::string_view kParakeetTdtModelName =
    "parakeet-tdt-0.6b-v3.litertlm";
inline constexpr absl::string_view kParakeetCtcModelName =
    "parakeet-ctc-0.6b.litertlm";
inline constexpr absl::string_view kWhisperTinyModelName =
    "whisper-tiny.litertlm";
inline constexpr absl::string_view kMoonshineTinyModelName =
    "moonshine-tiny.litertlm";
inline constexpr absl::string_view kQwen3AsrModelName =
    "qwen3-asr-0.6b.litertlm";
inline constexpr absl::string_view kTinyGemmaAsrModelName =
    "tinygemma-asr.litertlm";
inline constexpr absl::string_view kKokoroBaseModelName = "kokoro.litertlm";
inline constexpr absl::string_view kKokoroNoEspeakModelName =
    "kokoro_int8_cpu_gpu_vocoder_no_espeak.litertlm";
inline constexpr absl::string_view kQwen3TtsModelName =
    "qwen3_tts_int4.litertlm";
inline constexpr absl::string_view kPocketTtsModelName =
    "pocket_tts_int8.litertlm";
inline constexpr absl::string_view kPhononModelName = "phonon_int8.litertlm";
inline constexpr absl::string_view kDefaultKokoroVoice = "af_heart";
inline constexpr absl::string_view kDefaultKokoroLocale = "en-US";
inline constexpr absl::string_view kDefaultPocketTtsVoice = "alba";
inline constexpr absl::string_view kDefaultPocketTtsLocale = "en-US";
inline constexpr absl::string_view kDefaultPhononVoice = "freya";
inline constexpr absl::string_view kDefaultPhononLocale = "en-US";

inline constexpr int kDefaultCpuThreads = 4;
inline constexpr int kSampleRate16kHz = 16000;
inline constexpr int kSampleRate24kHz = 24000;
inline constexpr int kSampleRate48kHz = 48000;
inline constexpr float kDefaultAsrOverlapRatio = 0.75f;

struct AsrEvalCase {
  absl::string_view wav_name;
  absl::string_view expected_substr;
  absl::string_view reference_transcript;
};

struct AsrRunMetrics {
  std::string final_transcript;
  int interim_updates = 0;
  double first_interim_latency_ms = 0.0;
  double first_interim_audio_sec = 0.0;
  double total_latency_ms = 0.0;
  double finish_stream_ms = 0.0;
  double p50_chunk_ms = 0.0;
  double p90_chunk_ms = 0.0;
  double max_chunk_ms = 0.0;
  double rtf = 0.0;
};

struct WaveformComparison {
  size_t omni_samples = 0;
  size_t mp_samples = 0;
  float max_abs_diff = 0.0f;
  double snr_db = 0.0;
};

std::string GetBenchmarkCacheDir();
absl::StatusOr<std::string> ResolveStagedModel(absl::string_view filename);
int GetCurrentProcessThreadCount();

std::string NormalizeTranscript(absl::string_view text);
std::vector<std::string> SplitNormalizedWords(absl::string_view text);
double ComputeWordErrorRate(absl::string_view reference,
                            absl::string_view hypothesis);
double ComputeMultilingualErrorRate(absl::string_view reference,
                                    absl::string_view hypothesis,
                                    absl::string_view locale = "en-US");

double Percentile(std::vector<double> vals, double p);
double Mean(absl::Span<const double> vals);
double ComputeRmsEnergy(absl::Span<const float> samples);
WaveformComparison CompareWaveforms(absl::Span<const float> a,
                                    absl::Span<const float> b);

void AppendSegmentTranscript(std::string& dst, absl::string_view segment);

absl::StatusOr<std::unique_ptr<::odml::audio::transcription::SpeechProcessor>>
CreatePrewarmedOmniAsr(absl::string_view model_path, int sample_rate_hz);

absl::StatusOr<std::unique_ptr<
    ::mediapipe::tasks::audio::audio_transcriber::AudioTranscriber>>
CreatePrewarmedMediaPipeAsr(absl::string_view model_path, int sample_rate_hz);

absl::StatusOr<AsrRunMetrics> RunOmniAsr(
    ::odml::audio::transcription::SpeechProcessor& omni_proc,
    absl::Span<const float> audio_samples, int sample_rate_hz,
    size_t chunk_samples);

absl::StatusOr<AsrRunMetrics> RunMediaPipeAsr(
    ::mediapipe::tasks::audio::audio_transcriber::AudioTranscriber&
        mp_transcriber,
    absl::Span<const float> audio_samples, int sample_rate_hz,
    size_t chunk_samples);

absl::StatusOr<std::unique_ptr<::litert::omni::OmniEngine>>
CreateOmniKokoroEngine(absl::string_view model_path, absl::string_view voice,
                       absl::string_view locale);

absl::StatusOr<std::string> EnsurePocketTtsLitertLmModel();
absl::StatusOr<std::string> EnsurePhononLitertLmModel();

absl::StatusOr<
    std::unique_ptr<::mediapipe::tasks::audio::text_to_speech::TextToSpeech>>
CreateMediaPipeKokoroTts(
    absl::string_view model_path,
    absl::string_view default_voice = kDefaultKokoroVoice,
    absl::string_view default_locale = kDefaultKokoroLocale);

absl::StatusOr<std::vector<float>> RunOmniKokoroSynthesize(
    ::litert::omni::OmniEngine& omni_engine, absl::string_view text);

}  // namespace mediapipe::tasks::audio::benchmarks

#endif  // MEDIAPIPE_TASKS_CC_AUDIO_BENCHMARKS_EVAL_BENCHMARK_UTILS_H_
