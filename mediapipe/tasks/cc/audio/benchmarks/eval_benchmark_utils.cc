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

#include "mediapipe/tasks/cc/audio/benchmarks/eval_benchmark_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mediapipe/tasks/cc/audio/audio_transcriber/audio_transcriber.h"
#include "mediapipe/tasks/cc/audio/audio_transcriber/mediapipe_asr_processor.h"
#include "mediapipe/tasks/cc/audio/text_to_speech/mediapipe_tts_session.h"
#include "mediapipe/tasks/cc/audio/text_to_speech/text_to_speech.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "odml/audio/transcription/omni/speech_processor_omni.h"  // from @odml
#include "odml/audio/transcription/proto/omni_options.pb.h"       // from @odml
#include "odml/audio/transcription/speech_processor.h"            // from @odml
#include "odml/audio/transcription/transcription_result.h"        // from @odml
#include "omni/base/io_types.h"                           // from @litert_lm
#include "omni/base/model_utils.h"                        // from @litert_lm
#include "omni/multi_staged_session.h"                    // from @litert_lm
#include "omni/omni_engine.h"                             // from @litert_lm
#include "omni/omni_session.h"                            // from @litert_lm
#include "omni/text2speech/stream_text_source.h"          // from @litert_lm
#include "runtime/proto/text2speech_metadata.pb.h"        // from @litert_lm
#include "runtime/proto/text2speech_model_type.pb.h"      // from @litert_lm
#include "runtime/proto/text2speech_session_config.pb.h"  // from @litert_lm

ABSL_FLAG(std::string, cache_dir,
          std::string(mediapipe::tasks::audio::benchmarks::kDefaultCacheDir),
          "Directory containing staged .litertlm models.");

namespace mediapipe::tasks::audio::benchmarks {
namespace {

using ::litert::omni::AudioOutput;
using ::litert::omni::MultiStagedSession;
using ::litert::omni::OmniEngine;
using ::mediapipe::tasks::audio::audio_transcriber::AudioTranscriber;
using ::mediapipe::tasks::audio::audio_transcriber::AudioTranscriberOptions;
using ::mediapipe::tasks::audio::audio_transcriber::AudioTranscriberResult;
using ::mediapipe::tasks::audio::text_to_speech::TextToSpeech;
using ::odml::audio::transcription::OmniSpeechProcessorOptions;
using ::odml::audio::transcription::SpeechProcessor;
using ::odml::audio::transcription::TranscriptionResult;

constexpr double kMinNoisePower = 1e-20;
constexpr double kMaxSnrDb = 200.0;
constexpr double kMsPerSecond = 1000.0;
constexpr absl::string_view kLatin1FoldTable =
    "aaaaaa.ceeeeiiii.nooooo..uuuuy.y";

bool FoldLatin1Diacritic(unsigned char c2, std::string& out) {
  const unsigned char lower = static_cast<unsigned char>(c2 | 0x20);
  if (lower < 0xA0 || lower > 0xBF || c2 == 0x9F) return false;
  const char folded = kLatin1FoldTable[lower - 0xA0];
  if (folded == '.') return false;
  out.push_back(folded);
  return true;
}

const auto& ExtractSegments(const std::vector<TranscriptionResult>& res) {
  return res;
}
const auto& ExtractSegments(const AudioTranscriberResult& res) {
  return res.segments;
}

template <typename SegmentContainer>
void CollectSegments(const SegmentContainer& segments, AsrRunMetrics& metrics,
                     bool count_interims) {
  for (const auto& seg : segments) {
    if (seg.is_final) {
      AppendSegmentTranscript(metrics.final_transcript, seg.transcript);
    } else if (count_interims) {
      ++metrics.interim_updates;
    }
  }
}

template <typename Engine>
absl::StatusOr<AsrRunMetrics> RunAsrEngine(Engine& engine,
                                           absl::Span<const float> samples,
                                           int sample_rate_hz,
                                           size_t chunk_samples) {
  AsrRunMetrics metrics;
  std::vector<double> chunk_ms;
  const double dur_s = static_cast<double>(samples.size()) / sample_rate_hz;
  const bool is_unary = (chunk_samples == 0);
  const size_t step = is_unary ? samples.size() : chunk_samples;
  absl::Time t_start = absl::Now();
  for (size_t off = 0; off < samples.size(); off += step) {
    size_t n = std::min(step, samples.size() - off);
    absl::Time c0 = absl::Now();
    ABSL_ASSIGN_OR_RETURN(auto res,
                          engine.Transcribe(samples.subspan(off, n), is_unary));
    chunk_ms.push_back(absl::ToDoubleMilliseconds(absl::Now() - c0));
    CollectSegments(ExtractSegments(res), metrics, true);
  }
  if (!is_unary) {
    absl::Time f0 = absl::Now();
    ABSL_ASSIGN_OR_RETURN(auto final_res, engine.FinishStream());
    metrics.finish_stream_ms = absl::ToDoubleMilliseconds(absl::Now() - f0);
    CollectSegments(ExtractSegments(final_res), metrics, false);
  }
  metrics.total_latency_ms = absl::ToDoubleMilliseconds(absl::Now() - t_start);
  metrics.p50_chunk_ms = Percentile(chunk_ms, 0.50);
  metrics.p90_chunk_ms = Percentile(chunk_ms, 0.90);
  metrics.max_chunk_ms = Percentile(chunk_ms, 1.00);
  metrics.rtf = (metrics.total_latency_ms / kMsPerSecond) / dur_s;
  return metrics;
}

}  // namespace

std::string GetBenchmarkCacheDir() {
  std::string dir = absl::GetFlag(FLAGS_cache_dir);
  return dir.empty() ? std::string(kDefaultCacheDir) : dir;
}

absl::StatusOr<std::string> ResolveStagedModel(absl::string_view filename) {
  std::string cache_dir = GetBenchmarkCacheDir();
  std::string local = absl::StrCat(cache_dir, "/", filename);
  if (std::ifstream(local).good()) return local;
  std::string remote = absl::StrCat(
      "/", "cns", "/py-d/home/tensorflow-mobile-eng/litert-omni/models/",
      filename);
  return ::litert::omni::StageCnsFileIfNeeded(remote, cache_dir);
}

int GetCurrentProcessThreadCount() {
  std::ifstream status_file("/proc/self/status");
  for (std::string line; std::getline(status_file, line);) {
    if (!absl::StartsWith(line, "Threads:")) continue;
    int count = 0;
    for (char c : absl::string_view(line).substr(8)) {
      if (absl::ascii_isdigit(c)) count = count * 10 + (c - '0');
    }
    return count;
  }
  return -1;
}

std::string NormalizeTranscript(absl::string_view text) {
  std::string out;
  bool last_was_space = true;
  for (size_t i = 0; i < text.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    if (absl::ascii_isalnum(c)) {
      out.push_back(absl::ascii_tolower(c));
      last_was_space = false;
    } else if (c == 0xC3 && i + 1 < text.size()) {
      if (FoldLatin1Diacritic(static_cast<unsigned char>(text[++i]), out)) {
        last_was_space = false;
      }
    } else if (c == 0xC5 && i + 1 < text.size()) {
      const unsigned char c2 = static_cast<unsigned char>(text[++i]);
      if (c2 == 0x92 || c2 == 0x93) {
        out.append("oe");
        last_was_space = false;
      }
    } else if ((absl::ascii_isspace(c) || c == '\'' || c == '-') &&
               !last_was_space) {
      out.push_back(' ');
      last_was_space = true;
    }
  }
  if (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

std::vector<std::string> SplitNormalizedWords(absl::string_view text) {
  std::string norm = NormalizeTranscript(text);
  if (norm.empty()) return {};
  return absl::StrSplit(norm, ' ', absl::SkipEmpty());
}

double ComputeWordErrorRate(absl::string_view reference,
                            absl::string_view hypothesis) {
  std::vector<std::string> ref = SplitNormalizedWords(reference);
  std::vector<std::string> hyp = SplitNormalizedWords(hypothesis);
  if (ref.empty()) return hyp.empty() ? 0.0 : 1.0;
  const size_t n = ref.size();
  const size_t m = hyp.size();
  std::vector<std::vector<size_t>> dp(n + 1, std::vector<size_t>(m + 1, 0));
  for (size_t i = 0; i <= n; ++i) dp[i][0] = i;
  for (size_t j = 0; j <= m; ++j) dp[0][j] = j;
  for (size_t i = 1; i <= n; ++i) {
    for (size_t j = 1; j <= m; ++j) {
      size_t cost = (ref[i - 1] == hyp[j - 1]) ? 0 : 1;
      dp[i][j] = std::min(
          {dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost});
    }
  }
  return static_cast<double>(dp[n][m]) / static_cast<double>(n);
}

double Percentile(std::vector<double> vals, double p) {
  if (vals.empty()) return 0.0;
  std::sort(vals.begin(), vals.end());
  const int64_t max_idx = static_cast<int64_t>(vals.size() - 1);
  const int64_t idx = std::clamp(
      static_cast<int64_t>(std::llround(p * max_idx)), int64_t{0}, max_idx);
  return vals[idx];
}

double Mean(absl::Span<const double> vals) {
  if (vals.empty()) return 0.0;
  return std::accumulate(vals.begin(), vals.end(), 0.0) /
         static_cast<double>(vals.size());
}

double ComputeRmsEnergy(absl::Span<const float> samples) {
  if (samples.empty()) return 0.0;
  double sum_sq = 0.0;
  for (float s : samples) sum_sq += static_cast<double>(s) * s;
  return std::sqrt(sum_sq / static_cast<double>(samples.size()));
}

WaveformComparison CompareWaveforms(absl::Span<const float> a,
                                    absl::Span<const float> b) {
  WaveformComparison cmp{a.size(), b.size(), 0.0f, 0.0};
  const size_t n = std::min(a.size(), b.size());
  double sig_pow = 0.0;
  double noise_pow = 0.0;
  for (size_t i = 0; i < n; ++i) {
    float diff = std::abs(a[i] - b[i]);
    cmp.max_abs_diff = std::max(cmp.max_abs_diff, diff);
    sig_pow += static_cast<double>(a[i]) * a[i];
    noise_pow += static_cast<double>(diff) * diff;
  }
  cmp.snr_db = (noise_pow <= kMinNoisePower)
                   ? kMaxSnrDb
                   : 10.0 * std::log10(sig_pow / noise_pow);
  return cmp;
}

void AppendSegmentTranscript(std::string& dst, absl::string_view segment) {
  if (segment.empty()) return;
  if (!dst.empty() && dst.back() != ' ' && segment.front() != ' ') {
    dst.push_back(' ');
  }
  absl::StrAppend(&dst, segment);
}

absl::StatusOr<std::unique_ptr<SpeechProcessor>> CreatePrewarmedOmniAsr(
    absl::string_view model_path, int sample_rate_hz) {
  OmniSpeechProcessorOptions opts;
  opts.set_model_path(std::string(model_path));
  opts.set_sample_rate_hz(sample_rate_hz);
  opts.set_backend(OmniSpeechProcessorOptions::CPU);
  opts.set_overlap_ratio(kDefaultAsrOverlapRatio);
  opts.set_enable_interim_results(true);
  opts.mutable_advanced_options()->set_num_cpu_threads(kDefaultCpuThreads);
  opts.mutable_advanced_options()->set_cache_dir(GetBenchmarkCacheDir());
  ABSL_ASSIGN_OR_RETURN(
      auto proc,
      ::odml::audio::transcription::omni::CreateSpeechProcessorOmni(opts));
  ABSL_RETURN_IF_ERROR(proc->Prewarm());
  return proc;
}

absl::StatusOr<std::unique_ptr<AudioTranscriber>> CreatePrewarmedMediaPipeAsr(
    absl::string_view model_path, int sample_rate_hz) {
  absl::SetFlag(&FLAGS_mediapipe_asr_use_staged_scheduler, false);
  auto opts = std::make_unique<AudioTranscriberOptions>();
  opts->base_options.model_asset_path = std::string(model_path);
  opts->base_options.delegate = tasks::core::BaseOptions::Delegate::CPU;
  opts->base_options.delegate_options =
      tasks::core::BaseOptions::CpuOptions{.num_threads = kDefaultCpuThreads};
  opts->sample_rate_hz = sample_rate_hz;
  opts->model_architecture =
      ::mediapipe::tasks::audio::audio_transcriber::ModelArchitecture::kOmni;
  opts->enable_interim_results = true;
  opts->cache_dir = GetBenchmarkCacheDir();
  ABSL_ASSIGN_OR_RETURN(auto tr, AudioTranscriber::Create(std::move(opts)));
  ABSL_RETURN_IF_ERROR(tr->Prewarm());
  return tr;
}

absl::StatusOr<AsrRunMetrics> RunOmniAsr(SpeechProcessor& omni_proc,
                                         absl::Span<const float> audio_samples,
                                         int sample_rate_hz,
                                         size_t chunk_samples) {
  return RunAsrEngine(omni_proc, audio_samples, sample_rate_hz, chunk_samples);
}

absl::StatusOr<AsrRunMetrics> RunMediaPipeAsr(
    AudioTranscriber& mp_transcriber, absl::Span<const float> audio_samples,
    int sample_rate_hz, size_t chunk_samples) {
  return RunAsrEngine(mp_transcriber, audio_samples, sample_rate_hz,
                      chunk_samples);
}

absl::StatusOr<std::unique_ptr<OmniEngine>> CreateOmniKokoroEngine(
    absl::string_view model_path, absl::string_view voice,
    absl::string_view locale) {
  OmniEngine::Settings settings;
  settings.model_path = std::string(model_path);
  settings.cache_dir = GetBenchmarkCacheDir();
  settings.num_threads = kDefaultCpuThreads;
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  ::litert::lm::proto::Text2SpeechSessionConfig cfg;
  cfg.set_language(locale);
  cfg.set_voice(voice);
  ::litert::lm::proto::Text2SpeechMetadata meta;
  auto* kokoro_cfg = meta.mutable_tts_model_type()->mutable_kokoro();
  kokoro_cfg->set_voice_name(voice);
  kokoro_cfg->set_language(locale);
  return OmniEngine::Create("kokoro", settings, &cfg, &meta);
}

absl::StatusOr<std::unique_ptr<TextToSpeech>> CreateMediaPipeKokoroTts(
    absl::string_view model_path, absl::string_view default_voice,
    absl::string_view default_locale) {
  absl::SetFlag(&FLAGS_mediapipe_tts_use_staged_scheduler, false);
  ::mediapipe::tasks::audio::text_to_speech::TextToSpeechOptions opts;
  opts.base_options.model_asset_path = std::string(model_path);
  opts.base_options.delegate = tasks::core::BaseOptions::Delegate::CPU;
  opts.base_options.delegate_options =
      tasks::core::BaseOptions::CpuOptions{.num_threads = kDefaultCpuThreads};
  opts.cache_directory = GetBenchmarkCacheDir();
  opts.default_voice_id = std::string(default_voice);
  opts.default_locale = std::string(default_locale);
  return ::mediapipe::tasks::audio::text_to_speech::CreateTextToSpeech(opts);
}

absl::StatusOr<std::vector<float>> RunOmniKokoroSynthesize(
    OmniEngine& omni_engine, absl::string_view text) {
  ABSL_ASSIGN_OR_RETURN(
      auto session, omni_engine.CreateSession(
                        std::make_unique<::litert::omni::PushInputSource>()));
  std::unique_ptr<MultiStagedSession> owned(
      static_cast<MultiStagedSession*>(session.release()));
  auto* src = static_cast<::litert::omni::tts::StreamTextSource*>(
      owned->stages()[0].get());
  ABSL_RETURN_IF_ERROR(src->PushText(text));
  src->Finish();

  std::vector<float> samples;
  auto append_if_audio = [&](const MultiStagedSession::Output& out) {
    if (const auto* audio = std::get_if<AudioOutput>(&out)) {
      samples.insert(samples.end(), audio->pcm_samples.begin(),
                     audio->pcm_samples.end());
    }
  };
  while (true) {
    auto out_or = owned->ProcessNext();
    if (!out_or.ok()) {
      if (absl::IsOutOfRange(out_or.status()) ||
          absl::IsNotFound(out_or.status())) {
        break;
      }
      return out_or.status();
    }
    append_if_audio(*out_or);
    if (std::holds_alternative<::litert::omni::EndOfOutput>(*out_or)) break;
  }
  if (auto flush_or = owned->Flush(); flush_or.ok()) {
    append_if_audio(*flush_or);
  }
  return samples;
}

}  // namespace mediapipe::tasks::audio::benchmarks
